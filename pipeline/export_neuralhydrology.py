"""Modelling frame -> NeuralHydrology GenericDataset, for the EA-LSTM (models/ealstm.py).

Run (after `python -m pipeline.build_dataset --city coimbra`):

    python -m pipeline.export_neuralhydrology --city coimbra

LAYOUT (neuralhydrology 1.13, datasetzoo/genericdataset.py - read, not guessed)
-------------------------------------------------------------------------------
  data/processed/nh/
    time_series/<reach_id>.nc     one netCDF per reach; ONE coordinate named `date`,
                                  continuous daily; every dynamic input and both targets
                                  as float32 variables. Missing = NaN (GenericDataset
                                  only recognises NaN as missing - never a sentinel).
    attributes/attributes.csv     one row per reach; first column is the reach id (read
                                  as str and used as the index). GenericDataset reads
                                  EVERY csv in this folder, so nothing else goes here.
    train_reaches.txt             observable reaches with >= 1 OK target in the training
                                  period, minus the spatial holdout. Train + early stopping.
    spatial_holdout.txt           ~20 % of observable reaches with targets. Never trained
                                  on, never used for early stopping.
    all_reaches.txt               every reach the model can simulate (complete statics),
                                  observable or not. The test / inference basin list.
    export_manifest.json          counts, NaN fractions, static-attribute decisions,
                                  excluded reaches and why.

TARGETS
-------
  turbidity_log1p  = log1p(turbidity_proxy)   (config/modelling.yaml ealstm.targets)
  ndci             = ndci
Both are NaN on every reach-date without an OK Sentinel-2 observation - CLOUD,
NO_WATER_PIXELS, OUT_OF_RANGE, or no acquisition at all. Never filled. The flag itself
is written alongside as `quality_code` (see QUALITY_CODES) so the netCDF says WHY a
target is missing. The export asserts, per reach and per target,

    NaN target count == non-OK flagged dates + dates with no observation row

and that the OK dates in the frame are exactly the OK dates in `observations`.

Unobservable reaches are exported too: all-NaN targets, so the model can simulate them.

STATIC ATTRIBUTES
-----------------
NeuralHydrology rejects an attribute with zero or NaN spread, but it does NOT reject a
NaN value on one basin - that NaN would flow into the input gate and poison the
reach's output. So the export resolves the requested list first (resolve_static_attributes):
an attribute is dropped if it is NULL on too many reaches, constant, or a land-cover
proxy under the baseline's drop_when_proxy policy; a reach with a NULL in a kept
attribute is left out of every basin list. Both decisions are recorded, never filled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.config import load_config
from core.logging import get_logger, stage
from core.settings import CONFIG_DIR, REPO_ROOT
from pipeline.build_dataset import PROCESSED_DIR, frame_meta_path

log = get_logger(__name__)

# quality_code in the netCDF. 0 = no observation row at all for that reach-date.
QUALITY_CODES: dict[str, int] = {
    "NO_ROW": 0,
    "OK": 1,
    "CLOUD": 2,
    "NO_WATER_PIXELS": 3,
    "OUT_OF_RANGE": 4,
}
# Inputs a pure driver -> state simulator must never see: the reach's identity, and any
# observation of the state (lagged, as-of, upstream). tests/test_ealstm_export.py.
FORBIDDEN_INPUT_MARKERS = ("reach_id", "upstream_state", "_asof", "lag", "obs_")
TRANSFORMS = {"log1p": np.log1p, "identity": lambda x: x}
INVERSE_TRANSFORMS = {"log1p": np.expm1, "identity": lambda x: x}


class ExportError(RuntimeError):
    """The export would be wrong or empty. Raised, never papered over."""


@dataclass(frozen=True)
class TargetSpec:
    variable: str  # frame / observations name, e.g. turbidity_proxy
    nh_name: str  # netCDF + NH config name, e.g. turbidity_log1p
    transform: str  # log1p | identity


@dataclass(frozen=True)
class ExportSettings:
    nh_dir: Path
    targets: tuple[TargetSpec, ...]
    dynamic_inputs: tuple[str, ...]
    static_attributes: tuple[str, ...]
    train_start: date
    train_end: date
    holdout_fraction: float
    holdout_salt: str
    max_null_reach_fraction: float
    drop_when_proxy: tuple[str, ...]

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> ExportSettings:
        cfg = cfg or load_config("modelling")
        e = cfg["ealstm"]
        targets = tuple(
            TargetSpec(var, str(t["nh_name"]), str(t["transform"]))
            for var, t in e["targets"].items()
        )
        for t in targets:
            if t.transform not in TRANSFORMS:
                raise ValueError(f"{t.variable}: unknown transform {t.transform!r}")
        nh_cfgs = {t.variable: load_nh_config(t.variable) for t in targets}
        dyn, stat = _agreed_inputs(nh_cfgs)
        first = next(iter(nh_cfgs.values()))
        return cls(
            nh_dir=REPO_ROOT / e["nh_dir"],
            targets=targets,
            dynamic_inputs=dyn,
            static_attributes=stat,
            train_start=_nh_date(first["train_start_date"]),
            train_end=_nh_date(first["train_end_date"]),
            holdout_fraction=float(e["spatial_holdout_fraction"]),
            holdout_salt=str(e["spatial_holdout_salt"]),
            max_null_reach_fraction=float(e["max_null_reach_fraction"]),
            drop_when_proxy=tuple(cfg["baseline_gbm"].get("drop_when_proxy") or ()),
        )


def nh_config_path(variable: str) -> Path:
    return CONFIG_DIR / f"ealstm_{variable}.yml"


def load_nh_config(variable: str) -> dict[str, Any]:
    import yaml

    path = nh_config_path(variable)
    if not path.exists():
        raise FileNotFoundError(f"missing NeuralHydrology config {path}")
    cfg: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cfg


def _nh_date(value: str) -> date:
    return datetime.strptime(str(value), "%d/%m/%Y").date()


def _agreed_inputs(
    nh_cfgs: dict[str, dict[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """One export serves every target, so every target's NH config must list the same
    inputs - and none of them may be reach identity or an observation of the state."""
    dyns = {v: tuple(c["dynamic_inputs"]) for v, c in nh_cfgs.items()}
    stats = {v: tuple(c.get("static_attributes") or ()) for v, c in nh_cfgs.items()}
    if len(set(dyns.values())) != 1 or len(set(stats.values())) != 1:
        raise ValueError(f"EA-LSTM configs disagree on inputs: {dyns} / {stats}")
    for v, c in nh_cfgs.items():
        check_simulator_inputs(dyns[v], stats[v], c)
    return next(iter(dyns.values())), next(iter(stats.values()))


def check_simulator_inputs(
    dynamic: Sequence[str], static: Sequence[str], nh_cfg: dict[str, Any]
) -> None:
    """Raise if any input carries reach identity or an observation of the state."""
    bad = [f for f in [*dynamic, *static] if any(m in f for m in FORBIDDEN_INPUT_MARKERS)]
    if bad:
        raise ValueError(f"EA-LSTM inputs must be drivers and statics only; found {bad}")
    if nh_cfg.get("use_basin_id_encoding"):
        raise ValueError("use_basin_id_encoding must be false - reach identity via statics only")
    targets = set(nh_cfg.get("target_variables") or ())
    leak = targets & (set(dynamic) | set(static))
    if leak:
        raise ValueError(f"target(s) {sorted(leak)} are also inputs")
    for key in ("lagged_features", "autoregressive_inputs", "evolving_attributes"):
        if nh_cfg.get(key):
            raise ValueError(f"{key} must be empty for a driver -> state simulator")


# ---------------------------------------------------------------------------
# pure pieces
# ---------------------------------------------------------------------------
def select_holdout(reach_ids: Iterable[str], fraction: float, salt: str) -> list[str]:
    """Deterministic, unhand-picked: order by sha256(salt + id), take round(fraction*n),
    at least one if there are two or more candidates."""
    ids = sorted(set(reach_ids))
    if not ids or fraction <= 0:
        return []
    k = int(round(fraction * len(ids)))
    k = max(k, 1) if len(ids) >= 2 else 0
    ranked = sorted(ids, key=lambda r: hashlib.sha256(f"{salt}{r}".encode()).hexdigest())
    return sorted(ranked[:k])


def resolve_static_attributes(
    attrs: pd.DataFrame,
    requested: Sequence[str],
    *,
    max_null_fraction: float,
    drop_when_proxy: Sequence[str] = (),
    static_flags: dict[str, dict[str, int]] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """(kept, {dropped: reason}). attrs: one row per reach, indexed by reach_id.

    Dropped: missing column; NULL on more than max_null_fraction of reaches; zero spread
    over the reaches that have a value; or listed in drop_when_proxy while any reach's
    value is a PROXY_ (a duplicate of another feature - the baseline's policy)."""
    flags = static_flags or {}
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for a in requested:
        if a not in attrs:
            dropped[a] = "NOT_IN_FRAME"
            continue
        col = pd.to_numeric(attrs[a], errors="raise").astype("float64")
        null = float(col.isna().mean())
        flag_note = ", ".join(f"{k} x{n}" for k, n in sorted(flags.get(a, {}).items()))
        note = f" [{flag_note}]" if flag_note else ""
        if null > max_null_fraction:
            dropped[a] = f"NULL on {null:.1%} of reaches (> {max_null_fraction:.0%}){note}"
            continue
        values = col.dropna()
        if values.nunique() <= 1:
            dropped[a] = f"CONSTANT ({values.iloc[0] if len(values) else 'no value'}){note}"
            continue
        proxies = {k: n for k, n in flags.get(a, {}).items() if k.startswith("PROXY_")}
        if a in drop_when_proxy and proxies:
            dropped[a] = f"PROXY duplicate under baseline_gbm.drop_when_proxy{note}"
            continue
        kept.append(a)
    return kept, dropped


def ok_observations_from_frame(rows: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    """date -> value per variable for one reach's OK observations. An OK observation on
    day t is exactly the frame's as-of value with age 0 on day t."""
    today = rows[rows["obs_asof_age_days"] == 0]
    out = pd.DataFrame(
        {v: today[f"{v}_asof"].to_numpy(dtype="float64") for v in variables},
        index=pd.DatetimeIndex(pd.to_datetime(today["date"]), name="date"),
    )
    return out


def reach_timeseries(
    rows: pd.DataFrame,
    flags: pd.DataFrame | None,
    dynamic_inputs: Sequence[str],
    targets: Sequence[TargetSpec],
) -> pd.DataFrame:
    """One reach's NH time series: continuous daily `date` index, dynamic inputs,
    transformed targets (NaN unless OK), quality_code. Pure.

    rows   the reach's frame rows (date, dynamic inputs, <var>_asof, obs_asof_age_days)
    flags  the reach's observation rows: obs_date, quality_flag (any source S2); None or
           empty for a reach that was never observed
    """
    rows = rows.sort_values("date")
    idx = pd.DatetimeIndex(pd.to_datetime(rows["date"]), name="date")
    if idx.has_duplicates:
        raise ExportError("duplicate dates in the frame for one reach")
    full = pd.date_range(idx[0], idx[-1], freq="D", name="date")
    if len(full) != len(idx):
        raise ExportError(f"frame is not a continuous daily series ({len(idx)} of {len(full)})")
    out = pd.DataFrame(index=idx)
    for col in dynamic_inputs:
        if col not in rows:
            raise ExportError(f"dynamic input {col!r} is not in the frame")
        out[col] = rows[col].to_numpy(dtype="float64")

    ok = ok_observations_from_frame(rows, [t.variable for t in targets])
    for t in targets:
        values = ok[t.variable].reindex(idx)  # NaN wherever there is no OK observation
        if t.transform == "log1p" and (values < -1).any():
            raise ExportError(f"{t.variable} < -1 cannot be log1p-transformed")
        out[t.nh_name] = TRANSFORMS[t.transform](values.to_numpy(dtype="float64"))

    code = np.full(len(idx), QUALITY_CODES["NO_ROW"], dtype="int8")
    if flags is not None and not flags.empty:
        f = flags.assign(d=pd.to_datetime(flags["obs_date"]))
        if f["d"].duplicated().any():
            raise ExportError("more than one observation row per reach-date")
        unknown = set(f["quality_flag"]) - set(QUALITY_CODES)
        if unknown:
            raise ExportError(f"unknown quality_flag(s) {sorted(unknown)}")
        mapped = f.set_index("d")["quality_flag"].map(QUALITY_CODES).reindex(idx)
        code = np.where(mapped.notna(), mapped.to_numpy(dtype="float64"), 0).astype("int8")
    out["quality_code"] = code
    return out


def check_target_nans(
    series: pd.DataFrame, targets: Sequence[TargetSpec], reach_id: str
) -> dict[str, int]:
    """The invariant: per target, NaN count == non-OK flagged dates + dates with no row,
    and the OK dates are exactly the non-NaN dates. Raises ExportError otherwise."""
    code = series["quality_code"].to_numpy()
    non_ok = int(
        np.isin(code, [c for k, c in QUALITY_CODES.items() if k not in ("OK", "NO_ROW")]).sum()
    )
    no_row = int((code == QUALITY_CODES["NO_ROW"]).sum())
    is_ok = code == QUALITY_CODES["OK"]
    counts = {"days": len(series), "ok": int(is_ok.sum()), "non_ok": non_ok, "no_row": no_row}
    for t in targets:
        nan = series[t.nh_name].isna().to_numpy()
        if int(nan.sum()) != non_ok + no_row:
            raise ExportError(
                f"{reach_id} {t.nh_name}: {int(nan.sum())} NaN targets but {non_ok} non-OK "
                f"+ {no_row} missing reach-dates - a target was filled or an OK value lost"
            )
        if not np.array_equal(~nan, is_ok):
            raise ExportError(f"{reach_id} {t.nh_name}: non-NaN targets are not the OK dates")
        counts[f"nan_{t.nh_name}"] = int(nan.sum())
    return counts


def build_attributes(frame: pd.DataFrame, static_attributes: Sequence[str]) -> pd.DataFrame:
    """One row per reach (index reach_id), the requested statics. A static that varies
    within a reach in the frame is an error, not something to average."""
    cols = [c for c in static_attributes if c in frame]
    g = frame[["reach_id", *cols]].assign(reach_id=frame["reach_id"].astype(str))
    per = g.groupby("reach_id")[cols]
    distinct = per.nunique(dropna=False)
    varying = [c for c in cols if (distinct[c] > 1).any()]
    if varying:
        raise ExportError(f"static attributes vary within a reach: {varying}")
    attrs = per.first().astype("float64")
    for c in static_attributes:
        if c not in attrs:
            attrs[c] = np.nan
    attrs.index.name = "reach_id"
    return attrs[list(static_attributes)]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_observation_flags(city: str) -> pd.DataFrame:
    from sqlalchemy import text

    from core.db import session_scope

    with session_scope() as session:
        flags = pd.read_sql(
            text(
                "SELECT o.reach_id, o.obs_date, o.quality_flag FROM observations o "
                "JOIN reaches r USING (reach_id) WHERE r.city = :city AND o.source = 'S2'"
            ),
            session.connection(),
            params={"city": city},
        )
    if flags.empty:
        raise ExportError(f"no S2 observation rows for {city} - run pipeline.l1_satellite")
    return flags


def _write_netcdf(series: pd.DataFrame, path: Path, reach_id: str) -> None:
    import xarray as xr

    data = series.copy()
    floats = [c for c in data.columns if c != "quality_code"]
    data[floats] = data[floats].astype("float32")
    ds = xr.Dataset.from_dataframe(data)
    ds["quality_code"].attrs["flag_meanings"] = " ".join(QUALITY_CODES)
    ds["quality_code"].attrs["flag_values"] = list(QUALITY_CODES.values())
    ds.attrs["reach_id"] = reach_id
    ds.attrs["note"] = "targets are NaN on every reach-date without an OK S2 observation"
    tmp = path.with_suffix(".nc.part")
    ds.to_netcdf(tmp)
    ds.close()
    tmp.replace(path)


def _write_list(path: Path, ids: Iterable[str]) -> int:
    ids = sorted(ids)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return len(ids)


def export(city: str, *, es: ExportSettings | None = None) -> dict[str, Any]:
    es = es or ExportSettings.from_config()
    frame_file = PROCESSED_DIR / f"frame_{city}.parquet"
    if not frame_file.exists():
        raise FileNotFoundError(f"{frame_file} missing - run `python -m pipeline.build_dataset`")
    variables = [t.variable for t in es.targets]
    cols = [
        "reach_id",
        "date",
        "observable",
        "obs_asof_age_days",
        *[f"{v}_asof" for v in variables],
        *es.dynamic_inputs,
        *es.static_attributes,
    ]
    meta = frame_meta_path(city)
    static_flags = json.loads(meta.read_text())["static_flags"] if meta.exists() else {}

    with stage(log, "export_neuralhydrology", city=city) as counters:
        frame = pd.read_parquet(frame_file, columns=list(dict.fromkeys(cols)))
        if frame.empty:
            raise ExportError("empty modelling frame")
        flags = load_observation_flags(city)
        flags["obs_date"] = pd.to_datetime(flags["obs_date"])
        first, last = pd.to_datetime(frame["date"]).min(), pd.to_datetime(frame["date"]).max()
        outside = (flags["obs_date"] < first) | (flags["obs_date"] > last)
        if outside.any():
            counters.drop(int(outside.sum()), "OBS_OUTSIDE_FRAME_DATES")
        flags = flags[~outside]
        flags_by_reach = {str(r): g for r, g in flags.groupby("reach_id")}

        attrs = build_attributes(frame, es.static_attributes)
        kept, dropped = resolve_static_attributes(
            attrs,
            es.static_attributes,
            max_null_fraction=es.max_null_reach_fraction,
            drop_when_proxy=es.drop_when_proxy,
            static_flags=static_flags,
        )
        if not kept:
            raise ExportError(f"every static attribute was dropped: {dropped}")
        incomplete = attrs[kept].isna()
        excluded = {
            str(r): sorted(incomplete.columns[incomplete.loc[r]].tolist())
            for r in attrs.index[incomplete.any(axis=1)]
        }

        ts_dir = es.nh_dir / "time_series"
        at_dir = es.nh_dir / "attributes"
        ts_dir.mkdir(parents=True, exist_ok=True)
        at_dir.mkdir(parents=True, exist_ok=True)
        for stale in ts_dir.glob("*.nc"):
            stale.unlink()

        per_reach: dict[str, dict[str, Any]] = {}
        observable = frame.groupby(frame["reach_id"].astype(str))["observable"].first()
        train_lo, train_hi = pd.Timestamp(es.train_start), pd.Timestamp(es.train_end)
        for rid, rows in frame.groupby(frame["reach_id"].astype(str), observed=True):
            fl = flags_by_reach.get(rid)
            series = reach_timeseries(rows, fl, es.dynamic_inputs, es.targets)
            n_flag_ok = 0 if fl is None else int((fl["quality_flag"] == "OK").sum())
            counts = check_target_nans(series, es.targets, rid)
            if counts["ok"] != n_flag_ok:
                raise ExportError(
                    f"{rid}: frame has {counts['ok']} OK dates, observations has {n_flag_ok}"
                    " - the frame is stale; rebuild it"
                )
            in_train = (series.index >= train_lo) & (series.index <= train_hi)
            first_target = es.targets[0].nh_name
            counts["ok_train"] = int(series.loc[in_train, first_target].notna().sum())
            counts["observable"] = (
                bool(observable.get(rid)) if pd.notna(observable.get(rid)) else None
            )
            counts["dynamic_nan"] = int(series[list(es.dynamic_inputs)].isna().any(axis=1).sum())
            per_reach[rid] = counts
            _write_netcdf(series, ts_dir / f"{rid}.nc", rid)

        attrs.to_csv(at_dir / "attributes.csv", float_format="%.8g")

        simulatable = sorted(r for r in per_reach if r not in excluded)
        with_targets = [
            r for r in simulatable if per_reach[r]["observable"] and per_reach[r]["ok"] > 0
        ]
        holdout = select_holdout(with_targets, es.holdout_fraction, es.holdout_salt)
        train = [r for r in with_targets if r not in holdout and per_reach[r]["ok_train"] > 0]
        if not train:
            raise ExportError("no observable reach has a training-period target")
        n_train = _write_list(es.nh_dir / "train_reaches.txt", train)
        n_hold = _write_list(es.nh_dir / "spatial_holdout.txt", holdout)
        n_all = _write_list(es.nh_dir / "all_reaches.txt", simulatable)

        days = sum(c["days"] for c in per_reach.values())
        nan_fraction = {
            t.nh_name: {
                "all_reaches": sum(c[f"nan_{t.nh_name}"] for c in per_reach.values()) / days,
                "observable_reaches": (
                    sum(c[f"nan_{t.nh_name}"] for r, c in per_reach.items() if c["observable"])
                    / max(1, sum(c["days"] for c in per_reach.values() if c["observable"]))
                ),
            }
            for t in es.targets
        }
        manifest = {
            "city": city,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "frame": str(frame_file.relative_to(REPO_ROOT)),
            "dates": [str(first.date()), str(last.date())],
            "targets": {
                t.nh_name: {"variable": t.variable, "transform": t.transform} for t in es.targets
            },
            "dynamic_inputs": list(es.dynamic_inputs),
            "static_attributes_requested": list(es.static_attributes),
            "static_attributes_used": kept,
            "static_attributes_dropped": dropped,
            "excluded_reaches_null_static": excluded,
            "quality_codes": QUALITY_CODES,
            "rows_in": int(len(frame)),
            "rows_out": int(days),
            "reaches": len(per_reach),
            "nan_fraction": nan_fraction,
            "basin_lists": {
                "train_reaches": train,
                "spatial_holdout": holdout,
                "all_reaches": n_all,
            },
            "per_reach": per_reach,
        }
        (es.nh_dir / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        counters.record(
            rows_in=len(frame),
            rows_out=days,
            reaches=len(per_reach),
            train_reaches=n_train,
            holdout_reaches=n_hold,
            simulatable_reaches=n_all,
            excluded_reaches=len(excluded),
            statics_used=kept,
            statics_dropped=list(dropped),
            nan_fraction={k: round(v["all_reaches"], 4) for k, v in nan_fraction.items()},
        )
        for name, why in dropped.items():
            log.warning("export_nh.static_dropped", attribute=name, reason=why)
        for rid, attrs_missing in excluded.items():
            log.warning("export_nh.reach_excluded", reach_id=rid, null_statics=attrs_missing)
        log.info("export_nh.training_reaches", n=n_train, reaches=train)
        log.info("export_nh.holdout_reaches", n=n_hold, reaches=holdout)
        for t, fr in nan_fraction.items():
            log.info(
                "export_nh.target_nan_fraction", target=t, **{k: round(v, 4) for k, v in fr.items()}
            )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the frame for NeuralHydrology")
    parser.add_argument("--city", default="coimbra")
    args = parser.parse_args(argv)
    m = export(args.city)
    summary = {
        k: m[k]
        for k in (
            "reaches",
            "rows_out",
            "static_attributes_used",
            "static_attributes_dropped",
            "excluded_reaches_null_static",
            "nan_fraction",
        )
    }
    summary["train_reaches"] = len(m["basin_lists"]["train_reaches"])
    summary["spatial_holdout"] = m["basin_lists"]["spatial_holdout"]
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
