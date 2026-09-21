"""Entity-Aware LSTM with a CMAL head - the driver -> state simulator (P5.1).

Kratzert, Klotz, Shalev, Klambauer, Hochreiter & Nearing (2019), "Towards learning
universal, regional, and local hydrological behaviors via machine learning applied to
large-sample datasets", HESS 23:5089 - the EA-LSTM: the input gate is a function of the
static attributes only, so catchment character modulates which weather signals are
written into the cell state. Klotz et al. (2022), HESS 26:1673 - the CMAL head
(models/cmal.py). Trained through NeuralHydrology 1.13 (Kratzert et al. 2022, JOSS).

Run:

    python -m models.ealstm smoke                                # 1 epoch, 3 reaches, CPU
    python -m models.ealstm train --target all --device cuda:0   # 2 targets x 5 seeds
    python -m models.ealstm status                               # what is trained
    python -m models.ealstm hindcast --city coimbra              # simulations for eval

WHAT IT IS
----------
A pure driver -> state SIMULATOR. Inputs: 15 daily weather drivers over the previous 365
days (config/ealstm_<target>.yml) and the reach's static catchment attributes. It never
sees reach_id (use_basin_id_encoding: false) nor any observation of the state - lagged,
as-of or upstream (pipeline.export_neuralhydrology.check_simulator_inputs). Reach
identity comes only through the statics, which is what lets it simulate the driver-only
reaches and what makes perturbing a static (the scenario engine) mean something.
Satellite information enters afterwards, through models/assimilation.py.

One NeuralHydrology run per (target, seed); targets log1p(turbidity_proxy) and ndci
(config/modelling.yaml ealstm.targets - transforms documented there). The served
distribution for a target is the equal-weight mixture of the 5 seeds' CMAL mixtures
(models/cmal.ensemble); quantiles by CDF inversion, never by sampling.

EPOCH SELECTION
---------------
NeuralHydrology stops early on the validation loss (training reaches, 2024) but keeps the
weights of EVERY epoch and does not log the validation loss unless metrics are
configured. select_epoch() recomputes the same quantity - the CMAL negative
log-likelihood in NH's normalised target space, over the training reaches' OK 2024
targets - for every saved epoch, and the lowest one is served. The spatial-holdout
reaches are never part of it. NH 1.13 also switches on ReduceLROnPlateau whenever
early_stopping is on (utils/config.py dynamic_learning_rate) - the fixed learning-rate
schedule in the YAML is then ignored; recorded in selection.json.

FORECAST MODE
-------------
The forecast for issue date t and horizon h is the simulation for target date t+h: a
365-day driver window ending at t+h. Drivers up to t are the archive. Drivers for
t+1..t+h are:
  hindcast   the archive (observed) weather - every metric computed this way carries
             HINDCAST_LABEL, in metrics.json and not only in prose
  live       drivers_daily rows with source='FORECAST' (the Open-Meteo forecast endpoint,
             pipeline.l2_drivers), passed in as `future_drivers`
Under hindcast weather the simulation for a target date does not depend on t, so it is
computed once per (reach, date) and reused for every horizon that points at that date.

OUTPUTS
-------
  artifacts/ealstm/runs/<target>/seed<k>/<nh run>/   NeuralHydrology run (config, scaler,
                                                     every epoch's weights) + selection.json
  artifacts/ealstm/<target>__seed<k>.json            pointer: run dir + served epoch
  data/processed/ealstm_sim_<city>.parquet           hindcast simulations at OK obs dates
  data/processed/ealstm_run_<city>.json              run manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.config import load_config
from core.logging import get_logger, stage
from core.settings import REPO_ROOT
from models import cmal
from pipeline.export_neuralhydrology import (
    INVERSE_TRANSFORMS,
    TRANSFORMS,
    check_simulator_inputs,
    load_nh_config,
)

log = get_logger(__name__)

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "ealstm"
RUNS_DIR = ARTIFACT_DIR / "runs"
HINDCAST_LABEL = "hindcast with observed weather - upper bound on live skill"
WEATHER_HINDCAST = "ARCHIVE_HINDCAST"
WEATHER_LIVE = "FORECAST"
LEVELS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


class NotTrained(FileNotFoundError):
    """A (target, seed) member has no trained run."""


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetInfo:
    variable: str  # turbidity_proxy | ndci
    nh_name: str  # turbidity_log1p | ndci
    transform: str  # log1p | identity

    def forward(self, x: Any) -> Any:
        return TRANSFORMS[self.transform](x)

    def inverse(self, x: Any) -> Any:
        return INVERSE_TRANSFORMS[self.transform](x)


@dataclass(frozen=True)
class EALSTMSettings:
    version: str
    nh_dir: Path
    targets: dict[str, TargetInfo]
    seeds: tuple[int, ...]
    batch_size: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> EALSTMSettings:
        cfg = cfg or load_config("modelling")
        e = cfg["ealstm"]
        return cls(
            version=str(e["version"]),
            nh_dir=REPO_ROOT / e["nh_dir"],
            targets={
                v: TargetInfo(v, str(t["nh_name"]), str(t["transform"]))
                for v, t in e["targets"].items()
            },
            seeds=tuple(int(s) for s in e["seeds"]),
            batch_size=int(e["inference_batch_size"]),
        )

    def target(self, variable: str) -> TargetInfo:
        if variable not in self.targets:
            raise ValueError(f"unknown target {variable!r}; have {sorted(self.targets)}")
        return self.targets[variable]


def load_manifest(es: EALSTMSettings) -> dict[str, Any]:
    path = es.nh_dir / "export_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m pipeline.export_neuralhydrology`")
    m: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return m


def _nh(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _from_nh(s: str) -> date:
    return datetime.strptime(str(s), "%d/%m/%Y").date()


def run_config(
    variable: str,
    seed: int,
    *,
    es: EALSTMSettings,
    manifest: dict[str, Any],
    device: str,
    run_dir: Path,
    epochs: int | None = None,
    train_basin_file: Path | None = None,
) -> dict[str, Any]:
    """The NH config for one (target, seed): config/ealstm_<target>.yml with only the
    per-run keys overridden (see that file's header). Pure."""
    cfg = dict(load_nh_config(variable))
    used = list(manifest["static_attributes_used"])
    requested = list(cfg.get("static_attributes") or [])
    unknown = set(used) - set(requested)
    if unknown:
        raise ValueError(f"export used statics the config never asked for: {sorted(unknown)}")
    nh_dir = es.nh_dir.resolve()
    cfg.update(
        seed=int(seed),
        device=device,
        experiment_name=f"ealstm_{variable}_seed{seed}",
        run_dir=str(run_dir.resolve()),
        data_dir=str(nh_dir),
        train_basin_file=str((train_basin_file or nh_dir / "train_reaches.txt").resolve()),
        validation_basin_file=str((train_basin_file or nh_dir / "train_reaches.txt").resolve()),
        test_basin_file=str((nh_dir / "all_reaches.txt").resolve()),
        test_end_date=_nh(date.fromisoformat(manifest["dates"][1])),
        static_attributes=used,
    )
    if epochs is not None:
        cfg["epochs"] = int(epochs)
    check_run_config(cfg, es.target(variable))
    return cfg


def check_run_config(cfg: dict[str, Any], target: TargetInfo) -> None:
    """The things P5.1 fixes, asserted rather than trusted."""
    expect = {
        "model": "ealstm",
        "head": "cmal",
        "loss": "CMALLoss",
        "hidden_size": 64,
        "output_dropout": 0.4,
        "seq_length": 365,
        "predict_last_n": 1,
        "use_basin_id_encoding": False,
    }
    bad = {k: (cfg.get(k), v) for k, v in expect.items() if cfg.get(k) != v}
    if bad:
        raise ValueError(f"EA-LSTM config deviates from the P5.1 spec: {bad}")
    if list(cfg["target_variables"]) != [target.nh_name]:
        raise ValueError(f"target_variables {cfg['target_variables']} != [{target.nh_name}]")
    check_simulator_inputs(cfg["dynamic_inputs"], cfg["static_attributes"], cfg)
    tr = (_from_nh(cfg["train_start_date"]), _from_nh(cfg["train_end_date"]))
    va = (_from_nh(cfg["validation_start_date"]), _from_nh(cfg["validation_end_date"]))
    te = _from_nh(cfg["test_start_date"])
    if not (tr[0] <= tr[1] < va[0] <= va[1] < te):
        raise ValueError(f"periods overlap or are out of order: train {tr} val {va} test {te}")


# ---------------------------------------------------------------------------
# training (NeuralHydrology)
# ---------------------------------------------------------------------------
def member_root(variable: str, seed: int, root: Path = RUNS_DIR) -> Path:
    return root / variable / f"seed{seed}"


def pointer_path(variable: str, seed: int, root: Path = ARTIFACT_DIR) -> Path:
    return root / f"{variable}__seed{seed}.json"


def _nh_run_dir(parent: Path) -> Path:
    """NH writes <run_dir>/<experiment>_<ddmm_HHMMSS>/; the newest one with a config."""
    runs = sorted(
        (p for p in parent.iterdir() if p.is_dir() and (p / "config.yml").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        raise NotTrained(f"no NeuralHydrology run under {parent}")
    return runs[-1]


def train_member(
    variable: str,
    seed: int,
    *,
    device: str = "cpu",
    epochs: int | None = None,
    es: EALSTMSettings | None = None,
    runs_root: Path = RUNS_DIR,
    pointer_root: Path = ARTIFACT_DIR,
    train_basin_file: Path | None = None,
) -> dict[str, Any]:
    """Train one (target, seed), select its epoch, write the pointer. Returns the
    selection record."""
    import yaml
    from neuralhydrology.training.train import start_training
    from neuralhydrology.utils.config import Config

    es = es or EALSTMSettings.from_config()
    manifest = load_manifest(es)
    parent = member_root(variable, seed, runs_root)
    if parent.exists():
        shutil.rmtree(parent)  # a retrain replaces the member; never mix two runs
    parent.mkdir(parents=True)
    cfg = run_config(
        variable,
        seed,
        es=es,
        manifest=manifest,
        device=device,
        run_dir=parent,
        epochs=epochs,
        train_basin_file=train_basin_file,
    )
    cfg_path = parent / "config_in.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log.info("ealstm.train_start", target=variable, seed=seed, device=device, epochs=cfg["epochs"])
    t0 = datetime.now(UTC)
    start_training(Config(cfg_path))
    run_dir = _nh_run_dir(parent)
    sel = select_epoch(run_dir, variable, es=es, device="cpu")
    sel["train_seconds"] = round((datetime.now(UTC) - t0).total_seconds(), 1)
    sel["device"] = device
    (run_dir / "selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    pointer = {
        "target": variable,
        "seed": seed,
        "run_dir": run_dir.relative_to(pointer_root).as_posix(),
        "epoch": sel["best_epoch"],
        "trained_at": t0.isoformat(timespec="seconds"),
    }
    pointer_path(variable, seed, pointer_root).write_text(json.dumps(pointer, indent=2))
    log.info(
        "ealstm.train_done",
        target=variable,
        seed=seed,
        best_epoch=sel["best_epoch"],
        epochs_run=sel["epochs_run"],
        val_nll=sel["val_nll"][str(sel["best_epoch"])],
        seconds=sel["train_seconds"],
    )
    return sel


# ---------------------------------------------------------------------------
# loading a trained member
# ---------------------------------------------------------------------------
@dataclass
class Member:
    variable: str
    seed: int
    run_dir: Path
    epoch: int
    model: Any  # neuralhydrology EALSTM (torch.nn.Module), eval mode
    dynamic_inputs: tuple[str, ...]
    static_attributes: tuple[str, ...]  # NH's order: alphabetical
    dyn_center: np.ndarray
    dyn_scale: np.ndarray
    stat_mean: np.ndarray
    stat_std: np.ndarray
    y_center: float
    y_scale: float
    seq_length: int

    @property
    def tag(self) -> str:
        return f"{self.run_dir.name}@{self.epoch}"


def _epoch_file(run_dir: Path, epoch: int) -> Path:
    return run_dir / f"model_epoch{epoch:03d}.pt"


def saved_epochs(run_dir: Path) -> list[int]:
    return sorted(int(p.stem[-3:]) for p in run_dir.glob("model_epoch*.pt"))


def load_member(run_dir: Path, epoch: int, variable: str, seed: int) -> Member:
    import torch
    from neuralhydrology.datautils.utils import load_scaler
    from neuralhydrology.modelzoo import get_model
    from neuralhydrology.utils.config import Config

    cfg = Config(run_dir / "config.yml")
    cfg.device = "cpu"
    model = get_model(cfg)
    state = torch.load(_epoch_file(run_dir, epoch), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    scaler = load_scaler(run_dir)
    dyn = tuple(cfg.dynamic_inputs)
    stats = tuple(sorted(cfg.static_attributes))  # basedataset sorts attribute columns
    (target,) = cfg.target_variables
    center, scale = scaler["xarray_feature_center"], scaler["xarray_feature_scale"]
    return Member(
        variable=variable,
        seed=seed,
        run_dir=run_dir,
        epoch=epoch,
        model=model,
        dynamic_inputs=dyn,
        static_attributes=stats,
        dyn_center=np.array([float(center[f].values) for f in dyn]),
        dyn_scale=np.array([float(scale[f].values) for f in dyn]),
        stat_mean=scaler["attribute_means"][list(stats)].to_numpy(dtype="float64"),
        stat_std=scaler["attribute_stds"][list(stats)].to_numpy(dtype="float64"),
        y_center=float(center[target].values),
        y_scale=float(scale[target].values),
        seq_length=int(cfg.seq_length),
    )


def load_pointer(variable: str, seed: int, root: Path = ARTIFACT_DIR) -> dict[str, Any]:
    p = pointer_path(variable, seed, root)
    if not p.exists():
        raise NotTrained(
            f"{variable} seed {seed} is not trained ({p}) - run "
            f"`python -m models.ealstm train --target {variable}` (or on Colab, see "
            "notebooks/ealstm_colab.md)"
        )
    d: dict[str, Any] = json.loads(p.read_text())
    return d


@dataclass
class Ensemble:
    variable: str
    target: TargetInfo
    members: list[Member]
    version: str
    data: ReachData = field(repr=False)

    @property
    def seq_length(self) -> int:
        return self.members[0].seq_length

    def simulate(
        self,
        reach_id: str,
        end_dates: Sequence[date],
        *,
        future_drivers: pd.DataFrame | None = None,
        static_overrides: dict[str, float] | None = None,
        per_member: bool = False,
    ) -> cmal.Mixture | list[cmal.Mixture]:
        """The ensemble mixture (transformed target space) for each end date."""
        X, dates = self.data.drivers(reach_id, future_drivers)
        s_raw = self.data.statics(reach_id, static_overrides)
        mixes = [
            simulate_member(m, X, dates, s_raw, end_dates, batch_size=self.data.batch_size)
            for m in self.members
        ]
        return mixes if per_member else cmal.ensemble(mixes)


def _check_members_agree(members: Sequence[Member]) -> None:
    """Seeds differ only in initialisation: the inputs and scalers must be identical,
    or the equal-weight mixture would mix different models of different inputs."""
    ref = members[0]
    for m in members[1:]:
        same = (
            m.dynamic_inputs == ref.dynamic_inputs
            and m.static_attributes == ref.static_attributes
            and np.allclose(m.dyn_center, ref.dyn_center)
            and np.allclose(m.dyn_scale, ref.dyn_scale)
            and np.allclose(m.stat_mean, ref.stat_mean)
            and np.allclose(m.stat_std, ref.stat_std)
            and np.isclose(m.y_center, ref.y_center)
            and np.isclose(m.y_scale, ref.y_scale)
            and m.seq_length == ref.seq_length
        )
        if not same:
            raise RuntimeError(f"seed {m.seed} disagrees with seed {ref.seed} on inputs/scaler")


def ensemble_version(es: EALSTMSettings, variable: str, members: Sequence[Member]) -> str:
    h = hashlib.sha256()
    for m in sorted(members, key=lambda m: m.seed):
        h.update(f"{m.seed}:{m.tag}:".encode())
        h.update(hashlib.sha256(_epoch_file(m.run_dir, m.epoch).read_bytes()).digest())
    return f"ealstm-{es.version}+{variable}.{h.hexdigest()[:10]}"


def load_ensemble(
    variable: str,
    *,
    es: EALSTMSettings | None = None,
    seeds: Sequence[int] | None = None,
    root: Path = ARTIFACT_DIR,
    require_all: bool = True,
) -> Ensemble:
    es = es or EALSTMSettings.from_config()
    target = es.target(variable)
    members, missing = [], []
    for seed in seeds or es.seeds:
        try:
            p = load_pointer(variable, seed, root)
        except NotTrained:
            missing.append(seed)
            continue
        members.append(load_member(root / p["run_dir"], int(p["epoch"]), variable, seed))
    if missing and require_all:
        raise NotTrained(f"{variable}: seeds {missing} are not trained - the ensemble is 5 seeds")
    if not members:
        raise NotTrained(f"{variable}: no trained seed")
    _check_members_agree(members)
    data = ReachData(
        es.nh_dir, members[0].dynamic_inputs, members[0].static_attributes, es.batch_size
    )
    return Ensemble(variable, target, members, ensemble_version(es, variable, members), data)


# ---------------------------------------------------------------------------
# inputs + forward pass
# ---------------------------------------------------------------------------
def read_series(path: Path) -> pd.DataFrame:
    """One exported reach series, read with netCDF4 directly: ~0.05 s per file against
    1.5-3 s through xarray on Windows (348 reaches x 2 targets made the hindcast crawl).
    Auto-masking is off: the export writes missing values as NaN (xarray's float
    _FillValue), so NaN comes back as NaN - nothing is filled either way."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        t = ds.variables["date"]
        dates = netCDF4.num2date(
            t[:], t.units, getattr(t, "calendar", "standard"),
            only_use_cftime_datetimes=False, only_use_python_datetimes=True,
        )
        cols = {k: np.asarray(v[:]) for k, v in ds.variables.items() if k != "date"}
    return pd.DataFrame(cols, index=pd.DatetimeIndex(pd.to_datetime(list(dates)), name="date"))


@dataclass
class ReachData:
    """Reads the NH export (the same files the model was trained on)."""

    nh_dir: Path
    dynamic_inputs: tuple[str, ...]
    static_attributes: tuple[str, ...]
    batch_size: int = 2048
    _series: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _attrs: pd.DataFrame | None = field(default=None, repr=False)

    def series(self, reach_id: str) -> pd.DataFrame:
        if reach_id not in self._series:
            path = self.nh_dir / "time_series" / f"{reach_id}.nc"
            if not path.exists():
                raise LookupError(f"{reach_id} has no exported time series ({path})")
            self._series[reach_id] = read_series(path)
        return self._series[reach_id]

    def attributes(self) -> pd.DataFrame:
        if self._attrs is None:
            a = pd.read_csv(self.nh_dir / "attributes" / "attributes.csv", dtype={"reach_id": str})
            self._attrs = a.set_index("reach_id")
        return self._attrs

    def drivers(
        self, reach_id: str, future: pd.DataFrame | None = None
    ) -> tuple[np.ndarray, pd.DatetimeIndex]:
        """(T, F) raw dynamic inputs on a continuous daily index; `future` rows (index =
        date) are appended after the archive and must continue it without a gap."""
        s = self.series(reach_id)[list(self.dynamic_inputs)]
        if future is not None and not future.empty:
            f = future.copy()
            f.index = pd.DatetimeIndex(pd.to_datetime(f.index), name="date")
            missing = set(self.dynamic_inputs) - set(f.columns)
            if missing:
                raise ValueError(f"future drivers lack {sorted(missing)}")
            f = f[list(self.dynamic_inputs)]
            f = f[f.index > s.index[-1]]
            if not f.empty:
                expect = pd.date_range(s.index[-1] + pd.Timedelta(days=1), f.index[-1], freq="D")
                if not f.index.equals(expect):
                    raise ValueError("future drivers must continue the archive day by day")
                s = pd.concat([s, f.astype("float64")])
        return s.to_numpy(dtype="float64"), pd.DatetimeIndex(s.index)

    def statics(self, reach_id: str, overrides: dict[str, float] | None = None) -> np.ndarray:
        a = self.attributes()
        if reach_id not in a.index:
            raise LookupError(f"{reach_id} has no static attributes")
        row = a.loc[reach_id, list(self.static_attributes)].astype("float64").copy()
        for k, v in (overrides or {}).items():
            if k not in row.index:
                raise KeyError(f"{k} is not a static input of the model ({list(row.index)})")
            row[k] = float(v)
        return row.to_numpy(dtype="float64")


def build_windows(
    X: np.ndarray, dates: pd.DatetimeIndex, end_dates: Sequence[date], seq_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """(n, L, F) windows ending at each end date, and a validity mask: a window is valid
    only if it lies inside the series and has no missing driver. Invalid windows are
    returned as zeros and their outputs are set to NaN by the caller - never filled."""
    idx = dates.get_indexer(pd.DatetimeIndex(pd.to_datetime(list(end_dates))))
    n, L, F = len(idx), seq_length, X.shape[1]
    out = np.zeros((n, L, F), dtype="float32")
    ok = idx >= L - 1
    nan_row = np.isnan(X).any(axis=1)
    # count of NaN rows in each trailing window of length L
    csum = np.concatenate([[0], np.cumsum(nan_row)])
    for j in np.flatnonzero(ok):
        e = idx[j]
        if csum[e + 1] - csum[e + 1 - L] > 0:
            ok[j] = False
            continue
        out[j] = X[e + 1 - L : e + 1]
    return out, ok


def forward_member(
    m: Member, windows: np.ndarray, x_s_norm: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw NH head output at the last time step, normalised target space. windows: (n, L,
    F) raw drivers; x_s_norm: (S,) normalised statics shared by every row, or (n, S)."""
    import torch

    xn = (windows - m.dyn_center) / m.dyn_scale
    xs = np.broadcast_to(x_s_norm, (len(xn), len(m.static_attributes)))
    outs: list[list[np.ndarray]] = [[], [], [], []]
    with torch.no_grad():
        for i in range(0, len(xn), batch_size):
            xb = torch.from_numpy(xn[i : i + batch_size].astype("float32"))
            data = {
                "x_d": {f: xb[:, :, j : j + 1] for j, f in enumerate(m.dynamic_inputs)},
                "x_s": torch.from_numpy(np.ascontiguousarray(xs[i : i + batch_size], "float32")),
            }
            pred = m.model(data)
            for k, key in enumerate(("mu", "b", "tau", "pi")):
                outs[k].append(pred[key][:, -1, :].double().numpy())
    if not outs[0]:
        k = m.model.head.fc2.out_features // 4
        return tuple(np.zeros((0, k)) for _ in range(4))  # type: ignore[return-value]
    return tuple(np.concatenate(o) for o in outs)  # type: ignore[return-value]


def simulate_reaches(
    ens: Ensemble,
    reach_ids: Sequence[str],
    end_dates: Sequence[date],
    static_overrides: dict[str, dict[str, float]] | None = None,
) -> tuple[list[str], cmal.Mixture]:
    """Every reach x every end date in ONE batched pass per member (reach-major rows) -
    for many-reach work like the scenario sensitivity check. Rows with an invalid window
    or a NULL static are NaN, never filled."""
    rows_w, rows_s, ok_all, rid_rows = [], [], [], []
    for rid in reach_ids:
        X, dates = ens.data.drivers(rid)
        s_raw = ens.data.statics(rid, (static_overrides or {}).get(rid))
        w, ok = build_windows(X, dates, end_dates, ens.seq_length)
        if np.isnan(s_raw).any():
            ok[:] = False
        rows_w.append(w)
        rows_s.append(np.repeat(s_raw[None, :], len(end_dates), axis=0))
        ok_all.append(ok)
        rid_rows += [rid] * len(end_dates)
    W, S, OK = np.concatenate(rows_w), np.concatenate(rows_s), np.concatenate(ok_all)
    mixes = []
    for m in ens.members:
        k = m.model.head.fc2.out_features // 4
        arrays = [np.full((len(W), k), np.nan) for _ in range(4)]
        if OK.any():
            xs = (S[OK] - m.stat_mean) / m.stat_std
            out = forward_member(m, W[OK], xs, ens.data.batch_size)
            for a, v in zip(arrays, out, strict=True):
                a[OK] = v
        mixes.append(cmal.rescale(cmal.Mixture(*arrays), m.y_center, m.y_scale))
    return rid_rows, cmal.ensemble(mixes)


def simulate_member(
    m: Member,
    X: np.ndarray,
    dates: pd.DatetimeIndex,
    s_raw: np.ndarray,
    end_dates: Sequence[date],
    *,
    batch_size: int,
) -> cmal.Mixture:
    """One member's CMAL mixture per end date, in the TRANSFORMED target space (NH's
    normalisation undone). Rows whose window is invalid or statics are NULL are NaN."""
    windows, ok = build_windows(X, dates, end_dates, m.seq_length)
    if np.isnan(s_raw).any():
        ok[:] = False
    x_s = (s_raw - m.stat_mean) / m.stat_std
    n = len(end_dates)
    k = m.model.head.fc2.out_features // 4
    arrays = [np.full((n, k), np.nan) for _ in range(4)]
    if ok.any():
        mu, b, tau, pi = forward_member(m, windows[ok], np.nan_to_num(x_s), batch_size)
        for a, v in zip(arrays, (mu, b, tau, pi), strict=True):
            a[ok] = v
    mix = cmal.Mixture(*arrays)
    return cmal.rescale(mix, m.y_center, m.y_scale)


# ---------------------------------------------------------------------------
# epoch selection
# ---------------------------------------------------------------------------
def cmal_nll(mix: cmal.Mixture, y: np.ndarray) -> np.ndarray:
    """Per-row negative log-likelihood, exactly MaskedCMALLoss's expression."""
    w = cmal.normalise_weights(mix.pi)
    e = y[:, None] - mix.mu
    ll = (
        np.log(mix.tau)
        + np.log(1.0 - mix.tau)
        - np.log(mix.b)
        - np.maximum(mix.tau * e, (mix.tau - 1.0) * e) / mix.b
    )
    a = np.log(w + 1e-8) + ll
    amax = a.max(axis=1, keepdims=True)
    return np.asarray(-(amax[:, 0] + np.log(np.exp(a - amax).sum(axis=1))))


def select_epoch(
    run_dir: Path, variable: str, *, es: EALSTMSettings | None = None, device: str = "cpu"
) -> dict[str, Any]:
    """Validation NLL (normalised target space, training reaches, validation period) for
    every saved epoch; the lowest is served."""
    from neuralhydrology.utils.config import Config

    es = es or EALSTMSettings.from_config()
    target = es.target(variable)
    cfg = Config(run_dir / "config.yml")
    v0, v1 = cfg.validation_start_date, cfg.validation_end_date
    basins = [
        b.strip() for b in Path(cfg.validation_basin_file).read_text().splitlines() if b.strip()
    ]
    epochs = saved_epochs(run_dir)
    if not epochs:
        raise NotTrained(f"{run_dir} has no saved epochs")
    first = load_member(run_dir, epochs[0], variable, -1)
    data = ReachData(es.nh_dir, first.dynamic_inputs, first.static_attributes, es.batch_size)
    cases = []
    for rid in basins:
        s = data.series(rid)
        y = s[target.nh_name]
        y = y[(y.index >= pd.Timestamp(v0)) & (y.index <= pd.Timestamp(v1)) & y.notna()]
        if not y.empty:
            X, dates = data.drivers(rid)
            cases.append((X, dates, data.statics(rid), list(y.index.date), y.to_numpy()))
    if not cases:
        raise RuntimeError("no OK validation targets on the training reaches - cannot select")
    val: dict[str, float] = {}
    n_rows = 0
    for ep in epochs:
        m = first if ep == epochs[0] else load_member(run_dir, ep, variable, -1)
        tot, n = 0.0, 0
        for X, dates, s_raw, ends, y in cases:
            mix = simulate_member(m, X, dates, s_raw, ends, batch_size=es.batch_size)
            ok = mix.valid()
            # NH's loss is in normalised space: undo our rescale on both sides
            mn = cmal.rescale(mix.rows(np.flatnonzero(ok)), -m.y_center / m.y_scale, 1 / m.y_scale)
            yn = (y[ok] - m.y_center) / m.y_scale
            nll = cmal_nll(mn, yn)
            tot += float(nll.sum())
            n += int(len(nll))
        val[str(ep)] = tot / max(n, 1)
        n_rows = n
    best = min(val, key=lambda k: val[k])
    return {
        "target": variable,
        "run_dir": run_dir.name,
        "criterion": "mean CMAL NLL, normalised target space, training reaches, validation period",
        "validation_rows": n_rows,
        "val_nll": val,
        "best_epoch": int(best),
        "epochs_run": len(epochs),
        "note": (
            "NH 1.13: early_stopping enables ReduceLROnPlateau; the fixed LR schedule is unused"
        ),
    }


# ---------------------------------------------------------------------------
# serving
# ---------------------------------------------------------------------------
def distribution(
    ens: Ensemble, mix: cmal.Mixture, levels: Sequence[float] = LEVELS
) -> dict[str, Any]:
    """Quantiles in ORIGINAL units (the transform is monotone, so quantiles map exactly)
    and a cdf(x) in original units."""
    qt = cmal.quantiles(mix, levels)
    q = ens.target.inverse(qt)
    return {"quantiles_t": qt, "quantiles": q, "levels": tuple(levels)}


def make_cdf(ens: Ensemble, mix: cmal.Mixture) -> Callable[[Any], np.ndarray]:
    """cdf(x) = P(state <= x), x in the variable's ORIGINAL units. For log1p, the event
    {state <= x} is {log1p(state) <= log1p(x)} exactly."""

    def f(x: Any) -> np.ndarray:
        xt = ens.target.forward(np.asarray(x, dtype="float64"))
        return cmal.cdf(mix, xt)

    return f


@lru_cache(maxsize=4)
def _cached_ensemble(variable: str) -> Ensemble:
    return load_ensemble(variable)


def archive_end(ens: Ensemble, reach_id: str) -> date:
    return ens.data.series(reach_id).index[-1].date()


def load_future_drivers(reach_id: str, after: date) -> pd.DataFrame:
    """drivers_daily FORECAST rows after `after` (the Open-Meteo forecast endpoint,
    written by pipeline.l2_drivers). Raises if there are none - never a fabricated day."""
    from sqlalchemy import text

    from core.db import session_scope

    with session_scope() as session:
        rows = pd.read_sql(
            text(
                "SELECT * FROM drivers_daily WHERE reach_id = :r AND source = 'FORECAST' "
                "AND date > :d ORDER BY date"
            ),
            session.connection(),
            params={"r": reach_id, "d": after},
        )
    if rows.empty:
        raise LookupError(
            f"{reach_id}: no FORECAST drivers after {after} - run pipeline.l2_drivers"
        )
    return rows.set_index(pd.DatetimeIndex(pd.to_datetime(rows["date"]), name="date"))


def predict(
    reach_id: str,
    issued_date: date,
    horizon: int,
    variable: str = "turbidity_proxy",
    *,
    future_drivers: pd.DataFrame | None = None,
    ensemble: Ensemble | None = None,
) -> dict[str, Any]:
    """{p10, p50, p90, cdf, model_version} for one reach, issue date and horizon (same
    interface as models.baseline_gbm.predict), plus provenance.

    Un-assimilated and un-calibrated, like baseline_gbm.predict: models/assimilation.py
    and models/calibration.py post-process both models identically. If the target date
    is inside the archive, the drivers for t+1..t+h are OBSERVED weather and the output
    says weather=ARCHIVE_HINDCAST with HINDCAST_LABEL. Beyond the archive they must come
    from the forecast (`future_drivers`, or drivers_daily FORECAST rows)."""
    ens = ensemble or _cached_ensemble(variable)
    if not 1 <= horizon <= 10:
        raise ValueError(f"horizon {horizon} outside 1..10")
    target_date = issued_date + timedelta(days=horizon)
    end = archive_end(ens, reach_id)
    if issued_date > end:
        raise LookupError(f"issue date {issued_date} is after the archive ({end})")
    if target_date <= end and future_drivers is None:
        weather, fut = WEATHER_HINDCAST, None
    else:
        weather = WEATHER_LIVE
        fut = future_drivers if future_drivers is not None else load_future_drivers(reach_id, end)
        # drivers after the issue date must be the forecast, never the archive
        if issued_date < end:
            raise ValueError(
                "live forecast mode needs issued_date == the archive's last day; for an "
                "earlier issue date the archive already holds observed t+1..t+h weather"
            )
    mix = ens.simulate(reach_id, [target_date], future_drivers=fut)
    assert isinstance(mix, cmal.Mixture)
    d = distribution(ens, mix)
    q = {
        f"p{round(a * 100):02d}": float(v)
        for a, v in zip(d["levels"], d["quantiles"][0], strict=True)
    }
    if not np.isfinite(q["p50"]):
        raise LookupError(
            f"{reach_id} {target_date}: no simulation - a driver in the 365-day window or a "
            "static attribute is NULL (never filled)"
        )
    train_end = member_train_end(ens.members[0])
    return {
        **q,
        "cdf": make_cdf(ens, mix),
        "reach_id": reach_id,
        "variable": variable,
        "issued_date": issued_date,
        "horizon": horizon,
        "target_date": target_date,
        "model_version": ens.version,
        "weather": weather,
        "weather_label": HINDCAST_LABEL if weather == WEATHER_HINDCAST else "live forecast weather",
        "transform": ens.target.transform,
        "in_sample": target_date <= train_end,
    }


def member_train_end(m: Member) -> date:
    from neuralhydrology.utils.config import Config

    return pd.Timestamp(Config(m.run_dir / "config.yml").train_end_date).date()


# ---------------------------------------------------------------------------
# hindcast simulations (evaluation, assimilation, anomaly)
# ---------------------------------------------------------------------------
def observable_reaches(manifest: dict[str, Any]) -> dict[str, bool | None]:
    return {r: c.get("observable") for r, c in manifest["per_reach"].items()}


def hindcast(
    city: str,
    *,
    start: date,
    es: EALSTMSettings | None = None,
    variables: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Simulate every simulatable reach at every OK-observation date >= start, both
    targets. One row per (variable, reach, date): observed (transformed and original),
    7 quantiles (original units), PIT of the observation, and the ensemble mixture
    parameters (transformed space) so the exact CDF can be re-evaluated later."""
    es = es or EALSTMSettings.from_config()
    manifest = load_manifest(es)
    lists = manifest["basin_lists"]
    holdout, train = set(lists["spatial_holdout"]), set(lists["train_reaches"])
    reaches = [
        r.strip() for r in (es.nh_dir / "all_reaches.txt").read_text().splitlines() if r.strip()
    ]
    obs_flag = observable_reaches(manifest)
    parts = []
    for variable in variables or es.targets:
        ens = load_ensemble(variable, es=es)
        with stage(log, "ealstm_hindcast", target=variable) as counters:
            n_in = 0
            for i, rid in enumerate(reaches, 1):
                if i % 25 == 0 or i == len(reaches):
                    log.info("ealstm.hindcast_progress", target=variable, reach=i, of=len(reaches))
                y = ens.data.series(rid)[ens.target.nh_name]
                y = y[(y.index >= pd.Timestamp(start)) & y.notna()]
                n_in += len(y)
                if y.empty:
                    continue
                mix = ens.simulate(rid, list(y.index.date))
                assert isinstance(mix, cmal.Mixture)
                d = distribution(ens, mix)
                part = pd.DataFrame(
                    {
                        "variable": variable,
                        "reach_id": rid,
                        "date": y.index.date,
                        "observed_t": y.to_numpy(),
                        "observed": ens.target.inverse(y.to_numpy()),
                        "pit": cmal.cdf(mix, y.to_numpy()),
                        "reach_observable": obs_flag.get(rid),
                        "split": "holdout"
                        if rid in holdout
                        else ("train" if rid in train else "other"),
                        "model_version": ens.version,
                    }
                )
                for j, a in enumerate(d["levels"]):
                    part[f"p{round(a * 100):02d}"] = d["quantiles"][:, j]
                for name in ("mu", "b", "tau", "pi"):
                    arr = getattr(mix, name)
                    for c in range(arr.shape[1]):
                        part[f"{name}_{c}"] = arr[:, c]
                parts.append(part)
            out_rows = sum(len(p) for p in parts if p["variable"].iat[0] == variable)
            no_sim = sum(
                int(p["p50"].isna().sum()) for p in parts if p["variable"].iat[0] == variable
            )
            counters.record(rows_in=n_in, rows_out=out_rows, no_simulation=no_sim)
            if no_sim:
                counters.drop(no_sim, "NULL_DRIVER_OR_STATIC_IN_WINDOW")
    if not parts:
        raise RuntimeError(f"no OK observations on or after {start} to simulate")
    return pd.concat(parts, ignore_index=True)


def mixture_from_rows(rows: pd.DataFrame) -> cmal.Mixture:
    """Rebuild the stored ensemble mixture of hindcast rows."""

    def arr(name: str) -> np.ndarray:
        cols = sorted(
            (c for c in rows.columns if c.startswith(f"{name}_") and c[len(name) + 1 :].isdigit()),
            key=lambda c: int(c.rsplit("_", 1)[1]),
        )
        return rows[cols].to_numpy(dtype="float64")

    return cmal.Mixture(arr("mu"), arr("b"), arr("tau"), arr("pi"))


def forecast_rows(
    sim: pd.DataFrame,
    horizons: Sequence[int],
    folds: dict[str, tuple[date, date]],
) -> pd.DataFrame:
    """Hindcast simulations -> forecast rows in the evaluation harness's layout, one per
    (fold, variable, reach, issue date, horizon) whose target date t+h has an OK
    observation. The fold is the ISSUE date's, and a target past the fold's end is
    dropped (the same embargo as pipeline.build_dataset.embargo). Pure."""
    q = [c for c in sim.columns if c[0] == "p" and c[1:].isdigit()]
    parts = []
    tdate = pd.to_datetime(sim["date"])
    for h in horizons:
        issued = tdate - pd.Timedelta(days=int(h))
        for fold, (lo, hi) in folds.items():
            m = (
                (issued >= pd.Timestamp(lo))
                & (issued <= pd.Timestamp(hi))
                & (tdate <= pd.Timestamp(hi))
            )
            if not m.any():
                continue
            s = sim[m.to_numpy()]
            parts.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "variable": s["variable"].to_numpy(),
                        "reach_id": s["reach_id"].astype(str).to_numpy(),
                        "issued_date": issued[m].dt.date.to_numpy(),
                        "horizon": int(h),
                        "target_date": tdate[m].dt.date.to_numpy(),
                        "target": s["observed"].to_numpy(),
                        "reach_observable": s["reach_observable"].to_numpy(),
                        "split": s["split"].to_numpy(),
                        "model_version": s["model_version"].to_numpy(),
                        "weather": WEATHER_HINDCAST,
                        **{c: s[c].to_numpy() for c in q},
                    }
                )
            )
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def run_hindcast(city: str) -> dict[str, Any]:
    from pipeline.build_dataset import PROCESSED_DIR, FrameSettings

    es = EALSTMSettings.from_config()
    fs = FrameSettings.from_config()
    sim = hindcast(city, start=fs.val_start, es=es)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sim.to_parquet(PROCESSED_DIR / f"ealstm_sim_{city}.parquet", index=False)
    run = {
        "city": city,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "weather": WEATHER_HINDCAST,
        "weather_label": HINDCAST_LABEL,
        "versions": sim.groupby("variable")["model_version"].first().to_dict(),
        "rows": int(len(sim)),
        "rows_without_simulation": int(sim["p50"].isna().sum()),
        "start": str(fs.val_start),
    }
    (PROCESSED_DIR / f"ealstm_run_{city}.json").write_text(json.dumps(run, indent=2))
    return run


# ---------------------------------------------------------------------------
# attribution: integrated gradients on the ensemble P50
# ---------------------------------------------------------------------------
IG_WINDOW_DAYS = 30
IG_STEPS = 64
# |sum(attributions) - (P50(x) - P50(baseline))| must be within this fraction of the
# difference (transformed space; floor 1e-3) for the attribution to count as complete.
COMPLETENESS_TOL = 0.10


def _torch_mixture_quantile(mu: Any, b: Any, tau: Any, pi: Any, level: float) -> Any:
    """Differentiable mixture quantile: bisection without gradient, then one implicit-
    function step q = q0 - (F(q0) - level) / f(q0) with f detached, whose gradient is
    exactly dq/dtheta = -(dF/dtheta) / f (the implicit function theorem) and whose value
    is q0."""
    import torch

    w = pi / pi.sum(dim=-1, keepdim=True)

    def F(y: Any) -> Any:
        z = (y[:, None] - mu) / b
        lower = tau * torch.exp(torch.clamp((1 - tau) * z, max=0.0))
        upper = 1 - (1 - tau) * torch.exp(torch.clamp(-tau * z, max=0.0))
        return (w * torch.where(z < 0, lower, upper)).sum(dim=-1)

    def pdf(y: Any) -> Any:
        e = y[:, None] - mu
        rho = torch.maximum(tau * e, (tau - 1) * e) / b
        return (w * tau * (1 - tau) / b * torch.exp(-rho)).sum(dim=-1)

    with torch.no_grad():
        reach = 40.0 * b / torch.minimum(tau, 1 - tau)
        lo = (mu - reach).min(dim=-1).values
        hi = (mu + reach).max(dim=-1).values
        for _ in range(cmal.BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            below = F(mid) < level
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        q0 = 0.5 * (lo + hi)
    return q0 - (F(q0) - level) / pdf(q0).detach()


def ensemble_p50_torch(members: Sequence[Member]) -> Callable[..., Any]:
    """f(x_d_norm [n, L, F], x_s_norm [n, S]) -> ensemble P50, transformed target space."""
    import torch

    def f(x_d: Any, x_s: Any) -> Any:
        mus, bs, taus, pis = [], [], [], []
        for m in members:
            data = {
                "x_d": {name: x_d[:, :, j : j + 1] for j, name in enumerate(m.dynamic_inputs)},
                "x_s": x_s,
            }
            out = m.model(data)
            mus.append(out["mu"][:, -1, :] * m.y_scale + m.y_center)
            bs.append(out["b"][:, -1, :] * m.y_scale)
            taus.append(out["tau"][:, -1, :])
            p = out["pi"][:, -1, :]
            pis.append(p / p.sum(dim=-1, keepdim=True) / len(members))
        return _torch_mixture_quantile(
            torch.cat(mus, -1), torch.cat(bs, -1), torch.cat(taus, -1), torch.cat(pis, -1), 0.5
        )

    return f


def climatological_baseline(
    X: np.ndarray,
    dates: pd.DatetimeIndex,
    end: pd.Timestamp,
    L: int,
    train_end: date,
    half_window: int = 7,
) -> np.ndarray:
    """(L, F) baseline window: for each day, the mean of that input over the TRAINING
    years within +/- half_window days of the same day-of-year - 'what the weather usually
    is on this date'. Days whose climatology is undefined keep the actual value (they
    then get no attribution)."""
    train = dates <= pd.Timestamp(train_end)
    doy = np.minimum(dates.dayofyear.to_numpy(), 365)
    e = dates.get_loc(end)
    window_doy = doy[e + 1 - L : e + 1]
    out = X[e + 1 - L : e + 1].copy()
    tdoy, tX = doy[train], X[train]
    for i, d in enumerate(window_doy):
        dist = np.abs(tdoy - d)
        near = np.minimum(dist, 365 - dist) <= half_window
        if near.any():
            out[i] = np.nanmean(tX[near], axis=0)
    return out


def explain(
    reach_id: str,
    target_date: date,
    variable: str = "turbidity_proxy",
    *,
    ensemble: Ensemble | None = None,
    future_drivers: pd.DataFrame | None = None,
    window_days: int = IG_WINDOW_DAYS,
    n_steps: int = IG_STEPS,
) -> dict[str, Any]:
    """Integrated gradients (Sundararajan et al. 2017, via captum) of the ensemble P50
    with respect to the dynamic inputs over the last `window_days` days of the sequence
    plus the statics. Baseline: the day-of-year climatological mean of each driver
    (training years) and the training-reach mean of each static; days before the window
    are held at their actual values. Per-feature sums over the window, normalised: each
    feature keeps its share of the attribution and the total is P50(x) - P50(baseline) in
    the variable's ORIGINAL units - emitted in the alert engine's DriverContribution
    schema. IG is exact only along a smooth path; the convergence delta is reported and
    `attribution_complete` is false when it exceeds COMPLETENESS_TOL of the difference."""
    import torch
    from captum.attr import IntegratedGradients

    from engine.alerts import build_attribution

    ens = ensemble or _cached_ensemble(variable)
    m0 = ens.members[0]
    L, W = m0.seq_length, window_days
    X, dates = ens.data.drivers(reach_id, future_drivers)
    s_raw = ens.data.statics(reach_id)
    windows, ok = build_windows(X, dates, [target_date], L)
    if not ok[0] or np.isnan(s_raw).any():
        raise LookupError(f"{reach_id} {target_date}: incomplete inputs - no attribution")
    train_end = member_train_end(m0)
    base_win = climatological_baseline(X, dates, pd.Timestamp(target_date), L, train_end)
    xd = (windows[0] - m0.dyn_center) / m0.dyn_scale
    xb = (base_win - m0.dyn_center) / m0.dyn_scale
    xs = (s_raw - m0.stat_mean) / m0.stat_std
    t = lambda a: torch.from_numpy(np.asarray(a, dtype="float32"))[None]  # noqa: E731
    prefix = t(xd[: L - W])
    f = ensemble_p50_torch(ens.members)

    def fwd(last: Any, stat: Any, pre: Any) -> Any:
        return f(torch.cat([pre.expand(last.shape[0], -1, -1), last], dim=1), stat)

    ig = IntegratedGradients(fwd)
    inp = (t(xd[L - W :]), t(xs))
    base = (t(xb[L - W :]), torch.zeros_like(inp[1]))
    (a_dyn, a_stat), delta = ig.attribute(
        inp,
        baselines=base,
        additional_forward_args=(prefix,),
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    with torch.no_grad():
        p50_x = float(fwd(*inp, prefix)[0])
        p50_b = float(fwd(*base, prefix)[0])
    dyn = a_dyn[0].sum(dim=0).double().numpy()
    stat = a_stat[0].double().numpy()
    names = [*m0.dynamic_inputs, *m0.static_attributes]
    contrib_t = np.concatenate([dyn, stat])
    diff_t = p50_x - p50_b
    diff_o = float(ens.target.inverse(p50_x) - ens.target.inverse(p50_b))
    # Normalise: keep each feature's SHARE of the attribution, scale the total to the
    # P50 difference in original units (the SHAP-like "contributions sum to the change").
    total = float(contrib_t.sum())
    contrib_o = contrib_t / total * diff_o if abs(total) > 1e-12 else contrib_t * 0.0
    # IG is complete only if the P50 is smooth along the path. A 365-step LSTM can switch
    # regime abruptly (a near-discontinuity gradients cannot see); then sum(attr) misses
    # f(x) - f(baseline) and the shares are unreliable. Reported, never hidden.
    delta = float(delta.abs().max())
    complete = delta <= COMPLETENESS_TOL * max(abs(diff_t), 1e-3)
    values = [*windows[0][-1], *s_raw]  # each driver's value ON the target date
    drivers = build_attribution(contrib_o, names, [float(v) for v in values])
    return {
        "reach_id": reach_id,
        "variable": variable,
        "target_date": target_date,
        "method": f"integrated gradients (captum), {n_steps} steps, last {W} days + statics",
        "baseline": "day-of-year climatological drivers (training years); mean statics",
        "p50": float(ens.target.inverse(p50_x)),
        "p50_baseline": float(ens.target.inverse(p50_b)),
        "completeness_delta_transformed": delta,
        "p50_difference_transformed": float(diff_t),
        "attribution_complete": bool(complete),
        "drivers": drivers,
        "model_version": ens.version,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def status(es: EALSTMSettings | None = None) -> dict[str, Any]:
    es = es or EALSTMSettings.from_config()
    out: dict[str, Any] = {}
    for v in es.targets:
        for s in es.seeds:
            try:
                p = load_pointer(v, s)
                sel = json.loads((ARTIFACT_DIR / p["run_dir"] / "selection.json").read_text())
                out[f"{v}/seed{s}"] = {
                    "epoch": p["epoch"],
                    "epochs_run": sel["epochs_run"],
                    "val_nll": sel["val_nll"][str(p["epoch"])],
                    "trained_at": p["trained_at"],
                    "device": sel.get("device"),
                    "train_seconds": sel.get("train_seconds"),
                }
            except (NotTrained, FileNotFoundError):
                out[f"{v}/seed{s}"] = "NOT TRAINED"
    return out


def smoke_basins(es: EALSTMSettings, manifest: dict[str, Any], k: int = 3) -> list[str]:
    """The first training reaches, extended until every used static varies across them
    (NH refuses an attribute with zero spread over the training basins)."""
    train = manifest["basin_lists"]["train_reaches"]
    used = manifest["static_attributes_used"]
    a = pd.read_csv(es.nh_dir / "attributes" / "attributes.csv", dtype={"reach_id": str})
    a = a.set_index("reach_id").loc[train, used]
    chosen = list(train[:k])
    for r in train[k:]:
        if (a.loc[chosen].std() > 0).all():
            break
        if any(a.loc[chosen, c].nunique() == 1 and a.at[r, c] != a.loc[chosen[0], c] for c in used):
            chosen.append(r)
    if not (a.loc[chosen].std() > 0).all():
        raise RuntimeError("no subset of training reaches has spread in every static")
    return chosen


def smoke(device: str = "cpu") -> dict[str, Any]:
    """1 epoch, 1 seed, 3 training reaches, into artifacts/ealstm/smoke/ (never served):
    proves the install, the export, the config and the inference path end to end."""
    es = EALSTMSettings.from_config()
    root = ARTIFACT_DIR / "smoke"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    basins = smoke_basins(es, load_manifest(es))
    bf = root / "smoke_basins.txt"
    bf.write_text("\n".join(basins) + "\n")
    variable = next(iter(es.targets))
    seed = es.seeds[0]
    sel = train_member(
        variable,
        seed,
        device=device,
        epochs=1,
        es=es,
        runs_root=root / "runs",
        pointer_root=root,
        train_basin_file=bf,
    )
    ens = load_ensemble(variable, es=es, seeds=[seed], root=root)
    series = ens.data.series(basins[0])
    ends = list(series[series[ens.target.nh_name].notna()].index.date[-5:])
    mix = ens.simulate(basins[0], ends)
    assert isinstance(mix, cmal.Mixture)
    d = distribution(ens, mix)
    if not np.isfinite(d["quantiles"]).all():
        raise RuntimeError(f"smoke: non-finite quantiles {d['quantiles']}")
    if not (np.diff(d["quantiles"], axis=1) >= 0).all():
        raise RuntimeError("smoke: quantiles not monotone")
    return {
        "selection": sel,
        "reach": basins[0],
        "dates": [str(x) for x in ends],
        "p10_p50_p90": d["quantiles"][:, [1, 3, 5]].round(4).tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EA-LSTM (NeuralHydrology) - P5.1")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train target x seeds")
    t.add_argument("--target", default="all", help="turbidity_proxy | ndci | all")
    t.add_argument("--seeds", type=int, nargs="*", help="default: config ealstm.seeds")
    t.add_argument("--device", default="cpu", help="cpu | cuda:0")
    t.add_argument("--epochs", type=int, help="override the config's max epochs")
    t.add_argument(
        "--skip-trained", action="store_true", help="resume: skip members with a pointer"
    )
    s = sub.add_parser("smoke", help="1-epoch end-to-end check")
    s.add_argument("--device", default="cpu")
    sub.add_parser("status", help="list trained members")
    h = sub.add_parser("hindcast", help="simulate every OK-observation date >= val start")
    h.add_argument("--city", default="coimbra")
    args = parser.parse_args(argv)

    if args.cmd == "status":
        print(json.dumps(status(), indent=2))
    elif args.cmd == "smoke":
        print(json.dumps(smoke(args.device), indent=2, default=str))
    elif args.cmd == "hindcast":
        print(json.dumps(run_hindcast(args.city), indent=2, default=str))
    else:
        es = EALSTMSettings.from_config()
        targets = list(es.targets) if args.target == "all" else [args.target]
        for v in targets:
            for seed in args.seeds or es.seeds:
                if args.skip_trained and pointer_path(v, seed).exists():
                    log.info("ealstm.skip_trained", target=v, seed=seed)
                    continue
                train_member(v, seed, device=args.device, epochs=args.epochs, es=es)
        print(json.dumps(status(es), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
