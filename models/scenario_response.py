"""MASTERSPEC 9.6 rule 2 - the scenario response check, run and persisted.

    python -m models.scenario_response --city coimbra        (make response-check)

For every STATIC feature a lever in config/intervention_coefficients.yaml perturbs, and
every target variable: move the feature -1 SD and +1 SD (SD over the reaches the checked
fit was trained on, training fold) on every held-out reach-day of the walk-forward TEST
fold at the scenario horizon, re-infer with the variant-B fit whose training ended before
that fold (config modelling.yaml -> scenario.response_check.fit), everything else fixed,
and record the sign and size of the mean P50 change and of the mean change in
P(exceedance). engine.scenarios.response_check does the arithmetic and gives the verdict
against the negligible threshold fixed in config before the check was first run.

The file records the checked fit AND the serving fit (scenario.fit): the scenario engine
refuses a check whose serving version is not the model it is about to run.

Output: results/scenario_response_check_<city>.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from core.config import load_config, load_intervention_coefficients
from core.logging import get_logger, stage
from engine.alerts import STATIC_FEATURES
from engine.scenarios import expected_response_sign, response_check
from engine.thresholds import lookup as threshold_lookup
from engine.thresholds import seasonal_thresholds
from models.baseline_gbm import file_sha256, frame_path, load, load_frame, to_long
from models.scenario_model import GBMScenarioModel, check_path, observations, training_ranges

log = get_logger(__name__)


def run(city: str) -> dict[str, Any]:
    modelling = load_config("modelling")
    thresholds_cfg = load_config("thresholds")
    sc = modelling["scenario"]
    rc = sc["response_check"]
    horizon = int(sc["horizon"])
    negligible = float(rc["negligible_effect_sd"])
    if rc["fold"] != "test":
        raise ValueError("the response check runs on the walk-forward test fold")
    table = load_intervention_coefficients()
    features = sorted({i.effect.feature for i in table.interventions} & set(STATIC_FEATURES))
    if not features:
        raise RuntimeError("no lever perturbs a static feature - nothing to check")

    checked = load(city, str(rc["fit"]))
    serving = load(city, str(sc["fit"]))
    for m in (checked, serving):
        if m.manifest.get("variant", {}).get("name") != "B":
            raise RuntimeError(f"{m.version} is not variant B (MASTERSPEC 9.6 rule 1)")
    fs = checked.frame_settings
    if checked.train_end >= fs.test_start:
        raise RuntimeError(
            f"{checked.version} was trained through {checked.train_end}; the test fold "
            f"starts {fs.test_start} - it is not held out"
        )
    frame = load_frame(city)
    dates = pd.to_datetime(frame["date"]).dt.date
    test_end = fs.test_end or max(dates)
    model = GBMScenarioModel(checked, training_ranges(frame, checked))

    # held-out rows: every reach-day whose horizon-h target date lies in the test fold
    lag = timedelta(days=horizon)
    rows = frame[(dates >= fs.test_start - lag) & (dates <= test_end - lag)]
    held = to_long(rows, fs, fs.variables[0], [horizon], checked.categories, require_target=False)
    held = held[(held["target_date"] >= fs.test_start) & (held["target_date"] <= test_end)]
    held = held.drop(columns=["target"]).reset_index(drop=True)
    held["reach_id"] = held["reach_id"].astype(str)
    if held.empty:
        raise RuntimeError("no held-out rows in the test fold")

    # training-fold statistics
    train = frame[dates <= checked.train_end]
    obs = observations(frame, fs.variables)
    obs_train = obs[pd.to_datetime(obs["date"]).dt.date <= checked.train_end]
    thr_table = seasonal_thresholds(obs, checked.train_end, thresholds_cfg)

    results: list[dict[str, Any]] = []
    with stage(log, "scenario_response_check", city=city) as counters:
        for variable in fs.variables:
            trained_on = to_long(
                train, fs, variable, fs.horizons, checked.categories, require_target=True
            )
            train_reaches = set(trained_on["reach_id"].astype(str))
            target = obs_train.loc[obs_train["variable"] == variable, "value"].to_numpy()
            if len(target) < 2:
                raise RuntimeError(f"too few {variable} training observations for a target SD")
            target_sd = float(np.std(target, ddof=1))
            thr = threshold_lookup(
                held[["reach_id", "target_date"]].assign(variable=variable),
                thr_table,
                thresholds_cfg["seasons"],
            )["threshold"].to_numpy(dtype="float64")
            for feature in features:
                per_reach = (
                    train[train["reach_id"].astype(str).isin(train_reaches)]
                    .groupby("reach_id", observed=True)[feature]
                    .first()
                    .dropna()
                )
                if len(per_reach) < 2:
                    raise RuntimeError(f"{feature}: fewer than two training reaches have a value")
                chk = response_check(
                    model,
                    held,
                    feature=feature,
                    variable=variable,
                    feature_sd=float(per_reach.std(ddof=1)),
                    target_sd=target_sd,
                    expected_sign=expected_response_sign(table, feature),
                    negligible_effect_sd=negligible,
                    fold=str(rc["fold"]),
                    serving_model_version=serving.version,
                    thresholds=thr,
                )
                log.info(
                    "response_check",
                    feature=feature,
                    variable=variable,
                    verdict=str(chk.verdict),
                    effect_sd=round(chk.standardised_effect, 4),
                    sign=chk.observed_sign,
                    expected=chk.expected_sign,
                )
                chk = chk.model_copy(
                    update={
                        "feature_sd_reaches": len(per_reach),
                        "target_sd_observations": len(target),
                    }
                )
                results.append(chk.model_dump(mode="json"))
        counters.record(rows_in=len(held), rows_out=len(results))

    out: dict[str, Any] = {
        "city": city,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rule": "MASTERSPEC 9.6 rule 2",
        "method": (
            f"each static feature moved -1 SD and +1 SD (SD over the {checked.version} "
            "training reaches, training fold) on every held-out reach-day of the test fold "
            f"at horizon {horizon}, everything else fixed; effect = (mean P50 at +1 SD - mean "
            "P50 at -1 SD) / SD of the target's OK training observations; NEGLIGIBLE below "
            f"{negligible}; WRONG_SIGN if the sign contradicts the coefficient table"
        ),
        "config": {"horizon": horizon, **rc},
        "checked_model_version": checked.version,
        "serving_model_version": serving.version,
        "test_fold": {"start": str(fs.test_start), "end": str(test_end)},
        "held_out_rows": len(held),
        "frame_sha256": file_sha256(frame_path(city)),
        "checks": results,
    }
    path = check_path(city)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    log.info("response_check.written", path=str(path))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MASTERSPEC 9.6 scenario response check")
    parser.add_argument("--city", default="coimbra")
    args = parser.parse_args(argv)
    out = run(args.city)
    for c in out["checks"]:
        dp = c["mean_exceedance_prob_change"]
        print(
            f"{c['feature']:<20} {c['variable']:<16} {c['verdict']:<11} "
            f"effect {c['standardised_effect']:+.4f} SD  sign {c['observed_sign']:+d} "
            f"(literature {c['expected_sign']:+d})  "
            f"dP(exceed) {'n/a' if dp is None else f'{dp:+.4f}'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
