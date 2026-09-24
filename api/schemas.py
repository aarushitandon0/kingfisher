"""Pydantic schemas for every API boundary (MASTERSPEC section 12).

Every route declares one of these as its response_model, so a response that does not match
its contract fails inside the server, not in the browser. tests/test_api_contract.py
checks each endpoint against them.

Spatial endpoints return GeoJSON (RFC 7946). Feature properties are FLAT scalars: MapLibre
stringifies nested objects in properties, so anything a map layer styles on is top level.

Language: scenario outputs are planning estimates (engine.scenarios); alerts carry a
probability of threshold exceedance, never a statement that water is safe or unsafe, and
exposure is pathways and proximity only - no health outcome (CLAUDE.md #8).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engine.priorities import NotRanked, RankedReach
from engine.scenarios import InterventionRequest, ScenarioResult

Severity = Literal["ALERT", "WATCH", "INSUFFICIENT_EVIDENCE"]
Observability = Literal["OPTICALLY_OBSERVABLE", "DRIVER_ONLY", "NOT_ASSESSED"]
Variable = Literal["turbidity_proxy", "ndci"]
QualityFlag = Literal["OK", "CLOUD", "NO_WATER_PIXELS", "OUT_OF_RANGE"]


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


def observability(observable: bool | None) -> Observability:
    if observable is True:
        return "OPTICALLY_OBSERVABLE"
    if observable is False:
        return "DRIVER_ONLY"
    return "NOT_ASSESSED"


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------
class Geometry(_Out):
    type: Literal["Point", "LineString", "Polygon", "MultiPoint", "MultiLineString", "MultiPolygon"]
    coordinates: list[Any]


# ---------------------------------------------------------------------------
# cities
# ---------------------------------------------------------------------------
class BBox(_Out):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class CityOut(_Out):
    city: str
    name: str
    country: str
    reach_id_prefix: str
    bbox: BBox
    timezone: str | None
    status: Literal["READY", "NOT_BUILT"]  # NOT_BUILT: config exists, no reaches in the DB
    reaches: int
    optically_observable: int
    driver_only: int
    not_assessed: int


class CitiesResponse(_Out):
    cities: list[CityOut]


# ---------------------------------------------------------------------------
# reaches
# ---------------------------------------------------------------------------
class ReachProperties(_Out):
    reach_id: str
    city: str
    name: str | None
    strahler_order: int | None
    length_m: float
    catchment_area_km2: float | None
    observable: bool | None
    observability: Observability
    median_water_pixels: float | None
    # Current status. The probabilities are the forecast's peak daily P(exceed) over the
    # latest window - a map colour, not an alert (alerts pass the guardrails; see
    # alert_severity). NULL with exceedance_status NO_THRESHOLD where the reach has no
    # seasonal threshold (every driver-only reach) - never 0.
    exceedance_status: Literal["OK", "NO_THRESHOLD", "NO_FORECAST"]
    p_exceed_turbidity_proxy: float | None
    p_exceed_ndci: float | None
    p_exceed_max: float | None
    alert_severity: Severity | None  # from the latest alert run; NULL = no alert issued
    alert_id: UUID | None


class ReachFeature(_Out):
    type: Literal["Feature"] = "Feature"
    id: str
    geometry: Geometry
    properties: ReachProperties


class ReachCollection(_Out):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[ReachFeature]
    city: str
    forecast_issued_date: date | None
    forecast_model_version: str | None
    alert_run: Literal["OK", "NO_ALERT_RUN"]
    alert_run_issued_date: date | None


class ExposureLayerProperties(_Out):
    feature_type: str
    reaches: int  # reaches for which this is the nearest feature of its type
    reach_ids: str  # comma-separated (flat scalar for MapLibre)


class ExposureLayerFeature(_Out):
    type: Literal["Feature"] = "Feature"
    geometry: Geometry
    properties: ExposureLayerProperties


class ExposureLayer(_Out):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    city: str
    note: str = (
        "The nearest OpenStreetMap feature of each type within the exposure buffer of each "
        "reach. Exposure pathways and proximity only - no health outcome is predicted."
    )
    features: list[ExposureLayerFeature]


class CatchmentFeature(_Out):
    type: Literal["Feature"] = "Feature"
    id: str
    geometry: Geometry | None
    properties: dict[str, Any]


class ObservationOut(_Out):
    obs_date: date
    source: str
    quality_flag: QualityFlag
    turbidity_proxy: float | None
    ndci: float | None
    mndwi: float | None
    water_pixel_count: int | None
    cloud_fraction: float | None


class ThresholdOut(_Out):
    variable: Variable
    season: str
    threshold: float
    n_obs: int
    percentile: float
    fit_end: date


class ExposureFeatureOut(_Out):
    count: int | None
    nearest_distance_m: float | None


class ExposureOut(_Out):
    buffer_m: float | None
    population: int | None
    population_source: str | None
    features: dict[str, ExposureFeatureOut]
    note: str = (
        "Exposure pathways and proximity only (OpenStreetMap, GHS-POP). No health outcome is "
        "predicted and no water is declared safe or unsafe."
    )


class ReachDetailProperties(ReachProperties):
    upstream_ids: list[str]
    downstream_ids: list[str]
    catchment_attributes: dict[str, Any] | None
    thresholds: list[ThresholdOut]
    threshold_derivation: str
    exposure: ExposureOut | None
    history_start: date | None
    history_end: date | None
    history: list[ObservationOut]


class ReachDetail(_Out):
    type: Literal["Feature"] = "Feature"
    id: str
    geometry: Geometry
    properties: ReachDetailProperties
    catchment: CatchmentFeature  # RFC 7946 foreign member: the upstream catchment polygon


class ForecastDay(_Out):
    target_date: date
    horizon: int
    p05: float | None
    p10: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p90: float | None
    p95: float | None
    threshold: float | None
    exceedance_prob: float | None
    future_drivers_missing: int | None


class ForecastSeries(_Out):
    variable: Variable
    display_name: str
    units: str
    threshold_derivation: str
    days: list[ForecastDay]


class ForecastResponse(_Out):
    reach_id: str
    issued_date: date
    model_version: str
    variant: str
    weather: str
    horizon: int
    observable: bool | None
    observability: Observability
    notes: list[str]
    series: list[ForecastSeries]


# ---------------------------------------------------------------------------
# timeline (the hydrograph rail) - columnar: index i across every list is one row
# ---------------------------------------------------------------------------
class TimelineObservations(_Out):
    reach_id: list[str]
    date: list[date]
    variable: list[Variable]
    value: list[float | None]


class TimelineForecast(_Out):
    reach_id: list[str]
    target_date: list[date]
    variable: list[Variable]
    p10: list[float | None]
    p50: list[float | None]
    p90: list[float | None]
    threshold: list[float | None]  # NULL where the reach-season has none - never 0
    exceedance_prob: list[float | None]


class TimelineThresholds(_Out):
    reach_id: list[str]
    variable: list[Variable]
    season: list[str]
    threshold: list[float | None]


class TimelineResponse(_Out):
    city: str
    issued_date: date | None
    model_version: str | None
    seasons: dict[str, list[int]]
    threshold_note: str
    observations: TimelineObservations
    forecast: TimelineForecast
    thresholds: TimelineThresholds

    @model_validator(mode="after")
    def _columns_align(self) -> TimelineResponse:
        for block in (self.observations, self.forecast, self.thresholds):
            lengths = {k: len(v) for k, v in block.model_dump().items()}
            if len(set(lengths.values())) > 1:
                raise ValueError(f"{type(block).__name__} columns differ in length: {lengths}")
        return self


# ---------------------------------------------------------------------------
# alerts
# ---------------------------------------------------------------------------
class AlertSummary(_Out):
    alert_id: UUID
    reach_id: str
    reach_name: str | None
    city: str
    issued_date: date
    window_start: date
    window_end: date
    variable: str
    severity: Severity
    exceedance_prob: float | None  # always NULL for INSUFFICIENT_EVIDENCE
    threshold_value: float | None
    suppressed_reason: str | None
    model_version: str | None
    # Stored exposure context (engine.exposure) so a list can summarise it without a
    # request per alert. Pathways and proximity only.
    exposure: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _insufficient_has_no_probability(self) -> AlertSummary:
        if self.severity == "INSUFFICIENT_EVIDENCE":
            if self.exceedance_prob is not None:
                raise ValueError("INSUFFICIENT_EVIDENCE must not carry an exceedance probability")
            if not self.suppressed_reason:
                raise ValueError("INSUFFICIENT_EVIDENCE must say why")
        elif self.exceedance_prob is None:
            raise ValueError(f"{self.severity} needs an exceedance probability")
        return self


class DriverContributionOut(_Out):
    feature: str
    contribution: float
    value: float | None
    kind: str


class AttributionSeries(_Out):
    variable: Variable
    # PEAK_EXCEEDANCE: the day of the highest P(exceed) in the window (what the alert run
    # attributes). PEAK_P90: no threshold on this reach-season (every driver-only reach), so
    # the day with the highest P90 instead - drivers of the upper forecast, not of a risk.
    selection: Literal["PEAK_EXCEEDANCE", "PEAK_P90"]
    target_date: date
    horizon: int
    quantile: str
    contributions: list[DriverContributionOut]


class AttributionResponse(_Out):
    reach_id: str
    issued_date: date
    model_version: str
    basis: str
    series: list[AttributionSeries]


class AlertDetail(AlertSummary):
    threshold_derivation: str | None
    attribution: list[DriverContributionOut]
    exposure: dict[str, Any] | None
    basis: dict[str, Any] | None
    guardrails: dict[str, str] | None
    # INSUFFICIENT_EVIDENCE only: the probability the forecast would have given, kept apart
    # so it can never be read as a weak alert.
    withheld_exceedance_prob: float | None


class AlertsResponse(_Out):
    city: str | None
    active: bool
    alert_run: Literal["OK", "NO_ALERT_RUN"]
    alert_run_issued_date: date | None
    counts: dict[Severity, int]
    suppressed: dict[str, int] | None = Field(
        default=None,
        description="candidates each guardrail stopped in this run - not alert rows, "
        "counted so they are not lost; null when the run has no alert_runs record",
    )
    alerts: list[AlertSummary]


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
class ScenarioRequest(_Out):
    """POST /api/scenarios. Each intervention: {type, reach_ids, extent}. `extent` is how
    much of the lever (see config/intervention_coefficients.yaml, OPERATIONS); effect sizes
    come only from that table and cannot be sent."""

    name: str | None = Field(default=None, max_length=200)
    variables: list[Variable] | None = None
    interventions: list[InterventionRequest] = Field(min_length=1, max_length=20)

    @model_validator(mode="before")
    @classmethod
    def _no_effect_sizes(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for i in data.get("interventions") or []:
                if isinstance(i, dict) and "magnitude" in i:
                    raise ValueError(
                        "interventions take `extent` (how much of the lever), not `magnitude`: "
                        "effect sizes come only from the cited coefficient table"
                    )
        return data


class LeverPathOut(_Out):
    path: Literal["MODEL_PERTURBATION", "LITERATURE_DIRECT", "NOT_ESTIMABLE"]
    reason: str


class InterventionOut(_Out):
    """One lever of config/intervention_coefficients.yaml as validated by
    engine.coefficients, plus what a run in this city would do with it (MASTERSPEC 9.6)."""

    id: str
    name: str
    feature: str
    operation: Literal["subtract_treated_share", "floor", "scale_down"]
    direction: Literal["increase", "decrease"]
    effect_unit: str
    description: str
    # How `extent` is read for this operation; None where the operation takes no extent.
    extent_unit: str | None
    extent_min: float | None
    extent_max: float | None
    magnitude: float
    uncertainty_range: tuple[float, float]
    applies_to: dict[str, Any]
    cost_per_unit: dict[str, Any]
    citations: list[dict[str, Any]]
    direct_effect: dict[str, Any] | None
    paths: dict[str, LeverPathOut]


class NotQuantifiedOut(_Out):
    id: str
    name: str
    reason: str
    citations: list[dict[str, Any]]


class InterventionCatalogue(_Out):
    city: str
    coefficient_table_version: str
    caveat: str
    interventions: list[InterventionOut]
    not_quantified: list[NotQuantifiedOut]


class ScenarioResponse(_Out):
    scenario_id: UUID
    name: str | None
    city: str
    created_at: datetime
    result: ScenarioResult


# ---------------------------------------------------------------------------
# priorities
# ---------------------------------------------------------------------------
class RankedReachOut(RankedReach):
    name: str | None
    observability: Observability


class PrioritiesResponse(_Out):
    city: str
    issued_date: date | None
    model_version: str | None
    method: str
    weights: dict[str, float]
    weights_note: str = "exposure weights are a stated planning choice (config/priorities.yaml)"
    population_unit: float
    ranked: list[RankedReachOut]
    not_ranked: list[NotRanked]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
class MetricsDocument(BaseModel):
    """results/metrics.json, as written by models/evaluate.py. The top-level sections the
    UI reads are required; everything else passes through unchanged."""

    model_config = ConfigDict(extra="allow")

    city: str
    generated_at: str
    model_versions: dict[str, str]
    headline_run: str
    folds: dict[str, Any]
    runs: dict[str, Any]
    losses: list[Any]
    not_computed: dict[str, Any]


class SourceFile(_Out):
    path: str
    sha256: str
    modified_at: datetime
    size_bytes: int


class ValidationMetricsResponse(_Out):
    served_at: datetime
    source: SourceFile
    metrics: MetricsDocument
    scenario_response_check: dict[str, Any] | None
    scenario_response_check_source: SourceFile | None


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class ErrorResponse(_Out):
    detail: str
