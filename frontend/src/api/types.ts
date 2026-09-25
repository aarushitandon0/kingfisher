// Mirrors api/schemas.py. The server validates every response against those Pydantic
// models; these types only describe what arrives.

export type Severity = "ALERT" | "WATCH" | "INSUFFICIENT_EVIDENCE";
export type Observability = "OPTICALLY_OBSERVABLE" | "DRIVER_ONLY" | "NOT_ASSESSED";
export type Variable = "turbidity_proxy" | "ndci";
export type QualityFlag = "OK" | "CLOUD" | "NO_WATER_PIXELS" | "OUT_OF_RANGE";
export type ISODate = string;

export interface Geometry {
  type: "Point" | "LineString" | "Polygon" | "MultiPoint" | "MultiLineString" | "MultiPolygon";
  coordinates: unknown[];
}

export interface City {
  city: string;
  name: string;
  country: string;
  reach_id_prefix: string;
  bbox: { min_lon: number; min_lat: number; max_lon: number; max_lat: number };
  timezone: string | null;
  status: "READY" | "NOT_BUILT";
  reaches: number;
  optically_observable: number;
  driver_only: number;
  not_assessed: number;
}

export interface ReachProperties {
  reach_id: string;
  city: string;
  name: string | null;
  strahler_order: number | null;
  length_m: number;
  catchment_area_km2: number | null;
  observable: boolean | null;
  observability: Observability;
  median_water_pixels: number | null;
  exceedance_status: "OK" | "NO_THRESHOLD" | "NO_FORECAST";
  p_exceed_turbidity_proxy: number | null;
  p_exceed_ndci: number | null;
  p_exceed_max: number | null;
  alert_severity: Severity | null;
  alert_id: string | null;
}

export interface ReachFeature {
  type: "Feature";
  id: string;
  geometry: Geometry;
  properties: ReachProperties;
}

export interface ReachCollection {
  type: "FeatureCollection";
  features: ReachFeature[];
  city: string;
  forecast_issued_date: ISODate | null;
  forecast_model_version: string | null;
  alert_run: "OK" | "NO_ALERT_RUN";
  alert_run_issued_date: ISODate | null;
}

export interface CatchmentFeature {
  type: "Feature";
  id: string;
  geometry: Geometry | null;
  properties: Record<string, unknown>;
}

export interface Observation {
  obs_date: ISODate;
  source: string;
  quality_flag: QualityFlag;
  turbidity_proxy: number | null;
  ndci: number | null;
  mndwi: number | null;
  water_pixel_count: number | null;
  cloud_fraction: number | null;
}

export interface Threshold {
  variable: Variable;
  season: string;
  threshold: number;
  n_obs: number;
  percentile: number;
  fit_end: ISODate;
}

export interface ExposureFeatureSummary {
  count: number | null;
  nearest_distance_m: number | null;
}

export interface Exposure {
  buffer_m: number | null;
  population: number | null;
  population_source: string | null;
  features: Record<string, ExposureFeatureSummary>;
  note: string;
}

export interface ReachDetailProperties extends ReachProperties {
  upstream_ids: string[];
  downstream_ids: string[];
  catchment_attributes: Record<string, unknown> | null;
  thresholds: Threshold[];
  threshold_derivation: string;
  exposure: Exposure | null;
  history_start: ISODate | null;
  history_end: ISODate | null;
  history: Observation[];
}

export interface ReachDetail {
  type: "Feature";
  id: string;
  geometry: Geometry;
  properties: ReachDetailProperties;
  catchment: CatchmentFeature;
}

export interface ForecastDay {
  target_date: ISODate;
  horizon: number;
  p05: number | null;
  p10: number | null;
  p25: number | null;
  p50: number | null;
  p75: number | null;
  p90: number | null;
  p95: number | null;
  threshold: number | null;
  exceedance_prob: number | null;
  future_drivers_missing: number | null;
}

export interface ForecastSeries {
  variable: Variable;
  display_name: string;
  units: string;
  threshold_derivation: string;
  days: ForecastDay[];
}

export interface ForecastResponse {
  reach_id: string;
  issued_date: ISODate;
  model_version: string;
  variant: string;
  weather: string;
  horizon: number;
  observable: boolean | null;
  observability: Observability;
  notes: string[];
  series: ForecastSeries[];
}

export interface DriverContribution {
  feature: string;
  contribution: number;
  value: number | null;
  kind: string;
}

export interface AttributionSeries {
  variable: Variable;
  selection: "PEAK_EXCEEDANCE" | "PEAK_P90";
  target_date: ISODate;
  horizon: number;
  quantile: string;
  contributions: DriverContribution[];
}

export interface AttributionResponse {
  reach_id: string;
  issued_date: ISODate;
  model_version: string;
  basis: string;
  series: AttributionSeries[];
}

export interface ExposureLayer {
  type: "FeatureCollection";
  city: string;
  note: string;
  features: {
    type: "Feature";
    geometry: Geometry;
    properties: { feature_type: string; reaches: number; reach_ids: string };
  }[];
}

export interface Timeline {
  city: string;
  issued_date: ISODate | null;
  model_version: string | null;
  seasons: Record<string, number[]>;
  threshold_note: string;
  observations: { reach_id: string[]; date: ISODate[]; variable: Variable[]; value: (number | null)[] };
  forecast: {
    reach_id: string[];
    target_date: ISODate[];
    variable: Variable[];
    p10: (number | null)[];
    p50: (number | null)[];
    p90: (number | null)[];
    threshold: (number | null)[];
    exceedance_prob: (number | null)[];
  };
  thresholds: { reach_id: string[]; variable: Variable[]; season: string[]; threshold: (number | null)[] };
}

export interface AlertSummary {
  alert_id: string;
  reach_id: string;
  reach_name: string | null;
  city: string;
  issued_date: ISODate;
  window_start: ISODate;
  window_end: ISODate;
  variable: string;
  severity: Severity;
  exceedance_prob: number | null;
  threshold_value: number | null;
  suppressed_reason: string | null;
  model_version: string | null;
  exposure: {
    buffer_m?: number;
    population?: number | null;
    population_source?: string;
    features?: Record<string, ExposureFeatureSummary>;
  } | null;
}

export interface AlertDetail extends AlertSummary {
  threshold_derivation: string | null;
  attribution: DriverContribution[];
  basis: Record<string, unknown> | null;
  guardrails: Record<string, string> | null;
  withheld_exceedance_prob: number | null;
}

export interface AlertsResponse {
  city: string | null;
  active: boolean;
  alert_run: "OK" | "NO_ALERT_RUN";
  alert_run_issued_date: ISODate | null;
  counts: Record<Severity, number>;
  suppressed: Record<string, number> | null;
  alerts: AlertSummary[];
}

// ---- scenarios -----------------------------------------------------------------------
export type LeverPath = "MODEL_PERTURBATION" | "LITERATURE_DIRECT" | "NOT_ESTIMABLE";

export interface Citation {
  authors: string;
  year: number;
  title: string;
  source: string;
  doi: string | null;
  url: string | null;
  locator: string;
  supports: string[];
  intervention_id?: string;
  role?: "effect" | "cost" | "direct_effect";
}

export interface InterventionInfo {
  id: string;
  name: string;
  feature: string;
  operation: "subtract_treated_share" | "floor" | "scale_down";
  direction: "increase" | "decrease";
  effect_unit: string;
  description: string;
  extent_unit: string | null;
  extent_min: number | null;
  extent_max: number | null;
  magnitude: number;
  uncertainty_range: [number, number];
  applies_to: { reaches: string; conditions: string[] };
  cost_per_unit: {
    low: number;
    high: number;
    currency: string;
    unit: string;
    price_basis: string;
    note?: string | null;
    citation: Citation[];
  };
  citations: Citation[];
  direct_effect: Record<string, unknown> | null;
  paths: Record<string, { path: LeverPath; reason: string }>;
}

export interface InterventionCatalogue {
  city: string;
  coefficient_table_version: string;
  caveat: string;
  interventions: InterventionInfo[];
  not_quantified: { id: string; name: string; reason: string; citations: Citation[] }[];
}

export interface InterventionRequest {
  type: string;
  reach_ids: string[];
  extent: number | null;
}

export interface Interval {
  low: number;
  high: number;
}

export interface Estimate {
  exceedance_days: number;
  interval: Interval;
}

export interface LeverOutcome {
  intervention_id: string;
  path: LeverPath;
  flags: string[];
  detail: string[];
  feature: string;
  value_before: number | null;
  value_after: number | null;
}

export interface ReachResult {
  reach_id: string;
  variable: Variable;
  status: "OK" | "INSUFFICIENT_EVIDENCE" | "NOT_ESTIMABLE" | string;
  reason: string | null;
  days_evaluated: number;
  complete: boolean;
  baseline: Estimate | null;
  scenario: Estimate | null;
  delta_days: number | null;
  delta_pct: number | null;
  delta_coefficient_range: Interval | null;
  levers: LeverOutcome[];
}

export interface CostEstimate {
  intervention_id: string;
  reach_id: string;
  currency: string;
  unit: string;
  unit_cost_low: number;
  unit_cost_high: number;
  price_basis: string;
  quantity: number | null;
  quantity_unit: string | null;
  total_low: number | null;
  total_high: number | null;
  reason: string | null;
  note: string | null;
}

export interface LeverSummary {
  intervention_id: string;
  name: string;
  variable: Variable;
  feature: string;
  operation: string;
  extent: number | null;
  reach_ids: string[];
  path: LeverPath;
  reason: string;
  magnitude: number;
  uncertainty_range: [number, number];
  direct_magnitude: number | null;
  direct_uncertainty_range: [number, number] | null;
  citations: Citation[];
}

export interface ScenarioResult {
  label: "planning estimate";
  caveat: string;
  units: "exceedance days per year";
  model_version: string;
  variant: string;
  coefficient_table_version: string;
  reference_start: ISODate;
  reference_end: ISODate;
  horizon: number;
  ci_level: number;
  widening_factor: number;
  coefficient_samples: number;
  interval_method: string;
  threshold_derivation: string;
  spatial_scope: string;
  variables: Variable[];
  levers: LeverSummary[];
  reaches: ReachResult[];
  costs: CostEstimate[];
  citations: Citation[];
}

export interface ScenarioResponse {
  scenario_id: string;
  name: string | null;
  city: string;
  created_at: string;
  result: ScenarioResult;
}

// ---- validation ----------------------------------------------------------------------
export interface PointMetrics {
  mae: number;
  rmse: number;
  bias: number;
  crps: number;
  coverage_80?: number;
  coverage_90?: number;
}

export interface ScoredBlock {
  n_model: number;
  n_reaches?: number;
  n_common?: number;
  common?: {
    model: PointMetrics;
    seasonal_naive: PointMetrics;
    climatology: PointMetrics;
  };
  skill?: Record<"seasonal_naive" | "climatology", { mae: number; rmse: number; crps: number }>;
  model_all_rows?: PointMetrics;
}

export interface ReliabilityBin {
  bin_low: number;
  bin_high: number;
  n: number;
  mean_forecast: number | null;
  observed_frequency: number | null;
}

export interface ProbabilityBlock {
  threshold_derivation: string;
  n: number;
  event_rate: number;
  bins: ReliabilityBin[];
  brier: number;
  brier_climatology: number;
  brier_skill_vs_climatology: number;
  operating_point_pre_guardrail?: {
    min_exceedance_prob: number;
    flagged: number;
    hits: number;
    false_alarms: number;
    misses: number;
    false_alarm_ratio: number;
    probability_of_detection: number;
  };
}

export interface VariableMetrics {
  by_horizon: Record<string, ScoredBlock>;
  by_bucket: Record<string, ScoredBlock>;
  by_observability: Record<"observable" | "driver_only", ScoredBlock>;
  all_horizons: ScoredBlock;
  per_reach: {
    min_rows: number;
    reaches_scored: number;
    model_loses_to_seasonal_naive: string[];
    model_loses_to_climatology: string[];
  };
  probability: ProbabilityBlock;
}

export interface AnomalyVariable {
  reference: string;
  detections: number;
  events: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  ci95?: Record<"precision" | "recall" | "f1", [number, number]>;
  labels?: Record<string, number>;
}

export interface MetricsDocument {
  city: string;
  generated_at: string;
  model_versions: Record<string, string>;
  headline_run: string;
  folds: Record<string, Record<Variable, VariableMetrics>>;
  runs: Record<string, unknown>;
  losses: {
    run: string;
    fold: string;
    variable: string;
    scope: string;
    baseline: string;
    metric: string;
    model: number;
    baseline_value: number;
    skill: number;
    n: number;
  }[];
  not_computed: Record<string, string>;
  splits?: Record<string, { train_end: string }>;
  variables?: Record<string, { display_name: string; units: string }>;
  /** models/production_calibration.py - CQR applied to what the forecasts table serves. */
  production_calibration?: {
    method: string;
    live_rows_use_fit: string;
    live_note: string | null;
    note: string;
    coverage_80: Record<string, Record<string, Record<string, { coverage_80_raw: number | null; coverage_80_calibrated: number | null }>>>;
  };
  anomaly?: {
    reference: string;
    incidents_available: number;
    weather_label?: string;
    by_variable: Record<string, AnomalyVariable>;
  };
  production_model?: string;
  [k: string]: unknown;
}

export interface ValidationResponse {
  served_at: string;
  source: { path: string; sha256: string; modified_at: string; size_bytes: number };
  metrics: MetricsDocument;
  scenario_response_check: {
    checks?: {
      feature: string;
      variable: string;
      verdict: string;
      standardised_effect: number;
      observed_sign: number;
      expected_sign: number;
    }[];
    [k: string]: unknown;
  } | null;
}
