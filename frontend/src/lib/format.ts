// Formatting only. Nothing here computes a quantity the API did not provide.

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "2026-09-21" -> Date at UTC midnight. All dates are UTC calendar dates. */
export const parseDate = (d: string) => new Date(`${d.slice(0, 10)}T00:00:00Z`);
export const isoDate = (d: Date) => d.toISOString().slice(0, 10);
export const addDays = (d: string, n: number) => isoDate(new Date(parseDate(d).getTime() + n * 86400000));
export const daysBetween = (a: string, b: string) =>
  Math.round((parseDate(b).getTime() - parseDate(a).getTime()) / 86400000);

/** "21 Sep" / "21 Sep 2024" */
export function fmtDay(d: string | null | undefined, withYear = false): string {
  if (!d) return "";
  const x = parseDate(d);
  const s = `${x.getUTCDate()} ${MONTHS[x.getUTCMonth()]}`;
  return withYear ? `${s} ${x.getUTCFullYear()}` : s;
}

/** "21–30 Sep" or "28 Sep – 2 Oct" */
export function fmtRange(a: string, b: string): string {
  const x = parseDate(a);
  const y = parseDate(b);
  if (x.getUTCMonth() === y.getUTCMonth() && x.getUTCFullYear() === y.getUTCFullYear())
    return `${x.getUTCDate()}–${y.getUTCDate()} ${MONTHS[y.getUTCMonth()]}`;
  return `${fmtDay(a)} – ${fmtDay(b)}`;
}

export const fmtMonth = (d: string) => {
  const x = parseDate(d);
  return x.getUTCMonth() === 0 ? String(x.getUTCFullYear()) : MONTHS[x.getUTCMonth()];
};

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

/** Significant-figure formatting for values of unknown scale (index values, SHAP). */
export function fmtSig(v: number | null | undefined, sig = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 1) return Number(v.toPrecision(sig)).toString();
  return Number(v.toPrecision(Math.max(1, sig - 1))).toString();
}

export const fmtSigned = (v: number, sig = 2) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmtSig(Math.abs(v), sig);

export const fmtProb = (p: number | null | undefined) => fmtNum(p, 2);

export function fmtDistance(m: number | null | undefined): string {
  if (m === null || m === undefined) return "—";
  return m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`;
}

export function fmtMoney(v: number, currency: string): string {
  const n = v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e4 ? `${Math.round(v / 1e3)}k` : v.toLocaleString("en-GB");
  return `${currency} ${n}`;
}

export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** Variable labels as the metrics document names them; units are dimensionless indices. */
export const VARIABLE_LABEL: Record<string, string> = {
  turbidity_proxy: "turbidity index",
  ndci: "chlorophyll index (NDCI)",
};

/** Model feature names -> plain labels. Unknown names fall back to the raw name, so a new
 * feature is never hidden. */
const FEATURE_LABEL: Record<string, string> = {
  precip_mm: "rain, today",
  precip_mm_fut: "forecast rain",
  precip_max_hourly: "peak hourly rain",
  precip_max_hourly_fut: "forecast peak hourly rain",
  precip_duration_h: "rain duration",
  precip_fut_cum_mm: "forecast rain, cumulative",
  temp_mean_c: "air temperature",
  temp_mean_c_fut: "forecast air temperature",
  temp_max_c: "max air temperature",
  soil_moisture: "soil moisture",
  et0: "evapotranspiration",
  api_7: "7-day antecedent rain",
  api_7_fut: "forecast 7-day antecedent rain",
  api_14: "14-day antecedent rain",
  api_30: "30-day antecedent rain",
  antecedent_dry_days: "dry days",
  first_flush_index: "first-flush index",
  first_flush_index_fut: "forecast first-flush index",
  temp_low_flow_index: "warm low-flow index",
  doy_sin: "season (sin)",
  doy_cos: "season (cos)",
  upstream_state_lag1_turbidity: "upstream turbidity, yesterday",
  upstream_state_lag1_ndci: "upstream NDCI, yesterday",
  imperviousness_pct: "imperviousness",
  riparian_ndvi_mean: "riparian greenness",
  riparian_width_m: "riparian width",
  road_density_km_km2: "road density",
  alan_radiance: "night-time light",
  population: "catchment population",
  catchment_area_km2: "catchment area",
  strahler_order: "stream order",
  observable: "observable",
  median_water_pixels: "water pixels",
  turbidity_proxy_asof: "last turbidity reading",
  ndci_asof: "last NDCI reading",
  obs_asof_age_days: "age of last reading",
};
export const featureLabel = (f: string) => FEATURE_LABEL[f] ?? f.replaceAll("_", " ");

export const EXPOSURE_LABEL: Record<string, string> = {
  school: "school",
  kindergarten: "kindergarten",
  playground: "playground",
  park: "park",
  healthcare: "healthcare",
  footway: "footpath",
  cycleway: "cycleway",
  water_access: "water access point",
  population: "residents",
};
export const exposureLabel = (t: string) => EXPOSURE_LABEL[t] ?? t.replaceAll("_", " ");

/** "INSUFFICIENT_EVIDENCE" -> "Insufficient evidence" */
export const humanize = (s: string) => {
  const t = s.replaceAll("_", " ").toLowerCase();
  return t.charAt(0).toUpperCase() + t.slice(1);
};
