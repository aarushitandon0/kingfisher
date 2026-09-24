// The sediment ramp (design.md): clear water -> silt. The colour IS the measurement.
// Break points are a fixed display scale, stated in the legend - never fitted to the data
// on screen, so the same colour means the same thing on every city and every date.

export const RAMP = {
  clear: "#A8C4CC",
  slight: "#C4C2AE",
  turbid: "#C79A5B",
  heavy: "#9A6636",
  severe: "#6B4423",
} as const;

export const UNKNOWN = "#9B9A94";
export const ALERT = "#B03A2E";
export const WATCH = "#C79A5B";
export const KINGFISHER = "#1B6B8C";
export const SCENARIO = "#4A7C59";
export const INK = "#1C1F1E";
export const INK_MUTED = "#5A605D";
export const HAIRLINE = "#C8C5BB";
export const PAPER = "#F2F0EA";
export const PAPER_ALT = "#E8E5DC";

const STOPS = [RAMP.clear, RAMP.slight, RAMP.turbid, RAMP.heavy, RAMP.severe];

/** P(exceed) bands: <0.1, 0.1–0.3, 0.3–0.5, 0.5–0.7, >=0.7. */
export const PROB_BREAKS = [0.1, 0.3, 0.5, 0.7];
/** Observed value / its seasonal threshold: <0.6, 0.6–0.9, 0.9–1.0, 1.0–1.3, >=1.3. */
export const RATIO_BREAKS = [0.6, 0.9, 1.0, 1.3];
/** Exceedance days per year, for the scenario map: <10, 10–30, 30–60, 60–120, >=120. */
export const DAYS_BREAKS = [10, 30, 60, 120];

export function rampColor(v: number | null | undefined, breaks: number[]): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return UNKNOWN;
  let i = 0;
  while (i < breaks.length && v >= breaks[i]) i++;
  return STOPS[i];
}

/** MapLibre step expression over a numeric property, UNKNOWN where it is null. */
export function rampExpression(prop: string, breaks: number[]): unknown {
  return [
    "case",
    ["==", ["typeof", ["get", prop]], "number"],
    ["step", ["get", prop], STOPS[0], breaks[0], STOPS[1], breaks[1], STOPS[2], breaks[2], STOPS[3], breaks[3], STOPS[4]],
    UNKNOWN,
  ];
}

export const legendStops = (breaks: number[], fmt: (v: number) => string) => [
  { color: STOPS[0], label: `< ${fmt(breaks[0])}` },
  { color: STOPS[1], label: `${fmt(breaks[0])}–${fmt(breaks[1])}` },
  { color: STOPS[2], label: `${fmt(breaks[1])}–${fmt(breaks[2])}` },
  { color: STOPS[3], label: `${fmt(breaks[2])}–${fmt(breaks[3])}` },
  { color: STOPS[4], label: `≥ ${fmt(breaks[3])}` },
];

export const SEVERITY_COLOR: Record<string, string> = {
  ALERT,
  WATCH,
  INSUFFICIENT_EVIDENCE: UNKNOWN,
};
