// The exceedance ramp and the theme's colours, in two forms:
//  - CSS variables (src/theme.css) for the DOM and SVG charts: they follow light/dark.
//  - hex PALETTE twins for MapLibre, which cannot read CSS variables.
// Break points are a fixed display scale, stated in the legend - never fitted to the data
// on screen, so the same colour means the same thing on every city and every date.

export type Theme = "light" | "dark";

export const PALETTE = {
  dark: {
    ramp: ["#2FA8E0", "#8DB9C9", "#E0B24A", "#E8823A", "#E8503A"],
    change: ["#2E9E6B", "#6FC79E", "#565B64", "#E0975A", "#E0573A"],
    unknown: "#4A6B85", // no value: dim water, hatched over
    hatch: "#9AA3AE",
    alert: "#E85D3D",
    watch: "#E8A33D",
    brand: "#2E8FD6",
    brandLight: "#6FB8E8",
    ink: "#F2F2F0",
    inkMuted: "#A7AAB2",
    surface: "#12151B",
    // basemap: water-first on warm graphite land, so the map never merges with the navy/black
    // chrome and blue water + the severity ramp are the only cool, saturated marks.
    bg: "#24252A",
    landuse: "#2A2B2F",
    park: "#1F3A2B",
    building: "#34353A",
    water: "#0F4470",
    waterway: "#3A8FCC",
    road: "#50525A",
    roadOpacity: 0.7,
    rail: "#44464D",
    boundary: "#5A5D66",
    label: "#C9C6BD",
    labelMinor: "#8E8C86",
    waterLabel: "#7CC4F0",
    halo: "#1B1C20",
  },
  light: {
    ramp: ["#1583C4", "#7FA6BD", "#DCA23A", "#D9702A", "#BF321F"],
    change: ["#1F7F54", "#52B085", "#9AA0A8", "#D9844A", "#C2412A"],
    unknown: "#8EAEC6",
    hatch: "#5E6670",
    alert: "#C6432B",
    watch: "#D98C2B",
    brand: "#1C6FA8",
    brandLight: "#4A9BD4",
    ink: "#161A20",
    inkMuted: "#525862",
    surface: "#FFFFFF",
    // warm paper land against the cool grey chrome; white roads; green parks; clear blue water.
    bg: "#F3EFE6",
    landuse: "#ECE7DB",
    park: "#CFE5C6",
    building: "#E2DCCF",
    water: "#9FCBEC",
    waterway: "#5AA0D6",
    road: "#FFFFFF",
    roadOpacity: 0.95,
    rail: "#C9C2B3",
    boundary: "#B3AA98",
    label: "#4A4740",
    labelMinor: "#7E7A70",
    waterLabel: "#155E94",
    halo: "#F3EFE6",
  },
} as const;

export type Palette = (typeof PALETTE)[Theme];

/** DOM/SVG colours - CSS variables, so they follow the theme without a re-render. */
const STOPS = ["var(--exceed-0)", "var(--exceed-1)", "var(--exceed-2)", "var(--exceed-3)", "var(--exceed-4)"];
export const RAMP = {
  clear: STOPS[0],
  slight: STOPS[1],
  turbid: STOPS[2],
  heavy: STOPS[3],
  severe: STOPS[4],
} as const;

export const UNKNOWN = "var(--severity-insufficient)";
export const ALERT = "var(--severity-critical)";
export const WATCH = "var(--severity-watch)";
export const KINGFISHER = "var(--brand-500)";
export const BRAND = "var(--brand-500)";
export const BRAND_700 = "var(--brand-700)";
export const SCENARIO = "var(--scenario)";
export const INK = "var(--text-primary)";
export const INK_MUTED = "var(--text-secondary)";
export const INK_FAINT = "var(--text-tertiary)";
export const HAIRLINE = "var(--border)";
export const PAPER = "var(--surface)";
export const PAPER_ALT = "var(--surface-sunken)";

/** P(exceed) bands: <0.1, 0.1–0.3, 0.3–0.5, 0.5–0.7, >=0.7. */
export const PROB_BREAKS = [0.1, 0.3, 0.5, 0.7];
/** Observed value / its seasonal threshold: <0.6, 0.6–0.9, 0.9–1.0, 1.0–1.3, >=1.3. */
export const RATIO_BREAKS = [0.6, 0.9, 1.0, 1.3];
/** Exceedance days per year, for the scenario map: <10, 10–30, 30–60, 60–120, >=120. */
export const DAYS_BREAKS = [10, 30, 60, 120];

/** Line weight per ramp band: severity also reads as weight, not colour alone. */
export const RAMP_WIDTH = [3, 3.5, 4, 5, 6];

function band(v: number, breaks: number[]): number {
  let i = 0;
  while (i < breaks.length && v >= breaks[i]) i++;
  return i;
}

export function rampColor(v: number | null | undefined, breaks: number[]): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return UNKNOWN;
  return STOPS[band(v, breaks)];
}

/** MapLibre step expression over a numeric property, `unknown` where it is null. */
export function rampExpression(prop: string, breaks: number[], p: Palette, scheme: Scheme = "exceed"): unknown {
  const s = scheme === "change" ? p.change : p.ramp;
  return [
    "case",
    ["==", ["typeof", ["get", prop]], "number"],
    ["step", ["get", prop], s[0], breaks[0], s[1], breaks[1], s[2], breaks[2], s[3], breaks[3], s[4]],
    p.unknown,
  ];
}

/** MapLibre line width by band, scaled (driver-predicted lines are drawn thinner). */
export function widthExpression(prop: string, breaks: number[], scale = 1): unknown {
  const w = RAMP_WIDTH.map((x) => x * scale);
  return [
    "case",
    ["==", ["typeof", ["get", prop]], "number"],
    ["step", ["get", prop], w[0], breaks[0], w[1], breaks[1], w[2], breaks[2], w[3], breaks[3], w[4]],
    w[0],
  ];
}

export const legendStops = (breaks: number[], fmt: (v: number) => string) => [
  { color: STOPS[0], width: RAMP_WIDTH[0], label: `< ${fmt(breaks[0])}` },
  { color: STOPS[1], width: RAMP_WIDTH[1], label: `${fmt(breaks[0])}–${fmt(breaks[1])}` },
  { color: STOPS[2], width: RAMP_WIDTH[2], label: `${fmt(breaks[1])}–${fmt(breaks[2])}` },
  { color: STOPS[3], width: RAMP_WIDTH[3], label: `${fmt(breaks[2])}–${fmt(breaks[3])}` },
  { color: STOPS[4], width: RAMP_WIDTH[4], label: `≥ ${fmt(breaks[3])}` },
];

export type Scheme = "exceed" | "change";

/** Scenario change in exceedance days per year (scenario - baseline). Negative = fewer
 * days = improvement. |change| below the middle pair is "negligible": the map says so. */
export const CHANGE_BREAKS = [-5, -0.05, 0.05, 5];
export const NEGLIGIBLE_DAYS = 0.05;
const CHANGE_STOPS = ["var(--change-improve-strong)", "var(--change-improve)", "var(--change-neutral)", "var(--change-worsen)", "var(--change-worsen-strong)"];

export function changeColor(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return UNKNOWN;
  return CHANGE_STOPS[band(v, CHANGE_BREAKS)];
}

export const changeLegend = [
  { color: CHANGE_STOPS[4], label: "worse by 5+ days" },
  { color: CHANGE_STOPS[3], label: "worse" },
  { color: CHANGE_STOPS[2], label: "negligible (|Δ| < 0.05 d)" },
  { color: CHANGE_STOPS[1], label: "better" },
  { color: CHANGE_STOPS[0], label: "better by 5+ days" },
];

export const SEVERITY_COLOR: Record<string, string> = {
  ALERT,
  WATCH,
  INSUFFICIENT_EVIDENCE: UNKNOWN,
};
