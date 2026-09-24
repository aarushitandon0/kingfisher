// The exceedance ramp and the theme's colours, in two forms:
//  - CSS variables (src/theme.css) for the DOM and SVG charts: they follow light/dark.
//  - hex PALETTE twins for MapLibre, which cannot read CSS variables.
// Break points are a fixed display scale, stated in the legend - never fitted to the data
// on screen, so the same colour means the same thing on every city and every date.

export type Theme = "light" | "dark";

export const PALETTE = {
  light: {
    ramp: ["#CBD9E6", "#8FB7D6", "#E8B24A", "#D9772E", "#B8331F"],
    unknown: "#9A9CA3",
    hatch: "#7D7F86",
    alert: "#C6432B",
    watch: "#D98C2B",
    brand: "#1C6FA8",
    ink: "#1B1D22",
    inkMuted: "#5D6068",
    surface: "#FFFFFF",
    // basemap
    bg: "#F4F3EF",
    land: "#EDEDE8",
    landuse: "#EAE9E3",
    park: "#E3E8DF",
    building: "#E4E2DC",
    water: "#C9DDEC",
    waterway: "#B7D1E4",
    road: "#FFFFFF",
    roadCasing: "#DCDAD3",
    rail: "#D2D0C8",
    label: "#6E7178",
    waterLabel: "#4F7394",
    shadow: "#6A6F78",
    highlight: "#FFFFFF",
  },
  dark: {
    ramp: ["#3A5068", "#5E93BF", "#E8B24A", "#E0833A", "#E8503A"],
    unknown: "#80838B",
    hatch: "#8A8D95",
    alert: "#E8503A",
    watch: "#E59A3C",
    brand: "#4FA3DC",
    ink: "#F2F2F0",
    inkMuted: "#A7AAB2",
    surface: "#1C1F25",
    bg: "#15171B",
    land: "#181A1F",
    landuse: "#1A1D22",
    park: "#1A2020",
    building: "#20232A",
    water: "#1B2B3B",
    waterway: "#24405A",
    road: "#2A2D34",
    roadCasing: "#1F2228",
    rail: "#2C2F36",
    label: "#8A8E97",
    waterLabel: "#6F9CC4",
    shadow: "#000000",
    highlight: "#3A3E47",
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
export const RAMP_WIDTH = [2, 2.5, 3, 4, 5];

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
export function rampExpression(prop: string, breaks: number[], p: Palette): unknown {
  const s = p.ramp;
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

export const SEVERITY_COLOR: Record<string, string> = {
  ALERT,
  WATCH,
  INSUFFICIENT_EVIDENCE: UNKNOWN,
};
