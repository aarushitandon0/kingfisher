import type { Timeline, Variable } from "../api/types";
import { addDays, parseDate } from "./format";

/** How far back the map looks for a reach's last OK reading when drawn at a past date.
 * Older than this and the reach is drawn as "no recent observation" - a gap stays a gap. */
export const LOOKBACK_DAYS = 10;

export interface ReachAtDate {
  /** past: observed / seasonal threshold (max over variables); future: daily P(exceed). */
  value: number | null;
  /** future only: (P90 - P10) / threshold, the soft edge width; null in the past. */
  spread: number | null;
  /** past: date of the reading used. */
  obsDate: string | null;
}

interface Series {
  dates: string[];
  values: number[];
}

export interface TimelineIndex {
  issued: string | null;
  first: string | null;
  last: string | null;
  obs: Map<string, Record<Variable, Series>>;
  /** reach -> target_date -> variable -> row */
  fc: Map<string, Map<string, Partial<Record<Variable, { p: number | null; p10: number | null; p50: number | null; p90: number | null; thr: number | null }>>>>;
  thresholds: Map<string, number>; // `${reach}|${variable}|${season}`
  seasonOfMonth: string[]; // 1..12 -> season
  obsCountByDate: Map<string, number>;
  note: string;
}

export function buildIndex(t: Timeline): TimelineIndex {
  const obs = new Map<string, Record<Variable, Series>>();
  const counts = new Map<string, Set<string>>();
  const o = t.observations;
  for (let i = 0; i < o.reach_id.length; i++) {
    const v = o.value[i];
    if (v === null) continue;
    const r = o.reach_id[i];
    let rec = obs.get(r);
    if (!rec) {
      rec = { turbidity_proxy: { dates: [], values: [] }, ndci: { dates: [], values: [] } };
      obs.set(r, rec);
    }
    rec[o.variable[i]].dates.push(o.date[i]);
    rec[o.variable[i]].values.push(v);
    let c = counts.get(o.date[i]);
    if (!c) counts.set(o.date[i], (c = new Set()));
    c.add(r);
  }
  const fc: TimelineIndex["fc"] = new Map();
  const f = t.forecast;
  for (let i = 0; i < f.reach_id.length; i++) {
    let byDate = fc.get(f.reach_id[i]);
    if (!byDate) fc.set(f.reach_id[i], (byDate = new Map()));
    let row = byDate.get(f.target_date[i]);
    if (!row) byDate.set(f.target_date[i], (row = {}));
    row[f.variable[i]] = { p: f.exceedance_prob[i], p10: f.p10[i], p50: f.p50[i], p90: f.p90[i], thr: f.threshold[i] };
  }
  const thresholds = new Map<string, number>();
  const th = t.thresholds;
  for (let i = 0; i < th.reach_id.length; i++) {
    const v = th.threshold[i];
    if (v !== null) thresholds.set(`${th.reach_id[i]}|${th.variable[i]}|${th.season[i]}`, v);
  }
  const seasonOfMonth: string[] = [];
  for (const [s, months] of Object.entries(t.seasons)) for (const m of months) seasonOfMonth[m] = s;
  const allDates = [...o.date, ...f.target_date].sort();
  return {
    issued: t.issued_date,
    first: allDates[0] ?? null,
    last: allDates[allDates.length - 1] ?? null,
    obs,
    fc,
    thresholds,
    seasonOfMonth,
    obsCountByDate: new Map([...counts].map(([d, s]) => [d, s.size])),
    note: t.threshold_note,
  };
}

/** Index of the last element <= target in a sorted string array, or -1. */
function lastAtOrBefore(dates: string[], target: string): number {
  let lo = 0;
  let hi = dates.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (dates[mid] <= target) {
      ans = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return ans;
}

const VARS: Variable[] = ["turbidity_proxy", "ndci"];

export function reachAtDate(ix: TimelineIndex, reach: string, date: string): ReachAtDate {
  const future = ix.issued !== null && date > ix.issued;
  if (future) {
    const row = ix.fc.get(reach)?.get(date);
    if (!row) return { value: null, spread: null, obsDate: null };
    let p: number | null = null;
    let spread: number | null = null;
    for (const v of VARS) {
      const r = row[v];
      if (!r) continue;
      if (r.p !== null) p = p === null ? r.p : Math.max(p, r.p);
      if (r.thr !== null && r.thr !== 0 && r.p10 !== null && r.p90 !== null) {
        const s = Math.abs(r.p90 - r.p10) / Math.abs(r.thr);
        spread = spread === null ? s : Math.max(spread, s);
      }
    }
    return { value: p, spread, obsDate: null };
  }
  const rec = ix.obs.get(reach);
  if (!rec) return { value: null, spread: null, obsDate: null };
  const earliest = addDays(date, -LOOKBACK_DAYS);
  let ratio: number | null = null;
  let used: string | null = null;
  for (const v of VARS) {
    const s = rec[v];
    const i = lastAtOrBefore(s.dates, date);
    if (i < 0 || s.dates[i] < earliest) continue;
    const month = parseDate(s.dates[i]).getUTCMonth() + 1;
    const thr = ix.thresholds.get(`${reach}|${v}|${ix.seasonOfMonth[month]}`);
    if (thr === undefined || thr === 0) continue;
    const r = s.values[i] / thr;
    if (ratio === null || r > ratio) {
      ratio = r;
      used = s.dates[i];
    }
  }
  return { value: ratio, spread: null, obsDate: used };
}
