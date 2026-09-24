import { useEffect, useMemo, useRef, useState } from "react";
import type { Variable } from "../api/types";
import { addDays, fmtDay, fmtMonth, fmtSig, parseDate, VARIABLE_LABEL } from "../lib/format";
import { INK, INK_MUTED, RAMP } from "../lib/ramp";
import type { TimelineIndex } from "../lib/timeline";
import { useStore } from "../store";

const SPANS = [
  { id: "90d", label: "90 days", days: 90 },
  { id: "1y", label: "1 year", days: 365 },
  { id: "3y", label: "3 years", days: 3 * 365 },
  { id: "all", label: "All", days: Infinity },
] as const;

const H = 140;
const PAD = { l: 44, r: 12, t: 20, b: 22 };

/** The hydrograph rail (design.md): one continuous axis, observed history then the forecast
 * fan. The head is the date the map is drawn at. Gaps in observation stay gaps. */
export function HydrographRail({ ix, loading, error }: { ix: TimelineIndex | null; loading: boolean; error: string | null }) {
  const { selected, railDate, setRailDate, railVariable, setRailVariable } = useStore();
  const [span, setSpan] = useState<(typeof SPANS)[number]["id"]>("1y");
  const wrap = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(800);
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    if (wrap.current) ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);

  const issued = ix?.issued ?? null;
  const end = issued ? addDays(issued, 10) : ix?.last ?? null;
  const sp = SPANS.find((s) => s.id === span)!;
  const start = end && ix?.first ? (Number.isFinite(sp.days) ? addDays(issued ?? end, -sp.days) : ix.first) : null;

  const x = useMemo(() => {
    if (!start || !end) return null;
    const t0 = parseDate(start).getTime();
    const t1 = parseDate(end).getTime();
    return (d: string) => PAD.l + ((parseDate(d).getTime() - t0) / (t1 - t0)) * (w - PAD.l - PAD.r);
  }, [start, end, w]);

  const toDate = (px: number) => {
    if (!start || !end) return null;
    const t0 = parseDate(start).getTime();
    const t1 = parseDate(end).getTime();
    const f = Math.min(1, Math.max(0, (px - PAD.l) / (w - PAD.l - PAD.r)));
    return new Date(t0 + f * (t1 - t0)).toISOString().slice(0, 10);
  };

  const head = railDate ?? issued;
  // Pushing past "now" widens the forecast: the rail zooms to 90 days so the fan is legible.
  const pastNow = !!(railDate && issued && railDate > issued);
  useEffect(() => {
    if (pastNow) setSpan("90d");
  }, [pastNow]);
  const setHead = (d: string | null) => setRailDate(d === issued ? null : d);

  // ---- data for the selected reach -------------------------------------------------
  const reach = selected && ix ? { obs: ix.obs.get(selected)?.[railVariable], fc: ix.fc.get(selected) } : null;
  const inRange = (d: string) => !!start && !!end && d >= start && d <= end;
  const pts = reach?.obs ? reach.obs.dates.map((d, i) => [d, reach.obs!.values[i]] as const).filter(([d]) => inRange(d)) : [];
  const fcRows = reach?.fc
    ? [...reach.fc.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([d, r]) => [d, r[railVariable]] as const).filter(([, r]) => r)
    : [];
  const values = [...pts.map(([, v]) => v), ...fcRows.flatMap(([, r]) => [r!.p10, r!.p90, r!.thr]).filter((v): v is number => v !== null)];
  let lo = values.length ? Math.min(...values) : 0;
  let hi = values.length ? Math.max(...values) : 1;
  if (hi === lo) hi = lo + 1;
  const padv = (hi - lo) * 0.08;
  lo -= padv;
  hi += padv;
  const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b);

  // City mode (no reach): reaches with a clear-sky reading per date.
  const counts = !selected && ix ? [...ix.obsCountByDate].filter(([d]) => inRange(d)) : [];
  const maxCount = counts.length ? Math.max(...counts.map(([, n]) => n)) : 1;

  const ticks = useMemo(() => {
    if (!start || !end) return [];
    const out: string[] = [];
    const days = (parseDate(end).getTime() - parseDate(start).getTime()) / 86400000;
    const step = days > 800 ? 12 : days > 200 ? 1 : 0;
    const s = parseDate(start);
    const d = new Date(Date.UTC(s.getUTCFullYear(), s.getUTCMonth() + 1, 1));
    while (d.toISOString().slice(0, 10) <= end) {
      out.push(d.toISOString().slice(0, 10));
      if (step === 0) d.setUTCDate(d.getUTCDate() + 14);
      else d.setUTCMonth(d.getUTCMonth() + step);
    }
    return out;
  }, [start, end]);

  const dragging = useRef(false);
  function onPointer(e: React.PointerEvent<SVGSVGElement>, kind: "down" | "move" | "up") {
    if (kind === "down") {
      dragging.current = true;
      e.currentTarget.setPointerCapture(e.pointerId);
    }
    if (kind === "up") dragging.current = false;
    if (kind !== "up" && dragging.current) {
      const r = e.currentTarget.getBoundingClientRect();
      const d = toDate(e.clientX - r.left);
      if (d) setHead(d);
    }
  }
  function onKey(e: React.KeyboardEvent) {
    if (!head) return;
    const step = e.shiftKey ? 30 : 1;
    if (e.key === "ArrowLeft") setHead(clamp(addDays(head, -step)));
    else if (e.key === "ArrowRight") setHead(clamp(addDays(head, step)));
    else if (e.key === "Home" && start) setHead(start);
    else if (e.key === "End" && issued) setHead(issued);
    else return;
    e.preventDefault();
  }
  const clamp = (d: string) => (start && d < start ? start : end && d > end ? end : d);

  const future = head && issued && head > issued;
  const headLabel = !head
    ? ""
    : head === issued && !railDate
      ? `now — ${fmtDay(head, true)}, map shows the peak over the forecast window`
      : future
        ? `${fmtDay(head, true)} — forecast day ${Math.round((parseDate(head).getTime() - parseDate(issued!).getTime()) / 86400000)}, map shows P(exceed) that day`
        : `${fmtDay(head, true)} — map shows the last clear-sky reading within 10 days, against its seasonal threshold`;

  return (
    <section className="flex h-[172px] shrink-0 flex-col border-t border-hairline" aria-label="Hydrograph rail">
      <div className="flex items-center gap-3 px-3 pt-1.5">
        <span className="t-dense text-ink">
          {selected ? `${selected}, ${VARIABLE_LABEL[railVariable]}` : "Clear-sky readings across the city"}
        </span>
        <span className="t-dense truncate text-muted">{headLabel}</span>
        <div className="ml-auto flex items-center gap-1">
          {selected &&
            (["turbidity_proxy", "ndci"] as Variable[]).map((v) => (
              <button key={v} type="button" onClick={() => setRailVariable(v)} aria-pressed={railVariable === v} className={`t-dense whitespace-nowrap px-1.5 ${railVariable === v ? "text-ink underline underline-offset-4" : "text-muted hover:text-ink"}`}>
                {v === "ndci" ? "NDCI" : "turbidity"}
              </button>
            ))}
          <span className="mx-1 h-3 w-px bg-hairline" />
          {SPANS.map((s) => (
            <button key={s.id} type="button" onClick={() => setSpan(s.id)} aria-pressed={span === s.id} className={`t-dense whitespace-nowrap px-1.5 ${span === s.id ? "text-ink underline underline-offset-4" : "text-muted hover:text-ink"}`}>
              {s.label}
            </button>
          ))}
          <button type="button" onClick={() => setRailDate(null)} disabled={!railDate} className="t-dense ml-1 border border-hairline px-1.5 text-ink disabled:text-muted">
            Now
          </button>
        </div>
      </div>
      <div ref={wrap} className="relative min-h-0 flex-1">
        {(loading || error || !x) && (
          <p className="t-dense absolute left-3 top-2 text-muted">{error ? `Timeline unavailable. ${error}` : loading ? "Loading timeline" : "No observations or forecast for this city."}</p>
        )}
        {x && start && end && (
          <svg
            width={w}
            height={H}
            className="block touch-none select-none outline-offset-[-2px]"
            role="slider"
            tabIndex={0}
            aria-label="Rendered date"
            aria-valuetext={head ? fmtDay(head, true) : undefined}
            aria-valuemin={parseDate(start).getTime()}
            aria-valuemax={parseDate(end).getTime()}
            aria-valuenow={head ? parseDate(head).getTime() : undefined}
            onKeyDown={onKey}
            onPointerDown={(e) => onPointer(e, "down")}
            onPointerMove={(e) => onPointer(e, "move")}
            onPointerUp={(e) => onPointer(e, "up")}
            style={{ cursor: "ew-resize" }}
          >
            {/* month ticks */}
            {ticks.map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={H - PAD.b} y2={H - PAD.b + 4} stroke={INK_MUTED} strokeWidth={1} />
                <text x={x(t) + 2} y={H - 6} fontSize={10} fill={INK_MUTED} fontFamily="IBM Plex Mono">
                  {fmtMonth(t)}
                </text>
              </g>
            ))}
            <line x1={PAD.l} x2={w - PAD.r} y1={H - PAD.b} y2={H - PAD.b} stroke="#C8C5BB" />

            {selected ? (
              <>
                {/* y labels */}
                {[lo + padv, hi - padv].map((v) => (
                  <text key={v} x={PAD.l - 6} y={y(v) + 3} fontSize={10} textAnchor="end" fill={INK_MUTED} fontFamily="IBM Plex Mono">
                    {fmtSig(v, 2)}
                  </text>
                ))}
                {/* forecast fan */}
                {fcRows.length > 1 && (
                  <>
                    <path
                      d={
                        fcRows.map(([d, r], i) => `${i ? "L" : "M"}${x(d)},${y(r!.p90 ?? r!.p50 ?? lo)}`).join("") +
                        [...fcRows].reverse().map(([d, r]) => `L${x(d)},${y(r!.p10 ?? r!.p50 ?? lo)}`).join("") +
                        "Z"
                      }
                      fill={RAMP.turbid}
                      fillOpacity={0.3}
                    />
                    <path d={fcRows.map(([d, r], i) => `${i ? "L" : "M"}${x(d)},${y(r!.p50 ?? lo)}`).join("")} fill="none" stroke={RAMP.heavy} strokeWidth={2} />
                    {fcRows[0][1]!.thr !== null && (
                      <line x1={x(fcRows[0][0])} x2={x(fcRows.at(-1)![0])} y1={y(fcRows[0][1]!.thr!)} y2={y(fcRows[0][1]!.thr!)} stroke={INK_MUTED} strokeDasharray="4 3" />
                    )}
                  </>
                )}
                {/* observations: points, never joined */}
                {pts.map(([d, v]) => (
                  <circle key={d} cx={x(d)} cy={y(v)} r={span === "all" || span === "3y" ? 1.5 : 2.2} fill={INK} />
                ))}
                {!pts.length && !fcRows.length && (
                  <text x={PAD.l + 8} y={PAD.t + 20} fontSize={11} fill={INK_MUTED}>
                    No clear-sky reading of this reach in this span — it is driver-predicted, or cloud and narrow water hid it.
                  </text>
                )}
              </>
            ) : (
              counts.map(([d, n]) => (
                <line key={d} x1={x(d)} x2={x(d)} y1={H - PAD.b} y2={H - PAD.b - (n / maxCount) * (H - PAD.t - PAD.b)} stroke={RAMP.clear} strokeWidth={1.5} />
              ))
            )}
            {!selected && (
              <text x={PAD.l - 6} y={PAD.t + 4} fontSize={10} textAnchor="end" fill={INK_MUTED} fontFamily="IBM Plex Mono">
                {maxCount}
              </text>
            )}

            {/* now */}
            {issued && (
              <g>
                <line x1={x(issued)} x2={x(issued)} y1={PAD.t - 8} y2={H - PAD.b} stroke={INK} strokeWidth={1} />
                <text x={x(issued) + 3} y={PAD.t - 10} fontSize={10} fill={INK}>
                  now
                </text>
              </g>
            )}
            {/* the head */}
            {head && head !== issued && (
              <g>
                <line x1={x(head)} x2={x(head)} y1={PAD.t - 8} y2={H - PAD.b} stroke="#1B6B8C" strokeWidth={2} />
                <rect x={x(head) - 4} y={PAD.t - 12} width={8} height={8} fill="#1B6B8C" />
              </g>
            )}
          </svg>
        )}
      </div>
    </section>
  );
}
