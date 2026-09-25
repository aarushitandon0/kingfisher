import { useEffect, useMemo, useState } from "react";
import { api, apiHref } from "../api/client";
import type { AlertSummary, AlertsResponse, ReachCollection, Severity } from "../api/types";
import { Card, Chevron, ErrorNote, Loading, PageHeader, Segmented, severityBorder, SeverityLabel, Skeleton, Stat } from "../components/bits";
import { AttributionBars } from "../components/charts";
import { Splitter, usePanelSize } from "../components/Splitter";
import { useShortcut } from "../lib/shortcuts";
import { exposureLabel, fmtDay, fmtDistance, fmtProb, fmtRange, fmtSig, humanize, VARIABLE_LABEL } from "../lib/format";
import { PROB_BREAKS, SEVERITY_COLOR } from "../lib/ramp";
import { useApi, type Loadable } from "../lib/useApi";
import { ReachMap, type ReachValue } from "../map/ReachMap";
import { useStore } from "../store";

const SEVERITIES: Severity[] = ["ALERT", "WATCH", "INSUFFICIENT_EVIDENCE"];
/** Long groups show this many rows until expanded, so 600 hatched rows never bury the table. */
const GROUP_PREVIEW = 25;
const VAR_SHORT: Record<string, string> = { turbidity_proxy: "Turbidity", ndci: "NDCI" };

function exposureSummary(e: AlertSummary["exposure"]): string {
  if (!e) return "not computed";
  const feats = (e.features ?? {}) as Record<string, { count: number | null; nearest_distance_m: number | null }>;
  const near = Object.entries(feats)
    .filter(([, f]) => f.count && f.nearest_distance_m !== null)
    .sort((a, b) => (a[1].nearest_distance_m ?? 0) - (b[1].nearest_distance_m ?? 0))
    .slice(0, 2)
    .map(([k, f]) => `${exposureLabel(k)} ${fmtDistance(f.nearest_distance_m)}`);
  const pop = typeof e.population === "number" ? `${e.population} residents` : null;
  const parts = [...near, pop].filter(Boolean);
  return parts.length ? parts.join(", ") : `nothing within ${e.buffer_m ?? "—"} m`;
}

const reasonCode = (reason: string | null) => (reason ? reason.split(":")[0] : "UNSPECIFIED");

export function AlertsView({ city, reaches }: { city: string; reaches: Loadable<ReachCollection> }) {
  const res = useApi(`alerts:${city}`, () => api.alerts(city));
  const [shown, setShown] = useState<Record<Severity, boolean>>({ ALERT: true, WATCH: true, INSUFFICIENT_EVIDENCE: true });
  const [variable, setVariable] = useState<"all" | "turbidity_proxy" | "ndci">("all");
  const [q, setQ] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [side, setSide] = usePanelSize("alerts.side", null);
  const sideW = side ?? 360;
  useShortcut("b", () => setSide(side === 0 ? null : 0), { ctrl: true });
  const vw = typeof window !== "undefined" ? window.innerWidth : 1440;
  const r = res.data;

  const rows = useMemo(() => {
    const t = q.trim().toLowerCase();
    return (r?.alerts ?? []).filter(
      (a) => (variable === "all" || a.variable === variable) && (!t || a.reach_id.toLowerCase().includes(t) || (a.reach_name ?? "").toLowerCase().includes(t)),
    );
  }, [r, variable, q]);
  // Grouped by severity for scanning; within a group the API's order is kept.
  const groups = useMemo(() => SEVERITIES.filter((s) => shown[s]).map((s) => [s, rows.filter((a) => a.severity === s)] as const), [rows, shown]);
  const empty = groups.every(([, g]) => g.length === 0);
  const openAlert = r?.alerts.find((a) => a.alert_id === open) ?? null;

  return (
    <div className="page page-wide min-h-0 flex-1 overflow-y-auto">
      <PageHeader
        title="Alerts"
        actions={
          r?.alert_run === "OK" && (
            <>
              <div className="search w-full sm:w-[240px]">
                <svg aria-hidden viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
                  <circle cx="7" cy="7" r="4.5" />
                  <path d="m10.5 10.5 3 3" />
                </svg>
                <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Find a reach or stream" aria-label="Find a reach or stream" data-search />
              </div>
              <Segmented
                label="Variable"
                value={variable}
                onChange={setVariable}
                options={[
                  { value: "all", label: "All variables" },
                  { value: "turbidity_proxy", label: "Turbidity" },
                  { value: "ndci", label: "NDCI" },
                ]}
              />
            </>
          )
        }
      >
        {r?.alert_run === "OK" && (
          <>
            Latest run, issued <span className="text-ink">{fmtDay(r.alert_run_issued_date, true)}</span>. Each reach is assessed for both variables.
          </>
        )}
      </PageHeader>
      {res.loading && (
        <div>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            {[0, 1, 2, 3].map((i) => (
              <span key={i} className="skeleton block h-[116px] !rounded-[var(--radius-md)]" />
            ))}
          </div>
          <div className="card mt-6 p-5">
            <Skeleton label="alerts" lines={8} />
          </div>
        </div>
      )}
      <ErrorNote error={res.error} what="Alerts" />
      {r && <RunSummary r={r} shown={shown} toggle={(s) => setShown({ ...shown, [s]: !shown[s] })} />}

      {r && r.alert_run === "OK" && (
        <div className="mt-6 flex flex-col gap-6 min-[1280px]:flex-row min-[1280px]:items-start min-[1280px]:gap-2" style={{ ["--side-w" as string]: `${sideW}px` }}>
          <div className="card min-w-0 overflow-hidden min-[1280px]:flex-1">
            <AlertList groups={groups} expanded={expanded} expand={(s) => setExpanded({ ...expanded, [s]: true })} onOpen={setOpen} />
            <div className="hidden max-h-[calc(100vh-150px)] overflow-auto md:block">
              <table className="dtable min-w-[860px] text-[13px]">
                <thead>
                  <tr>
                    <th scope="col">Reach</th>
                    <th scope="col">Severity</th>
                    <th scope="col">Variable</th>
                    <th scope="col" className="!text-right">
                      P(exceed)
                    </th>
                    <th scope="col">Window</th>
                    <th scope="col">Exposure or reason</th>
                    <th scope="col">
                      <span className="sr-only">Details</span>
                    </th>
                  </tr>
                </thead>
                {groups.map(([s, g]) => {
                  if (g.length === 0) return null;
                  const all = expanded[s] || g.length <= GROUP_PREVIEW + 5;
                  const visible = all ? g : g.slice(0, GROUP_PREVIEW);
                  return (
                    <tbody key={s}>
                      <tr>
                        <th scope="colgroup" colSpan={7} className="border-b border-hairline bg-raised py-3 pl-4 text-left font-normal">
                          <span className="flex items-center gap-3">
                            <span className="t-h3">{s === "ALERT" ? "Alert" : s === "WATCH" ? "Watch" : "Insufficient evidence"}</span>
                            <span className="t-value-sm rounded-full bg-surface px-2 py-0.5 text-muted">{g.length}</span>
                            {s === "INSUFFICIENT_EVIDENCE" && <span className="t-dense text-muted">the system declined to alert: an output, not a gap</span>}
                          </span>
                        </th>
                      </tr>
                      {visible.map((a) => {
                        const b = severityBorder(a.severity);
                        const reason = a.severity === "INSUFFICIENT_EVIDENCE" ? reasonText(a.suppressed_reason) : exposureSummary(a.exposure);
                        return (
                          <tr
                            key={a.alert_id}
                            className={`row-hover group h-11 cursor-pointer ${open === a.alert_id ? "bg-raised" : ""}`}
                            onClick={() => setOpen(a.alert_id)}
                            onMouseEnter={() => useStore.getState().hover(a.reach_id)}
                            onMouseLeave={() => useStore.getState().hover(null)}
                          >
                            <td className={b.className} style={b.style}>
                              <span className="font-mono text-[12.5px] font-medium whitespace-nowrap text-ink">{a.reach_id}</span> <span className="text-muted">{a.reach_name ?? ""}</span>
                            </td>
                            <td className="whitespace-nowrap">
                              <SeverityLabel s={a.severity} small />
                            </td>
                            <td className="whitespace-nowrap text-muted">{VAR_SHORT[a.variable] ?? a.variable}</td>
                            <td className="num">{a.exceedance_prob !== null ? <span className="font-semibold text-ink">{fmtProb(a.exceedance_prob)}</span> : <span className="font-sans text-faint italic">withheld</span>}</td>
                            <td className="t-value-sm whitespace-nowrap text-muted">{fmtRange(a.window_start, a.window_end)}</td>
                            <td className="max-w-[340px] truncate text-muted" title={reason}>
                              {a.severity === "INSUFFICIENT_EVIDENCE" ? <span className="text-ink">{reason}</span> : reason}
                            </td>
                            <td className="w-10 text-right">
                              <button
                                type="button"
                                aria-label={`Details for ${a.reach_id}, ${VARIABLE_LABEL[a.variable] ?? a.variable}`}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setOpen(a.alert_id);
                                }}
                                className="btn btn-ghost btn-icon !h-8 !w-8 text-faint group-hover:text-ink hover:!text-brand-300"
                              >
                                <Chevron />
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                      {!all && (
                        <tr>
                          <td colSpan={7} className="!py-2">
                            <button type="button" className="btn btn-sm" onClick={() => setExpanded({ ...expanded, [s]: true })}>
                              Show all {g.length} rows
                            </button>
                          </td>
                        </tr>
                      )}
                    </tbody>
                  );
                })}
              </table>
            </div>
            {empty && <p className="t-ui px-4 py-6 text-muted">No rows match. Turn a severity tile back on, or clear the search.</p>}
          </div>

          <Splitter
            axis="x"
            reverse
            collapsible
            size={sideW}
            onSize={setSide}
            min={280}
            max={Math.max(300, vw - 760)}
            label="Summary column"
            className="hidden !self-stretch min-[1280px]:flex"
          />
          <div className={`flex min-w-0 flex-col gap-6 min-[1280px]:w-(--side-w) min-[1280px]:shrink-0 ${sideW === 0 ? "min-[1280px]:hidden" : ""}`}>
            {reaches.data && <MiniMap fc={reaches.data} />}
            <RunBreakdown r={r} />
            <ByVariable r={r} />
            <Reasons r={r} />
          </div>
        </div>
      )}
      {openAlert && <AlertDrawer a={openAlert} onClose={() => setOpen(null)} />}
    </div>
  );
}

type Group = readonly [Severity, AlertSummary[]];

/** Phones: the same rows as the table, stacked, so nothing scrolls sideways. */
function AlertList({ groups, expanded, expand, onOpen }: { groups: Group[]; expanded: Record<string, boolean>; expand: (s: Severity) => void; onOpen: (id: string) => void }) {
  return (
    <div className="md:hidden">
      {groups.map(([s, g]) => {
        if (g.length === 0) return null;
        const all = expanded[s] || g.length <= GROUP_PREVIEW + 5;
        const visible = all ? g : g.slice(0, GROUP_PREVIEW);
        return (
          <section key={s} aria-label={`${s === "ALERT" ? "Alert" : s === "WATCH" ? "Watch" : "Insufficient evidence"} rows`}>
            <header className="sticky top-0 z-[1] flex items-center gap-2.5 border-b border-hairline bg-raised px-4 py-2.5">
              <h2 className="t-h3">{s === "ALERT" ? "Alert" : s === "WATCH" ? "Watch" : "Insufficient evidence"}</h2>
              <span className="t-value-sm rounded-full bg-surface px-2 py-0.5 text-muted">{g.length}</span>
            </header>
            <ul>
              {visible.map((a) => {
                const b = severityBorder(a.severity);
                const reason = a.severity === "INSUFFICIENT_EVIDENCE" ? reasonText(a.suppressed_reason) : exposureSummary(a.exposure);
                return (
                  <li key={a.alert_id} className="border-b border-hairline last:border-b-0">
                    <button type="button" onClick={() => onOpen(a.alert_id)} className={`row-hover flex w-full items-start gap-3 py-3 pr-3 pl-4 text-left ${b.className}`} style={b.style}>
                      <span className="min-w-0 flex-1">
                        <span className="flex items-baseline gap-2">
                          <span className="font-mono text-[13px] font-medium whitespace-nowrap text-ink">{a.reach_id}</span>
                          <span className="truncate text-[13px] text-muted">{a.reach_name ?? ""}</span>
                        </span>
                        <span className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px] text-muted">
                          <span>{VAR_SHORT[a.variable] ?? a.variable}</span>
                          <span className="t-value-sm">{fmtRange(a.window_start, a.window_end)}</span>
                          {a.exceedance_prob !== null ? (
                            <span>
                              P <span className="t-value-sm font-semibold text-ink">{fmtProb(a.exceedance_prob)}</span>
                            </span>
                          ) : (
                            <span className="text-faint italic">withheld</span>
                          )}
                        </span>
                        <span className="t-dense mt-1 line-clamp-2 block text-muted">{reason}</span>
                      </span>
                      <Chevron className="mt-1 text-faint" />
                    </button>
                  </li>
                );
              })}
            </ul>
            {!all && (
              <div className="border-b border-hairline px-4 py-2.5">
                <button type="button" className="btn btn-sm w-full" onClick={() => expand(s)}>
                  Show all {g.length} rows
                </button>
              </div>
            )}
          </section>
        );
      })}
    </div>
  );
}

function reasonText(reason: string | null): string {
  if (!reason) return "";
  const [code, ...rest] = reason.split(":");
  return rest.length ? `${humanize(code)}: ${rest.join(":").trim()}` : humanize(code);
}

function RunSummary({ r, shown, toggle }: { r: AlertsResponse; shown: Record<Severity, boolean>; toggle: (s: Severity) => void }) {
  if (r.alert_run === "NO_ALERT_RUN")
    return (
      <p className="card card-pad t-body mt-2 text-muted">
        No alert run on record for this city. This is not "no alerts": nothing has been assessed. Run <code className="t-value-sm text-ink">make alerts city={r.city}</code>.
      </p>
    );
  const supp = Object.entries(r.suppressed ?? {});
  return (
    <fieldset>
      <legend className="sr-only">Select a tile to show or hide its rows.</legend>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Alert" value={r.counts.ALERT} note={`${r.counts.ALERT === 1 ? "reach-variable" : "reach-variables"} likely to exceed`} severity="ALERT" pressed={shown.ALERT} onToggle={() => toggle("ALERT")} />
        <Stat label="Watch" value={r.counts.WATCH} note="elevated, below alert" severity="WATCH" pressed={shown.WATCH} onToggle={() => toggle("WATCH")} />
        <Stat
          label="Insufficient evidence"
          value={r.counts.INSUFFICIENT_EVIDENCE}
          note="too little evidence to issue either"
          severity="INSUFFICIENT_EVIDENCE"
          pressed={shown.INSUFFICIENT_EVIDENCE}
          onToggle={() => toggle("INSUFFICIENT_EVIDENCE")}
        />
        <div className="card relative flex min-w-0 flex-col overflow-hidden p-4 pt-[18px]">
          <span aria-hidden className="absolute inset-x-0 top-0 h-[3px] bg-[var(--border-strong)]" />
          <span className="t-eyebrow">Stopped by guardrails</span>
          {supp.length ? (
            <div className="mt-2 grid grid-cols-2 gap-3">
              {supp.map(([g, n]) => (
                <div key={g}>
                  <span className="t-stat block text-ink">{n}</span>
                  <span className="t-dense block text-muted">{humanize(g)}</span>
                </div>
              ))}
            </div>
          ) : (
            <span className="t-stat mt-2 block text-ink">0</span>
          )}
          <span className="t-dense mt-1 block text-faint">counted, not listed</span>
        </div>
      </div>
    </fieldset>
  );
}

/** Where the issued alerts and watches are, at a glance. Click through to the full map. */
function MiniMap({ fc }: { fc: ReachCollection }) {
  const values = useMemo(() => {
    const m = new Map<string, ReachValue>();
    for (const f of fc.features) m.set(f.id, { value: f.properties.p_exceed_max });
    return m;
  }, [fc]);
  const sounding = useMemo(() => (v: number) => fmtProb(v), []);
  const hovered = useStore((s) => s.hovered);
  const first = fc.features.find((f) => f.properties.alert_severity === "ALERT") ?? fc.features.find((f) => f.properties.alert_severity === "WATCH");
  const go = () => {
    const s = useStore.getState();
    if (first) s.select(first.id);
    s.setView("map");
  };
  return (
    <Card title="Where" caption="Alert and watch pins over the network" bodyClass="!px-0 !pb-0" className="overflow-hidden">
      <div className="map-frame relative h-[240px] !rounded-none !shadow-none">
        <ReachMap label="Alert locations" reaches={fc} values={values} breaks={PROB_BREAKS} sounding={sounding} interactive={false} hovered={hovered} showCatchments={false} />
        <button type="button" onClick={go} className="btn btn-sm map-panel absolute right-3 bottom-3 z-10 !border-0">
          Open full map
        </button>
      </div>
    </Card>
  );
}

/** Every candidate the run assessed, split by outcome: one stacked bar plus the counts. */
function RunBreakdown({ r }: { r: AlertsResponse }) {
  const parts: { label: string; n: number; color: string; hatch?: boolean }[] = [
    { label: "Alert", n: r.counts.ALERT, color: SEVERITY_COLOR.ALERT },
    { label: "Watch", n: r.counts.WATCH, color: SEVERITY_COLOR.WATCH },
    ...Object.entries(r.suppressed ?? {}).map(([g, n], i) => ({ label: `Stopped: ${g}`, n, color: i === 0 ? "var(--brand-500)" : "var(--brand-300)" })),
    { label: "Insufficient evidence", n: r.counts.INSUFFICIENT_EVIDENCE, color: "var(--severity-insufficient)", hatch: true },
  ];
  const total = parts.reduce((t, p) => t + p.n, 0);
  return (
    <Card title="What the run assessed" caption={`${total} reach-variable candidates`}>
      <div className="flex h-3 w-full overflow-hidden rounded-full bg-raised" role="img" aria-label={parts.map((p) => `${p.label} ${p.n}`).join(", ")}>
        {parts.map((p) =>
          p.n ? <span key={p.label} className={p.hatch ? "hatch" : ""} style={{ width: `${(p.n / Math.max(1, total)) * 100}%`, background: p.hatch ? undefined : p.color, minWidth: 3 }} /> : null,
        )}
      </div>
      <ul className="mt-4 flex flex-col gap-2">
        {parts.map((p) => (
          <li key={p.label} className="flex items-center gap-2.5 text-[13px]">
            <span aria-hidden className={`h-2.5 w-2.5 shrink-0 rounded-[3px] ${p.hatch ? "hatch" : ""}`} style={{ background: p.hatch ? undefined : p.color }} />
            <span className="flex-1 text-muted">{p.label}</span>
            <span className="t-value-sm text-ink">{p.n}</span>
            <span className="t-value-sm w-12 text-right text-faint">{total ? `${Math.round((p.n / total) * 100)}%` : "—"}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function ByVariable({ r }: { r: AlertsResponse }) {
  const vars = [...new Set(r.alerts.map((a) => a.variable))];
  return (
    <Card title="By variable">
      <table className="w-full text-[13px]">
        <thead>
          <tr className="t-eyebrow text-left">
            <th className="pb-2 font-semibold" />
            <th className="pb-2 text-right font-semibold">Alert</th>
            <th className="pb-2 text-right font-semibold">Watch</th>
            <th className="pb-2 text-right font-semibold">Insuff.</th>
          </tr>
        </thead>
        <tbody>
          {vars.map((v) => (
            <tr key={v} className="border-t border-hairline">
              <td className="py-2 text-ink">{VAR_SHORT[v] ?? v}</td>
              {SEVERITIES.map((s) => {
                const n = r.alerts.filter((a) => a.variable === v && a.severity === s).length;
                return (
                  <td key={s} className={`t-value-sm py-2 text-right ${n && s === "ALERT" ? "text-alert" : n && s === "WATCH" ? "text-watch-text" : "text-muted"}`}>
                    {n}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

/** Why the system declined to alert, counted by reason code. */
function Reasons({ r }: { r: AlertsResponse }) {
  const counts = new Map<string, number>();
  for (const a of r.alerts) if (a.severity === "INSUFFICIENT_EVIDENCE") counts.set(reasonCode(a.suppressed_reason), (counts.get(reasonCode(a.suppressed_reason)) ?? 0) + 1);
  const list = [...counts].sort((a, b) => b[1] - a[1]);
  if (!list.length) return null;
  const max = Math.max(...list.map(([, n]) => n));
  return (
    <Card title="Why evidence is insufficient" caption="Reason codes on the declined rows">
      <ul className="flex flex-col gap-3">
        {list.map(([code, n]) => (
          <li key={code}>
            <div className="flex items-baseline justify-between gap-2 text-[13px]">
              <span className="text-ink">{humanize(code)}</span>
              <span className="t-value-sm text-muted">{n}</span>
            </div>
            <div className="mt-1 h-1.5 rounded-full bg-raised">
              <div className="hatch h-full rounded-full" style={{ width: `${(n / max) * 100}%` }} />
            </div>
          </li>
        ))}
      </ul>
    </Card>
  );
}

/** Right-side drawer: the full alert without leaving the table. Esc or the scrim closes it. */
function AlertDrawer({ a, onClose }: { a: AlertSummary; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label={`Alert detail, ${a.reach_id}`}>
      <div className="scrim absolute inset-0 bg-[var(--scrim)]" onClick={onClose} />
      <aside className="drawer absolute inset-y-0 right-0 flex w-full max-w-[480px] flex-col bg-raised shadow-[var(--shadow-lg)]">
        <header className="flex items-start justify-between gap-3 border-b border-hairline px-6 py-5">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <SeverityLabel s={a.severity} />
              <span className="t-dense text-muted">{VARIABLE_LABEL[a.variable] ?? a.variable}</span>
            </div>
            <h2 className="t-title mt-2 break-words">{a.reach_name ?? "Unnamed channel"}</h2>
            <p className="t-ui text-muted">
              <span className="font-mono text-ink">{a.reach_id}</span>, window {fmtRange(a.window_start, a.window_end)}
            </p>
          </div>
          <button type="button" onClick={onClose} className="btn btn-ghost btn-icon shrink-0" aria-label="Close (Esc)" autoFocus>
            <svg aria-hidden viewBox="0 0 12 12" width="12" height="12">
              <path d="M2.5 2.5l7 7M9.5 2.5l-7 7" stroke="currentColor" strokeWidth="1.5" />
            </svg>
          </button>
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">
          <AlertDetailBlock id={a.alert_id} />
        </div>
      </aside>
    </div>
  );
}

function AlertDetailBlock({ id }: { id: string }) {
  const res = useApi(`alert:${id}`, () => api.alert(id));
  const a = res.data;
  if (res.loading) return <Loading what="alert" />;
  if (res.error) return <ErrorNote error={res.error} what="Alert detail" />;
  if (!a) return null;
  const feats = (a.exposure?.features ?? {}) as Record<string, { count: number | null; nearest_distance_m: number | null }>;
  return (
    <div className="flex flex-col gap-7">
      <section>
        {a.exceedance_prob !== null && (
          <div className="mb-4 flex items-baseline gap-3">
            <span className="t-reading">{fmtProb(a.exceedance_prob)}</span>
            <span className="t-ui text-muted">peak P(exceed) in the window</span>
          </div>
        )}
        <h3 className="t-eyebrow mb-3">Attribution</h3>
        {a.severity === "INSUFFICIENT_EVIDENCE" ? (
          <p className="t-ui text-muted">
            Not attributed: nothing was forecast to exceed. {reasonText(a.suppressed_reason)}.
            {a.withheld_exceedance_prob !== null && (
              <>
                {" "}
                The forecast alone would have given <span className="t-value-sm text-ink">{fmtProb(a.withheld_exceedance_prob)}</span>: withheld, not issued as a weak alert.
              </>
            )}
          </p>
        ) : (
          <>
            <AttributionBars items={a.attribution} units="index units" />
            <p className="t-dense mt-2 text-muted">{String(a.basis?.attribution ?? "")}</p>
          </>
        )}
        {a.threshold_value !== null && (
          <p className="t-dense mt-3 text-muted">
            Threshold <span className="t-value-sm text-ink">{fmtSig(a.threshold_value)}</span>. {a.threshold_derivation}
          </p>
        )}
      </section>
      <section>
        <h3 className="t-eyebrow mb-2">Guardrails</h3>
        <ul>
          {Object.entries(a.guardrails ?? {}).map(([g, s]) => (
            <li key={g} className="flex items-center justify-between gap-3 border-b border-hairline py-2 text-[13px]">
              <span>{g}</span>
              <span className={`pill pill-sm ${s === "pass" ? "pill-ok" : "pill-insufficient"}`}>{humanize(s)}</span>
            </li>
          ))}
        </ul>
      </section>
      <section>
        <h3 className="t-eyebrow mb-2">Exposure within {a.exposure?.buffer_m ?? "—"} m</h3>
        <table className="w-full text-[13px]">
          <tbody>
            {Object.entries(feats)
              .sort((x, y) => (x[1].nearest_distance_m ?? Infinity) - (y[1].nearest_distance_m ?? Infinity))
              .map(([k, f]) => (
                <tr key={k} className={`border-b border-hairline ${f.count ? "" : "text-faint"}`}>
                  <td className="py-1.5">{exposureLabel(k)}</td>
                  <td className="t-value-sm py-1.5 text-right">{f.count ?? "—"}</td>
                  <td className="t-value-sm py-1.5 text-right">{f.count ? fmtDistance(f.nearest_distance_m) : ""}</td>
                </tr>
              ))}
            <tr>
              <td className="py-1.5">residents</td>
              <td className="t-value-sm py-1.5 text-right">{a.exposure?.population ?? "—"}</td>
              <td />
            </tr>
          </tbody>
        </table>
        <p className="t-dense mt-2 text-muted">Pathways and proximity only. No health outcome is predicted; no water is declared safe or unsafe.</p>
      </section>
      <section>
        <details>
          <summary className="inline-flex cursor-pointer items-center gap-1.5 text-[13px] text-muted hover:text-ink">
            <Chevron />
            Basis of this row
          </summary>
          <table className="mt-2 w-full text-[12px]">
            <tbody>
              {Object.entries(a.basis ?? {})
                .filter(([k]) => k !== "attribution")
                .map(([k, v]) => (
                  <tr key={k} className="border-b border-hairline align-top">
                    <td className="py-1 pr-2 text-muted">{k.replaceAll("_", " ")}</td>
                    <td className="t-value-sm py-1 text-right break-all">{typeof v === "object" ? Object.entries(v as object).map(([a2, b2]) => `${a2}: ${b2}`).join(", ") : String(v)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </details>
      </section>
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => {
            useStore.getState().select(a.reach_id);
            useStore.getState().setView("map");
          }}
        >
          Open reach on map
        </button>
        <a className="btn" href={apiHref(`/api/export/fhir/${a.alert_id}`)} target="_blank" rel="noreferrer">
          FHIR bundle
        </a>
      </div>
    </div>
  );
}
