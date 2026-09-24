import { Fragment, useMemo, useState } from "react";
import { api } from "../api/client";
import type { AlertSummary, AlertsResponse, Severity } from "../api/types";
import { Chevron, ErrorNote, Loading, PageHeader, severityBorder, SeverityLabel, Stat } from "../components/bits";
import { AttributionBars } from "../components/charts";
import { exposureLabel, fmtDay, fmtDistance, fmtProb, fmtRange, fmtSig, humanize, VARIABLE_LABEL } from "../lib/format";
import { useApi } from "../lib/useApi";
import { useStore } from "../store";

const SEVERITIES: Severity[] = ["ALERT", "WATCH", "INSUFFICIENT_EVIDENCE"];

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

export function AlertsView({ city }: { city: string }) {
  const res = useApi(`alerts:${city}`, () => api.alerts(city));
  const [shown, setShown] = useState<Record<Severity, boolean>>({ ALERT: true, WATCH: true, INSUFFICIENT_EVIDENCE: true });
  const [open, setOpen] = useState<string | null>(null);
  const r = res.data;
  // Grouped by severity for scanning; within a group the API's order is kept.
  const groups = useMemo(
    () => SEVERITIES.filter((s) => shown[s]).map((s) => [s, (r?.alerts ?? []).filter((a) => a.severity === s)] as const),
    [r, shown],
  );
  const empty = groups.every(([, g]) => g.length === 0);

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 md:px-8">
      <div className="max-w-7xl">
        <PageHeader title="Alerts">
          {r?.alert_run === "OK" && (
            <>
              Latest run, issued <span className="t-value-sm text-ink">{fmtDay(r.alert_run_issued_date, true)}</span>.
            </>
          )}
        </PageHeader>
        {res.loading && <Loading what="alerts" />}
        <ErrorNote error={res.error} what="Alerts" />
        {r && <RunSummary r={r} shown={shown} toggle={(s) => setShown({ ...shown, [s]: !shown[s] })} />}

        {r && r.alert_run === "OK" && (
          <>
            <div className="mt-6 overflow-x-auto md:overflow-visible">
              <table className="w-full min-w-[900px] t-dense">
                <thead className="sticky top-0 z-10 bg-paper">
                  <tr className="text-left text-muted shadow-[inset_0_-1px_0_var(--hairline)]">
                    <th scope="col" className="py-2 pr-4 pl-3 font-normal">Reach</th>
                    <th scope="col" className="py-2 pr-4 font-normal">Severity</th>
                    <th scope="col" className="py-2 pr-4 font-normal">Variable</th>
                    <th scope="col" className="py-2 pr-4 text-right font-normal">Probability</th>
                    <th scope="col" className="py-2 pr-4 font-normal">Window</th>
                    <th scope="col" className="py-2 font-normal">Exposure or reason</th>
                    <th scope="col" className="py-2 font-normal">
                      <span className="sr-only">Expand</span>
                    </th>
                  </tr>
                </thead>
                {groups.map(([s, rows]) =>
                  rows.length === 0 ? null : (
                    <tbody key={s}>
                      <tr>
                        <th scope="colgroup" colSpan={7} className="hairline-b bg-paper-alt py-1.5 pl-3 text-left font-normal">
                          <SeverityLabel s={s} /> <span className="t-value-sm ml-1 text-muted">{rows.length}</span>
                        </th>
                      </tr>
                      {rows.map((a) => {
                        const b = severityBorder(a.severity);
                        const isOpen = open === a.alert_id;
                        const toggle = () => setOpen(isOpen ? null : a.alert_id);
                        return (
                          <Fragment key={a.alert_id}>
                            <tr className={`hairline-b cursor-pointer align-top hover:bg-paper-alt ${isOpen ? "bg-paper-alt" : ""}`} onClick={toggle}>
                              <td className={`py-2 pr-4 pl-3 ${b.className}`} style={b.style}>
                                <span className="text-ink">{a.reach_id}</span> <span className="text-muted">{a.reach_name ?? ""}</span>
                              </td>
                              <td className="py-2 pr-4 whitespace-nowrap">
                                <SeverityLabel s={a.severity} />
                              </td>
                              <td className="py-2 pr-4 whitespace-nowrap text-muted">{VARIABLE_LABEL[a.variable] ?? a.variable}</td>
                              <td className="py-2 pr-4 text-right t-value-sm">{a.exceedance_prob !== null ? fmtProb(a.exceedance_prob) : <span className="text-muted">withheld</span>}</td>
                              <td className="py-2 pr-4 t-value-sm whitespace-nowrap">{fmtRange(a.window_start, a.window_end)}</td>
                              <td className="py-2 pr-2 text-muted">
                                {a.severity === "INSUFFICIENT_EVIDENCE" ? <span className="text-ink">{reasonText(a.suppressed_reason)}</span> : exposureSummary(a.exposure)}
                              </td>
                              <td className="py-1 pr-1 text-right">
                                <button
                                  type="button"
                                  aria-expanded={isOpen}
                                  aria-label={`${isOpen ? "Hide" : "Show"} details for ${a.reach_id}, ${VARIABLE_LABEL[a.variable] ?? a.variable}`}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggle();
                                  }}
                                  className="btn btn-ghost btn-sm"
                                >
                                  <Chevron />
                                  Details
                                </button>
                              </td>
                            </tr>
                            {isOpen && (
                              <tr className="hairline-b bg-paper-alt">
                                <td colSpan={7} className={`pr-3 pb-5 pl-3 ${b.className}`} style={b.style}>
                                  <AlertDetailBlock id={a.alert_id} />
                                </td>
                              </tr>
                            )}
                          </Fragment>
                        );
                      })}
                    </tbody>
                  ),
                )}
              </table>
            </div>
            {empty && <p className="t-ui mt-3 text-muted">No alerts of the selected severities in this run. Select a reading above to show its rows.</p>}
          </>
        )}
      </div>
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
      <p className="t-body mt-2 text-muted">
        No alert run on record for this city. This is not "no alerts": nothing has been assessed. Run <code className="t-value-sm text-ink">make alerts city={r.city}</code>.
      </p>
    );
  const supp = Object.entries(r.suppressed ?? {});
  return (
    <div>
      <fieldset>
        <legend className="t-dense mb-2 text-muted">Select a reading to show or hide its rows.</legend>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
          <Stat label={<SeverityLabel s="ALERT" />} value={r.counts.ALERT} note={r.counts.ALERT === 1 ? "alert" : "alerts"} severity="ALERT" pressed={shown.ALERT} onToggle={() => toggle("ALERT")} />
          <Stat label={<SeverityLabel s="WATCH" />} value={r.counts.WATCH} note={r.counts.WATCH === 1 ? "watch" : "watches"} severity="WATCH" pressed={shown.WATCH} onToggle={() => toggle("WATCH")} />
          <Stat
            label={<SeverityLabel s="INSUFFICIENT_EVIDENCE" />}
            value={r.counts.INSUFFICIENT_EVIDENCE}
            note="reach-variables with too little evidence to issue either"
            severity="INSUFFICIENT_EVIDENCE"
            pressed={shown.INSUFFICIENT_EVIDENCE}
            onToggle={() => toggle("INSUFFICIENT_EVIDENCE")}
          />
          {supp.length > 0 && (
            <Stat
              label={<span className="text-muted">Stopped by guardrails</span>}
              value={
                <span className="flex flex-wrap gap-x-4">
                  {supp.map(([g, n]) => (
                    <span key={g}>
                      {n}
                      <span className="t-dense ml-1 font-sans text-muted">{g}</span>
                    </span>
                  ))}
                </span>
              }
              note="counted before they became alerts, not listed"
            />
          )}
        </div>
      </fieldset>
      <p className="t-ui mt-3 max-w-4xl text-muted">
        Insufficient evidence is an output, not a gap in the table: the system declines to alert where it cannot see the water or has no threshold to judge it by.
      </p>
    </div>
  );
}

function AlertDetailBlock({ id }: { id: string }) {
  const res = useApi(`alert:${id}`, () => api.alert(id));
  const a = res.data;
  if (res.loading) return <Loading what="alert" className="pt-3" />;
  if (res.error) return <ErrorNote error={res.error} what="Alert detail" />;
  if (!a) return null;
  const feats = (a.exposure?.features ?? {}) as Record<string, { count: number | null; nearest_distance_m: number | null }>;
  return (
    <div className="grid gap-6 pt-3 md:grid-cols-[1.2fr_1fr_1fr]">
      <section>
        <h3 className="t-ui mb-1 text-ink">Attribution</h3>
        {a.severity === "INSUFFICIENT_EVIDENCE" ? (
          <p className="t-ui text-muted">
            Not attributed: nothing was forecast to exceed. {reasonText(a.suppressed_reason)}.
            {a.withheld_exceedance_prob !== null && (
              <>
                {" "}
                The forecast alone would have given <span className="t-value-sm">{fmtProb(a.withheld_exceedance_prob)}</span> — withheld, not issued as a weak alert.
              </>
            )}
          </p>
        ) : (
          <>
            <AttributionBars items={a.attribution} units="index units" />
            <p className="t-dense mt-1 text-muted">{String(a.basis?.attribution ?? "")}</p>
          </>
        )}
        {a.threshold_value !== null && (
          <p className="t-dense mt-2 text-muted">
            Threshold <span className="t-value-sm text-ink">{fmtSig(a.threshold_value)}</span>. {a.threshold_derivation}
          </p>
        )}
      </section>
      <section>
        <h3 className="t-ui mb-1 text-ink">Guardrails</h3>
        <table className="w-full t-dense">
          <tbody>
            {Object.entries(a.guardrails ?? {}).map(([g, s]) => (
              <tr key={g} className="hairline-b">
                <td className="py-0.5">{g}</td>
                <td className={`py-0.5 text-right ${s === "pass" ? "text-ink" : "text-muted"}`}>{humanize(s)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h3 className="t-ui mt-3 mb-1 text-ink">Basis</h3>
        <table className="w-full t-dense">
          <tbody>
            {Object.entries(a.basis ?? {})
              .filter(([k]) => k !== "attribution")
              .map(([k, v]) => (
                <tr key={k} className="hairline-b align-top">
                  <td className="py-0.5 pr-2">{k.replaceAll("_", " ")}</td>
                  <td className="py-0.5 text-right t-value-sm">{typeof v === "object" ? Object.entries(v as object).map(([a2, b2]) => `${a2}: ${b2}`).join(", ") : String(v)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </section>
      <section>
        <h3 className="t-ui mb-1 text-ink">Exposure within {a.exposure?.buffer_m ?? "—"} m</h3>
        <table className="w-full t-dense">
          <tbody>
            {Object.entries(feats)
              .sort((x, y) => (x[1].nearest_distance_m ?? Infinity) - (y[1].nearest_distance_m ?? Infinity))
              .map(([k, f]) => (
                <tr key={k} className={`hairline-b ${f.count ? "" : "text-muted"}`}>
                  <td className="py-0.5">{exposureLabel(k)}</td>
                  <td className="py-0.5 text-right t-value-sm">{f.count ?? "—"}</td>
                  <td className="py-0.5 text-right t-value-sm">{f.count ? fmtDistance(f.nearest_distance_m) : ""}</td>
                </tr>
              ))}
            <tr>
              <td className="py-0.5">residents</td>
              <td className="py-0.5 text-right t-value-sm">{a.exposure?.population ?? "—"}</td>
              <td />
            </tr>
          </tbody>
        </table>
        <p className="t-dense mt-1 text-muted">Pathways and proximity only. No health outcome is predicted; no water is declared safe or unsafe.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          <button type="button" className="btn btn-sm btn-primary" onClick={() => { useStore.getState().select(a.reach_id); useStore.getState().setView("map"); }}>
            Open reach on map
          </button>
          <a className="btn btn-sm" href={`/api/export/fhir/${a.alert_id}`} target="_blank" rel="noreferrer">
            FHIR bundle
          </a>
        </div>
      </section>
    </div>
  );
}
