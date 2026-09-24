import { Fragment, useMemo, useState } from "react";
import { api } from "../api/client";
import type { AlertSummary, AlertsResponse, Severity } from "../api/types";
import { ErrorNote, Loading, severityBorder, SeverityLabel } from "../components/bits";
import { AttributionBars } from "../components/charts";
import { exposureLabel, fmtDay, fmtDistance, fmtProb, fmtRange, fmtSig, humanize, plural, VARIABLE_LABEL } from "../lib/format";
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
  const rows = useMemo(() => (r?.alerts ?? []).filter((a) => shown[a.severity]), [r, shown]);

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 md:px-6">
      <div className="max-w-6xl">
        <h1 className="t-title">Alerts</h1>
        {res.loading && <Loading what="alerts" />}
        <ErrorNote error={res.error} what="Alerts" />
        {r && <RunSummary r={r} />}

        {r && r.alert_run === "OK" && (
          <>
            <fieldset className="mt-4 flex flex-wrap gap-4 t-ui">
              <legend className="sr-only">Show severities</legend>
              {SEVERITIES.map((s) => (
                <label key={s} className="flex items-center gap-1.5">
                  <input type="checkbox" checked={shown[s]} onChange={() => setShown({ ...shown, [s]: !shown[s] })} />
                  <SeverityLabel s={s} /> <span className="t-value-sm text-muted">{r.counts[s]}</span>
                </label>
              ))}
            </fieldset>

            <table className="mt-3 w-full t-dense">
              <thead>
                <tr className="hairline-b text-left text-muted">
                  <th className="py-1.5 pl-3 font-normal">Reach</th>
                  <th className="py-1.5 font-normal">Severity</th>
                  <th className="py-1.5 font-normal">Variable</th>
                  <th className="py-1.5 pr-4 text-right font-normal">Probability</th>
                  <th className="py-1.5 font-normal">Window</th>
                  <th className="hidden py-1.5 font-normal md:table-cell">Exposure</th>
                  <th className="py-1.5 font-normal">
                    <span className="sr-only">Expand</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((a) => {
                  const b = severityBorder(a.severity);
                  const isOpen = open === a.alert_id;
                  return (
                    <Fragment key={a.alert_id}>
                      <tr className="hairline-b align-top">
                        <td className={`py-1.5 pl-3 ${b.className}`} style={b.style}>
                          <span className="text-ink">{a.reach_id}</span> <span className="text-muted">{a.reach_name ?? ""}</span>
                        </td>
                        <td className="py-1.5">
                          <SeverityLabel s={a.severity} />
                        </td>
                        <td className="py-1.5 text-muted">{VARIABLE_LABEL[a.variable] ?? a.variable}</td>
                        <td className="py-1.5 pr-4 text-right t-value-sm">{a.exceedance_prob !== null ? fmtProb(a.exceedance_prob) : <span className="text-muted">withheld</span>}</td>
                        <td className="py-1.5 t-value-sm">{fmtRange(a.window_start, a.window_end)}</td>
                        <td className="hidden py-1.5 text-muted md:table-cell">
                          {a.severity === "INSUFFICIENT_EVIDENCE" ? <span className="text-ink">{reasonText(a.suppressed_reason)}</span> : exposureSummary(a.exposure)}
                        </td>
                        <td className="py-1.5 pr-2 text-right">
                          <button type="button" aria-expanded={isOpen} onClick={() => setOpen(isOpen ? null : a.alert_id)} className="text-muted hover:text-ink">
                            {isOpen ? "Close" : "Details"}
                          </button>
                        </td>
                      </tr>
                      {isOpen && (
                        <tr className="hairline-b">
                          <td colSpan={7} className={`pb-4 pl-3 ${b.className}`} style={b.style}>
                            <AlertDetailBlock id={a.alert_id} />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
            {!rows.length && <p className="t-ui mt-3 text-muted">No alerts of the selected severities in this run.</p>}
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

function RunSummary({ r }: { r: AlertsResponse }) {
  if (r.alert_run === "NO_ALERT_RUN")
    return (
      <p className="t-body mt-2 text-muted">
        No alert run on record for this city. This is not "no alerts": nothing has been assessed. Run <code className="t-value-sm">make alerts city={r.city}</code>.
      </p>
    );
  const supp = Object.entries(r.suppressed ?? {});
  return (
    <div className="t-body mt-1">
      <p>
        Run issued {fmtDay(r.alert_run_issued_date, true)}: <span className="text-alert">{plural(r.counts.ALERT, "alert")}</span>, {plural(r.counts.WATCH, "watch", "watches")},{" "}
        {r.counts.INSUFFICIENT_EVIDENCE} reach-variables with insufficient evidence to issue either.
      </p>
      {supp.length > 0 && (
        <p className="t-ui text-muted">
          Guardrails stopped {supp.map(([g, n]) => `${n} on ${g}`).join(", ")} before they became alerts. Those are counted, not listed.
        </p>
      )}
      <p className="t-ui text-muted">
        Insufficient evidence is an output, not a gap in the table: the system declines to alert where it cannot see the water or has no threshold to judge it by.
      </p>
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
    <div className="grid gap-6 pt-2 md:grid-cols-[1.2fr_1fr_1fr]">
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
        <p className="t-dense mt-2">
          <a className="text-kf underline underline-offset-2" href={`/api/export/fhir/${a.alert_id}`} target="_blank" rel="noreferrer">
            FHIR bundle
          </a>{" "}
          <button type="button" className="ml-2 text-kf underline underline-offset-2" onClick={() => { useStore.getState().select(a.reach_id); useStore.getState().setView("map"); }}>
            Open reach on map
          </button>
        </p>
      </section>
    </div>
  );
}
