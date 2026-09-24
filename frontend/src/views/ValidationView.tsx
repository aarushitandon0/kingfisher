import { useState } from "react";
import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis, Bar, BarChart } from "recharts";
import { api } from "../api/client";
import type { MetricsDocument, ScoredBlock, Variable, VariableMetrics } from "../api/types";
import { Chevron, ErrorNote, Loading, PageHeader, Segmented } from "../components/bits";
import { TooltipLines } from "../components/charts";
import { fmtDay, fmtNum, humanize, VARIABLE_LABEL } from "../lib/format";
import { HAIRLINE, INK, INK_MUTED, RAMP } from "../lib/ramp";
import { useApi } from "../lib/useApi";

const AXIS = { stroke: HAIRLINE, tick: { fill: INK_MUTED, fontSize: 11, fontFamily: "IBM Plex Mono" }, tickLine: false };
const VARS: Variable[] = ["turbidity_proxy", "ndci"];

export function ValidationView({ city }: { city: string }) {
  const res = useApi(`validation:${city}`, () => api.validation(city));
  const [fold, setFold] = useState("test");
  const [variable, setVariable] = useState<Variable>("turbidity_proxy");
  const doc = res.data?.metrics;
  const vm: VariableMetrics | undefined = doc?.folds[fold]?.[variable];

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 md:px-8">
      <div className="max-w-6xl">
        <PageHeader title="Validation" />
        {res.loading && <Loading what="metrics" />}
        <ErrorNote error={res.error} what="Validation metrics" />
        {res.data && doc && (
          <>
            <p className="t-ui -mt-3 max-w-4xl text-muted">
              Served live from <span className="t-value-sm text-ink">results/{city === "coimbra" ? "" : `${city}/`}{res.data.source.path}</span>, sha256{" "}
              <span className="t-value-sm">{res.data.source.sha256.slice(0, 12)}</span>, written {fmtDay(res.data.source.modified_at.slice(0, 10), true)}. Walk-forward folds; headline run{" "}
              <span className="t-value-sm">{doc.headline_run}</span> (weather as observed — an upper bound on live skill). Production model: {doc.production_model ?? "LightGBM"}.
            </p>

            <div className="t-ui mt-5 flex flex-wrap items-center gap-x-6 gap-y-3 border-y border-hairline py-3">
              <span className="flex flex-wrap items-center gap-2">
                <span className="text-muted">Fold</span>
                <Segmented label="Fold" value={fold} onChange={setFold} options={Object.keys(doc.folds).map((f) => ({ value: f, label: f }))} />
                {doc.splits?.[fold] && (
                  <span className="t-dense text-muted">
                    trained to <span className="t-value-sm text-ink">{doc.splits[fold].train_end}</span>
                  </span>
                )}
              </span>
              <span className="flex flex-wrap items-center gap-2">
                <span className="text-muted">Variable</span>
                <Segmented label="Variable" value={variable} onChange={setVariable} options={VARS.map((v) => ({ value: v, label: VARIABLE_LABEL[v] }))} />
              </span>
            </div>

            {!vm ? (
              <p className="t-ui mt-4 text-muted">No {variable} metrics in the {fold} fold.</p>
            ) : (
              <>
                <div className="mt-6 grid gap-10 lg:grid-cols-[1.5fr_1fr]">
                  <Reliability vm={vm} />
                  <div>
                    <Observability vm={vm} />
                    <LeadTime doc={doc} vm={vm} />
                  </div>
                </div>
                <Skill vm={vm} doc={doc} fold={fold} variable={variable} />
                <Anomaly doc={doc} />
                <ResponseCheck check={res.data.scenario_response_check} />
                <NotComputed doc={doc} />
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/** Section title: one style for every section on the page. `first` drops the top margin
 * for sections that open a grid row. */
function H({ children, first = false }: { children: React.ReactNode; first?: boolean }) {
  return <h2 className={`t-title mb-2 border-b border-hairline pb-1.5 ${first ? "" : "mt-12"}`}>{children}</h2>;
}

function Reliability({ vm }: { vm: VariableMetrics }) {
  const p = vm.probability;
  const bins = p.bins.filter((b) => b.n > 0 && b.mean_forecast !== null && b.observed_frequency !== null);
  const data = bins.map((b) => ({ x: b.mean_forecast!, y: b.observed_frequency!, n: b.n, lo: b.bin_low, hi: b.bin_high }));
  return (
    <section aria-label="Reliability diagram">
      <H first>Reliability</H>
      <p className="t-dense text-muted">
        Forecast P(exceed threshold) against how often it was exceeded, <span className="t-value-sm">{p.n.toLocaleString("en-GB")}</span> reach-days. On the diagonal is calibrated; below it, the forecast is overconfident.
      </p>
      <div className="mt-2 aspect-square max-h-[440px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
            <CartesianGrid stroke={HAIRLINE} strokeOpacity={0.5} />
            <XAxis dataKey="x" type="number" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} {...AXIS} label={{ value: "forecast probability", position: "insideBottom", offset: -14, fill: INK_MUTED, fontSize: 11 }} />
            <YAxis type="number" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} width={40} {...AXIS} label={{ value: "observed frequency", angle: -90, position: "insideLeft", fill: INK_MUTED, fontSize: 11 }} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={INK_MUTED} strokeDasharray="4 3" />
            <ReferenceLine y={p.event_rate} stroke={HAIRLINE} label={{ value: "base rate", position: "insideTopLeft", fill: INK_MUTED, fontSize: 10 }} />
            <Line dataKey="y" stroke={RAMP.heavy} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Scatter dataKey="y" fill={RAMP.heavy} isAnimationActive={false} />
            <Tooltip
              content={({ active, payload }) => {
                const d = active ? (payload?.[0]?.payload as (typeof data)[number] | undefined) : undefined;
                return d
                  ? TooltipLines([
                      ["bin", `${d.lo.toFixed(1)}–${d.hi.toFixed(1)}`],
                      ["mean forecast", d.x.toFixed(3)],
                      ["observed", d.y.toFixed(3)],
                      ["reach-days", d.n.toLocaleString("en-GB")],
                    ])
                  : null;
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="h-16 w-full" aria-label="Reach-days per probability bin">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={p.bins} margin={{ top: 0, right: 16, bottom: 0, left: 48 }}>
            <XAxis dataKey="bin_low" hide />
            <YAxis hide scale="sqrt" />
            <Bar dataKey="n" fill={RAMP.clear} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="t-dense text-muted">Reach-days per bin (square-root scale).</p>
      <table className="mt-2 t-dense">
        <tbody>
          <tr>
            <td className="pr-4">Brier score</td>
            <td className="t-value-sm text-right">{fmtNum(p.brier, 4)}</td>
          </tr>
          <tr>
            <td className="pr-4">Brier, climatology</td>
            <td className="t-value-sm text-right">{fmtNum(p.brier_climatology, 4)}</td>
          </tr>
          <tr>
            <td className="pr-4">Brier skill vs climatology</td>
            <td className="t-value-sm text-right">{fmtNum(p.brier_skill_vs_climatology, 3)}</td>
          </tr>
          <tr>
            <td className="pr-4">Event rate</td>
            <td className="t-value-sm text-right">{fmtNum(p.event_rate, 3)}</td>
          </tr>
        </tbody>
      </table>
    </section>
  );
}

const BUCKETS: [string, string][] = [
  ["h01_03", "days 1–3"],
  ["h04_07", "days 4–7"],
  ["h08_10", "days 8–10"],
];

function Skill({ vm, doc, fold, variable }: { vm: VariableMetrics; doc: MetricsDocument; fold: string; variable: Variable }) {
  const rows: [string, ScoredBlock | undefined][] = [...BUCKETS.map(([k, l]) => [l, vm.by_bucket[k]] as [string, ScoredBlock | undefined]), ["all horizons", vm.all_horizons]];
  const losses = doc.losses.filter((l) => l.fold === fold && l.variable === variable);
  return (
    <section>
      <H>Skill against baselines</H>
      <p className="t-dense text-muted">
        Scored on the rows every forecaster has. Skill = 1 − model / baseline: positive beats the baseline, negative loses to it. Seasonal-naive is a point forecast, so its CRPS equals its MAE.
      </p>
      <div className="overflow-x-auto">
        <table className="mt-2 w-full min-w-[720px] t-dense">
          <thead>
            <tr className="hairline-b text-muted">
              <th className="py-1 text-left font-normal">Lead time</th>
              <th className="py-1 text-right font-normal">n</th>
              <th className="py-1 text-right font-normal">MAE model</th>
              <th className="py-1 text-right font-normal">seasonal-naive</th>
              <th className="py-1 text-right font-normal">climatology</th>
              <th className="py-1 text-right font-normal">CRPS model</th>
              <th className="py-1 text-right font-normal">skill vs seasonal-naive</th>
              <th className="py-1 text-right font-normal">skill vs climatology</th>
              <th className="py-1 text-right font-normal">80% coverage</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, b]) => (
              <tr key={label} className="hairline-b">
                <td className="py-1">{label}</td>
                <td className="py-1 text-right t-value-sm">{b?.n_common?.toLocaleString("en-GB") ?? "—"}</td>
                <td className="py-1 text-right t-value-sm">{fmtNum(b?.common?.model.mae, 3)}</td>
                <td className="py-1 text-right t-value-sm">{fmtNum(b?.common?.seasonal_naive.mae, 3)}</td>
                <td className="py-1 text-right t-value-sm">{fmtNum(b?.common?.climatology.mae, 3)}</td>
                <td className="py-1 text-right t-value-sm">{fmtNum(b?.common?.model.crps, 3)}</td>
                <td className="py-1 text-right t-value-sm">{fmtSkill(b?.skill?.seasonal_naive?.crps)}</td>
                <td className="py-1 text-right t-value-sm">{fmtSkill(b?.skill?.climatology?.crps)}</td>
                <td className="py-1 text-right t-value-sm">{fmtNum(b?.common?.model.coverage_80, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="t-dense mt-1 text-muted">Skill columns are on CRPS. Nominal 80% coverage is 0.80.</p>
      <p className="t-ui mt-3">
        {vm.per_reach.reaches_scored} reaches scored. The model loses to seasonal-naive on {vm.per_reach.model_loses_to_seasonal_naive.length} (
        <span className="t-value-sm">{vm.per_reach.model_loses_to_seasonal_naive.join(", ") || "none"}</span>) and to climatology on {vm.per_reach.model_loses_to_climatology.length} (
        <span className="t-value-sm">{vm.per_reach.model_loses_to_climatology.join(", ") || "none"}</span>).
      </p>
      <details className="mt-2 t-dense" open={losses.length > 0 && losses.length <= 12}>
        <summary className="inline-flex cursor-pointer items-center gap-1 hover:text-kf">
          <Chevron />
          {losses.length} scored comparisons in this fold where the model loses (all runs)
        </summary>
        <div className="overflow-x-auto">
        <table className="mt-1 w-full min-w-[640px] t-dense">
          <thead>
            <tr className="hairline-b text-muted">
              <th className="py-0.5 text-left font-normal">run</th>
              <th className="py-0.5 text-left font-normal">scope</th>
              <th className="py-0.5 text-left font-normal">baseline</th>
              <th className="py-0.5 text-left font-normal">metric</th>
              <th className="py-0.5 text-right font-normal">model</th>
              <th className="py-0.5 text-right font-normal">baseline</th>
              <th className="py-0.5 text-right font-normal">skill</th>
              <th className="py-0.5 text-right font-normal">n</th>
            </tr>
          </thead>
          <tbody>
            {losses.map((l, k) => (
              <tr key={k} className="hairline-b">
                <td className="py-0.5">{l.run}</td>
                <td className="py-0.5">{l.scope}</td>
                <td className="py-0.5">{l.baseline.replaceAll("_", "-")}</td>
                <td className="py-0.5">{l.metric.toUpperCase()}</td>
                <td className="py-0.5 text-right t-value-sm">{fmtNum(l.model, 3)}</td>
                <td className="py-0.5 text-right t-value-sm">{fmtNum(l.baseline_value, 3)}</td>
                <td className="py-0.5 text-right t-value-sm">{fmtSkill(l.skill)}</td>
                <td className="py-0.5 text-right t-value-sm">{l.n.toLocaleString("en-GB")}</td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      </details>
    </section>
  );
}

const fmtSkill = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(3)}`);

function Observability({ vm }: { vm: VariableMetrics }) {
  const o = vm.by_observability;
  const obs = o.observable;
  const drv = o.driver_only;
  return (
    <section>
      <H first>Observable vs driver-only</H>
      <table className="w-full t-dense">
        <thead>
          <tr className="hairline-b text-muted">
            <th className="py-1 text-left font-normal" />
            <th className="py-1 text-right font-normal">scored reach-days</th>
            <th className="py-1 text-right font-normal">MAE</th>
            <th className="py-1 text-right font-normal">CRPS skill vs climatology</th>
          </tr>
        </thead>
        <tbody>
          <tr className="hairline-b">
            <td className="py-1">optically observable</td>
            <td className="py-1 text-right t-value-sm">{obs?.n_model?.toLocaleString("en-GB") ?? "—"}</td>
            <td className="py-1 text-right t-value-sm">{fmtNum(obs?.common?.model.mae, 3)}</td>
            <td className="py-1 text-right t-value-sm">{fmtSkill(obs?.skill?.climatology?.crps)}</td>
          </tr>
          <tr className="hairline-b">
            <td className="py-1">driver-predicted</td>
            <td className="py-1 text-right t-value-sm">{drv?.n_model?.toLocaleString("en-GB") ?? "—"}</td>
            <td className="py-1 text-right t-value-sm">{fmtNum(drv?.common?.model.mae, 3)}</td>
            <td className="py-1 text-right t-value-sm">{fmtSkill(drv?.skill?.climatology?.crps)}</td>
          </tr>
        </tbody>
      </table>
      {(drv?.n_model ?? 0) === 0 && (
        <p className="t-ui mt-2">
          Driver-predicted reaches cannot be scored: they have no clean water pixels at 10 m, so there is nothing observed to score them against. Their forecasts are unvalidated here, which is why they are capped at watch.
        </p>
      )}
    </section>
  );
}

function LeadTime({ doc, vm }: { doc: MetricsDocument; vm: VariableMetrics }) {
  const op = vm.probability.operating_point_pre_guardrail;
  const note = doc.not_computed?.lead_time_distribution;
  return (
    <section className="mt-10">
      <H first>Lead time</H>
      {note ? (
        <p className="t-ui">
          Not computed: {note} <span className="text-muted">No incident record exists for these streams, so there is no event to measure a warning's lead time against.</span>
        </p>
      ) : (
        <p className="t-ui text-muted">See metrics.json.</p>
      )}
      <p className="t-dense mt-3 text-muted">Skill by lead day (CRPS skill vs climatology)</p>
      <div className="h-32 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={Object.entries(vm.by_horizon)
              .sort((a, b) => Number(a[0]) - Number(b[0]))
              .map(([h, b]) => ({ h: Number(h), s: b.skill?.climatology?.crps ?? null, n: b.n_common }))}
            margin={{ top: 4, right: 8, bottom: 0, left: 0 }}
          >
            <CartesianGrid vertical={false} stroke={HAIRLINE} strokeOpacity={0.5} />
            <XAxis dataKey="h" {...AXIS} />
            <YAxis width={40} {...AXIS} axisLine={false} tickFormatter={(v: number) => v.toFixed(2)} />
            <ReferenceLine y={0} stroke={INK} />
            <Bar dataKey="s" fill={RAMP.heavy} isAnimationActive={false} />
            <Tooltip content={({ active, payload }) => { const d = active ? (payload?.[0]?.payload as { h: number; s: number | null; n?: number } | undefined) : undefined; return d ? TooltipLines([["lead day", String(d.h)], ["skill", fmtSkill(d.s)], ["n", d.n?.toLocaleString("en-GB") ?? "—"]]) : null; }} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      {op && (
        <p className="t-dense mt-3">
          Before guardrails, at P ≥ <span className="t-value-sm">{op.min_exceedance_prob}</span>: <span className="t-value-sm">{op.hits}</span> hits, <span className="t-value-sm">{op.false_alarms}</span> false alarms, <span className="t-value-sm">{op.misses}</span> misses — false-alarm ratio{" "}
          <span className="t-value-sm">{fmtNum(op.false_alarm_ratio, 2)}</span>, probability of detection <span className="t-value-sm">{fmtNum(op.probability_of_detection, 2)}</span>.
        </p>
      )}
    </section>
  );
}

function Anomaly({ doc }: { doc: MetricsDocument }) {
  const a = doc.anomaly;
  return (
    <section>
      <H>Anomaly detection</H>
      {!a ? (
        <p className="t-ui text-muted">Not run for this city (make anomaly).</p>
      ) : (
        <>
          <p className="t-dense text-muted">
            Reference: {a.reference}. {a.incidents_available} recorded incidents are available, so precision and recall are measured against this proxy, not real events. {a.weather_label ? `${humanize(a.weather_label)}.` : ""}
          </p>
          <table className="mt-2 w-full max-w-3xl t-dense">
            <thead>
              <tr className="hairline-b text-muted">
                <th className="py-1 text-left font-normal">Variable</th>
                <th className="py-1 text-right font-normal">detections</th>
                <th className="py-1 text-right font-normal">events</th>
                <th className="py-1 text-right font-normal">precision</th>
                <th className="py-1 text-right font-normal">recall</th>
                <th className="py-1 text-right font-normal">F1</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(a.by_variable).map(([v, m]) => (
                <tr key={v} className="hairline-b">
                  <td className="py-1">{VARIABLE_LABEL[v] ?? v}</td>
                  <td className="py-1 text-right t-value-sm">{m.detections}</td>
                  <td className="py-1 text-right t-value-sm">{m.events}</td>
                  {(["precision", "recall", "f1"] as const).map((k) => (
                    <td key={k} className="py-1 text-right t-value-sm">
                      {fmtNum(m[k], 2)}
                      {m.ci95?.[k] && <span className="block text-muted">[{fmtNum(m.ci95[k][0], 2)}–{fmtNum(m.ci95[k][1], 2)}]</span>}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="t-dense mt-1 text-muted">95% bootstrap intervals in brackets. Labels only; no source or polluter is attributed.</p>
        </>
      )}
    </section>
  );
}

function ResponseCheck({ check }: { check: { checks?: { feature: string; variable: string; verdict: string; standardised_effect: number; observed_sign: number; expected_sign: number }[] } | null }) {
  if (!check?.checks?.length) return null;
  return (
    <section>
      <H>Scenario response check</H>
      <p className="t-dense text-muted">Before the scenario engine may perturb a catchment attribute through the model, the model's response to it must have the sign the literature expects and a non-negligible size. Where it fails, the lever is not estimated.</p>
      <table className="mt-2 w-full max-w-3xl t-dense">
        <thead>
          <tr className="hairline-b text-muted">
            <th className="py-1 text-left font-normal">Attribute</th>
            <th className="py-1 text-left font-normal">Variable</th>
            <th className="py-1 text-right font-normal">effect (target SD per ±1 SD)</th>
            <th className="py-1 text-right font-normal">sign, model / literature</th>
            <th className="py-1 text-left pl-4 font-normal">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {check.checks.map((c) => (
            <tr key={`${c.feature}-${c.variable}`} className="hairline-b">
              <td className="py-1">{c.feature.replaceAll("_", " ")}</td>
              <td className="py-1">{VARIABLE_LABEL[c.variable] ?? c.variable}</td>
              <td className="py-1 text-right t-value-sm">{fmtSkill(c.standardised_effect)}</td>
              <td className="py-1 text-right t-value-sm">
                {c.observed_sign > 0 ? "+" : c.observed_sign < 0 ? "−" : "0"} / {c.expected_sign > 0 ? "+" : "−"}
              </td>
              <td className="py-1 pl-4">{humanize(c.verdict)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function NotComputed({ doc }: { doc: MetricsDocument }) {
  const items = Object.entries(doc.not_computed ?? {});
  if (!items.length) return null;
  return (
    <section className="mb-10">
      <H>Not computed</H>
      <ul className="t-ui">
        {items.map(([k, v]) => (
          <li key={k} className="hairline-b py-1">
            <span className="text-ink">{humanize(k)}</span>: <span className="text-muted">{v}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
