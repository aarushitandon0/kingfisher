import { useRef, useState } from "react";
import { Area, CartesianGrid, Cell, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis, Bar, BarChart } from "recharts";
import { api } from "../api/client";
import type { MetricsDocument, ScoredBlock, Variable, VariableMetrics } from "../api/types";
import { Card, Chevron, ErrorNote, Loading, PageHeader, Segmented, VerdictBadge } from "../components/bits";
import { TooltipLines } from "../components/charts";
import { Splitter, usePanelSize } from "../components/Splitter";
import { fmtDay, fmtNum, humanize, VARIABLE_LABEL } from "../lib/format";
import { ALERT, BRAND, HAIRLINE, INK, INK_MUTED, PAPER, PAPER_ALT, PROB_BREAKS, rampColor } from "../lib/ramp";
import { useApi } from "../lib/useApi";

const AXIS = { stroke: HAIRLINE, tick: { fill: INK_MUTED, fontSize: 11, fontFamily: "IBM Plex Mono" }, tickLine: false };
const VARS: Variable[] = ["turbidity_proxy", "ndci"];

export function ValidationView({ city }: { city: string }) {
  const res = useApi(`validation:${city}`, () => api.validation(city));
  const [fold, setFold] = useState("test");
  const [variable, setVariable] = useState<Variable>("turbidity_proxy");
  const [leftW, setLeftW] = usePanelSize("validation.left", null);
  const grid = useRef<HTMLDivElement>(null);
  const gridW = () => grid.current?.clientWidth ?? 1200;
  const doc = res.data?.metrics;
  const vm: VariableMetrics | undefined = doc?.folds[fold]?.[variable];

  return (
    <div className="page min-h-0 flex-1 overflow-y-auto">
      <PageHeader
        title="Validation"
        actions={
          doc && (
            <>
              <span className="flex items-center gap-2">
                <span className="t-eyebrow">Fold</span>
                <Segmented label="Fold" value={fold} onChange={setFold} options={Object.keys(doc.folds).map((f) => ({ value: f, label: f }))} />
                {doc.splits?.[fold] && <span className="t-dense text-muted">trained to {doc.splits[fold].train_end}</span>}
              </span>
              <span className="flex items-center gap-2">
                <span className="t-eyebrow">Variable</span>
                <Segmented label="Variable" value={variable} onChange={setVariable} options={VARS.map((v) => ({ value: v, label: v === "ndci" ? "Chlorophyll (NDCI)" : "Turbidity index" }))} />
              </span>
            </>
          )
        }
      >
        {res.data && doc && (
          <>
            Walk-forward folds, weather as observed (an upper bound on live skill). Production model: {doc.production_model ?? "LightGBM"}.
          </>
        )}
      </PageHeader>
      {res.loading && <Loading what="metrics" />}
      <ErrorNote error={res.error} what="Validation metrics" />
      {res.data && doc && (
        <>
          {!vm ? (
            <p className="t-ui mt-4 text-muted">No {variable} metrics in the {fold} fold.</p>
          ) : (
            <>
              <Kpis vm={vm} />
              <div
                ref={grid}
                className={`relative mt-6 grid gap-6 lg:grid-cols-2 ${leftW !== null ? "lg:grid-cols-[var(--left-w)_minmax(0,1fr)]" : ""}`}
                style={{ ["--left-w" as string]: `${leftW ?? 0}px` }}
              >
                {/* The column gap is a drag handle (double-click: back to half and half). */}
                <Splitter
                  axis="x"
                  size={() => leftW ?? (gridW() - 24) / 2}
                  onSize={setLeftW}
                  min={360}
                  max={Math.max(400, gridW() - 24 - 360)}
                  label="Column split"
                  className="!absolute inset-y-0 hidden -translate-x-1/2 lg:flex"
                  style={{ left: leftW !== null ? leftW + 12 : "50%", height: "100%" }}
                />
                <div className="lg:row-span-2">
                  <Reliability vm={vm} />
                </div>
                <Observability vm={vm} />
                <LeadTime doc={doc} vm={vm} />
                <div className="lg:col-span-2">
                  <Skill vm={vm} doc={doc} fold={fold} variable={variable} />
                </div>
                <Anomaly doc={doc} />
                <ResponseCheck check={res.data.scenario_response_check} />
                <div className="lg:col-span-2">
                  <NotComputed doc={doc} />
                </div>
              </div>
              <p className="t-dense mt-6 text-faint">
                Served live from <span className="font-mono">results/{city === "coimbra" ? "" : `${city}/`}{res.data.source.path}</span>, sha256{" "}
                <span className="font-mono">{res.data.source.sha256.slice(0, 12)}</span>, written {fmtDay(res.data.source.modified_at.slice(0, 10), true)}. Headline run{" "}
                <span className="font-mono">{doc.headline_run}</span>.
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
}

/** Headline numbers for the chosen fold and variable, all read from metrics.json. Skill is
 * green where the model beats the baseline and red where it loses - the table's one job. */
function Kpis({ vm }: { vm: VariableMetrics }) {
  const a = vm.all_horizons;
  const tiles: [string, React.ReactNode, React.ReactNode, number | null][] = [
    ["CRPS skill vs seasonal-naive", fmtSkill(a?.skill?.seasonal_naive?.crps), "all lead days; > 0 beats it", a?.skill?.seasonal_naive?.crps ?? null],
    ["CRPS skill vs climatology", fmtSkill(a?.skill?.climatology?.crps), "all lead days; > 0 beats it", a?.skill?.climatology?.crps ?? null],
    ["Brier skill vs climatology", fmtSkill(vm.probability.brier_skill_vs_climatology), "exceedance probability", vm.probability.brier_skill_vs_climatology],
    ["80% interval coverage", fmtNum(a?.common?.model.coverage_80, 2), "nominal 0.80", null],
    [
      "Reaches scored",
      vm.per_reach.reaches_scored,
      <>
        loses to seasonal-naive on <span className="text-ink">{vm.per_reach.model_loses_to_seasonal_naive.length}</span>
      </>,
      null,
    ],
  ];
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-5">
      {tiles.map(([label, value, note, skill]) => (
        <div key={label} className="card relative overflow-hidden p-4 pt-[18px]">
          <span
            aria-hidden
            className="absolute inset-x-0 top-0 h-[3px]"
            style={{ background: skill === null ? "var(--border-strong)" : skill > 0 ? "var(--severity-ok)" : "var(--severity-critical)" }}
          />
          <span className="t-eyebrow block">{label}</span>
          <span className={`t-stat mt-2 block ${skill === null ? "text-ink" : skill > 0 ? "text-ok" : "text-alert"}`}>{value}</span>
          <span className="t-dense mt-1 block text-muted">{note}</span>
        </div>
      ))}
    </div>
  );
}

function Reliability({ vm }: { vm: VariableMetrics }) {
  const p = vm.probability;
  const bins = p.bins.filter((b) => b.n > 0 && b.mean_forecast !== null && b.observed_frequency !== null);
  const data = bins.map((b) => ({ x: b.mean_forecast!, y: b.observed_frequency!, n: b.n, lo: b.bin_low, hi: b.bin_high }));
  const shade = shadedGaps(data);
  return (
    <Card
      title="Reliability"
      caption={
        <>
          Forecast P(exceed threshold) against how often it was exceeded, {p.n.toLocaleString("en-GB")} reach-days. On the diagonal is calibrated; below it, overconfident.
        </>
      }
      className="h-full"
    >
      <p className="flex items-center gap-4 text-[12px] text-muted">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-[3px]" style={{ background: "color-mix(in srgb, var(--severity-critical) 30%, transparent)" }} aria-hidden />
          overconfident
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-[3px]" style={{ background: "color-mix(in srgb, var(--brand-500) 30%, transparent)" }} aria-hidden />
          underconfident
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0 w-4 border-t border-dashed border-[var(--text-secondary)]" aria-hidden />
          perfect calibration
        </span>
      </p>
      <div
        className="mt-3 aspect-square max-h-[460px] w-full"
        role="img"
        aria-label={`Reliability diagram. Brier skill against climatology ${fmtSkill(p.brier_skill_vs_climatology)}; points below the diagonal are overconfident.`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
            <CartesianGrid stroke={HAIRLINE} strokeOpacity={0.5} />
            <XAxis dataKey="x" type="number" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} {...AXIS} label={{ value: "forecast probability", position: "insideBottom", offset: -14, fill: INK_MUTED, fontSize: 11 }} />
            <YAxis type="number" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} width={40} {...AXIS} label={{ value: "observed frequency", angle: -90, position: "insideLeft", fill: INK_MUTED, fontSize: 11 }} />
            {/* Below the diagonal the forecast said more than happened (overconfident);
                above it, less (underconfident). Shaded between the curve and the diagonal. */}
            <Area data={shade} dataKey="over" stroke="none" fill={ALERT} fillOpacity={0.18} isAnimationActive={false} activeDot={false} tooltipType="none" />
            <Area data={shade} dataKey="under" stroke="none" fill={BRAND} fillOpacity={0.18} isAnimationActive={false} activeDot={false} tooltipType="none" />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={INK_MUTED} strokeDasharray="4 3" />
            <ReferenceLine y={p.event_rate} stroke={HAIRLINE} label={{ value: "base rate", position: "insideTopLeft", fill: INK_MUTED, fontSize: 10 }} />
            <Line dataKey="y" stroke={BRAND} strokeWidth={2.5} dot={false} isAnimationActive={false} />
            <Scatter dataKey="y" fill={BRAND} stroke={PAPER} strokeWidth={1.5} isAnimationActive={false} />
            <Tooltip
              content={({ active, payload }) => {
                const d = active ? (payload?.find((q) => (q.payload as { n?: number })?.n !== undefined)?.payload as (typeof data)[number] | undefined) : undefined;
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
            <Bar dataKey="n" isAnimationActive={false} radius={[2, 2, 0, 0]}>
              {p.bins.map((b) => (
                <Cell key={b.bin_low} fill={rampColor((b.bin_low + b.bin_high) / 2, PROB_BREAKS)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="t-dense text-muted">Reach-days per probability bin (square-root scale), coloured as on the map.</p>
      <dl className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {(
          [
            ["Brier score", fmtNum(p.brier, 4), null],
            ["Brier, climatology", fmtNum(p.brier_climatology, 4), null],
            ["Brier skill", fmtSkill(p.brier_skill_vs_climatology), p.brier_skill_vs_climatology],
            ["Event rate", fmtNum(p.event_rate, 3), null],
          ] as [string, string, number | null][]
        ).map(([k, v, sk]) => (
          <div key={k} className="rounded-md bg-raised px-3 py-2.5">
            <dt className="t-eyebrow">{k}</dt>
            <dd className={`mt-1 font-display text-[20px] leading-6 font-bold tabular-nums ${sk === null ? "text-ink" : sk > 0 ? "text-ok" : "text-alert"}`}>{v}</dd>
          </div>
        ))}
      </dl>
    </Card>
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
    <Card
      title="Skill against baselines"
      caption="Scored on the rows every forecaster has. Skill = 1 − model / baseline on CRPS: positive (green) beats the baseline, negative (red) loses to it. Seasonal-naive is a point forecast, so its CRPS equals its MAE."
      bodyClass="!px-0"
    >
      <div className="overflow-x-auto">
        <table className="dtable min-w-[760px] text-[13px]">
          <thead>
            <tr>
              <th>Lead time</th>
              <th className="!text-right">n</th>
              <th className="!text-right">MAE model</th>
              <th className="!text-right">Seasonal-naive</th>
              <th className="!text-right">Climatology</th>
              <th className="!text-right">CRPS model</th>
              <th className="!text-right">Skill vs naive</th>
              <th className="!text-right">Skill vs clim.</th>
              <th className="!text-right">80% coverage</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, b]) => (
              <tr key={label} className={label === "all horizons" ? "row-total" : "row-hover"}>
                <td>{label}</td>
                <td className="num">{b?.n_common?.toLocaleString("en-GB") ?? "—"}</td>
                <td className="num">{fmtNum(b?.common?.model.mae, 3)}</td>
                <td className="num">{fmtNum(b?.common?.seasonal_naive.mae, 3)}</td>
                <td className="num">{fmtNum(b?.common?.climatology.mae, 3)}</td>
                <td className="num">{fmtNum(b?.common?.model.crps, 3)}</td>
                <td className="num">
                  <SkillValue v={b?.skill?.seasonal_naive?.crps} />
                </td>
                <td className="num">
                  <SkillValue v={b?.skill?.climatology?.crps} />
                </td>
                <td className="num">{fmtNum(b?.common?.model.coverage_80, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="px-5">
      <p className="t-ui mt-4">
        {vm.per_reach.reaches_scored} reaches scored. The model loses to seasonal-naive on {vm.per_reach.model_loses_to_seasonal_naive.length} (
        <span className="t-value-sm">{vm.per_reach.model_loses_to_seasonal_naive.join(", ") || "none"}</span>) and to climatology on {vm.per_reach.model_loses_to_climatology.length} (
        <span className="t-value-sm">{vm.per_reach.model_loses_to_climatology.join(", ") || "none"}</span>).
      </p>
      <details className="mt-3 text-[12px]" open={losses.length > 0 && losses.length <= 12}>
        <summary className="inline-flex cursor-pointer items-center gap-1.5 font-medium text-muted hover:text-ink">
          <Chevron />
          {losses.length} scored comparisons in this fold where the model loses (all runs)
        </summary>
        <div className="mt-2 ml-4 overflow-x-auto">
          <table className="dtable min-w-[640px] text-[12px]">
            <thead>
              <tr>
                <th>Run</th>
                <th>Scope</th>
                <th>Baseline</th>
                <th>Metric</th>
                <th className="!text-right">Model</th>
                <th className="!text-right">Baseline</th>
                <th className="!text-right">Skill</th>
                <th className="!text-right">n</th>
              </tr>
            </thead>
            <tbody>
              {losses.map((l, k) => (
                <tr key={k} className="row-hover">
                  <td>{l.run}</td>
                  <td>{l.scope}</td>
                  <td>{l.baseline.replaceAll("_", "-")}</td>
                  <td>{l.metric.toUpperCase()}</td>
                  <td className="num">{fmtNum(l.model, 3)}</td>
                  <td className="num">{fmtNum(l.baseline_value, 3)}</td>
                  <td className="num">
                    <SkillValue v={l.skill} />
                  </td>
                  <td className="num">{l.n.toLocaleString("en-GB")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      </div>
    </Card>
  );
}

/** A signed skill value in the colour of its verdict: beats the baseline or loses to it. */
function SkillValue({ v }: { v: number | null | undefined }) {
  const c = v === null || v === undefined ? "text-muted" : v > 0 ? "text-ok" : v < 0 ? "text-alert" : "text-ink";
  return <span className={`font-semibold ${c}`}>{fmtSkill(v)}</span>;
}

const fmtSkill = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(3)}`);

function Observability({ vm }: { vm: VariableMetrics }) {
  const o = vm.by_observability;
  const obs = o.observable;
  const drv = o.driver_only;
  return (
    <Card title="Observable vs driver-only" caption="The satellite can only score reaches it can see." bodyClass="!px-0">
      <table className="dtable text-[13px]">
        <thead>
          <tr>
            <th />
            <th className="!text-right">Scored reach-days</th>
            <th className="!text-right">MAE</th>
            <th className="!text-right">CRPS skill vs clim.</th>
          </tr>
        </thead>
        <tbody>
          <tr className="row-hover">
            <td>Optically observable</td>
            <td className="num">{obs?.n_model?.toLocaleString("en-GB") ?? "—"}</td>
            <td className="num">{fmtNum(obs?.common?.model.mae, 3)}</td>
            <td className="num">
              <SkillValue v={obs?.skill?.climatology?.crps} />
            </td>
          </tr>
          <tr className="row-hover">
            <td>Driver-predicted</td>
            <td className="num">{drv?.n_model?.toLocaleString("en-GB") ?? "—"}</td>
            <td className="num">{fmtNum(drv?.common?.model.mae, 3)}</td>
            <td className="num">
              <SkillValue v={drv?.skill?.climatology?.crps} />
            </td>
          </tr>
        </tbody>
      </table>
      {(drv?.n_model ?? 0) === 0 && (
        <p className="t-dense mx-5 mt-3 hatch-border pl-3 text-muted">
          Driver-predicted reaches cannot be scored: they have no clean water pixels at 10 m, so there is nothing observed to score them against. Their forecasts are unvalidated here, which is why they are capped at watch.
        </p>
      )}
    </Card>
  );
}

function LeadTime({ doc, vm }: { doc: MetricsDocument; vm: VariableMetrics }) {
  const op = vm.probability.operating_point_pre_guardrail;
  const note = doc.not_computed?.lead_time_distribution;
  return (
    <Card title="Skill by lead day" caption="CRPS skill against climatology for each forecast day ahead.">
      <div
        className="h-40 w-full"
        role="img"
        aria-label="Bar chart of CRPS skill against climatology by forecast lead day, 1 to 10."
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={Object.entries(vm.by_horizon)
              .sort((a, b) => Number(a[0]) - Number(b[0]))
              .map(([h, b]) => ({ h: Number(h), s: b.skill?.climatology?.crps ?? null, n: b.n_common }))}
            margin={{ top: 4, right: 8, bottom: 10, left: 0 }}
          >
            <CartesianGrid vertical={false} stroke={HAIRLINE} strokeOpacity={0.5} />
            <XAxis dataKey="h" {...AXIS} label={{ value: "lead day", position: "insideBottomRight", offset: -2, fill: INK_MUTED, fontSize: 10 }} />
            <YAxis width={40} {...AXIS} axisLine={false} tickFormatter={(v: number) => v.toFixed(2)} />
            <ReferenceLine y={0} stroke={INK} />
            <Bar dataKey="s" fill={BRAND} radius={[3, 3, 0, 0]} isAnimationActive={false} activeBar={{ fill: "var(--brand-300)" }} />
            <Tooltip cursor={{ fill: PAPER_ALT }} content={({ active, payload }) => { const d = active ? (payload?.[0]?.payload as { h: number; s: number | null; n?: number } | undefined) : undefined; return d ? TooltipLines([["lead day", String(d.h)], ["skill", fmtSkill(d.s)], ["n", d.n?.toLocaleString("en-GB") ?? "—"]]) : null; }} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      {op && (
        <p className="t-dense mt-3 text-muted">
          Before guardrails, at P ≥ <span className="t-value-sm">{op.min_exceedance_prob}</span>: <span className="t-value-sm">{op.hits}</span> hits, <span className="t-value-sm">{op.false_alarms}</span> false alarms, <span className="t-value-sm">{op.misses}</span> misses — false-alarm ratio{" "}
          <span className="t-value-sm">{fmtNum(op.false_alarm_ratio, 2)}</span>, probability of detection <span className="t-value-sm">{fmtNum(op.probability_of_detection, 2)}</span>.
        </p>
      )}
      <div className="mt-4 rounded-md bg-raised px-3 py-2.5">
        <p className="t-eyebrow">Warning lead time</p>
        {note ? (
          <p className="t-dense mt-1 text-muted">
            <span className="text-ink">Not computed:</span> {note} No incident record exists for these streams, so there is no event to measure a warning's lead time against.
          </p>
        ) : (
          <p className="t-dense mt-1 text-muted">See metrics.json.</p>
        )}
      </div>
    </Card>
  );
}

function Anomaly({ doc }: { doc: MetricsDocument }) {
  const a = doc.anomaly;
  return (
    <Card title="Anomaly detection" bodyClass="!px-0">
      {!a ? (
        <p className="t-ui px-5 text-muted">Not run for this city (make anomaly).</p>
      ) : (
        <>
          <p className="t-dense mb-3 px-5 text-muted">
            Reference: {a.reference}. {a.incidents_available} recorded incidents are available, so precision and recall are measured against this proxy, not real events. {a.weather_label ? `${humanize(a.weather_label)}.` : ""}
          </p>
          <div className="overflow-x-auto">
            <table className="dtable min-w-[480px] text-[13px]">
              <thead>
                <tr>
                  <th>Variable</th>
                  <th className="!text-right">Detections</th>
                  <th className="!text-right">Events</th>
                  <th className="!text-right">Precision</th>
                  <th className="!text-right">Recall</th>
                  <th className="!text-right">F1</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(a.by_variable).map(([v, m]) => (
                  <tr key={v} className="row-hover">
                    <td>{v === "ndci" ? "NDCI" : v === "turbidity_proxy" ? "Turbidity" : (VARIABLE_LABEL[v] ?? v)}</td>
                    <td className="num">{m.detections}</td>
                    <td className="num">{m.events}</td>
                    {(["precision", "recall", "f1"] as const).map((k) => (
                      <td key={k} className="num">
                        <span className="text-ink">{fmtNum(m[k], 2)}</span>
                        {m.ci95?.[k] && <span className="block text-[11px] text-faint">[{fmtNum(m.ci95[k][0], 2)}–{fmtNum(m.ci95[k][1], 2)}]</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="t-dense mt-3 px-5 text-muted">95% bootstrap intervals in brackets. Labels only; no source or polluter is attributed.</p>
        </>
      )}
    </Card>
  );
}

function ResponseCheck({ check }: { check: { checks?: { feature: string; variable: string; verdict: string; standardised_effect: number; observed_sign: number; expected_sign: number }[] } | null }) {
  if (!check?.checks?.length) return null;
  return (
    <Card
      title="Scenario response check"
      caption="The model-trust gate. Before the scenario engine may perturb a catchment attribute through the model, the response must have the sign the literature expects and a non-negligible size. Where it fails, the lever is not estimated."
      bodyClass="!px-0"
    >
      <div className="overflow-x-auto">
        <table className="dtable min-w-[520px] text-[13px]">
          <thead>
            <tr>
              <th>Attribute</th>
              <th>Variable</th>
              <th className="!text-right">Effect (SD)</th>
              <th className="!text-center">Sign: model / lit.</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {check.checks.map((c) => {
              const agree = c.observed_sign === c.expected_sign;
              return (
                <tr key={`${c.feature}-${c.variable}`} className="row-hover">
                  <td>{c.feature.replaceAll("_", " ")}</td>
                  <td className="text-muted">{c.variable === "ndci" ? "NDCI" : "Turbidity"}</td>
                  <td className="num">{fmtSkill(c.standardised_effect)}</td>
                  <td className="text-center">
                    <span className="inline-flex items-center gap-1.5" aria-label={`model ${signWord(c.observed_sign)}, literature ${signWord(c.expected_sign)}`}>
                      <SignGlyph s={c.observed_sign} tone={agree ? "ok" : "bad"} />
                      <span className="text-faint">/</span>
                      <SignGlyph s={c.expected_sign} tone="neutral" />
                    </span>
                  </td>
                  <td>
                    <VerdictBadge verdict={c.verdict} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="t-dense mt-3 px-5 text-muted">Arrows: direction of the response to more of the attribute. A red model arrow disagrees with the literature.</p>
    </Card>
  );
}

const signWord = (s: number) => (s > 0 ? "increases" : s < 0 ? "decreases" : "flat");

function SignGlyph({ s, tone }: { s: number; tone: "ok" | "bad" | "neutral" }) {
  const c = tone === "ok" ? "var(--severity-ok)" : tone === "bad" ? "var(--severity-critical)" : "var(--text-secondary)";
  return (
    <span aria-hidden className="inline-grid h-5 w-5 place-items-center rounded-full text-[11px] font-bold" style={{ color: c, background: `color-mix(in srgb, ${c} 16%, transparent)` }}>
      {s > 0 ? "▲" : s < 0 ? "▼" : "–"}
    </span>
  );
}

function NotComputed({ doc }: { doc: MetricsDocument }) {
  const items = Object.entries(doc.not_computed ?? {});
  if (!items.length) return null;
  return (
    <Card title="Not computed" caption="Listed, not hidden: what this validation cannot yet measure, and why.">
      <ul className="grid gap-x-8 md:grid-cols-2">
        {items.map(([k, v]) => (
          <li key={k} className="border-b border-hairline py-2 text-[13px]">
            <span className="font-medium text-ink">{humanize(k)}</span>: <span className="text-muted">{v}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

interface RelPoint {
  x: number;
  y: number;
}

/** Bands between the reliability curve and the diagonal, split where the curve crosses it,
 * so each side is shaded in its own colour. Pure geometry on the plotted points. */
function shadedGaps(pts: RelPoint[]): { x: number; over: [number, number]; under: [number, number] }[] {
  const out: { x: number; over: [number, number]; under: [number, number] }[] = [];
  const push = (x: number, y: number) =>
    out.push({ x, over: y < x ? [y, x] : [x, x], under: y > x ? [x, y] : [x, x] });
  pts.forEach((p, i) => {
    if (i > 0) {
      const a = pts[i - 1];
      const da = a.y - a.x;
      const db = p.y - p.x;
      if (da * db < 0) {
        const t = da / (da - db);
        const x = a.x + t * (p.x - a.x);
        push(x, x);
      }
    }
    push(p.x, p.y);
  });
  return out;
}
