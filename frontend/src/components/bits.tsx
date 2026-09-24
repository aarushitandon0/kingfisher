import { useRef, type ReactNode } from "react";
import type { ApiError } from "../api/client";
import type { Observability, Severity } from "../api/types";
import { rampColor, SEVERITY_COLOR } from "../lib/ramp";

/** A rule with a small label - section divider in dense panels. Rendered as a real heading
 * so screen-reader users can jump between sections. */
export function Rule({ children, as: Tag = "h3" }: { children: ReactNode; as?: "h2" | "h3" | "h4" | "p" }) {
  return (
    <div className="mt-7 mb-2 flex items-center gap-2">
      <Tag className="t-eyebrow shrink-0">{children}</Tag>
      <span className="h-px flex-1 bg-hairline" />
    </div>
  );
}

/** Page title + one line of context. Page titles sit one step above section titles. */
export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="mb-6">
      <h1 className="t-display">{title}</h1>
      {children && <div className="t-ui mt-1.5 max-w-4xl text-muted">{children}</div>}
    </header>
  );
}

/** Plainly stated - "optically observable" or "driver-predicted". */
export function ObservabilityBadge({ observability, medianPixels }: { observability: Observability; medianPixels?: number | null }) {
  if (observability === "OPTICALLY_OBSERVABLE")
    return (
      <p className="t-ui flex items-baseline gap-2">
        <span aria-hidden className="inline-block h-2 w-2 shrink-0 rotate-45 bg-brand" />
        <span>
          Optically observable
          {medianPixels != null && (
            <span className="text-muted">
              {" "}
              — median <span className="t-value-sm">{medianPixels}</span> clean water pixels per pass
            </span>
          )}
        </span>
      </p>
    );
  if (observability === "DRIVER_ONLY")
    return (
      <p className="t-ui flex items-baseline gap-2">
        <span aria-hidden className="inline-block h-2 w-2 shrink-0 rotate-45 border border-brand" />
        <span>
          Driver-predicted
          <span className="text-muted">
            {" "}
            — too narrow for clean water pixels at 10 m, so forecast from weather and catchment only.
            Capped at watch; never alerts.
          </span>
        </span>
      </p>
    );
  return <p className="t-ui text-muted">Observability not assessed.</p>;
}

/** Severity as a 3px left border (table rows) - hatched for insufficient evidence. */
export function severityBorder(s: Severity | null): { className: string; style?: React.CSSProperties } {
  if (s === "INSUFFICIENT_EVIDENCE") return { className: "hatch-border" };
  if (!s) return { className: "border-l-[3px] border-l-transparent" };
  return { className: "border-l-[3px]", style: { borderLeftColor: SEVERITY_COLOR[s] } };
}

const PILL: Record<Severity, string> = { ALERT: "pill-alert", WATCH: "pill-watch", INSUFFICIENT_EVIDENCE: "pill-insufficient" };
const SEVERITY_TEXT: Record<Severity, string> = { ALERT: "text-alert", WATCH: "text-watch-text", INSUFFICIENT_EVIDENCE: "text-muted" };

/** Severity as a pill: tinted ground, severity-coloured text, a dot. */
export function SeverityLabel({ s, small = false }: { s: Severity; small?: boolean }) {
  const text = s === "ALERT" ? "Alert" : s === "WATCH" ? "Watch" : "Insufficient evidence";
  return <span className={`pill ${PILL[s]} ${small ? "pill-sm" : ""}`}>{text}</span>;
}

export function ErrorNote({ error, what }: { error: ApiError | null; what: string }) {
  if (!error) return null;
  return (
    <div role="alert" className="t-ui my-2 rounded-md border-l-[3px] border-alert bg-paper-alt py-2 pr-3 pl-3">
      <p className="font-medium text-ink">{what} unavailable.</p>
      <p className="text-muted">{error.detail}</p>
    </div>
  );
}

export function Loading({ what, className = "" }: { what: string; className?: string }) {
  return (
    <div className={`w-full max-w-[240px] py-1 ${className}`} role="status" aria-live="polite">
      <p className="t-ui text-muted">Loading {what}</p>
      <div className="progress mt-1.5" aria-hidden />
    </div>
  );
}

/** A line swatch; `width` matches the map's line weight for that band. */
export function Swatch({ color, hatch, dashed, width = 3 }: { color?: string; hatch?: boolean; dashed?: boolean; width?: number }) {
  if (hatch) return <span aria-hidden className="hatch inline-block h-2.5 w-6 rounded-[2px] align-middle" />;
  if (dashed)
    return (
      <span
        aria-hidden
        className="inline-block h-0.5 w-6 align-middle"
        style={{ backgroundImage: `repeating-linear-gradient(90deg, ${color} 0 6px, transparent 6px 10px)` }}
      />
    );
  return <span aria-hidden className="inline-block w-6 rounded-full align-middle" style={{ background: color, height: width }} />;
}

/** Disclosure chevron; rotates with the aria-expanded / details[open] state it sits in. */
export function Chevron({ className = "" }: { className?: string }) {
  return (
    <svg aria-hidden viewBox="0 0 12 12" width="12" height="12" className={`chev inline-block shrink-0 ${className}`}>
      <path d="M4.5 2.5 8 6l-3.5 3.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="square" />
    </svg>
  );
}

export interface SegmentOption<T extends string> {
  value: T;
  label: ReactNode;
}

/** One-of-N choice. The single toggle control used everywhere (variable, span, fold, tool):
 * hairline group, the chosen segment on recessed paper with a kingfisher underline - the
 * same mark as the active nav tab. ARIA radiogroup with roving tabindex and arrow keys. */
export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
  dense = false,
  className = "",
}: {
  label: string;
  value: T;
  options: SegmentOption<T>[];
  onChange: (v: T) => void;
  dense?: boolean;
  className?: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const idx = Math.max(0, options.findIndex((o) => o.value === value));
  function onKey(e: React.KeyboardEvent) {
    const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const next = (idx + d + options.length) % options.length;
    onChange(options[next].value);
    refs.current[next]?.focus();
  }
  return (
    <div role="radiogroup" aria-label={label} onKeyDown={onKey} className={`inline-flex max-w-full shrink-0 self-start rounded-md bg-paper-alt p-0.5 ${className}`}>
      {options.map((o, i) => {
        const on = i === idx;
        return (
          <button
            key={o.value}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={on}
            tabIndex={on ? 0 : -1}
            onClick={() => onChange(o.value)}
            className={`${dense ? "t-dense h-6 px-2" : "t-ui h-7 px-3"} rounded-[5px] font-medium whitespace-nowrap ${
              on ? "bg-surface text-brand shadow-[0_1px_2px_rgba(0,0,0,0.08),0_0_0_1px_var(--border)]" : "text-muted hover:text-ink"
            }`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/** Where a value sits on the fixed display ramp: the same colours as the map, a tick at the
 * value. Pure colour lookup and position - the number itself is printed beside it. */
export function RampGauge({ value, breaks, max, fmt }: { value: number; breaks: number[]; max: number; fmt: (v: number) => string }) {
  const edges = [0, ...breaks, max];
  const pos = Math.min(1, Math.max(0, value / max)) * 100;
  return (
    <div aria-hidden className="mt-2 max-w-[280px]">
      <div className="relative flex h-1.5 overflow-visible [&>span:first-child]:rounded-l-full [&>span:nth-last-child(2)]:rounded-r-full">
        {edges.slice(0, -1).map((a, i) => (
          <span key={a} style={{ width: `${((edges[i + 1] - a) / max) * 100}%`, background: rampColor((a + edges[i + 1]) / 2, breaks) }} />
        ))}
        <span className="absolute -top-1 -bottom-1 w-0.5 -translate-x-1/2 rounded-full bg-ink" style={{ left: `${pos}%` }} />
      </div>
      <div className="relative mt-1 h-3 font-mono text-[10px] leading-3 text-muted">
        {breaks.map((b) => (
          <span key={b} className="absolute -translate-x-1/2" style={{ left: `${(b / max) * 100}%` }}>
            {fmt(b)}
          </span>
        ))}
      </div>
    </div>
  );
}

/** A counted reading: label, a mono number in its severity colour, a 3px top edge.
 * Optionally a filter toggle. */
export function Stat({
  label,
  value,
  note,
  severity,
  tone = "neutral",
  pressed,
  onToggle,
}: {
  label: ReactNode;
  value: ReactNode;
  note?: ReactNode;
  severity?: Severity | null;
  /** Counts that are not severities can carry the brand instead. */
  tone?: "brand" | "neutral";
  pressed?: boolean;
  onToggle?: () => void;
}) {
  const off = onToggle && !pressed;
  const edge = severity === "INSUFFICIENT_EVIDENCE" ? undefined : severity ? SEVERITY_COLOR[severity] : tone === "brand" ? "var(--brand-500)" : "var(--border)";
  const body = (
    <>
      {/* 3px top edge in the severity colour; hatched for insufficient evidence. */}
      <span aria-hidden className={`absolute inset-x-0 top-0 h-[3px] ${severity === "INSUFFICIENT_EVIDENCE" ? "hatch" : ""}`} style={{ background: edge }} />
      <span className="t-dense flex items-center gap-1.5">
        {onToggle && (
          <span aria-hidden className={`inline-grid h-3.5 w-3.5 shrink-0 place-items-center rounded-[3px] border ${pressed ? "border-brand bg-brand" : "border-faint bg-surface"}`}>
            {pressed && (
              <svg viewBox="0 0 10 10" width="8" height="8">
                <path d="M1.5 5.2 4 7.5l4.5-5" fill="none" stroke="var(--on-brand)" strokeWidth="1.8" />
              </svg>
            )}
          </span>
        )}
        {label}
      </span>
      <span className={`t-reading mt-1.5 block ${off ? "opacity-45" : ""} ${severity ? SEVERITY_TEXT[severity] : tone === "brand" ? "text-brand" : "text-ink"}`}>{value}</span>
      {note && <span className="t-dense mt-0.5 block text-muted">{note}</span>}
    </>
  );
  const cls = "card relative flex min-w-0 flex-col items-stretch justify-start overflow-hidden pt-3.5 pr-3 pb-3 pl-3 text-left";
  if (!onToggle) return <div className={cls}>{body}</div>;
  return (
    <button type="button" aria-pressed={pressed} onClick={onToggle} className={`${cls} ctl hover:shadow-[var(--shadow-float)] ${off ? "bg-paper-alt" : ""}`}>
      {body}
    </button>
  );
}
