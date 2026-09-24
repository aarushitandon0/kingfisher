import { useRef, type ReactNode } from "react";
import type { ApiError } from "../api/client";
import type { Observability, Severity } from "../api/types";
import { rampColor, SEVERITY_COLOR } from "../lib/ramp";

/** A rule with a small label - section divider in dense panels. Rendered as a real heading
 * so screen-reader users can jump between sections. */
export function Rule({ children, as: Tag = "h3" }: { children: ReactNode; as?: "h2" | "h3" | "h4" | "p" }) {
  return (
    <div className="mt-6 mb-2 flex items-center gap-2">
      <Tag className="t-dense shrink-0 font-medium text-ink">{children}</Tag>
      <span className="h-px flex-1 bg-hairline" />
    </div>
  );
}

/** Page title + one line of context. Page titles sit one step above section titles. */
export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="mb-5">
      <h1 className="t-display">{title}</h1>
      {children && <div className="t-ui mt-1 max-w-4xl text-muted">{children}</div>}
    </header>
  );
}

/** Plainly stated - "optically observable" or "driver-predicted". */
export function ObservabilityBadge({ observability, medianPixels }: { observability: Observability; medianPixels?: number | null }) {
  if (observability === "OPTICALLY_OBSERVABLE")
    return (
      <p className="t-ui flex items-baseline gap-2">
        <span aria-hidden className="inline-block h-2 w-2 shrink-0 rotate-45 bg-ink" />
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
        <span aria-hidden className="inline-block h-2 w-2 shrink-0 rotate-45 border border-ink" />
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

export function SeverityLabel({ s }: { s: Severity }) {
  if (s === "ALERT") return <span className="text-alert">Alert</span>;
  if (s === "WATCH") return <span className="text-watch-text">Watch</span>;
  return <span className="text-muted">Insufficient evidence</span>;
}

export function ErrorNote({ error, what }: { error: ApiError | null; what: string }) {
  if (!error) return null;
  return (
    <div role="alert" className="t-ui my-2 border-l-2 border-ink bg-paper-alt py-2 pr-3 pl-3">
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

export function Swatch({ color, hatch, dashed }: { color?: string; hatch?: boolean; dashed?: boolean }) {
  if (hatch) return <span aria-hidden className="hatch inline-block h-2.5 w-5 align-middle" />;
  if (dashed)
    return (
      <span
        aria-hidden
        className="inline-block h-0.5 w-5 align-middle"
        style={{ backgroundImage: `repeating-linear-gradient(90deg, ${color} 0 6px, transparent 6px 10px)` }}
      />
    );
  return <span aria-hidden className="inline-block h-[3px] w-5 align-middle" style={{ background: color }} />;
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
    <div role="radiogroup" aria-label={label} onKeyDown={onKey} className={`inline-flex max-w-full shrink-0 self-start border border-hairline bg-paper ${className}`}>
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
            className={`${dense ? "t-dense h-7 px-2" : "t-ui h-8 px-3"} whitespace-nowrap ${i ? "border-l border-hairline" : ""} ${
              on ? "bg-paper-alt text-ink shadow-[inset_0_-2px_0_var(--kingfisher)]" : "text-muted hover:bg-paper-alt hover:text-ink"
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
      <div className="relative flex h-1.5">
        {edges.slice(0, -1).map((a, i) => (
          <span key={a} style={{ width: `${((edges[i + 1] - a) / max) * 100}%`, background: rampColor((a + edges[i + 1]) / 2, breaks) }} />
        ))}
        <span className="absolute -top-1 -bottom-1 w-0.5 -translate-x-1/2 bg-ink" style={{ left: `${pos}%` }} />
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

/** A counted reading: label, a mono number, a 3px severity edge. Optionally a filter toggle. */
export function Stat({
  label,
  value,
  note,
  severity,
  pressed,
  onToggle,
}: {
  label: ReactNode;
  value: ReactNode;
  note?: ReactNode;
  severity?: Severity | null;
  pressed?: boolean;
  onToggle?: () => void;
}) {
  const b: { className: string; style?: React.CSSProperties } = severity ? severityBorder(severity) : { className: "" };
  const body = (
    <>
      <span className="t-dense flex items-center gap-1.5">
        {onToggle && (
          <span
            aria-hidden
            className={`inline-block h-3 w-3 shrink-0 border ${pressed ? "border-kf bg-kf shadow-[inset_0_0_0_2px_var(--chart-paper)]" : "border-muted bg-paper"}`}
          />
        )}
        {label}
      </span>
      <span className={`t-reading mt-1 block ${onToggle && !pressed ? "text-muted" : ""}`}>{value}</span>
      {note && <span className="t-dense mt-0.5 block text-muted">{note}</span>}
    </>
  );
  const cls = `flex min-w-0 flex-col items-stretch justify-start border border-hairline py-2.5 pr-3 pl-3 text-left ${b.className}`;
  if (!onToggle)
    return (
      <div className={cls} style={b.style}>
        {body}
      </div>
    );
  return (
    <button type="button" aria-pressed={pressed} onClick={onToggle} className={`${cls} ctl hover:bg-paper-alt ${pressed ? "" : "bg-paper-alt/50"}`} style={b.style}>
      {body}
    </button>
  );
}
