import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
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
export function PageHeader({ title, children, actions }: { title: string; children?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
      <div className="min-w-0">
        <h1 className="t-display">{title}</h1>
        {children && <div className="t-ui mt-1.5 max-w-3xl text-muted">{children}</div>}
      </div>
      {actions && <div className="flex w-full flex-wrap items-center gap-3 sm:w-auto">{actions}</div>}
    </header>
  );
}

/** A card with a real title (h3 scale) and an optional one-line caption under it. */
export function Card({
  title,
  caption,
  aside,
  children,
  className = "",
  bodyClass = "",
  as: Tag = "section",
}: {
  title?: ReactNode;
  caption?: ReactNode;
  aside?: ReactNode;
  children?: ReactNode;
  className?: string;
  bodyClass?: string;
  as?: "section" | "div" | "aside";
}) {
  return (
    <Tag className={`card flex min-w-0 flex-col ${className}`}>
      {(title || aside) && (
        <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2 px-5 pt-5">
          <div className="min-w-0">
            {title && <h2 className="t-h3">{title}</h2>}
            {caption && <p className="t-dense mt-1 text-muted">{caption}</p>}
          </div>
          {aside}
        </div>
      )}
      <div className={`min-w-0 flex-1 px-5 pt-4 pb-5 ${bodyClass}`}>{children}</div>
    </Tag>
  );
}

/** Scenario response-check verdict: fail / warn / pass, never plain grey text. */
export function VerdictBadge({ verdict }: { verdict: string }) {
  const v = verdict.toUpperCase();
  const cls = v === "PASS" || v === "OK" ? "pill-ok" : v === "NEGLIGIBLE" ? "pill-watch" : "pill-alert";
  return <span className={`pill pill-sm ${cls}`}>{v === "OK" ? "Pass" : v.charAt(0) + v.slice(1).toLowerCase().replaceAll("_", " ")}</span>;
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
const SEVERITY_TEXT: Record<Severity, string> = { ALERT: "text-alert", WATCH: "text-watch-text", INSUFFICIENT_EVIDENCE: "text-ink" };

/** Severity as a pill: tinted ground, severity-coloured text, a dot. */
export function SeverityLabel({ s, small = false }: { s: Severity; small?: boolean }) {
  const text = s === "ALERT" ? "Alert" : s === "WATCH" ? "Watch" : "Insufficient evidence";
  return <span className={`pill ${PILL[s]} ${small ? "pill-sm" : ""}`}>{text}</span>;
}

export function ErrorNote({ error, what }: { error: ApiError | null; what: string }) {
  if (!error) return null;
  return (
    <div role="alert" className="t-ui my-2 rounded-md border-l-[3px] border-alert bg-[var(--severity-critical-bg)] py-2 pr-3 pl-3">
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

/** Placeholder lines in the shape of the content that is loading. */
export function Skeleton({ lines = 3, className = "", label }: { lines?: number; className?: string; label: string }) {
  return (
    <div className={`flex flex-col gap-2.5 ${className}`} role="status" aria-live="polite">
      <span className="sr-only">Loading {label}</span>
      {Array.from({ length: lines }, (_, i) => (
        <span key={i} aria-hidden className="skeleton block h-3.5" style={{ width: `${[92, 78, 64, 86, 70][i % 5]}%` }} />
      ))}
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

/** One-of-N choice. The single toggle control used everywhere (variable, span, fold, tool,
 * scenario map mode): a pill track, the chosen option filled with the brand. ARIA
 * radiogroup with roving tabindex and arrow keys. */
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
  const track = useRef<HTMLDivElement>(null);
  const idx = Math.max(0, options.findIndex((o) => o.value === value));
  // The thumb follows the chosen option; measured, so labels of any width line up.
  const [thumb, setThumb] = useState<{ x: number; w: number } | null>(null);
  useLayoutEffect(() => {
    const measure = () => {
      const el = refs.current[idx];
      if (el) setThumb({ x: el.offsetLeft, w: el.offsetWidth });
    };
    measure();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    if (ro && track.current) ro.observe(track.current);
    return () => ro?.disconnect();
  }, [idx, options.length]);
  function onKey(e: React.KeyboardEvent) {
    const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const next = (idx + d + options.length) % options.length;
    onChange(options[next].value);
    refs.current[next]?.focus();
  }
  return (
    <div
      ref={track}
      role="radiogroup"
      aria-label={label}
      onKeyDown={onKey}
      className={`relative isolate inline-flex max-w-full shrink-0 self-start overflow-x-auto rounded-full bg-raised p-[3px] shadow-[inset_0_0_0_1px_var(--border)] ${className}`}
    >
      {thumb && <span aria-hidden className="seg-thumb -z-10" style={{ width: thumb.w, transform: `translateX(${thumb.x}px)` }} />}
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
            className={`${dense ? "h-7 px-3 text-[12px]" : "h-[30px] px-3.5 text-[13px]"} rounded-full leading-4 font-medium whitespace-nowrap ${
              on ? `text-[var(--on-brand)] ${thumb ? "" : "bg-brand"}` : "text-muted hover:text-ink"
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

/** A counted reading (§2.5 stat tile): eyebrow label, a big number, a caption, and a 3px
 * top edge in the tile's accent (severity colour, hatched for insufficient evidence).
 * With `onToggle` the whole tile is the toggle. */
export function Stat({
  label,
  value,
  note,
  severity,
  tone = "neutral",
  pressed,
  onToggle,
  className = "",
}: {
  label: ReactNode;
  value: ReactNode;
  note?: ReactNode;
  severity?: Severity | null;
  /** Counts that are not severities can carry the brand instead. */
  tone?: "brand" | "neutral";
  pressed?: boolean;
  onToggle?: () => void;
  className?: string;
}) {
  const off = onToggle && !pressed;
  const edge = severity === "INSUFFICIENT_EVIDENCE" ? undefined : severity ? SEVERITY_COLOR[severity] : tone === "brand" ? "var(--brand-500)" : "var(--border-strong)";
  const body = (
    <>
      <span aria-hidden className={`absolute inset-x-0 top-0 h-[3px] ${severity === "INSUFFICIENT_EVIDENCE" ? "hatch" : ""}`} style={{ background: edge }} />
      <span className="flex items-center justify-between gap-2">
        <span className="t-eyebrow min-w-0 break-words">{label}</span>
        {onToggle && (
          <span aria-hidden className={`inline-grid h-4 w-4 shrink-0 place-items-center rounded-[4px] border-2 ${pressed ? "border-brand bg-brand" : "border-hairline-strong"}`}>
            {pressed && (
              <svg viewBox="0 0 10 10" width="9" height="9">
                <path d="M1.5 5.2 4 7.5l4.5-5" fill="none" stroke="#fff" strokeWidth="1.8" />
              </svg>
            )}
          </span>
        )}
      </span>
      <span className={`t-stat mt-2 block ${off ? "opacity-40" : ""} ${severity ? SEVERITY_TEXT[severity] : tone === "brand" ? "text-brand-text" : "text-ink"}`}>{typeof value === "number" ? <CountUp value={value} /> : value}</span>
      {note && <span className="t-dense mt-1 block text-muted">{note}</span>}
    </>
  );
  const cls = `card relative flex min-w-0 flex-col items-stretch justify-start overflow-hidden p-4 pt-[18px] text-left ${className}`;
  if (!onToggle) return <div className={cls}>{body}</div>;
  return (
    <button type="button" aria-pressed={pressed} onClick={onToggle} className={`${cls} ctl hover:bg-raised ${off ? "opacity-80" : ""}`}>
      {body}
    </button>
  );
}

/** An integer that counts up to its value when it first appears or changes (~500 ms, eased).
 * Display only: the final number is the value given; nothing is computed here. */
export function CountUp({ value }: { value: number }) {
  const [shown, setShown] = useState(value);
  const from = useRef(0);
  useEffect(() => {
    if (!Number.isInteger(value) || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
      setShown(value);
      return;
    }
    const a = from.current;
    const t0 = performance.now();
    let raf = 0;
    const tick = (t: number) => {
      const k = Math.min(1, (t - t0) / 500);
      const e = 1 - Math.pow(1 - k, 3);
      setShown(Math.round(a + (value - a) * e));
      if (k < 1) raf = requestAnimationFrame(tick);
      else from.current = value;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value]);
  return <span aria-label={String(value)}>{shown}</span>;
}
