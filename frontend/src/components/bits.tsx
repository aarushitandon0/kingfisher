import type { ReactNode } from "react";
import type { ApiError } from "../api/client";
import type { Observability, Severity } from "../api/types";
import { SEVERITY_COLOR } from "../lib/ramp";

/** A rule with a small label - section divider in dense panels. */
export function Rule({ children }: { children: ReactNode }) {
  return (
    <div className="mt-5 mb-2 flex items-center gap-2">
      <span className="t-dense shrink-0 text-muted">{children}</span>
      <span className="h-px flex-1 bg-hairline" />
    </div>
  );
}

/** Plainly stated - "optically observable" or "driver-predicted". */
export function ObservabilityBadge({ observability, medianPixels }: { observability: Observability; medianPixels?: number | null }) {
  if (observability === "OPTICALLY_OBSERVABLE")
    return (
      <p className="t-ui flex items-baseline gap-2">
        <span aria-hidden className="inline-block h-2 w-2 rotate-45 bg-ink" />
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
        <span aria-hidden className="inline-block h-2 w-2 rotate-45 border border-ink" />
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
    <div className="t-ui border-l-2 border-hairline py-1 pl-3 text-muted">
      <p className="text-ink">{what} unavailable.</p>
      <p>{error.detail}</p>
    </div>
  );
}

export function Loading({ what }: { what: string }) {
  return <p className="t-ui text-muted">Loading {what}</p>;
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
