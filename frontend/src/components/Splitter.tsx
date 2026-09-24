import { useCallback, useRef, useState } from "react";

// VS Code-style sashes: the gap between two panels is the handle. Drag it to resize, double-
// click to reset, arrow keys for fine steps (Shift = large), Enter to collapse/expand when
// the panel allows it. Sizes are remembered per browser.

const PREFIX = "kingfisher-size:";

function read(key: string): number | null | undefined {
  try {
    const v = localStorage.getItem(PREFIX + key);
    if (v === null) return undefined;
    if (v === "auto") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  } catch {
    return undefined;
  }
}

/** A panel size in px that survives reloads. `null` means "use the layout's default". */
export function usePanelSize(key: string, initial: number | null): [number | null, (v: number | null) => void] {
  const [v, setV] = useState<number | null>(() => {
    const r = read(key);
    return r === undefined ? initial : r;
  });
  const set = useCallback(
    (n: number | null) => {
      setV(n);
      try {
        localStorage.setItem(PREFIX + key, n === null ? "auto" : String(Math.round(n)));
      } catch {
        /* private window: the size lasts for this page only */
      }
    },
    [key],
  );
  return [v, set];
}

export interface SplitterProps {
  /** "x": a vertical bar dragged left/right. "y": a horizontal bar dragged up/down. */
  axis: "x" | "y";
  /** Current size of the controlled panel in px (measured on drag start if a function). */
  size: number | (() => number);
  onSize: (px: number | null) => void;
  min: number;
  max: number;
  /** The controlled panel is after the handle (right / below): dragging towards it shrinks it. */
  reverse?: boolean;
  /** Dragging well below `min` collapses the panel to 0. */
  collapsible?: boolean;
  label: string;
  className?: string;
  style?: React.CSSProperties;
}

export function Splitter({ axis, size, onSize, min, max, reverse = false, collapsible = false, label, className = "", style }: SplitterProps) {
  const start = useRef<{ pos: number; size: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const current = () => (typeof size === "function" ? size() : size);
  const cur = typeof size === "number" ? size : null;
  const collapsed = collapsible && cur === 0;

  const clamp = (raw: number) => {
    if (collapsible && raw < min * 0.55) return 0;
    return Math.min(max, Math.max(min, raw));
  };

  function onPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    if (e.button !== 0) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    start.current = { pos: axis === "x" ? e.clientX : e.clientY, size: current() };
    setDragging(true);
    document.body.style.cursor = axis === "x" ? "col-resize" : "row-resize";
    document.body.style.userSelect = "none";
  }
  function onPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!start.current) return;
    const d = (axis === "x" ? e.clientX : e.clientY) - start.current.pos;
    onSize(clamp(start.current.size + (reverse ? -d : d)));
  }
  function end() {
    start.current = null;
    setDragging(false);
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }
  function onKeyDown(e: React.KeyboardEvent) {
    const step = e.shiftKey ? 64 : 16;
    const grow = axis === "x" ? (reverse ? "ArrowLeft" : "ArrowRight") : reverse ? "ArrowUp" : "ArrowDown";
    const shrink = axis === "x" ? (reverse ? "ArrowRight" : "ArrowLeft") : reverse ? "ArrowDown" : "ArrowUp";
    const now = current();
    if (e.key === grow) onSize(Math.min(max, Math.max(min, now + step)));
    else if (e.key === shrink) onSize(clamp(now - step));
    else if (e.key === "Home") onSize(collapsible ? 0 : min);
    else if (e.key === "End") onSize(max);
    else if (e.key === "Enter" && collapsible) onSize(collapsed ? null : 0);
    else return;
    e.preventDefault();
  }

  const x = axis === "x";
  return (
    <div
      role="separator"
      aria-orientation={x ? "vertical" : "horizontal"}
      aria-label={`${label}. Drag or use the arrow keys to resize; double-click to reset${collapsible ? "; Enter to collapse" : ""}.`}
      aria-valuenow={cur ?? undefined}
      aria-valuemin={collapsible ? 0 : min}
      aria-valuemax={max}
      tabIndex={0}
      title={`Drag to resize ${label.toLowerCase()}, double-click to reset`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={end}
      onPointerCancel={end}
      onDoubleClick={() => onSize(null)}
      onKeyDown={onKeyDown}
      data-dragging={dragging || undefined}
      className={`sash group relative z-10 flex shrink-0 touch-none items-center justify-center outline-none ${x ? "w-4 cursor-col-resize self-stretch" : "h-4 cursor-row-resize"} ${className}`}
      style={style}
    >
      {/* The line: invisible at rest, brand on hover / drag / focus - as in VS Code. */}
      <span
        aria-hidden
        className={`pointer-events-none absolute rounded-full bg-brand opacity-0 transition-opacity duration-150 group-hover:opacity-100 group-focus-visible:opacity-100 group-data-[dragging]:opacity-100 ${
          x ? "inset-y-1 left-1/2 w-[3px] -translate-x-1/2" : "inset-x-1 top-1/2 h-[3px] -translate-y-1/2"
        }`}
      />
      {/* A faint grip so the handle is discoverable before it is hovered. */}
      <span
        aria-hidden
        className={`pointer-events-none relative rounded-full bg-[var(--border-strong)] transition-opacity group-hover:opacity-0 group-data-[dragging]:opacity-0 ${x ? "h-8 w-1" : "h-1 w-8"}`}
      />
      {collapsed && (
        <span aria-hidden className="pointer-events-none absolute grid h-6 w-6 place-items-center rounded-full bg-raised text-muted shadow-[var(--shadow-md)]">
          <svg viewBox="0 0 12 12" width="10" height="10" className={x ? (reverse ? "" : "rotate-180") : reverse ? "-rotate-90" : "rotate-90"}>
            <path d="M7.5 2.5 4 6l3.5 3.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
          </svg>
        </span>
      )}
    </div>
  );
}
