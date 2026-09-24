import { useEffect, useState } from "react";
import { typing, useShortcut } from "../lib/shortcuts";
import { VIEWS, useStore } from "../store";

const LIST: [string, string][] = [
  ["1 – 4", "Map, Alerts, Scenarios, Validation"],
  ["/", "Search reaches, streams and places"],
  ["Ctrl + B", "Show or hide the side panel"],
  ["Ctrl + J", "Show or hide the timeline panel (map)"],
  ["Esc", "Close the open reach, drawer or dialog"],
  ["Drag a gap", "Resize the panels on either side"],
  ["Double-click a gap", "Reset that panel's size"],
  ["?", "This list"],
];

/** Page-wide shortcuts, and the "?" sheet that lists them. */
export function Shortcuts() {
  const [open, setOpen] = useState(false);
  const setView = useStore((s) => s.setView);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const i = Number(e.key) - 1;
      if (e.ctrlKey || e.metaKey || e.altKey || typing(e) || !(i >= 0 && i < VIEWS.length)) return;
      setView(VIEWS[i].id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setView]);
  useShortcut("/", () => {
    const el = document.querySelector<HTMLInputElement>("[data-search]");
    if (el) {
      el.focus();
      el.select();
    } else setView("map");
  });
  useShortcut("?", () => setOpen((o) => !o));
  useEffect(() => {
    const onOpen = () => setOpen(true);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("kf:shortcuts", onOpen);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("kf:shortcuts", onOpen);
      window.removeEventListener("keydown", onKey);
    };
  }, []);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 grid place-items-center p-4" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
      <div className="scrim absolute inset-0 bg-[var(--scrim)]" onClick={() => setOpen(false)} />
      <div className="drawer card relative w-full max-w-[440px] bg-raised p-6 shadow-[var(--shadow-lg)]">
        <div className="flex items-center justify-between">
          <h2 className="t-title">Keyboard shortcuts</h2>
          <button type="button" className="btn btn-ghost btn-icon" onClick={() => setOpen(false)} aria-label="Close" autoFocus>
            <svg aria-hidden viewBox="0 0 12 12" width="12" height="12">
              <path d="M2.5 2.5l7 7M9.5 2.5l-7 7" stroke="currentColor" strokeWidth="1.5" />
            </svg>
          </button>
        </div>
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-5 gap-y-2.5 text-[13px]">
          {LIST.map(([k, v]) => (
            <div key={k} className="contents">
              <dt>
                <kbd className="rounded-md border border-hairline-strong bg-surface px-2 py-0.5 font-mono text-[12px] whitespace-nowrap text-ink shadow-[0_1px_0_var(--border-strong)]">{k}</kbd>
              </dt>
              <dd className="text-muted">{v}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}
