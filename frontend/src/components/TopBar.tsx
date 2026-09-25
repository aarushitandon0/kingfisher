import type { City } from "../api/types";
import { setTheme, useTheme } from "../lib/theme";
import { VIEWS, useStore } from "../store";

/** The favicon's mark: a stream line and the bird - cobalt back, orange breast. */
function Mark() {
  return (
    <svg aria-hidden viewBox="0 0 32 32" width="30" height="30" className="shrink-0 self-center">
      <rect width="32" height="32" rx="8" fill="var(--brand-500)" />
      <path d="M5 22c3.7-2.6 7.3-2.6 11 0s7.3 2.6 11 0" fill="none" stroke="var(--on-brand)" strokeWidth="2.6" strokeLinecap="round" />
      <circle cx="21" cy="11.5" r="3.4" fill="var(--severity-watch)" />
    </svg>
  );
}

function ThemeToggle() {
  const theme = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      className="btn btn-ghost btn-icon"
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
    >
      {theme === "dark" ? (
        <svg aria-hidden viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
          <circle cx="8" cy="8" r="3" />
          <path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1" />
        </svg>
      ) : (
        <svg aria-hidden viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round">
          <path d="M13.5 9.5A5.5 5.5 0 0 1 6.5 2.5a5.5 5.5 0 1 0 7 7Z" />
        </svg>
      )}
    </button>
  );
}

const ICONS: Record<string, string> = {
  map: "M1.5 3.5 5.5 2l5 1.5 4-1.5v10.5l-4 1.5-5-1.5-4 1.5z M5.5 2v10.5 M10.5 3.5V14",
  alerts: "M8 2.2 14.3 13.3H1.7z M8 6.5v3 M8 11.3v.1",
  scenarios: "M2 12.5c2.5-4 4.5-4 6 0s3.5 4 6 0 M2 5.5h5 M11 5.5h3 M9 3.5v4",
  validation: "M2 14h12 M4 14V9 M8 14V4 M12 14V7",
};

export function TopBar({ cities }: { cities: City[] | null }) {
  const { view, city, setView, setCity } = useStore();
  const current = cities?.find((c) => c.city === city);
  return (
    <header className="relative z-30 flex shrink-0 flex-col border-b border-hairline bg-chrome md:h-16 md:flex-row md:items-stretch">
      <div className="flex h-14 min-w-0 items-center gap-3 px-4 md:h-auto md:px-6">
        <span className="flex items-center gap-2.5">
          <Mark />
          <span className="t-h3 text-[17px]">Kingfisher</span>
        </span>
        <span aria-hidden className="mx-1 h-6 w-px bg-hairline" />
        <label className="sr-only" htmlFor="city">
          City
        </label>
        <span className="relative inline-flex items-center">
          <select
            id="city"
            value={city}
            onChange={(e) => setCity(e.target.value)}
            className="ctl h-9 max-w-[12rem] appearance-none rounded-[var(--radius-md)] border border-hairline bg-surface pr-8 pl-3 text-[13px] font-semibold text-ink hover:border-hairline-strong hover:bg-raised"
          >
            {(cities ?? [{ city, name: city } as City]).map((c) => (
              <option key={c.city} value={c.city}>
                {c.name}
                {c.status === "NOT_BUILT" ? " (not built)" : ""}
              </option>
            ))}
          </select>
          <svg aria-hidden viewBox="0 0 12 12" width="12" height="12" className="pointer-events-none absolute right-3 text-muted">
            <path d="M3 4.5 6 7.5l3-3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </span>
        {current && current.status === "READY" && (
          <span className="t-dense hidden truncate text-faint lg:inline">
            <span className="t-value-sm text-ink">{current.reaches}</span> reaches, <span className="t-value-sm text-ink">{current.optically_observable}</span> optically observable
          </span>
        )}
        <span className="ml-auto md:hidden">
          <ThemeToggle />
        </span>
      </div>
      <nav className="flex h-12 items-stretch overflow-x-auto border-t border-hairline px-1 md:ml-auto md:h-auto md:gap-1 md:border-t-0 md:px-4" aria-label="Views">
        {VIEWS.map((v) => {
          const on = view === v.id;
          return (
            <button key={v.id} type="button" onClick={() => setView(v.id)} aria-current={on ? "page" : undefined} className="nav-tab flex-1 md:flex-none">
              <span aria-hidden className="nav-tab-bg" />
              <svg aria-hidden viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="hidden shrink-0 min-[400px]:block">
                <path d={ICONS[v.id]} />
              </svg>
              {v.label}
            </button>
          );
        })}
        <span className="hidden items-center gap-1 border-l border-hairline pl-3 ml-2 my-3.5 md:flex">
          <button
            type="button"
            onClick={() => window.dispatchEvent(new Event("kf:shortcuts"))}
            className="btn btn-ghost btn-icon"
            aria-label="Keyboard shortcuts"
            title="Keyboard shortcuts (?)"
          >
            <svg aria-hidden viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
              <rect x="1.5" y="3.5" width="13" height="9" rx="1.5" />
              <path d="M4 6.5h.01M6.5 6.5h.01M9 6.5h.01M11.5 6.5h.01M4.5 9.5h7" />
            </svg>
          </button>
          <ThemeToggle />
        </span>
      </nav>
    </header>
  );
}
