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
      className="btn btn-ghost btn-sm h-8 w-8 px-0"
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

export function TopBar({ cities }: { cities: City[] | null }) {
  const { view, city, setView, setCity } = useStore();
  const current = cities?.find((c) => c.city === city);
  return (
    <header className="relative z-30 flex shrink-0 flex-col border-b border-hairline bg-surface md:h-14 md:flex-row md:items-stretch">
      <div className="flex h-12 min-w-0 items-center gap-3 px-4 md:h-auto">
        <span className="flex items-center gap-2.5">
          <Mark />
          <span className="font-display text-[19px] leading-6 font-semibold tracking-[-0.02em]">Kingfisher</span>
        </span>
        <span aria-hidden className="h-5 w-px bg-hairline" />
        <label className="sr-only" htmlFor="city">
          City
        </label>
        <select
          id="city"
          value={city}
          onChange={(e) => setCity(e.target.value)}
          className="t-ui ctl h-8 max-w-[11rem] rounded-md border border-transparent bg-transparent pr-1 pl-1.5 font-medium text-ink hover:border-hairline hover:bg-paper-alt"
        >
          {(cities ?? [{ city, name: city } as City]).map((c) => (
            <option key={c.city} value={c.city}>
              {c.name}
              {c.status === "NOT_BUILT" ? " (not built)" : ""}
            </option>
          ))}
        </select>
        {current && current.status === "READY" && (
          <span className="t-dense hidden truncate text-muted lg:inline">
            <span className="t-value-sm text-ink">{current.reaches}</span> reaches, <span className="t-value-sm text-ink">{current.optically_observable}</span> optically observable
          </span>
        )}
        <span className="ml-auto md:hidden">
          <ThemeToggle />
        </span>
      </div>
      <nav className="flex h-11 items-stretch gap-1 border-t border-hairline px-2 md:ml-auto md:h-auto md:border-t-0" aria-label="Views">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            type="button"
            onClick={() => setView(v.id)}
            aria-current={view === v.id ? "page" : undefined}
            className={`t-ui relative flex-1 px-3 font-medium md:flex-none ${view === v.id ? "text-brand" : "text-muted hover:text-ink"}`}
          >
            <span className={`rounded-md px-2 py-1.5 ${view === v.id ? "" : "hover-tint"}`}>{v.label}</span>
            <span aria-hidden className={`absolute inset-x-2 -bottom-px h-0.5 rounded-full ${view === v.id ? "bg-brand" : "bg-transparent"}`} />
          </button>
        ))}
        <span className="hidden items-center pl-2 md:flex">
          <ThemeToggle />
        </span>
      </nav>
    </header>
  );
}
