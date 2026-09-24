import { useSyncExternalStore } from "react";
import type { Theme } from "./ramp";

// Dark by default (the design is dark-first: the map needs a dark ground for the water to
// be the brightest thing on it). The viewer can pin light; the pin is remembered per
// browser and written as `data-theme` on <html>, which theme.css reads.

const KEY = "kingfisher-theme";
const listeners = new Set<() => void>();

function readPin(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

let current: Theme = readPin() ?? "dark";
if (typeof document !== "undefined") document.documentElement.dataset.theme = current;

export function currentTheme(): Theme {
  return current;
}

export function setTheme(t: Theme) {
  current = t;
  document.documentElement.dataset.theme = t;
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* private window: the pin lasts for this page only */
  }
  listeners.forEach((l) => l());
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function useTheme(): Theme {
  return useSyncExternalStore(subscribe, currentTheme, () => "dark");
}
