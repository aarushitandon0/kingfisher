import { useSyncExternalStore } from "react";
import type { Theme } from "./ramp";

// Light/dark: follows the system unless the viewer pins one (remembered per browser).
// The pin is `data-theme` on <html>; theme.css reads it.

const KEY = "kingfisher-theme";
const listeners = new Set<() => void>();
const media = typeof window !== "undefined" ? window.matchMedia?.("(prefers-color-scheme: dark)") : undefined;

function readPin(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

let pin: Theme | null = readPin();
if (pin && typeof document !== "undefined") document.documentElement.dataset.theme = pin;

export function currentTheme(): Theme {
  return pin ?? (media?.matches ? "dark" : "light");
}

export function setTheme(t: Theme) {
  pin = t;
  document.documentElement.dataset.theme = t;
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* private window: the pin lasts for this page only */
  }
  listeners.forEach((l) => l());
}

media?.addEventListener?.("change", () => listeners.forEach((l) => l()));

function subscribe(l: () => void) {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function useTheme(): Theme {
  return useSyncExternalStore(subscribe, currentTheme, () => "light");
}
