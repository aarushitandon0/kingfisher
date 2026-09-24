import { useEffect, useRef } from "react";

/** True when a keypress belongs to a text field and should not trigger a shortcut. */
export function typing(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  return !!t && (/^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName) || t.isContentEditable);
}

/** A keyboard shortcut for as long as the component is mounted. With `ctrl`, Ctrl (or Cmd)
 * must be held and the shortcut also works inside text fields, as in VS Code. */
export function useShortcut(key: string, fn: () => void, opts: { ctrl?: boolean } = {}) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (e.key.toLowerCase() !== key.toLowerCase() || e.altKey) return;
      if (opts.ctrl ? !mod : mod || typing(e)) return;
      e.preventDefault();
      ref.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [key, opts.ctrl]);
}
