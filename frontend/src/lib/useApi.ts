import { useEffect, useState } from "react";
import { ApiError } from "../api/client";

export interface Loadable<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
}

const cache = new Map<string, Promise<unknown>>();

/** Fetch once per key per session. Errors are not cached: the next mount retries. */
export function cached<T>(key: string, load: () => Promise<T>): Promise<T> {
  let p = cache.get(key) as Promise<T> | undefined;
  if (!p) {
    p = load();
    cache.set(key, p);
    p.catch(() => cache.delete(key));
  }
  return p;
}

export function useApi<T>(key: string | null, load: () => Promise<T>): Loadable<T> {
  const [state, setState] = useState<Loadable<T>>({ data: null, error: null, loading: key !== null });
  useEffect(() => {
    if (key === null) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    let live = true;
    setState((s) => ({ data: s.data && key === lastKey.get(setState) ? s.data : null, error: null, loading: true }));
    lastKey.set(setState, key);
    cached(key, load).then(
      (data) => live && setState({ data, error: null, loading: false }),
      (e) =>
        live &&
        setState({
          data: null,
          error: e instanceof ApiError ? e : new ApiError(0, String(e)),
          loading: false,
        }),
    );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}

const lastKey = new WeakMap<object, string>();
