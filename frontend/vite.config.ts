import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// The API runs on :8000 (make api). Proxying keeps the browser on one origin.
// `vite build --mode snapshot` (npm run build:snapshot) is the static, server-less build:
// relative asset paths (it may be served from a sub-path) and VITE_SNAPSHOT=1.
export default defineConfig(({ mode }) => ({
  base: mode === "snapshot" ? "./" : "/",
  define: mode === "snapshot" ? { "import.meta.env.VITE_SNAPSHOT": JSON.stringify("1") } : {},
  plugins: [react(), tailwindcss()],
  // MapLibre 6 resolves its worker (and the worker's shared chunk) relative to
  // import.meta.url; pre-bundling moves the module and breaks that.
  optimizeDeps: { exclude: ["maplibre-gl"] },
  worker: { format: "es" },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000", "/health": "http://127.0.0.1:8000" },
  },
}));
