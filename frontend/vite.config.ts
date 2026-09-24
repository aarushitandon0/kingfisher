import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// The API runs on :8000 (make api). Proxying keeps the browser on one origin.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // MapLibre 6 resolves its worker (and the worker's shared chunk) relative to
  // import.meta.url; pre-bundling moves the module and breaks that.
  optimizeDeps: { exclude: ["maplibre-gl"] },
  worker: { format: "es" },
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000", "/health": "http://localhost:8000" },
  },
});
