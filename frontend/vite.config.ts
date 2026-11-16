import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In development the React app runs on :5173 and forwards API calls to the
// FastAPI backend on :8000. In production FastAPI serves the built files itself.
const backend = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": backend,
      "/files": backend,
      "/ws": { target: backend, ws: true },
    },
  },
});
