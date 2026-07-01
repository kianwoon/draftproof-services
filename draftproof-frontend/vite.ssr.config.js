// Dedicated Vite config for the build-time SSR bundle (scripts/prerender-render.mjs).
// The shared vite.config.js sets rollupOptions.output.manualChunks to split
// react/react-dom/react-router-dom into a vendor chunk for the CLIENT bundle.
// That manualChunks rule is invalid for an SSR build — react is externalized on
// the server, so Rollup refuses to put an external module into a manual chunk
// ("react cannot be included in manualChunks... resolved as an external module").
// This config reuses the same React plugin but omits the chunk-splitting.
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    target: 'es2020',
    minify: false,
    sourcemap: false,
    reportCompressedSize: false,
  },
});
