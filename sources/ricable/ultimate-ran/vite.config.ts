import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  root: './apps/dojo', // Set apps/dojo as the root for Vite
  build: {
    outDir: '../../dist/dojo-ui', // Output to dist/dojo-ui
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    open: false, // Don't open browser automatically
  },
  resolve: {
    alias: {
      // Alias for modules within the dojo app, if any
      '@dojo': path.resolve(__dirname, './apps/dojo/src'),
      // Alias for project root modules, useful for shared components/types
      '@src': path.resolve(__dirname, './src'),
    },
  },
});
