import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      // Proxy API requests to the backend server
      '/api': 'http://127.0.0.1:8000',
      '/configs': 'http://127.0.0.1:8000',
      '/files': 'http://127.0.0.1:8000',
      '/run': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
    }
  }
})