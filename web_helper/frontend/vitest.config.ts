/// <reference types="vitest" />
import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: true,
    exclude: ['tests/e2e/**', '**/node_modules/**', '**/dist/**'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.d.ts',
        '**/*.config.*',
        'dist/',
      ],
    },
  },
  resolve: {
    alias: [
      { find: /^@\/components\/ui$/, replacement: path.resolve(__dirname, './src/shared/ui') },
      { find: /^@\/components\/ui\/(.*)$/, replacement: path.resolve(__dirname, './src/shared/ui/$1') },
      { find: '@app', replacement: path.resolve(__dirname, './src/app') },
      { find: '@pages', replacement: path.resolve(__dirname, './src/pages') },
      { find: '@widgets', replacement: path.resolve(__dirname, './src/widgets') },
      { find: '@features', replacement: path.resolve(__dirname, './src/features') },
      { find: '@entities', replacement: path.resolve(__dirname, './src/entities') },
      { find: '@shared', replacement: path.resolve(__dirname, './src/shared') },
      { find: '@', replacement: path.resolve(__dirname, './src') },
    ],
  },
});
