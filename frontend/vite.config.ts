import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { loadEnv } from 'vite'
import { releaseApiUrl } from './releaseApiUrl.ts'
import { defineConfig } from 'vitest/config'

export default defineConfig(({ command, mode }) => ({
  define: command === 'build' ? {
    'import.meta.env.VITE_API_BASE_URL': JSON.stringify(releaseApiUrl(loadEnv(mode, '.', 'VITE_API_BASE_URL').VITE_API_BASE_URL)),
  } : undefined,
  plugins: [react(), tailwindcss()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    clearMocks: true,
    pool: 'threads',
    maxWorkers: 1,
  },
}))
