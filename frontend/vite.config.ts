import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Defaults: frontend on 5199, backend on 8765. Override with CLAIMGRAPH_FRONTEND_PORT and CLAIMGRAPH_API.
const port = Number(process.env.CLAIMGRAPH_FRONTEND_PORT ?? 5199)

export default defineConfig({
  plugins: [react()],
  server: {
    port,
    strictPort: true,
    proxy: {
      '/api': { target: process.env.CLAIMGRAPH_API ?? 'http://127.0.0.1:8765', changeOrigin: true },
    },
  },
  preview: { port },
})
