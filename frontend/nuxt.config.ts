import { defineNuxtConfig } from 'nuxt/config'

export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: false },
  ssr: false,
  runtimeConfig: {
    public: {
      // Set at build time: NUXT_PUBLIC_API_BASE=https://your-api.example.com
      // Default hits FastAPI directly in local dev (see backend CORS).
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:9000',
    },
  },
})
