export function getApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim()
  if (import.meta.env.DEV && !configured) return 'http://localhost:8000'
  if (!configured) throw new Error('VITE_API_BASE_URL is required')
  return configured.replace(/\/+$/, '')
}
