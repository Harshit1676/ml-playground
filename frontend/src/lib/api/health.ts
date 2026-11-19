export interface HealthResponse {
  status: 'ok'
}

const defaultApiBaseUrl = 'http://localhost:8000'

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()
  const baseUrl = (configuredBaseUrl || defaultApiBaseUrl).replace(/\/+$/, '')
  const response = await fetch(`${baseUrl}/api/v1/health`, {
    headers: { Accept: 'application/json' },
    signal,
  })

  if (!response.ok) {
    throw new Error(`Health check failed with HTTP ${response.status}.`)
  }

  const payload: unknown = await response.json()
  if (
    typeof payload !== 'object' ||
    payload === null ||
    !('status' in payload) ||
    payload.status !== 'ok'
  ) {
    throw new Error('Health endpoint returned an unexpected response.')
  }

  return payload as HealthResponse
}
