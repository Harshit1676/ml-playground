import { getApiBaseUrl } from './baseUrl'
export interface HealthResponse {
  status: 'ok'
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const baseUrl = getApiBaseUrl()
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
