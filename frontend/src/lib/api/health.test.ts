import { afterEach, describe, expect, it, vi } from 'vitest'
import { getHealth } from './health'

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe('getHealth', () => {
  it('requests the versioned health endpoint using the configured base URL', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.test/')
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'ok' }) })
    vi.stubGlobal('fetch', fetchMock)

    await expect(getHealth()).resolves.toEqual({ status: 'ok' })
    expect(fetchMock).toHaveBeenCalledWith('https://api.example.test/api/v1/health', {
      headers: { Accept: 'application/json' },
      signal: undefined,
    })
  })

  it('rejects non-success responses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 503 }))

    await expect(getHealth()).rejects.toThrow('HTTP 503')
  })

  it('rejects malformed health payloads', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'ready' }) }))

    await expect(getHealth()).rejects.toThrow('unexpected response')
  })
})
