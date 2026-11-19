import { afterEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_REQUEST } from '../../features/linear-regression/defaults'
import type { LinearRegressionRequest, LinearRegressionResponse } from '../../features/linear-regression/types'
import { LinearRegressionApiError, LinearRegressionNetworkError, runLinearRegression } from './linearRegression'

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

function responseFor(request: LinearRegressionRequest = DEFAULT_REQUEST): LinearRegressionResponse {
  const x = request.dataset.kind === 'points' ? [...request.dataset.x] : [-1, 0, 1]
  const y = request.dataset.kind === 'points' ? [...request.dataset.y] : [-1, 1, 3]
  return {
    algorithm: 'linear_regression',
    seed: request.seed,
    dataset: { x, y },
    parameters: request.parameters,
    model: { slope: 0.4, intercept: 0.6 },
    predictions: { y_hat: x.map((feature) => 0.4 * feature + 0.6) },
    metrics: { mse: 2.4 },
    training: { steps_completed: request.parameters.steps, loss_history: Array.from({ length: request.parameters.steps + 1 }, (_, index) => 4 - index * 0.01) },
  }
}

describe('runLinearRegression', () => {
  it('posts the typed request to the configured API and validates the response', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.test/')
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => responseFor() })
    vi.stubGlobal('fetch', fetchMock)

    await expect(runLinearRegression(DEFAULT_REQUEST)).resolves.toEqual(responseFor())
    expect(fetchMock).toHaveBeenCalledWith('https://api.example.test/api/v1/experiments/linear-regression', {
      method: 'POST',
      headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: JSON.stringify(DEFAULT_REQUEST),
      signal: undefined,
    })
  })

  it('exposes structured backend validation errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({ error: { code: 'VALIDATION_ERROR', message: 'Request validation failed.', details: { fields: [{ field: 'dataset.x', reason: 'Invalid value.' }] } } }),
    }))

    await expect(runLinearRegression(DEFAULT_REQUEST)).rejects.toMatchObject({
      name: 'LinearRegressionApiError',
      status: 422,
      code: 'VALIDATION_ERROR',
      details: { fields: [{ reason: 'Invalid value.' }] },
    })
  })

  it('maps a failed fetch to a retryable network error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('offline')))

    await expect(runLinearRegression(DEFAULT_REQUEST)).rejects.toBeInstanceOf(LinearRegressionNetworkError)
  })

  it('rejects a successful HTTP response that does not match the response contract', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ algorithm: 'linear_regression' }) }))

    await expect(runLinearRegression(DEFAULT_REQUEST)).rejects.toMatchObject({
      name: 'LinearRegressionApiError',
      code: 'INVALID_RESPONSE',
    } satisfies Partial<LinearRegressionApiError>)
  })

  it.each([Number.NaN, Number.POSITIVE_INFINITY, -0.01])(
    'rejects non-finite or negative loss values (%s)',
    async (loss) => {
      const response = responseFor()
      response.training.loss_history[1] = loss
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => response,
      }))

      await expect(runLinearRegression(DEFAULT_REQUEST)).rejects.toMatchObject({ code: 'INVALID_RESPONSE' })
    },
  )

  it('rejects a prediction array whose length differs from the observations', async () => {
    const response = responseFor()
    response.predictions.y_hat.pop()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => response }))

    await expect(runLinearRegression(DEFAULT_REQUEST)).rejects.toMatchObject({ code: 'INVALID_RESPONSE' })
  })

  it('rejects predictions inconsistent with the returned model parameters', async () => {
    const response = responseFor()
    response.predictions.y_hat[0] += 0.01
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => response }))

    await expect(runLinearRegression(DEFAULT_REQUEST)).rejects.toMatchObject({ code: 'INVALID_RESPONSE' })
  })

  it('accepts rounding differences within the model prediction tolerance', async () => {
    const response = responseFor()
    response.predictions.y_hat[0] += 5e-10
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => response }))

    await expect(runLinearRegression(DEFAULT_REQUEST)).resolves.toEqual(response)
  })

  it('rejects inline dataset observations that differ from the submitted points', async () => {
    const request: LinearRegressionRequest = {
      ...DEFAULT_REQUEST,
      dataset: { kind: 'points', x: [-1, 0, 1], y: [-1, 1, 3] },
    }
    const response = responseFor(request)
    response.dataset.x[0] = -0.5
    response.predictions.y_hat = response.dataset.x.map((feature) => 0.4 * feature + 0.6)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => response }))

    await expect(runLinearRegression(request)).rejects.toMatchObject({ code: 'INVALID_RESPONSE' })
  })
})
