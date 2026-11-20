import { afterEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_LOGISTIC_REQUEST } from '../../features/logistic-regression/defaults'
import type { LogisticRegressionRequest } from '../../features/logistic-regression/types'
import { LogisticRegressionApiError, LogisticRegressionNetworkError, runLogisticRegression } from './logisticRegression'
import { jsonResponse, responseForLogisticRequest } from '../../test/logisticRegressionFixtures'

afterEach(() => vi.unstubAllGlobals())

describe('Logistic Regression API client', () => {
  it('posts the typed request and accepts a semantically consistent response', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(responseForLogisticRequest(DEFAULT_LOGISTIC_REQUEST)))
    vi.stubGlobal('fetch', fetchMock)

    const result = await runLogisticRegression(DEFAULT_LOGISTIC_REQUEST)
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]

    expect(url).toBe('http://localhost:8000/api/v1/experiments/logistic-regression')
    expect(init.method).toBe('POST')
    expect(init.headers).toMatchObject({ 'Content-Type': 'application/json' })
    expect(JSON.parse(String(init.body))).toEqual(DEFAULT_LOGISTIC_REQUEST)
    expect(result.algorithm).toBe('logistic_regression')
    expect(result.training.steps_completed).toBe(DEFAULT_LOGISTIC_REQUEST.parameters.steps)
  })

  it('rejects an out-of-range probability', async () => {
    const response = responseForLogisticRequest(DEFAULT_LOGISTIC_REQUEST)
    response.predictions.probabilities[0] = 1.5
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(response)))

    await expect(runLogisticRegression(DEFAULT_LOGISTIC_REQUEST)).rejects.toMatchObject({
      name: 'LogisticRegressionApiError',
      code: 'INVALID_RESPONSE',
    })
  })

  it('checks initial BCE against the submitted parameters and accepts zero training steps', async () => {
    const zeroStepRequest = {
      ...DEFAULT_LOGISTIC_REQUEST,
      parameters: { ...DEFAULT_LOGISTIC_REQUEST.parameters, steps: 0 },
    }
    const validResponse = responseForLogisticRequest(zeroStepRequest)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(validResponse)))

    await expect(runLogisticRegression(zeroStepRequest)).resolves.toMatchObject({
      training: { steps_completed: 0, loss_history: [validResponse.training.loss_history[0]] },
    })

    const inconsistentResponse = responseForLogisticRequest(zeroStepRequest)
    inconsistentResponse.training.loss_history[0] += 0.1
    inconsistentResponse.metrics.binary_cross_entropy = inconsistentResponse.training.loss_history[0]
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(inconsistentResponse)))
    await expect(runLogisticRegression(zeroStepRequest)).rejects.toMatchObject({
      name: 'LogisticRegressionApiError',
      code: 'INVALID_RESPONSE',
    })
  })

  it('accepts the upper feature endpoint when uniform sampling rounds up, but rejects values above it', async () => {
    const upper = 1 + Number.EPSILON
    const request: LogisticRegressionRequest = {
      ...DEFAULT_LOGISTIC_REQUEST,
      dataset: { kind: 'synthetic_logistic', n_samples: 8, true_coefficient: 0, true_intercept: 0, x_min: 1, x_max: upper },
      parameters: { ...DEFAULT_LOGISTIC_REQUEST.parameters, steps: 0 },
    }
    // Effective points returned by FastAPI for seed 42 with adjacent float64 bounds.
    const x = [upper, 1, upper, upper, 1, upper, upper, upper]
    const response = responseForLogisticRequest({ ...request, dataset: { kind: 'points', x, y: [0, 0, 0, 1, 1, 1, 0, 0] } })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(response)))
    await expect(runLogisticRegression(request)).resolves.toEqual(response)

    response.dataset.x[0] = upper + Number.EPSILON
    await expect(runLogisticRegression(request)).rejects.toMatchObject({ code: 'INVALID_RESPONSE' })
  })

  it('rejects a zero-update response that changes a coefficient even when predictions are unaffected', async () => {
    const request: LogisticRegressionRequest = {
      ...DEFAULT_LOGISTIC_REQUEST,
      dataset: { kind: 'points', x: [0, 0], y: [0, 1] },
      parameters: { ...DEFAULT_LOGISTIC_REQUEST.parameters, steps: 0 },
    }
    const response = responseForLogisticRequest(request)
    response.model.coefficient += 1
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(response)))
    await expect(runLogisticRegression(request)).rejects.toMatchObject({ code: 'INVALID_RESPONSE' })
  })

  it('rejects final BCE inconsistent with the fitted model even when the history repeats the same wrong value', async () => {
    const response = responseForLogisticRequest(DEFAULT_LOGISTIC_REQUEST)
    response.metrics.binary_cross_entropy += 1
    response.training.loss_history[response.training.steps_completed] = response.metrics.binary_cross_entropy
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(response)))
    await expect(runLogisticRegression(DEFAULT_LOGISTIC_REQUEST)).rejects.toMatchObject({ code: 'INVALID_RESPONSE' })
  })

  it('preserves structured backend errors including the generated single-class case', async () => {
    const payload = {
      error: {
        code: 'SINGLE_CLASS_DATASET',
        message: 'Seeded data produced only class 0; change the seed or generator settings.',
        details: { class_label: 0, n_samples: 10 },
      },
    }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(payload, 422)))

    await expect(runLogisticRegression(DEFAULT_LOGISTIC_REQUEST)).rejects.toMatchObject({
      name: 'LogisticRegressionApiError',
      status: 422,
      code: 'SINGLE_CLASS_DATASET',
      details: { class_label: 0, n_samples: 10 },
    })
  })

  it('turns network failures into a retryable network error and preserves aborts', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('offline')))
    await expect(runLogisticRegression(DEFAULT_LOGISTIC_REQUEST)).rejects.toBeInstanceOf(LogisticRegressionNetworkError)

    const controller = new AbortController()
    controller.abort()
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new DOMException('Aborted', 'AbortError')))
    await expect(runLogisticRegression(DEFAULT_LOGISTIC_REQUEST, controller.signal)).rejects.toBeInstanceOf(DOMException)
  })

  it('uses the API error type for unreadable JSON', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => { throw new Error('invalid json') } }))

    await expect(runLogisticRegression(DEFAULT_LOGISTIC_REQUEST)).rejects.toBeInstanceOf(LogisticRegressionApiError)
  })

  it('preserves cancellation and reports lost connections while reading the response body', async () => {
    const abort = new DOMException('Aborted', 'AbortError')
    const controller = new AbortController()
    const json = vi.fn()
      .mockRejectedValueOnce(new TypeError('Connection terminated'))
      .mockImplementationOnce(() => { controller.abort(); throw abort })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json }))
    await expect(runLogisticRegression(DEFAULT_LOGISTIC_REQUEST)).rejects.toBeInstanceOf(LogisticRegressionNetworkError)
    await expect(runLogisticRegression(DEFAULT_LOGISTIC_REQUEST, controller.signal)).rejects.toBe(abort)
  })
})
