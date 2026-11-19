import type { LinearRegressionRequest, LinearRegressionResponse } from '../../features/linear-regression/types'

export interface StructuredApiError {
  code: string
  message: string
  details?: Record<string, unknown> | null
}

export class LinearRegressionApiError extends Error {
  readonly name = 'LinearRegressionApiError'

  constructor(
    message: string,
    readonly status: number,
    readonly code: string,
    readonly details?: Record<string, unknown> | null,
  ) {
    super(message)
  }
}

export class LinearRegressionNetworkError extends Error {
  readonly name = 'LinearRegressionNetworkError'

  constructor() {
    super('Could not reach the experiment service. Check that the backend is running, then retry.')
  }
}

const defaultApiBaseUrl = 'http://localhost:8000'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function finiteNumberArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every(isFiniteNumber)
}

function predictionsMatchModel(
  x: number[],
  predictions: number[],
  slope: number,
  intercept: number,
): boolean {
  return x.every((feature, index) => {
    const expected = slope * feature + intercept
    const actual = predictions[index]
    if (!Number.isFinite(expected)) return false
    const tolerance = 1e-9 * Math.max(1, Math.abs(expected), Math.abs(actual))
    return Math.abs(actual - expected) <= tolerance
  })
}

function isLinearRegressionResponse(value: unknown, request: LinearRegressionRequest): value is LinearRegressionResponse {
  if (!isRecord(value) || value.algorithm !== 'linear_regression' || value.seed !== request.seed) return false
  if (!isRecord(value.dataset) || !isRecord(value.parameters) || !isRecord(value.model)) return false
  if (!isRecord(value.predictions) || !isRecord(value.metrics) || !isRecord(value.training)) return false

  const x = value.dataset.x
  const y = value.dataset.y
  const predictions = value.predictions.y_hat
  const losses = value.training.loss_history
  const stepsCompleted = value.training.steps_completed
  const slope = value.model.slope
  const intercept = value.model.intercept
  const mse = value.metrics.mse
  const parameters = value.parameters

  if (
    !finiteNumberArray(x) || !finiteNumberArray(y) || !finiteNumberArray(predictions) ||
    !finiteNumberArray(losses) || x.length === 0 || x.length !== y.length ||
    x.length !== predictions.length || !Number.isInteger(stepsCompleted) ||
    stepsCompleted !== request.parameters.steps || losses.length !== stepsCompleted + 1 ||
    !isFiniteNumber(slope) || !isFiniteNumber(intercept) || !isFiniteNumber(mse) || mse < 0 ||
    !isRecord(parameters) || parameters.initial_slope !== request.parameters.initial_slope ||
    parameters.initial_intercept !== request.parameters.initial_intercept ||
    parameters.learning_rate !== request.parameters.learning_rate ||
    parameters.steps !== request.parameters.steps
  ) {
    return false
  }

  return losses.every((loss) => loss >= 0) && predictionsMatchModel(x, predictions, slope, intercept)
}

function parseStructuredError(value: unknown): StructuredApiError | undefined {
  if (!isRecord(value) || !isRecord(value.error)) return undefined
  const error = value.error
  if (typeof error.code !== 'string' || typeof error.message !== 'string') return undefined
  return {
    code: error.code,
    message: error.message,
    details: isRecord(error.details) ? error.details : null,
  }
}

export async function runLinearRegression(
  request: LinearRegressionRequest,
  signal?: AbortSignal,
): Promise<LinearRegressionResponse> {
  const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()
  const baseUrl = (configuredBaseUrl || defaultApiBaseUrl).replace(/\/+$/, '')
  let response: Response

  try {
    response = await fetch(`${baseUrl}/api/v1/experiments/linear-regression`, {
      method: 'POST',
      headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    })
  } catch (error) {
    if (signal?.aborted) throw error
    throw new LinearRegressionNetworkError()
  }

  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    throw new LinearRegressionApiError(
      `The experiment service returned an unreadable response (HTTP ${response.status}).`,
      response.status,
      'INVALID_RESPONSE',
    )
  }

  if (!response.ok) {
    const structuredError = parseStructuredError(payload)
    throw new LinearRegressionApiError(
      structuredError?.message ?? `Experiment request failed with HTTP ${response.status}.`,
      response.status,
      structuredError?.code ?? 'HTTP_ERROR',
      structuredError?.details,
    )
  }

  if (!isLinearRegressionResponse(payload, request)) {
    throw new LinearRegressionApiError(
      'The experiment service returned data that does not match the Linear Regression response contract.',
      response.status,
      'INVALID_RESPONSE',
    )
  }

  return payload
}
