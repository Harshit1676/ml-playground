import { getApiBaseUrl } from './baseUrl'
import { deriveThresholdMetrics, logisticProbability } from '../../features/logistic-regression/classification'
import type { BinaryLabel, LogisticRegressionRequest, LogisticRegressionResponse } from '../../features/logistic-regression/types'

export interface LogisticStructuredApiError {
  code: string
  message: string
  details?: Record<string, unknown> | null
}

export class LogisticRegressionApiError extends Error {
  readonly name = 'LogisticRegressionApiError'

  constructor(
    message: string,
    readonly status: number,
    readonly code: string,
    readonly details?: Record<string, unknown> | null,
  ) {
    super(message)
  }
}

export class LogisticRegressionNetworkError extends Error {
  readonly name = 'LogisticRegressionNetworkError'

  constructor() {
    super('Could not reach the experiment service. Check that the backend is running, then retry.')
  }
}

const RESPONSE_TOLERANCE = 1e-9

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function finiteNumberArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every(isFiniteNumber)
}

function binaryLabelArray(value: unknown): value is BinaryLabel[] {
  return Array.isArray(value) && value.every((label) => label === 0 || label === 1)
}

function closeEnough(actual: number, expected: number): boolean {
  return Math.abs(actual - expected) <= RESPONSE_TOLERANCE * Math.max(1, Math.abs(actual), Math.abs(expected))
}

function binaryCrossEntropyAtParameters(
  x: number[],
  y: BinaryLabel[],
  coefficient: number,
  intercept: number,
): number {
  let total = 0
  for (let index = 0; index < x.length; index += 1) {
    const logit = coefficient * x[index] + intercept
    if (!Number.isFinite(logit)) return Number.NaN
    total += Math.max(logit, 0) + Math.log1p(Math.exp(-Math.abs(logit))) - y[index] * logit
    if (!Number.isFinite(total)) return Number.NaN
  }
  return total / x.length
}

function sameParameters(value: unknown, request: LogisticRegressionRequest): boolean {
  if (!isRecord(value)) return false
  return value.initial_coefficient === request.parameters.initial_coefficient &&
    value.initial_intercept === request.parameters.initial_intercept &&
    value.learning_rate === request.parameters.learning_rate &&
    value.steps === request.parameters.steps
}

function isLogisticRegressionResponse(value: unknown, request: LogisticRegressionRequest): value is LogisticRegressionResponse {
  if (!isRecord(value) || value.algorithm !== 'logistic_regression' || value.seed !== request.seed || value.threshold !== request.threshold) return false
  if (!isRecord(value.dataset) || !isRecord(value.parameters) || !isRecord(value.model)) return false
  if (!isRecord(value.predictions) || !isRecord(value.metrics) || !isRecord(value.training)) return false

  const x = value.dataset.x
  const y = value.dataset.y
  const probabilities = value.predictions.probabilities
  const predictedClasses = value.predictions.predicted_classes
  const losses = value.training.loss_history
  const stepsCompleted = value.training.steps_completed
  const coefficient = value.model.coefficient
  const intercept = value.model.intercept
  const threshold = value.threshold
  const metrics = value.metrics
  const matrix = metrics.confusion_matrix

  if (
    !finiteNumberArray(x) || !binaryLabelArray(y) || !finiteNumberArray(probabilities) || !binaryLabelArray(predictedClasses) ||
    !finiteNumberArray(losses) || x.length === 0 || x.length !== y.length || probabilities.length !== x.length || predictedClasses.length !== x.length ||
    !Number.isInteger(stepsCompleted) || stepsCompleted !== request.parameters.steps || losses.length !== stepsCompleted + 1 ||
    !isFiniteNumber(coefficient) || !isFiniteNumber(intercept) || !isFiniteNumber(threshold) || threshold < 0 || threshold > 1 ||
    !sameParameters(value.parameters, request) || !isRecord(matrix)
  ) return false

  if (!losses.every((loss) => loss >= 0) || !isFiniteNumber(metrics.binary_cross_entropy) || metrics.binary_cross_entropy < 0) return false
  if (!closeEnough(metrics.binary_cross_entropy, losses[losses.length - 1])) return false
  if (![metrics.accuracy, metrics.precision, metrics.recall, metrics.f1].every((metric) => isFiniteNumber(metric) && metric >= 0 && metric <= 1)) return false

  if (request.dataset.kind === 'points') {
    const inlineDataset = request.dataset
    if (x.length !== inlineDataset.x.length || y.length !== inlineDataset.y.length ||
      !x.every((feature, index) => feature === inlineDataset.x[index]) ||
      !y.every((label, index) => label === inlineDataset.y[index])) return false
  } else {
    const syntheticDataset = request.dataset
    if (x.length !== syntheticDataset.n_samples || new Set(y).size !== 2) return false
    // A float64 uniform draw can round to the upper endpoint.
    if (!x.every((feature) => feature >= syntheticDataset.x_min && feature <= syntheticDataset.x_max)) return false
  }
  if (!x.every((feature) => Math.abs(feature) <= 1_000_000)) return false

  const initialLoss = binaryCrossEntropyAtParameters(
    x,
    y,
    request.parameters.initial_coefficient,
    request.parameters.initial_intercept,
  )
  if (!Number.isFinite(initialLoss) || !closeEnough(losses[0], initialLoss)) return false

  if (stepsCompleted === 0 && (coefficient !== request.parameters.initial_coefficient ||
    intercept !== request.parameters.initial_intercept)) return false
  const finalLoss = binaryCrossEntropyAtParameters(x, y, coefficient, intercept)
  if (!Number.isFinite(finalLoss) || !closeEnough(finalLoss, metrics.binary_cross_entropy)) return false

  const expectedProbabilities = x.map((feature) => logisticProbability(coefficient * feature + intercept))
  if (!expectedProbabilities.every(Number.isFinite) || !probabilities.every((probability) => probability >= 0 && probability <= 1)) return false
  if (!probabilities.every((probability, index) => closeEnough(probability, expectedProbabilities[index]))) return false

  const expectedMetrics = deriveThresholdMetrics(y, probabilities, threshold)
  if (!predictedClasses.every((label, index) => label === expectedMetrics.predictedClasses[index])) return false
  const expectedMatrix = expectedMetrics.confusionMatrix
  const matrixKeys = ['true_positive', 'true_negative', 'false_positive', 'false_negative'] as const
  if (!matrixKeys.every((key) => Number.isInteger(matrix[key]) && matrix[key] === expectedMatrix[key])) return false
  if (matrixKeys.reduce((sum, key) => sum + Number(matrix[key]), 0) !== x.length) return false
  return closeEnough(metrics.accuracy as number, expectedMetrics.accuracy) &&
    closeEnough(metrics.precision as number, expectedMetrics.precision) &&
    closeEnough(metrics.recall as number, expectedMetrics.recall) &&
    closeEnough(metrics.f1 as number, expectedMetrics.f1)
}

function parseStructuredError(value: unknown): LogisticStructuredApiError | undefined {
  if (!isRecord(value) || !isRecord(value.error)) return undefined
  const error = value.error
  if (typeof error.code !== 'string' || typeof error.message !== 'string') return undefined
  return { code: error.code, message: error.message, details: isRecord(error.details) ? error.details : null }
}

export async function runLogisticRegression(
  request: LogisticRegressionRequest,
  signal?: AbortSignal,
): Promise<LogisticRegressionResponse> {
  const baseUrl = getApiBaseUrl()
  let response: Response

  try {
    response = await fetch(`${baseUrl}/api/v1/experiments/logistic-regression`, {
      method: 'POST',
      headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    })
  } catch (error) {
    if (signal?.aborted) throw error
    throw new LogisticRegressionNetworkError()
  }

  let payload: unknown
  try {
    payload = await response.json()
  } catch (error) {
    if (signal?.aborted) throw error
    if (error instanceof TypeError) throw new LogisticRegressionNetworkError()
    throw new LogisticRegressionApiError(`The experiment service returned an unreadable response (HTTP ${response.status}).`, response.status, 'INVALID_RESPONSE')
  }

  if (!response.ok) {
    const structured = parseStructuredError(payload)
    throw new LogisticRegressionApiError(
      structured?.message ?? `Experiment request failed with HTTP ${response.status}.`,
      response.status,
      structured?.code ?? 'HTTP_ERROR',
      structured?.details,
    )
  }

  if (!isLogisticRegressionResponse(payload, request)) {
    throw new LogisticRegressionApiError(
      'The experiment service returned data that does not match the Logistic Regression response contract.',
      response.status,
      'INVALID_RESPONSE',
    )
  }
  return payload
}
