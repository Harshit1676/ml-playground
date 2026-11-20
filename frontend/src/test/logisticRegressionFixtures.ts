import { vi } from 'vitest'
import { deriveThresholdMetrics, logisticProbability } from '../features/logistic-regression/classification'
import type { LogisticRegressionRequest, LogisticRegressionResponse } from '../features/logistic-regression/types'

function binaryCrossEntropy(logits: number[], labels: number[]): number {
  return logits.reduce((sum, logit, index) => sum + Math.max(logit, 0) + Math.log1p(Math.exp(-Math.abs(logit))) - labels[index] * logit, 0) / logits.length
}

export function responseForLogisticRequest(request: LogisticRegressionRequest): LogisticRegressionResponse {
  let dataset: { x: number[]; y: (0 | 1)[] }
  if (request.dataset.kind === 'points') {
    dataset = { x: [...request.dataset.x], y: [...request.dataset.y] }
  } else {
    const synthetic = request.dataset
    dataset = {
      x: Array.from({ length: synthetic.n_samples }, (_, index) => synthetic.x_min + (synthetic.x_max - synthetic.x_min) * (index + 0.5) / synthetic.n_samples),
      y: Array.from({ length: synthetic.n_samples }, (_, index) => (index % 2) as 0 | 1),
    }
  }

  let coefficient = request.parameters.initial_coefficient
  let intercept = request.parameters.initial_intercept
  const calculateLogits = () => dataset.x.map((feature) => coefficient * feature + intercept)
  const calculateProbabilities = () => calculateLogits().map(logisticProbability)
  const lossHistory = [binaryCrossEntropy(calculateLogits(), dataset.y)]

  for (let step = 0; step < request.parameters.steps; step += 1) {
    const probabilities = calculateProbabilities()
    const residuals = probabilities.map((probability, index) => probability - dataset.y[index])
    const coefficientGradient = residuals.reduce((sum, residual, index) => sum + residual * dataset.x[index], 0) / residuals.length
    const interceptGradient = residuals.reduce((sum, residual) => sum + residual, 0) / residuals.length
    const nextCoefficient = coefficient - request.parameters.learning_rate * coefficientGradient
    const nextIntercept = intercept - request.parameters.learning_rate * interceptGradient
    coefficient = nextCoefficient
    intercept = nextIntercept
    lossHistory.push(binaryCrossEntropy(calculateLogits(), dataset.y))
  }

  const probabilities = calculateProbabilities()
  const thresholdMetrics = deriveThresholdMetrics(dataset.y, probabilities, request.threshold)
  return {
    algorithm: 'logistic_regression',
    seed: request.seed,
    dataset,
    parameters: { ...request.parameters },
    threshold: request.threshold,
    model: { coefficient, intercept },
    predictions: { probabilities, predicted_classes: thresholdMetrics.predictedClasses },
    metrics: {
      binary_cross_entropy: lossHistory[lossHistory.length - 1],
      confusion_matrix: thresholdMetrics.confusionMatrix,
      accuracy: thresholdMetrics.accuracy,
      precision: thresholdMetrics.precision,
      recall: thresholdMetrics.recall,
      f1: thresholdMetrics.f1,
    },
    training: { steps_completed: request.parameters.steps, loss_history: lossHistory },
  }
}

export function jsonResponse(payload: unknown, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => payload }
}

export function requestFromFetch(fetchMock: ReturnType<typeof vi.fn>, index = 0): LogisticRegressionRequest {
  const init = fetchMock.mock.calls[index][1] as RequestInit
  return JSON.parse(String(init.body)) as LogisticRegressionRequest
}
