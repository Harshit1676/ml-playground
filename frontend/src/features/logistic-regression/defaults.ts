import type { LogisticRegressionRequest } from './types'

export interface LogisticExperimentFormState {
  datasetKind: 'synthetic_logistic' | 'points'
  nSamples: string
  trueCoefficient: string
  trueIntercept: string
  xMin: string
  xMax: string
  seed: string
  inlineX: string
  inlineY: string
  initialCoefficient: string
  initialIntercept: string
  learningRate: string
  steps: string
  threshold: string
}

export const DEFAULT_LOGISTIC_FORM: LogisticExperimentFormState = {
  datasetKind: 'synthetic_logistic',
  nSamples: '100',
  trueCoefficient: '1.5',
  trueIntercept: '0',
  xMin: '-4',
  xMax: '4',
  seed: '42',
  inlineX: '-3, -2, -1, 1, 2, 3',
  inlineY: '0, 0, 0, 1, 1, 1',
  initialCoefficient: '0',
  initialIntercept: '0',
  learningRate: '0.1',
  steps: '25',
  threshold: '0.5',
}

export const DEFAULT_LOGISTIC_REQUEST: LogisticRegressionRequest = {
  dataset: {
    kind: 'synthetic_logistic',
    n_samples: 100,
    true_coefficient: 1.5,
    true_intercept: 0,
    x_min: -4,
    x_max: 4,
  },
  parameters: {
    initial_coefficient: 0,
    initial_intercept: 0,
    learning_rate: 0.1,
    steps: 25,
  },
  threshold: 0.5,
  seed: 42,
}
