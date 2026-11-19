import type { LinearRegressionRequest } from './types'

export interface ExperimentFormState {
  datasetKind: 'synthetic_linear' | 'points'
  nSamples: string
  trueSlope: string
  trueIntercept: string
  noiseStd: string
  xMin: string
  xMax: string
  seed: string
  inlineX: string
  inlineY: string
  initialSlope: string
  initialIntercept: string
  learningRate: string
  steps: string
}

export const DEFAULT_FORM: ExperimentFormState = {
  datasetKind: 'synthetic_linear',
  nSamples: '60',
  trueSlope: '1.8',
  trueIntercept: '0.7',
  noiseStd: '1.2',
  xMin: '-5',
  xMax: '5',
  seed: '42',
  inlineX: '-4, -2, 0, 2, 4',
  inlineY: '-6, -2, 1, 4, 8',
  initialSlope: '0',
  initialIntercept: '0',
  learningRate: '0.01',
  steps: '100',
}

export const DEFAULT_REQUEST: LinearRegressionRequest = {
  dataset: {
    kind: 'synthetic_linear',
    n_samples: 60,
    slope: 1.8,
    intercept: 0.7,
    noise_std: 1.2,
    x_min: -5,
    x_max: 5,
  },
  parameters: {
    initial_slope: 0,
    initial_intercept: 0,
    learning_rate: 0.01,
    steps: 100,
  },
  seed: 42,
}
