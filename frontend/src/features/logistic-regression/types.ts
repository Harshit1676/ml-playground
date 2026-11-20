export type BinaryLabel = 0 | 1

export interface LogisticPointsDatasetRequest {
  kind: 'points'
  x: number[]
  y: BinaryLabel[]
}

export interface SyntheticLogisticDatasetRequest {
  kind: 'synthetic_logistic'
  n_samples: number
  true_coefficient: number
  true_intercept: number
  x_min: number
  x_max: number
}

export interface LogisticRegressionParameters {
  initial_coefficient: number
  initial_intercept: number
  learning_rate: number
  steps: number
}

export interface LogisticRegressionRequest {
  dataset: LogisticPointsDatasetRequest | SyntheticLogisticDatasetRequest
  parameters: LogisticRegressionParameters
  threshold: number
  seed: number
}

export interface LogisticConfusionMatrix {
  true_positive: number
  true_negative: number
  false_positive: number
  false_negative: number
}

export interface LogisticRegressionResponse {
  algorithm: 'logistic_regression'
  seed: number
  dataset: { x: number[]; y: BinaryLabel[] }
  parameters: LogisticRegressionParameters
  threshold: number
  model: { coefficient: number; intercept: number }
  predictions: { probabilities: number[]; predicted_classes: BinaryLabel[] }
  metrics: {
    binary_cross_entropy: number
    confusion_matrix: LogisticConfusionMatrix
    accuracy: number
    precision: number
    recall: number
    f1: number
  }
  training: { steps_completed: number; loss_history: number[] }
}
