export interface PointsDatasetRequest {
  kind: 'points'
  x: number[]
  y: number[]
}

export interface SyntheticLinearDatasetRequest {
  kind: 'synthetic_linear'
  n_samples: number
  slope: number
  intercept: number
  noise_std: number
  x_min: number
  x_max: number
}

export interface LinearRegressionParameters {
  initial_slope: number
  initial_intercept: number
  learning_rate: number
  steps: number
}

export interface LinearRegressionRequest {
  dataset: PointsDatasetRequest | SyntheticLinearDatasetRequest
  parameters: LinearRegressionParameters
  seed: number
}

export interface LinearRegressionResponse {
  algorithm: 'linear_regression'
  seed: number
  dataset: { x: number[]; y: number[] }
  parameters: LinearRegressionParameters
  model: { slope: number; intercept: number }
  predictions: { y_hat: number[] }
  metrics: { mse: number }
  training: { steps_completed: number; loss_history: number[] }
}
