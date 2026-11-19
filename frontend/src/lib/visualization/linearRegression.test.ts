import { describe, expect, it } from 'vitest'
import type { LinearRegressionResponse } from '../../features/linear-regression/types'
import { buildLossLayout, buildLossTraces, buildRegressionLayout, buildRegressionTraces } from './linearRegression'

const result: LinearRegressionResponse = {
  algorithm: 'linear_regression',
  seed: 9,
  dataset: { x: [-2, 0, 3], y: [-4, 1, 7] },
  parameters: { initial_slope: 0, initial_intercept: 0, learning_rate: 0.1, steps: 2 },
  model: { slope: 1.5, intercept: 0.25 },
  predictions: { y_hat: [-2.75, 0.25, 4.75] },
  metrics: { mse: 1.4 },
  training: { steps_completed: 2, loss_history: [5.2, 2.3, 1.4] },
}

describe('Linear Regression chart transforms', () => {
  it('plots response observations and creates the fitted line from response model parameters', () => {
    const [observations, fittedLine] = buildRegressionTraces(result)

    expect(observations).toMatchObject({ x: [-2, 0, 3], y: [-4, 1, 7] })
    expect(fittedLine).toMatchObject({ x: [-2, 3], y: [-2.75, 4.75] })
    expect(observations).toMatchObject({ type: 'scatter', mode: 'markers' })
    expect(fittedLine).toMatchObject({ type: 'scatter', mode: 'lines' })
    expect(buildRegressionLayout()).toMatchObject({
      autosize: true,
      xaxis: { title: { text: 'Feature x' } },
      yaxis: { title: { text: 'Target y' } },
    })
  })

  it('plots each returned loss at its actual training-step index', () => {
    const [loss] = buildLossTraces(result)

    expect(loss).toMatchObject({ x: [0, 1, 2], y: [5.2, 2.3, 1.4] })
    expect(loss).toMatchObject({ type: 'scatter', mode: 'lines+markers' })
    expect(buildLossLayout()).toMatchObject({
      autosize: true,
      xaxis: { title: { text: 'Training step' } },
      yaxis: { title: { text: 'Mean squared error' } },
    })
  })
})
