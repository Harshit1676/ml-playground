import { describe, expect, it } from 'vitest'
import { DEFAULT_LOGISTIC_REQUEST } from '../../features/logistic-regression/defaults'
import type { LogisticRegressionRequest } from '../../features/logistic-regression/types'
import { logisticProbability } from '../../features/logistic-regression/classification'
import {
  buildLogisticLossTraces,
  buildLogisticRegressionLayout,
  buildLogisticRegressionTraces,
} from './logisticRegression'
import { responseForLogisticRequest } from '../../test/logisticRegressionFixtures'

interface XYTrace {
  x?: number[]
  y?: number[]
}

function inlineRequest(x: number[], y: (0 | 1)[], coefficient: number, steps = 0): LogisticRegressionRequest {
  return {
    ...DEFAULT_LOGISTIC_REQUEST,
    dataset: { kind: 'points', x, y },
    parameters: { ...DEFAULT_LOGISTIC_REQUEST.parameters, initial_coefficient: coefficient, steps },
    threshold: 0.5,
  }
}

describe('Logistic Regression Plotly transforms', () => {
  it('uses observed labels and a sorted smooth curve calculated from returned parameters', () => {
    const response = responseForLogisticRequest(inlineRequest([3, -2, 0], [1, 0, 1], -1))
    const traces = buildLogisticRegressionTraces(response, 0.7, [0, 1, 1])
    const probabilityTrace = traces[2] as XYTrace
    const observedZeroTrace = traces[0]

    expect(observedZeroTrace).toMatchObject({ x: [-2], y: [0], name: 'Observed class 0' })
    expect(probabilityTrace).toMatchObject({ type: 'scatter', mode: 'lines', name: 'Predicted probability' })
    expect(probabilityTrace.x).toHaveLength(101)
    expect(probabilityTrace.x?.[0]).toBeLessThan(probabilityTrace.x?.[100] as number)
    expect(probabilityTrace.y?.[0]).toBeCloseTo(logisticProbability(response.model.coefficient * Number(probabilityTrace.x?.[0]) + response.model.intercept))
    expect(traces[3]).toMatchObject({ x: [expect.any(Number), expect.any(Number)], y: [0.7, 0.7], name: 'Threshold 0.7' })
  })

  it.each([1, -1])('resolves the sigmoid transition even across a wide domain (coefficient %i)', (coefficient) => {
    const response = responseForLogisticRequest(inlineRequest([-1e6, 1e6], [0, 1], coefficient))
    const [,, curve] = buildLogisticRegressionTraces(response, 0.5, [0, 1]) as Required<XYTrace>[]
    expect(curve.x[0]).toBe(-1e6)
    expect(curve.x.at(-1)).toBe(1e6)
    expect(curve.x).toEqual([...new Set(curve.x)].sort((a, b) => a - b))
    // Plotly joins adjacent points with straight lines. Read that rendered value
    // at x=2, between the decision boundary and the sigmoid's near-flat tail.
    const right = curve.x.findIndex((x) => x >= 2)
    const fraction = (2 - curve.x[right - 1]) / (curve.x[right] - curve.x[right - 1])
    const displayed = curve.y[right - 1] + fraction * (curve.y[right] - curve.y[right - 1])
    expect(displayed).toBeCloseTo(1 / (1 + Math.exp(-2 * coefficient)), 2)
  })

  it('uses distinct, ordered coordinates for adjacent representable feature values', () => {
    const lower = 1
    const upper = 1 + Number.EPSILON
    const response = responseForLogisticRequest(inlineRequest([upper, lower], [1, 0], -1))
    const [,, curve] = buildLogisticRegressionTraces(response, 0.5, [1, 0]) as Required<XYTrace>[]
    expect(curve.x).toEqual([lower, upper])
    expect(curve.y.every(Number.isFinite)).toBe(true)
  })

  it('handles zero coefficients and constant, narrow feature ranges', () => {
    const response = responseForLogisticRequest(inlineRequest([2, 2], [0, 1], 0))
    const curve = buildLogisticRegressionTraces(response, 0.5, [0, 1])[2] as XYTrace

    expect(curve.x).toHaveLength(101)
    expect(curve.x?.[0]).toBeLessThan(2)
    expect(curve.x?.[100]).toBeGreaterThan(2)
    expect(new Set(curve.y).size).toBe(1)
    expect(curve.y?.[0]).toBeCloseTo(0.5)
  })

  it('anchors the bottom edge of the wrapping legend above the observations', () => {
    const legend = buildLogisticRegressionLayout().legend
    expect(legend?.yanchor).toBe('bottom')
    expect(legend?.y).toBeGreaterThan(1)
  })

  it('plots only the returned loss history, including the step-zero-only case', () => {
    const response = responseForLogisticRequest(inlineRequest([-1, 1], [0, 1], 0.25, 0))

    expect(buildLogisticLossTraces(response)).toEqual([expect.objectContaining({ x: [0], y: [response.training.loss_history[0]] })])
  })
})
