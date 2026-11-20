import { describe, expect, it } from 'vitest'
import { DEFAULT_LOGISTIC_FORM, DEFAULT_LOGISTIC_REQUEST } from './defaults'
import { buildLogisticRegressionRequest } from './request'

describe('Logistic Regression request construction', () => {
  it('builds the reproducible backend-compatible default request', () => {
    expect(buildLogisticRegressionRequest({ ...DEFAULT_LOGISTIC_FORM })).toEqual({ request: DEFAULT_LOGISTIC_REQUEST, errors: {} })
  })

  it('builds inline points and keeps the required seed field without showing it to users', () => {
    const result = buildLogisticRegressionRequest({
      ...DEFAULT_LOGISTIC_FORM,
      datasetKind: 'points',
      seed: '900',
      inlineX: '-1, 0, 2',
      inlineY: '0, 1, 1',
    })

    expect(result.errors).toEqual({})
    expect(result.request).toEqual({
      dataset: { kind: 'points', x: [-1, 0, 2], y: [0, 1, 1] },
      parameters: { initial_coefficient: 0, initial_intercept: 0, learning_rate: 0.1, steps: 25 },
      threshold: 0.5,
      seed: 42,
    })
  })

  it('rejects inline labels other than exactly zero or one', () => {
    const result = buildLogisticRegressionRequest({ ...DEFAULT_LOGISTIC_FORM, datasetKind: 'points', inlineY: '0, 0.5, 1' })

    expect(result.request).toBeUndefined()
    expect(result.errors.inlineY).toBe('Each label must be exactly 0 or 1.')
  })

  it('validates backend sample, range, seed, parameter, and threshold limits', () => {
    const result = buildLogisticRegressionRequest({
      ...DEFAULT_LOGISTIC_FORM,
      nSamples: '1',
      xMin: '4',
      xMax: '4',
      seed: '4294967296',
      learningRate: '0',
      steps: '1.5',
      threshold: '1.1',
    })

    expect(result.request).toBeUndefined()
    expect(result.errors).toMatchObject({
      nSamples: expect.any(String),
      xMin: expect.any(String),
      xMax: expect.any(String),
      seed: expect.any(String),
      learningRate: expect.any(String),
      steps: expect.any(String),
      threshold: expect.any(String),
    })
  })
})
