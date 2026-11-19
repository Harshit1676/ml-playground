import { describe, expect, it } from 'vitest'
import { DEFAULT_FORM, DEFAULT_REQUEST } from './defaults'
import { buildLinearRegressionRequest } from './request'

describe('buildLinearRegressionRequest', () => {
  it('builds the seeded default experiment using the documented API fields', () => {
    expect(buildLinearRegressionRequest(DEFAULT_FORM)).toEqual({ request: DEFAULT_REQUEST, errors: {} })
  })

  it('builds inline point data from paired comma-separated values', () => {
    const form = {
      ...DEFAULT_FORM,
      datasetKind: 'points' as const,
      inlineX: '-1, 0, 1',
      inlineY: '-1, 1, 3',
      steps: '0',
    }

    expect(buildLinearRegressionRequest(form).request).toEqual({
      dataset: { kind: 'points', x: [-1, 0, 1], y: [-1, 1, 3] },
      parameters: { initial_slope: 0, initial_intercept: 0, learning_rate: 0.01, steps: 0 },
      seed: 42,
    })
  })

  it('keeps the required seed field for inline data without validating a hidden seed control', () => {
    const form = {
      ...DEFAULT_FORM,
      datasetKind: 'points' as const,
      inlineX: '1, 2',
      inlineY: '3, 5',
      seed: 'not-visible',
    }

    expect(buildLinearRegressionRequest(form)).toEqual({
      request: {
        dataset: { kind: 'points', x: [1, 2], y: [3, 5] },
        parameters: { initial_slope: 0, initial_intercept: 0, learning_rate: 0.01, steps: 100 },
        seed: 42,
      },
      errors: {},
    })
  })

  it.each([
    ['learningRate', '0'],
    ['learningRate', '1.01'],
    ['steps', '501'],
    ['seed', '4294967296'],
    ['nSamples', '1'],
    ['noiseStd', '-1'],
  ] as const)('rejects invalid %s value %s before making a request', (field, value) => {
    const result = buildLinearRegressionRequest({ ...DEFAULT_FORM, [field]: value })

    expect(result.request).toBeUndefined()
    expect(result.errors[field]).toBeTruthy()
  })

  it('rejects inline arrays with mismatched lengths and non-finite values', () => {
    const result = buildLinearRegressionRequest({
      ...DEFAULT_FORM,
      datasetKind: 'points',
      inlineX: '0, 1',
      inlineY: '2',
    })

    expect(result.request).toBeUndefined()
    expect(result.errors.inlineY).toMatch(/one target/i)

    const nonFinite = buildLinearRegressionRequest({ ...DEFAULT_FORM, datasetKind: 'points', inlineX: '0, Infinity', inlineY: '1, 2' })
    expect(nonFinite.request).toBeUndefined()
    expect(nonFinite.errors.inlineX).toMatch(/finite/i)
  })

  it('rejects synthetic feature ranges where the minimum is not below the maximum', () => {
    const result = buildLinearRegressionRequest({ ...DEFAULT_FORM, xMin: '5', xMax: '5' })

    expect(result.request).toBeUndefined()
    expect(result.errors.xMin).toMatch(/less than/i)
  })
})
