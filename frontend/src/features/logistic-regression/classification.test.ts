import { describe, expect, it } from 'vitest'
import { deriveThresholdMetrics, logisticProbability } from './classification'

describe('local threshold metrics', () => {
  it('uses probability greater than or equal to threshold at the exact boundary', () => {
    expect(deriveThresholdMetrics([0, 1], [0.5, 0.5], 0.5).predictedClasses).toEqual([1, 1])
    expect(deriveThresholdMetrics([0, 1], [0.5, 0.5], 0.500001).predictedClasses).toEqual([0, 0])
    expect(deriveThresholdMetrics([0, 1, 1], [0, 0.999999, 1], 0).predictedClasses).toEqual([1, 1, 1])
    expect(deriveThresholdMetrics([0, 1, 1], [0, 0.999999, 1], 1).predictedClasses).toEqual([0, 0, 1])
  })

  it('computes correctly oriented confusion counts and all classification metrics', () => {
    const result = deriveThresholdMetrics([0, 1, 0, 1], [0.1, 0.2, 0.8, 0.9], 0.5)

    expect(result.predictedClasses).toEqual([0, 0, 1, 1])
    expect(result.confusionMatrix).toEqual({ true_positive: 1, true_negative: 1, false_positive: 1, false_negative: 1 })
    expect(result).toMatchObject({ accuracy: 0.5, precision: 0.5, recall: 0.5, f1: 0.5 })
  })

  it('returns zero for all zero-denominator metrics on a single-class sample', () => {
    const result = deriveThresholdMetrics([0, 0], [0.1, 0.2], 0.5)

    expect(result.confusionMatrix).toEqual({ true_positive: 0, true_negative: 2, false_positive: 0, false_negative: 0 })
    expect(result).toMatchObject({ accuracy: 1, precision: 0, recall: 0, f1: 0 })
  })

  it('computes the sigmoid stably at large positive and negative values', () => {
    expect(logisticProbability(1000)).toBe(1)
    expect(logisticProbability(-1000)).toBe(0)
    expect(logisticProbability(0)).toBe(0.5)
  })
})
