import type { BinaryLabel, LogisticConfusionMatrix } from './types'

export interface ThresholdMetrics {
  predictedClasses: BinaryLabel[]
  confusionMatrix: LogisticConfusionMatrix
  accuracy: number
  precision: number
  recall: number
  f1: number
}

export function logisticProbability(logit: number): number {
  if (!Number.isFinite(logit)) return Number.NaN
  if (logit >= 0) return 1 / (1 + Math.exp(-logit))
  const exponential = Math.exp(logit)
  return exponential / (1 + exponential)
}

export function deriveThresholdMetrics(
  labels: BinaryLabel[],
  probabilities: number[],
  threshold: number,
): ThresholdMetrics {
  if (labels.length === 0 || labels.length !== probabilities.length) {
    throw new Error('Labels and probabilities must have the same non-zero length.')
  }
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new Error('Threshold must be between 0 and 1.')
  }

  const predictedClasses = probabilities.map((probability) => (probability >= threshold ? 1 : 0) as BinaryLabel)
  const confusionMatrix: LogisticConfusionMatrix = {
    true_positive: 0,
    true_negative: 0,
    false_positive: 0,
    false_negative: 0,
  }

  labels.forEach((actual, index) => {
    const predicted = predictedClasses[index]
    if (actual === 1 && predicted === 1) confusionMatrix.true_positive += 1
    else if (actual === 0 && predicted === 0) confusionMatrix.true_negative += 1
    else if (actual === 0 && predicted === 1) confusionMatrix.false_positive += 1
    else confusionMatrix.false_negative += 1
  })

  const safeRatio = (numerator: number, denominator: number) => denominator === 0 ? 0 : numerator / denominator
  const accuracy = safeRatio(confusionMatrix.true_positive + confusionMatrix.true_negative, labels.length)
  const precision = safeRatio(confusionMatrix.true_positive, confusionMatrix.true_positive + confusionMatrix.false_positive)
  const recall = safeRatio(confusionMatrix.true_positive, confusionMatrix.true_positive + confusionMatrix.false_negative)
  const f1 = safeRatio(2 * precision * recall, precision + recall)

  return { predictedClasses, confusionMatrix, accuracy, precision, recall, f1 }
}
