import { DEFAULT_LOGISTIC_REQUEST } from './defaults'
import type { LogisticExperimentFormState } from './defaults'
import type { BinaryLabel, LogisticRegressionRequest } from './types'

export type LogisticFormErrors = Partial<Record<keyof LogisticExperimentFormState, string>>

export interface LogisticRequestBuildResult {
  request?: LogisticRegressionRequest
  errors: LogisticFormErrors
}

function readNumber(raw: string, field: keyof LogisticExperimentFormState, errors: LogisticFormErrors): number | undefined {
  const value = Number(raw)
  if (raw.trim() === '' || !Number.isFinite(value)) {
    errors[field] = 'Enter a finite number.'
    return undefined
  }
  return value
}

function readBoundedNumber(
  raw: string,
  field: keyof LogisticExperimentFormState,
  errors: LogisticFormErrors,
  minimum: number,
  maximum: number,
  minimumExclusive = false,
): number | undefined {
  const value = readNumber(raw, field, errors)
  if (value === undefined) return undefined
  if ((minimumExclusive ? value <= minimum : value < minimum) || value > maximum) {
    errors[field] = minimumExclusive
      ? `Enter a value greater than ${minimum} and at most ${maximum}.`
      : `Enter a value from ${minimum} to ${maximum}.`
    return undefined
  }
  return value
}

function readInteger(
  raw: string,
  field: keyof LogisticExperimentFormState,
  errors: LogisticFormErrors,
  minimum: number,
  maximum: number,
): number | undefined {
  const value = readBoundedNumber(raw, field, errors, minimum, maximum)
  if (value === undefined) return undefined
  if (!Number.isInteger(value)) {
    errors[field] = 'Enter a whole number.'
    return undefined
  }
  return value
}

function readList(
  raw: string,
  field: 'inlineX' | 'inlineY',
  errors: LogisticFormErrors,
  labelsOnly: boolean,
): number[] | undefined {
  if (!raw.trim()) {
    errors[field] = labelsOnly ? 'Enter at least one binary label.' : 'Enter at least one feature value.'
    return undefined
  }
  const parts = raw.split(',').map((part) => part.trim())
  const values = parts.map(Number)
  if (parts.some((part) => part === '') || values.some((value) => !Number.isFinite(value))) {
    errors[field] = 'Use comma-separated finite numbers.'
    return undefined
  }
  if (values.length < 1 || values.length > 1000) {
    errors[field] = 'Enter between 1 and 1,000 values.'
    return undefined
  }
  if (labelsOnly && values.some((value) => value !== 0 && value !== 1)) {
    errors[field] = 'Each label must be exactly 0 or 1.'
    return undefined
  }
  if (!labelsOnly && values.some((value) => Math.abs(value) > 1_000_000)) {
    errors[field] = 'Each feature must be between -1,000,000 and 1,000,000.'
    return undefined
  }
  return values
}

export function buildLogisticRegressionRequest(form: LogisticExperimentFormState): LogisticRequestBuildResult {
  const errors: LogisticFormErrors = {}
  const initialCoefficient = readBoundedNumber(form.initialCoefficient, 'initialCoefficient', errors, -1_000_000, 1_000_000)
  const initialIntercept = readBoundedNumber(form.initialIntercept, 'initialIntercept', errors, -1_000_000, 1_000_000)
  const learningRate = readBoundedNumber(form.learningRate, 'learningRate', errors, 0, 1, true)
  const steps = readInteger(form.steps, 'steps', errors, 0, 500)
  const threshold = readBoundedNumber(form.threshold, 'threshold', errors, 0, 1)
  const seed = form.datasetKind === 'synthetic_logistic'
    ? readInteger(form.seed, 'seed', errors, 0, 4_294_967_295)
    : DEFAULT_LOGISTIC_REQUEST.seed

  let dataset: LogisticRegressionRequest['dataset'] | undefined
  if (form.datasetKind === 'synthetic_logistic') {
    const nSamples = readInteger(form.nSamples, 'nSamples', errors, 2, 1000)
    const trueCoefficient = readBoundedNumber(form.trueCoefficient, 'trueCoefficient', errors, -1_000_000, 1_000_000)
    const trueIntercept = readBoundedNumber(form.trueIntercept, 'trueIntercept', errors, -1_000_000, 1_000_000)
    const xMin = readBoundedNumber(form.xMin, 'xMin', errors, -1_000_000, 1_000_000)
    const xMax = readBoundedNumber(form.xMax, 'xMax', errors, -1_000_000, 1_000_000)

    if (xMin !== undefined && xMax !== undefined && xMin >= xMax) {
      errors.xMin = 'Feature minimum must be less than feature maximum.'
      errors.xMax = 'Feature maximum must be greater than feature minimum.'
    }
    if (
      nSamples !== undefined && trueCoefficient !== undefined && trueIntercept !== undefined &&
      xMin !== undefined && xMax !== undefined && xMin < xMax
    ) {
      dataset = { kind: 'synthetic_logistic', n_samples: nSamples, true_coefficient: trueCoefficient, true_intercept: trueIntercept, x_min: xMin, x_max: xMax }
    }
  } else {
    const x = readList(form.inlineX, 'inlineX', errors, false)
    const rawLabels = readList(form.inlineY, 'inlineY', errors, true)
    if (x && rawLabels) {
      if (x.length !== rawLabels.length) {
        errors.inlineY = 'Provide one binary label for each feature value.'
      } else {
        dataset = { kind: 'points', x, y: rawLabels as BinaryLabel[] }
      }
    }
  }

  if (
    dataset && initialCoefficient !== undefined && initialIntercept !== undefined &&
    learningRate !== undefined && steps !== undefined && threshold !== undefined && seed !== undefined &&
    Object.keys(errors).length === 0
  ) {
    return {
      request: {
        dataset,
        parameters: { initial_coefficient: initialCoefficient, initial_intercept: initialIntercept, learning_rate: learningRate, steps },
        threshold,
        seed,
      },
      errors,
    }
  }

  return { errors }
}
