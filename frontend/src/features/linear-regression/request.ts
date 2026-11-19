import { DEFAULT_REQUEST } from './defaults'
import type { ExperimentFormState } from './defaults'
import type { LinearRegressionRequest } from './types'

export type FormErrors = Partial<Record<keyof ExperimentFormState, string>>

export interface RequestBuildResult {
  request?: LinearRegressionRequest
  errors: FormErrors
}

function readNumber(
  raw: string,
  field: keyof ExperimentFormState,
  errors: FormErrors,
): number | undefined {
  const value = Number(raw)
  if (raw.trim() === '' || !Number.isFinite(value)) {
    errors[field] = 'Enter a finite number.'
    return undefined
  }
  return value
}

function readBoundedNumber(
  raw: string,
  field: keyof ExperimentFormState,
  errors: FormErrors,
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
  field: keyof ExperimentFormState,
  errors: FormErrors,
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

function readInlineValues(
  raw: string,
  field: 'inlineX' | 'inlineY',
  errors: FormErrors,
): number[] | undefined {
  if (!raw.trim()) {
    errors[field] = 'Enter at least one value.'
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
  if (values.some((value) => Math.abs(value) > 1_000_000)) {
    errors[field] = 'Each value must be between -1,000,000 and 1,000,000.'
    return undefined
  }
  return values
}

export function buildLinearRegressionRequest(form: ExperimentFormState): RequestBuildResult {
  const errors: FormErrors = {}
  const initialSlope = readBoundedNumber(form.initialSlope, 'initialSlope', errors, -1_000_000, 1_000_000)
  const initialIntercept = readBoundedNumber(form.initialIntercept, 'initialIntercept', errors, -1_000_000, 1_000_000)
  const learningRate = readBoundedNumber(form.learningRate, 'learningRate', errors, 0, 1, true)
  const steps = readInteger(form.steps, 'steps', errors, 0, 500)
  const seed = form.datasetKind === 'synthetic_linear'
    ? readInteger(form.seed, 'seed', errors, 0, 4_294_967_295)
    : DEFAULT_REQUEST.seed

  let dataset: LinearRegressionRequest['dataset'] | undefined
  if (form.datasetKind === 'synthetic_linear') {
    const nSamples = readInteger(form.nSamples, 'nSamples', errors, 2, 1000)
    const slope = readBoundedNumber(form.trueSlope, 'trueSlope', errors, -1_000_000, 1_000_000)
    const intercept = readBoundedNumber(form.trueIntercept, 'trueIntercept', errors, -1_000_000, 1_000_000)
    const noiseStd = readBoundedNumber(form.noiseStd, 'noiseStd', errors, 0, 1000)
    const xMin = readBoundedNumber(form.xMin, 'xMin', errors, -1_000_000, 1_000_000)
    const xMax = readBoundedNumber(form.xMax, 'xMax', errors, -1_000_000, 1_000_000)

    if (xMin !== undefined && xMax !== undefined && xMin >= xMax) {
      errors.xMin = 'The feature minimum must be less than the maximum.'
      errors.xMax = 'The feature maximum must be greater than the minimum.'
    }
    if (
      nSamples !== undefined && slope !== undefined && intercept !== undefined &&
      noiseStd !== undefined && xMin !== undefined && xMax !== undefined &&
      xMin < xMax
    ) {
      dataset = {
        kind: 'synthetic_linear',
        n_samples: nSamples,
        slope,
        intercept,
        noise_std: noiseStd,
        x_min: xMin,
        x_max: xMax,
      }
    }
  } else {
    const x = readInlineValues(form.inlineX, 'inlineX', errors)
    const y = readInlineValues(form.inlineY, 'inlineY', errors)
    if (x && y) {
      if (x.length !== y.length) {
        errors.inlineY = 'Provide one target value for each x value.'
      } else {
        dataset = { kind: 'points', x, y }
      }
    }
  }

  if (
    dataset && initialSlope !== undefined && initialIntercept !== undefined &&
    learningRate !== undefined && steps !== undefined && seed !== undefined &&
    Object.keys(errors).length === 0
  ) {
    return {
      request: {
        dataset,
        parameters: {
          initial_slope: initialSlope,
          initial_intercept: initialIntercept,
          learning_rate: learningRate,
          steps,
        },
        seed,
      },
      errors,
    }
  }

  return { errors }
}
