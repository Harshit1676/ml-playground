import type { KMeansRequest, Point, Recipe } from './types'

export const defaults = {
  kind: 'synthetic_clusters' as 'synthetic_clusters' | 'points',
  samples: '150', centers: '-5,-4\n0,5\n6,-2', spread: '1.5', generationSeed: '42',
  points: '0,0\n2,4\n8,8\n10,12', k: '3', iterations: '30', seed: '7',
}
export type Form = typeof defaults
export type Errors = Partial<Record<keyof Form, string>>

function readPairs(raw: string, limit: number): Point[] {
  const lines = raw.trim().split(/\r?\n/)
  if (lines.length > limit) throw new Error(`Enter at most ${limit} pairs.`)
  return lines.map((line, i) => {
    const parts = line.split(',').map(v => v.trim())
    const [x, y] = parts.map(Number)
    if (parts.length !== 2 || parts.includes('') || ![x, y].every(n => Number.isFinite(n) && Math.abs(n) <= 1e6)) {
      throw new Error(`Line ${i + 1} needs two finite coordinates from −1,000,000 to 1,000,000, separated by a comma.`)
    }
    return [x, y]
  })
}

export function buildRequest(form: Form): { request?: KMeansRequest; errors: Errors } {
  const errors: Errors = {}
  function number(field: keyof Form, min: number, max: number, integer = true) {
    const n = Number(form[field])
    if (!form[field].trim() || !Number.isFinite(n) || n < min || n > max || (integer && !Number.isInteger(n))) {
      errors[field] = `Enter a ${integer ? 'whole number' : 'number'} from ${min} to ${max}.`
    }
    return n
  }
  const k = number('k', 1, 16)
  const max_iterations = number('iterations', 1, 100)
  const seed = number('seed', 0, 4294967295)
  const field = form.kind === 'points' ? 'points' : 'centers'
  let pairs: Point[] = []
  try { pairs = readPairs(form[field], field === 'points' ? 1000 : 16) }
  catch (e) { errors[field] = (e as Error).message }
  let dataset: KMeansRequest['dataset']
  let count = pairs.length
  if (form.kind === 'points') {
    dataset = { kind: 'points', points: pairs }
  } else {
    count = number('samples', 1, 1000)
    const spread = number('spread', 0, 1e6, false)
    const generation_seed = number('generationSeed', 0, 4294967295)
    if (pairs.some(pair => pair.some(v => Math.abs(v) + spread > 1e6))) {
      errors.spread = 'Each source square must fit within ±1,000,000 on both axes.'
    }
    dataset = { kind: 'synthetic_clusters', n_samples: count, centers: pairs, spread, generation_seed } satisfies Recipe
  }
  if (count > 0 && k > count) errors.k = 'Fitted clusters cannot exceed the observation count.'
  if (Object.keys(errors).length) return { errors }
  return { request: { dataset, parameters: { k, max_iterations }, seed }, errors }
}
