import type { KMeansRequest, KMeansResponse, Point, Snapshot } from '../../features/k-means/types'

const record = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const bounded = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && Math.abs(v) <= 1e6
const loss = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0
const points = (v: unknown, n: number): v is Point[] => Array.isArray(v) && v.length === n &&
  v.every(p => Array.isArray(p) && p.length === 2 && p.every(bounded))
const equalPoints = (a: Point[], b: Point[]) => a.length === b.length && a.every(([x,y], i) => x === b[i][0] && y === b[i][1])
const equalIds = (a: number[], b: number[]) => a.length === b.length && a.every((id, i) => id === b[i])
const ids = (v: unknown, n: number, k: number): v is number[] => Array.isArray(v) && v.length === n &&
  v.every(id => Number.isInteger(id) && id >= 0 && id < k)

export function validResponse(v: unknown, request: KMeansRequest): v is KMeansResponse {
  const { dataset, parameters: { k, max_iterations } } = request
  const n = dataset.kind === 'points' ? dataset.points.length : dataset.n_samples
  if (!record(v) || v.algorithm !== 'kmeans' || v.seed !== request.seed ||
      !record(v.parameters) || v.parameters.k !== k || v.parameters.max_iterations !== max_iterations) return false
  const d = v.dataset
  if (!record(d) || d.kind !== dataset.kind || !points(d.points, n)) return false
  if (dataset.kind === 'points') {
    if (!equalPoints(d.points, dataset.points)) return false
  } else {
    if (d.n_samples !== n || d.spread !== dataset.spread || d.generation_seed !== dataset.generation_seed ||
        !points(d.centers, dataset.centers.length) || !equalPoints(d.centers, dataset.centers)) return false
    // Include the endpoints: adding a uniform float64 offset can round to either edge.
    if (!d.points.every(([x, y]) => dataset.centers.some(([cx, cy]) =>
      x >= cx - dataset.spread && x <= cx + dataset.spread &&
      y >= cy - dataset.spread && y <= cy + dataset.spread))) return false
  }

  const t = v.training
  if (!record(t) || typeof t.converged !== 'boolean' || typeof t.iterations_completed !== 'number' ||
      !Number.isInteger(t.iterations_completed) || t.iterations_completed < 1 || t.iterations_completed > max_iterations ||
      !points(t.initial_centroids, k) || !record(v.model) || !points(v.model.centroids, k) ||
      !record(v.metrics) || !loss(v.metrics.inertia) || !ids(v.assignments, n, k)) return false
  const history = t.inertia_history
  const states = t.snapshots
  if (!Array.isArray(history) || history.length !== t.iterations_completed + 1 ||
      !Array.isArray(states) || states.length !== history.length) return false
  if (!states.every((s, i): s is Snapshot => record(s) && s.iteration === i && points(s.centroids, k) &&
      ids(s.assignments, n, k) && loss(s.inertia) && s.inertia === history[i])) return false
  const last = states[states.length - 1]
  // These are repeated values in the same JSON document, not independently computed estimates.
  return equalPoints(t.initial_centroids, states[0].centroids) && equalPoints(v.model.centroids, last.centroids) &&
    equalIds(v.assignments, last.assignments) && v.metrics.inertia === last.inertia &&
    t.converged === equalIds(last.assignments, states[states.length - 2].assignments) &&
    (t.converged || t.iterations_completed === max_iterations)
}

export class ExperimentError extends Error {
  constructor(message: string, readonly code: string, readonly retryable: boolean) { super(message) }
}

export async function runKMeans(request: KMeansRequest, signal: AbortSignal): Promise<KMeansResponse> {
  const base = (import.meta.env.VITE_API_BASE_URL?.trim() || 'http://localhost:8000').replace(/\/+$/, '')
  let response: Response
  let body: unknown
  try {
    response = await fetch(base + '/api/v1/experiments/k-means', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(request), signal,
    })
    body = await response.json()
  } catch (error) {
    if (signal.aborted) throw error
    if (error instanceof SyntaxError) throw new ExperimentError('The service returned unreadable JSON.', 'INVALID_RESPONSE', true)
    throw new ExperimentError('Could not reach the experiment service. Check the backend connection and retry.', 'NETWORK_ERROR', true)
  }
  if (!response.ok) {
    const error = record(body) && record(body.error) ? body.error : {}
    const fields = record(error.details) && Array.isArray(error.details.fields) ? error.details.fields : []
    const reason = record(fields[0]) && typeof fields[0].reason === 'string' ? ' ' + fields[0].reason : ''
    throw new ExperimentError((typeof error.message === 'string' ? error.message : 'Request failed (HTTP ' + response.status + ').') + reason,
      typeof error.code === 'string' ? error.code : 'HTTP_ERROR', response.status >= 500)
  }
  if (!validResponse(body, request)) throw new ExperimentError('The returned clustering history is inconsistent with this experiment.', 'INVALID_RESPONSE', false)
  return body
}
