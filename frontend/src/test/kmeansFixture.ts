import type { KMeansRequest, KMeansResponse, Point } from '../features/k-means/types'

// The seed-0, one-update capped fixture from the backend engine tests.
const points: Point[] = [[0,0],[1,0],[3,0],[9,0],[10,0]]
export const request: KMeansRequest = { dataset: { kind: 'points', points }, seed: 0, parameters: { k: 2, max_iterations: 1 } }
export const fixture: KMeansResponse = {
  ...request, dataset: { kind: 'points', points }, algorithm: 'kmeans',
  model: { centroids: [[3.25,0],[10,0]] }, assignments: [0,0,0,1,1], metrics: { inertia: 16.6875 },
  training: { initial_centroids: [[9,0],[10,0]], iterations_completed: 1, converged: false, inertia_history: [181,16.6875],
    snapshots: [
      { iteration: 0, centroids: [[9,0],[10,0]], assignments: [0,0,0,0,1], inertia: 181 },
      { iteration: 1, centroids: [[3.25,0],[10,0]], assignments: [0,0,0,1,1], inertia: 16.6875 },
    ] },
}
