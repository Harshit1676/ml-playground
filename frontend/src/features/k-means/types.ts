export type Point = [number, number]
export type Points = { kind: 'points'; points: Point[] }
export type Recipe = {
  kind: 'synthetic_clusters'
  centers: Point[]
  n_samples: number
  spread: number
  generation_seed: number
}
export interface KMeansRequest {
  dataset: Points | Recipe
  parameters: { k: number; max_iterations: number }
  seed: number
}
export interface Snapshot {
  iteration: number
  centroids: Point[]
  assignments: number[]
  inertia: number
}
export interface KMeansResponse extends KMeansRequest {
  algorithm: 'kmeans'
  dataset: Points | (Recipe & { points: Point[] })
  assignments: number[]
  model: { centroids: Point[] }
  metrics: { inertia: number }
  training: {
    initial_centroids: Point[]
    iterations_completed: number
    converged: boolean
    inertia_history: number[]
    snapshots: Snapshot[]
  }
}
