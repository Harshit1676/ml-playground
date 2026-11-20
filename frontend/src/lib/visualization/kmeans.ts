import type { Data, Layout } from 'plotly.js-dist-min'
import type { Point, Snapshot } from '../../features/k-means/types'

export const colors = ['#176e5a','#a65322','#35699e','#80549c','#9b3656','#68772c','#257e87','#8c622d',
  '#46633a','#755f9b','#a03d36','#517b91','#6c5652','#357052','#705275','#56618c']
const symbols = ['circle','square','diamond','triangle-up','triangle-down','hexagon','pentagon','star'] as const

export function scatterData(points: Point[], state: Snapshot): Data[] {
  return [
    { type: 'scatter', mode: 'markers', name: 'Observations',
      x: points.map(p => p[0]), y: points.map(p => p[1]), customdata: state.assignments,
      marker: { size: 8, color: state.assignments.map(i => colors[i]), symbol: state.assignments.map(i => symbols[i % symbols.length]), opacity: 0.8 },
      hovertemplate: 'Observation (%{x}, %{y})<br>Cluster %{customdata}<extra></extra>' },
    { type: 'scatter', mode: 'markers', name: 'Centroids',
      x: state.centroids.map(p => p[0]), y: state.centroids.map(p => p[1]), customdata: state.centroids.map((_, i) => i),
      marker: { size: 17, symbol: 'x', color: colors.slice(0, state.centroids.length), line: { color: '#19272b', width: 1.5 } },
      hovertemplate: 'Centroid %{customdata}<br>(%{x}, %{y})<extra></extra>' },
  ]
}

function axisRange(points: Point[], dimension: 0 | 1): [number, number] {
  const values = points.map(p => p[dimension])
  const lo = Math.min(...values), hi = Math.max(...values)
  const pad = lo === hi ? Math.max(Math.abs(lo) * 0.05, 0.001) :
    Math.max((hi - lo) * 0.1, Number.EPSILON * Math.max(Math.abs(lo), Math.abs(hi)) * 16, Number.MIN_VALUE)
  return [lo - pad, hi + pad]
}
export function scatterLayout(points: Point[]): Partial<Layout> {
  return { autosize: true, showlegend: false, margin: { t: 16, l: 62, r: 20, b: 54 },
    xaxis: { title: { text: 'Feature x' }, range: axisRange(points, 0) },
    yaxis: { title: { text: 'Feature y' }, range: axisRange(points, 1), scaleanchor: 'x', scaleratio: 1 } }
}
export function inertiaData(history: number[], iteration: number): Data[] {
  return [
    { type: 'scatter', mode: 'lines+markers', name: 'Inertia', x: history.map((_, i) => i), y: history, line: { color: '#176e5a' } },
    { type: 'scatter', mode: 'markers', name: 'Selected iteration', x: [iteration], y: [history[iteration]],
      marker: { symbol: 'diamond', size: 14, color: '#b46b29', line: { width: 1, color: '#19272b' } } },
  ]
}
