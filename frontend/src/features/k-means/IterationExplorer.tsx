import { useMemo, useState } from 'react'
import type { Layout } from 'plotly.js-dist-min'
import { PlotlyFigure } from '../../components/PlotlyFigure'
import { colors, inertiaData, scatterData, scatterLayout } from '../../lib/visualization/kmeans'
import type { KMeansResponse } from './types'

export function IterationExplorer({ run, visible }: { run: KMeansResponse; visible: boolean }) {
  const end = run.training.iterations_completed
  const [index, select] = useState(end)
  const [showAssignments, setShowAssignments] = useState(false)
  const state = run.training.snapshots[index]
  const previous = run.training.snapshots[index - 1]
  const reassigned = previous ? state.assignments.filter((id, i) => id !== previous.assignments[i]).length : 0
  const points = run.dataset.points
  const scatter = useMemo(() => scatterData(points, state, previous), [points, state, previous])
  const layout = useMemo(() => scatterLayout(points), [points])
  const history = useMemo(() => inertiaData(run.training.inertia_history, index), [run, index])
  const historyLayout = useMemo<Partial<Layout>>(() => ({ autosize: true, showlegend: false, margin: { t: 12, r: 20, b: 52, l: 62 },
    xaxis: { title: { text: 'Iteration' }, range: [-0.2, end + 0.2] }, yaxis: { title: { text: 'Inertia' }, rangemode: 'tozero' } }), [end])
  const counts = Array<number>(run.parameters.k).fill(0)
  for (const id of state.assignments) counts[id]++
  return <>
    <section className="chart-card km-explorer">
      <header className="results-heading"><h2>Explore the run</h2></header>
      <label className="field-label" htmlFor="km-step">Iteration {index} of {end}</label>
      <input id="km-step" type="range" min={0} max={end} value={index} onChange={e => select(Number(e.target.value))} />
      <strong>Inertia at iteration {index}: {state.inertia.toLocaleString(undefined, { maximumSignificantDigits: 8 })}</strong>
      {visible && <PlotlyFigure className="plot-frame" label="Observations and centroids at selected iteration" data={scatter} layout={layout} />}
      <div className="km-step-actions">
        <button disabled={!index} onClick={() => select(0)}>Initialization</button>
        <button disabled={!index} onClick={() => select(index - 1)}>Previous</button>
        <button disabled={index === end} onClick={() => select(index + 1)}>Next</button>
        <button disabled={index === end} onClick={() => select(end)}>Final result</button>
      </div>
      <p className="km-caption">Crosses mark centroids. Dotted trails join the previous and current positions.</p>
      <p className="snapshot-change" aria-live="polite">{previous
        ? `${reassigned} of ${points.length} observations changed cluster since iteration ${index - 1}.`
        : 'Starting assignments: each point joins its nearest seeded centroid.'}</p>
      <h3>Cluster assignments</h3>
      <p className="km-caption">{points.length} observations · {run.parameters.k} fitted clusters · Initialization seed {run.seed}
          {run.dataset.kind === 'synthetic_clusters' && <> · Generation seed {run.dataset.generation_seed}</>}</p>
      <table className="classification-table"><caption>Cluster membership at iteration {index}</caption>
        <thead><tr><th>Cluster</th><th>Centroid (x, y)</th><th>Observations</th><th>Distance moved</th></tr></thead>
        <tbody>{state.centroids.map(([x,y], id) => <tr key={id}><th scope="row"><i className="km-swatch" style={{ background: colors[id] }} />{id}</th>
          <td>({x}, {y})</td><td>{counts[id] || '0 (empty)'}</td>
          <td>{previous ? Math.hypot(x - previous.centroids[id][0], y - previous.centroids[id][1]).toLocaleString(undefined, { maximumSignificantDigits: 5 }) : 'Initial'}</td></tr>)}</tbody>
      </table>
      {counts.includes(0) && <p className="single-class-note">An empty cluster has zero assigned observations. It keeps its previous centroid and can gain points later.</p>}
      <details className="classification-details" onToggle={e => setShowAssignments(e.currentTarget.open)}>
        <summary>Inspect individual assignments</summary>
        {showAssignments && <div className="km-observations" tabIndex={0} role="region" aria-label="Observation assignments">
          <table className="classification-table"><caption>Assignments at iteration {index}</caption>
            <thead><tr><th>Observation</th><th>(x, y)</th><th>Cluster</th></tr></thead>
            <tbody>{points.map(([x, y], i) => <tr key={i}><th scope="row">{i + 1}</th><td>({x}, {y})</td><td>{state.assignments[i]}</td></tr>)}</tbody>
          </table></div>}
      </details>
    </section>
    <section className="chart-card"><h3>Inertia by iteration</h3>
      <p className="km-caption">Inertia sums the squared distances from observations to their assigned centroids. The diamond marks your selected step.
        More clusters can lower inertia even when the grouping is less useful.</p>
      {visible && <PlotlyFigure className="plot-frame" label="Inertia history with selected iteration" data={history} layout={historyLayout} />}
    </section>
    <section className="explanation-card"><div><h3>{run.training.converged ? 'Run outcome: converged' : 'Run outcome: iteration limit reached'}</h3>
      <p>{run.training.converged ? 'Assignments stopped changing after the final centroid update.'
        : 'Assignments were still changing at the limit. A returned centroid can differ from the mean of its newly assigned points.'}
        {' '}Try a different initialization seed to see whether the same observations settle into a different grouping. Equal distances favor the lowest cluster ID.</p>
    </div></section>
  </>
}
