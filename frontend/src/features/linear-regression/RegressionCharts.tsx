import { useState } from 'react'
import { PlotlyFigure } from '../../components/PlotlyFigure'
import {
  buildLossLayout,
  buildLossTraces,
  buildRegressionLayout,
  buildRegressionTraces,
} from '../../lib/visualization/linearRegression'
import type { LinearRegressionResponse } from './types'

interface RegressionChartsProps {
  response: LinearRegressionResponse
}

const format = (value: number) => value.toLocaleString(undefined, { maximumSignificantDigits: 6 })

export function RegressionChart({ response }: RegressionChartsProps) {
  const [index, setIndex] = useState(0)
  const { dataset, predictions } = response
  const observed = dataset.y[index]
  const predicted = predictions.y_hat[index]
  return <section className="chart-card primary-chart" aria-labelledby="fit-chart-title">
    <header className="chart-heading"><h3 id="fit-chart-title">A line through the data</h3></header>
    <p className="chart-description">Select a point to see the gap between its observed target and the fitted line.</p>
    <PlotlyFigure className="plot-frame" label="Scatterplot of the returned dataset with the returned fitted regression line"
      data={buildRegressionTraces(response, index)} layout={buildRegressionLayout()}
      onClick={event => {
        const point = event.points.find(item => item.data.name === 'Observations')
        if (point) setIndex(point.pointNumber)
      }} />
    <div className="inspector">
      <label htmlFor="inspect-observation">Observation {index + 1} of {dataset.x.length} · x = {format(dataset.x[index])}</label>
      <input id="inspect-observation" type="range" min={1} max={dataset.x.length} value={index + 1}
        onChange={event => setIndex(Number(event.target.value) - 1)} />
      <dl className="point-values" aria-live="polite">
        <div><dt>Observed y</dt><dd>{format(observed)}</dd></div>
        <div><dt>Predicted y</dt><dd>{format(predicted)}</dd></div>
        <div><dt>Residual</dt><dd>{format(observed - predicted)}</dd></div>
      </dl>
      <p className="field-hint">All three values use target (y) units. Residual = observed − predicted. A positive residual places the point above the line.</p>
    </div>
  </section>
}

export function LossChart({ response }: RegressionChartsProps) {
  return (
    <section className="chart-card loss-card" aria-labelledby="loss-chart-title">
      <header className="chart-heading">
        <div>
          <p className="section-kicker">The training</p>
          <h3 id="loss-chart-title">How the error changed</h3>
        </div>
        <span className="chart-unit">MSE per step</span>
      </header>
      <p className="chart-description">Step 0 is the initial mean squared error. Each following point records the error after one training update.</p>
      <PlotlyFigure
        className="plot-frame loss-plot"
        label="Mean squared error by training step, using the response loss history"
        data={buildLossTraces(response)}
        layout={buildLossLayout()}
      />
    </section>
  )
}
