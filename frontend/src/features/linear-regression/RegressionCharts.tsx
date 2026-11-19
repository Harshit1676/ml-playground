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

export function RegressionCharts({ response }: RegressionChartsProps) {
  return (
    <div className="charts-grid">
      <section className="chart-card" aria-labelledby="fit-chart-title">
        <header className="chart-heading">
          <div>
            <p className="section-kicker">The fit</p>
            <h3 id="fit-chart-title">A line through the data</h3>
          </div>
          <span className="chart-unit">x vs. y</span>
        </header>
        <p className="chart-description">Each point is an observation. The green line uses the final slope and intercept returned by the experiment service.</p>
        <PlotlyFigure
          className="plot-frame"
          label="Scatterplot of the returned dataset with the returned fitted regression line"
          data={buildRegressionTraces(response)}
          layout={buildRegressionLayout()}
        />
      </section>

      <section className="chart-card loss-card" aria-labelledby="loss-chart-title">
        <header className="chart-heading">
          <div>
            <p className="section-kicker">The training</p>
            <h3 id="loss-chart-title">How the error changed</h3>
          </div>
          <span className="chart-unit">MSE per step</span>
        </header>
        <p className="chart-description">The curve plots the backend's initial MSE and each post-update MSE. It does not estimate parameter values between steps.</p>
        <PlotlyFigure
          className="plot-frame loss-plot"
          label="Mean squared error by training step, using the response loss history"
          data={buildLossTraces(response)}
          layout={buildLossLayout()}
        />
      </section>
    </div>
  )
}
