import { PlotlyFigure } from '../../components/PlotlyFigure'
import {
  buildLogisticLossLayout,
  buildLogisticLossTraces,
  buildLogisticRegressionLayout,
  buildLogisticRegressionTraces,
} from '../../lib/visualization/logisticRegression'
import type { BinaryLabel, LogisticRegressionResponse } from './types'

interface LogisticRegressionChartsProps {
  response: LogisticRegressionResponse
  threshold: number
  predictedClasses: BinaryLabel[]
}

export function LogisticRegressionCharts({ response, threshold, predictedClasses }: LogisticRegressionChartsProps) {
  return (
    <div className="charts-grid logistic-charts-grid">
      <section className="chart-card" aria-labelledby="probability-chart-title">
        <header className="chart-heading">
          <div>
            <p className="section-kicker">The classifier</p>
            <h3 id="probability-chart-title">Probability curve and observations</h3>
          </div>
          <span className="chart-unit">x vs. probability</span>
        </header>
        <p className="chart-description">Markers show observed labels at 0 or 1. The curve uses the returned final coefficient and intercept; the dashed guide is the threshold you are exploring.</p>
        <PlotlyFigure
          className="plot-frame logistic-probability-plot"
          label="Observed binary labels, returned Logistic Regression probabilities, and the active classification threshold"
          data={buildLogisticRegressionTraces(response, threshold, predictedClasses)}
          layout={buildLogisticRegressionLayout()}
        />
      </section>

      <section className="chart-card loss-card" aria-labelledby="logistic-loss-title">
        <header className="chart-heading">
          <div>
            <p className="section-kicker">The training</p>
            <h3 id="logistic-loss-title">Binary cross-entropy by step</h3>
          </div>
          <span className="chart-unit">BCE per step</span>
        </header>
        <p className="chart-description">This chart uses the backend's initial BCE and each post-update loss. It does not infer intermediate coefficients.</p>
        <PlotlyFigure
          className="plot-frame loss-plot"
          label="Binary cross-entropy by training step, using the returned loss history"
          data={buildLogisticLossTraces(response)}
          layout={buildLogisticLossLayout()}
        />
      </section>
    </div>
  )
}
