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

export function ProbabilityChart({ response, threshold, predictedClasses }: LogisticRegressionChartsProps) {
  return (
    <section className="chart-card primary-chart" aria-labelledby="probability-chart-title">
      <header className="chart-heading">
        <div>
          <h3 id="probability-chart-title">Probability curve and observations</h3>
        </div>
        <span className="chart-unit">x vs. probability</span>
      </header>
      <p className="chart-description">Circles and diamonds mark observed classes. Where the curve meets or exceeds the dashed threshold, the predicted class is 1.</p>
      <PlotlyFigure
        className="plot-frame logistic-probability-plot"
        label="Observed binary labels, returned Logistic Regression probabilities, and the active classification threshold"
        data={buildLogisticRegressionTraces(response, threshold, predictedClasses)}
        layout={buildLogisticRegressionLayout()}
      />
    </section>
  )
}

export function LogisticLossChart({ response }: Pick<LogisticRegressionChartsProps, 'response'>) {
  return (
    <section className="chart-card loss-card" aria-labelledby="logistic-loss-title">
      <header className="chart-heading">
        <div>
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
  )
}
