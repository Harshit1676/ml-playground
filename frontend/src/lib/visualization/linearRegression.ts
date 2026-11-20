import type { Data, Layout } from 'plotly.js-dist-min'
import type { LinearRegressionResponse } from '../../features/linear-regression/types'

export function buildRegressionTraces(response: LinearRegressionResponse, selected?: number): Data[] {
  const minimumX = Math.min(...response.dataset.x)
  const maximumX = Math.max(...response.dataset.x)
  const lineX = [minimumX, maximumX]
  const lineY = lineX.map((x) => response.model.slope * x + response.model.intercept)

  const traces: Data[] = [
    {
      type: 'scatter',
      mode: 'markers',
      name: 'Observations',
      x: response.dataset.x,
      y: response.dataset.y,
      marker: { color: '#975320', size: 9, line: { color: '#fffaf2', width: 1.5 } },
      customdata: response.predictions.y_hat,
      hovertemplate: 'x = %{x:.4g}<br>Observed y = %{y:.4g}<br>Predicted y = %{customdata:.4g}<extra>Observation</extra>',
    },
    {
      type: 'scatter',
      mode: 'lines',
      name: 'Fitted line',
      x: lineX,
      y: lineY,
      line: { color: '#16715c', width: 3 },
      // A point on the fitted line must still receive the click.
      hoverinfo: 'skip',
    },
  ]
  if (selected !== undefined) {
    const x = response.dataset.x[selected]
    traces.push({
      type: 'scatter', mode: 'lines+markers', name: 'Residual', showlegend: false,
      x: [x, x], y: [response.predictions.y_hat[selected], response.dataset.y[selected]],
      line: { color: '#77345e', width: 3, dash: 'dot' },
      marker: { symbol: ['diamond', 'circle-open'], size: [8, 15], color: '#77345e', line: { width: 2 } },
      hoverinfo: 'skip',
    })
  }
  return traces
}

export function buildRegressionLayout(): Partial<Layout> {
  return {
    autosize: true,
    uirevision: 'inspect',
    margin: { l: 54, r: 18, t: 18, b: 48 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    legend: { orientation: 'h', y: 1.12, x: 0 },
    xaxis: { title: { text: 'Feature x' }, zeroline: false, gridcolor: '#e8ebe5' },
    yaxis: { title: { text: 'Target y' }, zeroline: false, gridcolor: '#e8ebe5' },
  }
}

export function buildLossTraces(response: LinearRegressionResponse): Data[] {
  return [
    {
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Mean squared error',
      x: response.training.loss_history.map((_, step) => step),
      y: response.training.loss_history,
      marker: { color: '#975320', size: 5 },
      line: { color: '#975320', width: 2 },
      hovertemplate: 'Step %{x}<br>MSE = %{y:.5f}<extra></extra>',
    },
  ]
}

export function buildLossLayout(): Partial<Layout> {
  return {
    autosize: true,
    margin: { l: 65, r: 18, t: 18, b: 48 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: false,
    xaxis: { title: { text: 'Training step' }, rangemode: 'tozero', gridcolor: '#e8ebe5' },
    yaxis: { title: { text: 'Mean squared error' }, rangemode: 'tozero', gridcolor: '#e8ebe5' },
  }
}
