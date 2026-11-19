import type { Data, Layout } from 'plotly.js-dist-min'
import type { LinearRegressionResponse } from '../../features/linear-regression/types'

export function buildRegressionTraces(response: LinearRegressionResponse): Data[] {
  const minimumX = Math.min(...response.dataset.x)
  const maximumX = Math.max(...response.dataset.x)
  const lineX = [minimumX, maximumX]
  const lineY = lineX.map((x) => response.model.slope * x + response.model.intercept)

  return [
    {
      type: 'scatter',
      mode: 'markers',
      name: 'Observations',
      x: response.dataset.x,
      y: response.dataset.y,
      marker: { color: '#c58043', size: 9, line: { color: '#fffaf2', width: 1.5 } },
      hovertemplate: 'x = %{x:.3f}<br>y = %{y:.3f}<extra>Observation</extra>',
    },
    {
      type: 'scatter',
      mode: 'lines',
      name: 'Fitted line',
      x: lineX,
      y: lineY,
      line: { color: '#16715c', width: 3 },
      hovertemplate: 'x = %{x:.3f}<br>ŷ = %{y:.3f}<extra>Model</extra>',
    },
  ]
}

export function buildRegressionLayout(): Partial<Layout> {
  return {
    autosize: true,
    margin: { l: 54, r: 18, t: 18, b: 48 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, ui-sans-serif, system-ui, sans-serif', color: '#536366', size: 11 },
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
      marker: { color: '#c58043', size: 5 },
      line: { color: '#c58043', width: 2 },
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
    font: { family: 'Inter, ui-sans-serif, system-ui, sans-serif', color: '#536366', size: 11 },
    showlegend: false,
    xaxis: { title: { text: 'Training step' }, rangemode: 'tozero', gridcolor: '#e8ebe5' },
    yaxis: { title: { text: 'Mean squared error' }, rangemode: 'tozero', gridcolor: '#e8ebe5' },
  }
}
