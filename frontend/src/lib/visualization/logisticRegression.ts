import type { Data, Layout } from 'plotly.js-dist-min'
import { logisticProbability } from '../../features/logistic-regression/classification'
import type { BinaryLabel, LogisticRegressionResponse } from '../../features/logistic-regression/types'

export function buildLogisticRegressionTraces(
  response: LogisticRegressionResponse,
  threshold: number,
  predictedClasses: BinaryLabel[],
): Data[] {
  const observedX = response.dataset.x
  const minimum = Math.min(...observedX)
  const maximum = Math.max(...observedX)
  const span = maximum - minimum
  const padding = Math.max(Math.abs(minimum) * 0.05, 0.001)
  const curveMinimum = span === 0 ? minimum - padding : minimum
  const curveMaximum = span === 0 ? maximum + padding : maximum
  const { coefficient, intercept } = response.model
  const grid = Array.from({ length: 101 }, (_, index) =>
    index === 100 ? curveMaximum : curveMinimum + (curveMaximum - curveMinimum) * index / 100)
  // Resolve the bend even when the displayed x range covers thousands of logits.
  if (Math.abs(coefficient) * (curveMaximum - curveMinimum) > 32) {
    for (let logit = -16; logit <= 16; logit += 0.25) {
      const feature = (logit - intercept) / coefficient
      if (feature > curveMinimum && feature < curveMaximum) grid.push(feature)
    }
  }
  const curveX = [...new Set(grid)].sort((a, b) => a - b)
  const curveY = curveX.map((feature) => logisticProbability(response.model.coefficient * feature + response.model.intercept))

  const observedTraces: Data[] = ([0, 1] as const).map((label) => {
    const indexes = response.dataset.y.flatMap((actual, index) => actual === label ? [index] : [])
    return {
      type: 'scatter',
      mode: 'markers',
      name: `Observed class ${label}`,
      x: indexes.map((index) => observedX[index]),
      y: indexes.map((index) => response.dataset.y[index]),
      text: indexes.map((index) => `Predicted class ${predictedClasses[index]} · probability ${response.predictions.probabilities[index]}`),
      marker: {
        color: label === 0 ? '#397a9b' : '#c58043',
        symbol: label === 0 ? 'circle' : 'diamond',
        size: 10,
        line: { color: '#fffefa', width: 1.5 },
      },
      hovertemplate: `Feature x = %{x:.3f}<br>Observed label = ${label}<br>%{text}<extra>Observation</extra>`,
    }
  })

  return [
    ...observedTraces,
    {
      type: 'scatter',
      mode: 'lines',
      name: 'Predicted probability',
      x: curveX,
      y: curveY,
      line: { color: '#176e5a', width: 3 },
      hovertemplate: 'Feature x = %{x:.3f}<br>Probability = %{y:.3f}<extra>Model</extra>',
    },
    {
      type: 'scatter',
      mode: 'lines',
      name: `Threshold ${threshold}`,
      x: [curveMinimum, curveMaximum],
      y: [threshold, threshold],
      line: { color: '#a7443b', width: 2, dash: 'dash' },
      hovertemplate: `Classification threshold = ${threshold}<extra></extra>`,
    },
  ]
}

export function buildLogisticRegressionLayout(): Partial<Layout> {
  return {
    autosize: true,
    margin: { l: 58, r: 18, t: 32, b: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, ui-sans-serif, system-ui, sans-serif', color: '#536366', size: 11 },
    legend: { orientation: 'h', y: 1.02, yanchor: 'bottom', x: 0 },
    xaxis: { title: { text: 'Feature x' }, zeroline: false, gridcolor: '#e8ebe5' },
    yaxis: {
      title: { text: 'Probability / observed class' },
      range: [-0.08, 1.08],
      tickvals: [0, 0.5, 1],
      ticktext: ['Class 0', '0.5', 'Class 1'],
      gridcolor: '#e8ebe5',
    },
  }
}

export function buildLogisticLossTraces(response: LogisticRegressionResponse): Data[] {
  return [{
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Binary cross-entropy',
    x: response.training.loss_history.map((_, step) => step),
    y: response.training.loss_history,
    marker: { color: '#c58043', size: 5 },
    line: { color: '#c58043', width: 2 },
    hovertemplate: 'Step %{x}<br>BCE = %{y:.5f}<extra></extra>',
  }]
}

export function buildLogisticLossLayout(): Partial<Layout> {
  return {
    autosize: true,
    margin: { l: 65, r: 18, t: 18, b: 48 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, ui-sans-serif, system-ui, sans-serif', color: '#536366', size: 11 },
    showlegend: false,
    xaxis: { title: { text: 'Training step' }, rangemode: 'tozero', gridcolor: '#e8ebe5' },
    yaxis: { title: { text: 'Binary cross-entropy' }, rangemode: 'tozero', gridcolor: '#e8ebe5' },
  }
}
