import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-basic-dist-min'
import type { PlotlyFigureProps } from './PlotlyFigure'

const Plot = createPlotlyComponent(Plotly)

export function PlotlyFigureImpl({ data, layout, label, className }: PlotlyFigureProps) {
  return (
    <div className={className} role="img" aria-label={label}>
      <Plot
        data={data}
        layout={layout}
        config={{ responsive: true, displaylogo: false, scrollZoom: false }}
        useResizeHandler
        style={{ width: '100%', height: '100%', minHeight: '280px' }}
      />
    </div>
  )
}

export default PlotlyFigureImpl
