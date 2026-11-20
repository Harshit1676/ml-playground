import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-basic-dist-min'
import type { PlotlyFigureProps } from './PlotlyFigure'

const Plot = createPlotlyComponent(Plotly)

export function PlotlyFigureImpl({ data, layout, label, className, onClick }: PlotlyFigureProps) {
  return (
    <div className={className} role="img" aria-label={label}>
      <Plot
        data={data}
        onClick={onClick}
        layout={{
          ...layout,
          font: { family: 'Inter, system-ui, sans-serif', size: 14, color: '#34494b' },
          hoverlabel: { bgcolor: '#203b36', font: { size: 14, color: '#ffffff' } },
          xaxis: { automargin: true, ...layout.xaxis },
          yaxis: { automargin: true, ...layout.yaxis },
        }}
        config={{ responsive: true, displaylogo: false, scrollZoom: false }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}

export default PlotlyFigureImpl
