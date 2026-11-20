import { lazy, Suspense } from 'react'
import type { Data, Layout, PlotMouseEvent } from 'plotly.js-dist-min'

const PlotlyFigureImpl = lazy(() => import('./PlotlyFigureImpl'))

export interface PlotlyFigureProps {
  onClick?: (event: PlotMouseEvent) => void
  data: Data[]
  layout: Partial<Layout>
  label: string
  className?: string
}

export function PlotlyFigure(props: PlotlyFigureProps) {
  return (
    <Suspense fallback={
      <div className={props.className} role="img" aria-label={props.label} aria-busy="true">
        <span className="chart-loading">Preparing chart…</span>
      </div>
    }>
      <PlotlyFigureImpl {...props} />
    </Suspense>
  )
}
