import { describe, expect, it } from 'vitest'
import Plotly from 'plotly.js-basic-dist-min'

interface ScatterSchema {
  attributes?: {
    mode?: {
      flags?: string[]
    }
  }
}

interface PlotlySchema {
  traces: Record<string, ScatterSchema>
}

interface PlotlyWithSchema {
  PlotSchema: {
    get: () => PlotlySchema
  }
  [method: string]: unknown
}

describe('Plotly basic bundle compatibility', () => {
  it('provides scatter marker/line modes and the React adapter update methods', () => {
    const plotly = Plotly as unknown as PlotlyWithSchema
    const scatter = plotly.PlotSchema.get().traces.scatter
    const requiredMethods = ['newPlot', 'react', 'restyle', 'relayout', 'update', 'purge']

    expect(scatter).toBeDefined()
    expect(scatter.attributes?.mode?.flags).toEqual(expect.arrayContaining(['markers', 'lines']))
    for (const method of requiredMethods) {
      expect(typeof plotly[method]).toBe('function')
    }
  })
})
