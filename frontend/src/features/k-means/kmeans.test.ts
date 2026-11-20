import { describe, expect, it } from 'vitest'
import { buildRequest, defaults } from './request'
import { validResponse } from '../../lib/api/kmeans'
import { inertiaData, scatterData, scatterLayout } from '../../lib/visualization/kmeans'
import { fixture, request } from '../../test/kmeansFixture'
import type { KMeansResponse, Recipe } from './types'

describe('K-Means contract', () => {
  it('builds reproducible synthetic and inline requests with separate seeds', () => {
    expect(buildRequest(defaults).request).toEqual({
      dataset: {kind:'synthetic_clusters', centers:[[-5,-4],[0,5],[6,-2]], n_samples:150, spread:1.5, generation_seed:42},
      parameters:{k:3, max_iterations:30}, seed:7,
    })
    expect(buildRequest({...defaults, seed:'13'}).request?.dataset).toEqual(buildRequest(defaults).request?.dataset)
    expect(buildRequest({...defaults, generationSeed:'13'}).request?.seed).toBe(7)
    expect(buildRequest({...defaults, kind:'points', points:'0,1\n0,1', k:'2'}).request?.dataset).toEqual({kind:'points',points:[[0,1],[0,1]]})
  })
  it.each([{points:'1'}, {points:'1,2,3'}, {points:'1, '}, {points:'NaN,1'}, {points:'1000001,0'}, {k:'5'}, {seed:'4294967296'}, {iterations:'1.5'}])(
    'rejects invalid inline controls %o', change => expect(buildRequest({...defaults,kind:'points',...change}).request).toBeUndefined())
  it.each([{samples:'0'}, {centers:'999999,0',spread:'2'}, {spread:'-1'}, {generationSeed:'-1'}, {centers:Array(17).fill('0,0').join('\n')}])(
    'rejects invalid synthetic controls %o', change => expect(buildRequest({...defaults,...change}).request).toBeUndefined())
  it('supports more sources than samples, with k bounded by samples', () => {
    expect(buildRequest({...defaults,samples:'1',k:'1'}).request).toBeDefined()
  })
  it('accepts the full capped response and rejects inconsistent summaries or snapshots', () => {
    expect(validResponse(fixture,request)).toBe(true)
    const corruptions: ((r: KMeansResponse) => void)[] = [
      r=>{r.seed++}, r=>{r.parameters.k++}, r=>{r.dataset.points.pop()}, r=>{r.dataset.points[0][0]++},
      r=>{r.training.snapshots[0].centroids[0][0]=Infinity}, r=>{r.training.snapshots[0].assignments[0]=2},
      r=>{r.training.snapshots[0].assignments.pop()}, r=>{r.training.snapshots[1].iteration=0},
      r=>{r.training.snapshots.pop()}, r=>{r.metrics.inertia=-1}, r=>{r.training.inertia_history[0]++},
      r=>{r.training.initial_centroids[0][0]++}, r=>{r.model.centroids[0][0]++}, r=>{r.assignments[0]=1},
      r=>{r.training.converged=true}, r=>{r.metrics.inertia=NaN},
    ]
    for (const corrupt of corruptions) {
      const r = structuredClone(fixture)
      corrupt(r)
      expect(validResponse(r,request)).toBe(false)
    }
    const parameters={k:2,max_iterations:10}
    expect(validResponse({...fixture,parameters},{...request,parameters})).toBe(false)
    expect(validResponse({...fixture,algorithm:'linear_regression'}, request)).toBe(false)
  })
  it('validates synthetic recipe, generation seed and returned coordinates', () => {
    const dataset: Recipe={kind:'synthetic_clusters',n_samples:5,centers:[[5,0]],spread:5,generation_seed:33}
    const req={...request,dataset}
    const result={...fixture,dataset:{...dataset,points:fixture.dataset.points}}
    expect(validResponse(result,req)).toBe(true)
    for (const change of [{generation_seed:1},{centers:[[0,0]]},{spread:9},{n_samples:6},{points:[[0,0]]},
      {points:[[11,0], ...fixture.dataset.points.slice(1)]},{kind:'points'}])
      expect(validResponse({...result,dataset:{...result.dataset,...change}},req)).toBe(false)
  })
  it('reads plot coordinates and highlighted inertia from the chosen backend state', () => {
    const first=fixture.training.snapshots[0]
    const traces=scatterData(fixture.dataset.points,first)
    expect(traces[0]).toMatchObject({x:[0,1,3,9,10],customdata:[0,0,0,0,1]})
    expect(traces[1]).toMatchObject({x:[9,10],y:[0,0]})
    expect(inertiaData(fixture.training.inertia_history,0)[1]).toMatchObject({x:[0],y:[181]})
    for(const points of [[[0,0],[0,0]],[[1,1],[1+Number.EPSILON,1]]] as [number,number][][]) {
      const range=scatterLayout(points).xaxis!.range!
      expect(range[1]).toBeGreaterThan(range[0])
    }
  })
})
