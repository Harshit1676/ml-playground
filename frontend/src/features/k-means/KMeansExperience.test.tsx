import { afterEach, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { KMeansExperience } from './KMeansExperience'
import { fixture, request } from '../../test/kmeansFixture'
import { IterationExplorer } from './IterationExplorer'
import type { PlotlyFigureProps } from '../../components/PlotlyFigure'
import App from '../../App'

vi.mock('../../components/PlotlyFigure', () => ({PlotlyFigure: ({label, data}: PlotlyFigureProps) =>
  <output aria-label={label}>{JSON.stringify(data)}</output>}))
afterEach(() => { cleanup(); vi.unstubAllGlobals(); window.history.replaceState({},'', '/') })
const json = (data: unknown) => ({ok:true, status:200,json:async()=>data})
const change = (name: string, value: string) => fireEvent.change(
  within(screen.getByRole('region', {name:'K-Means Lab'})).getByLabelText(name), {target:{value}})
function inline() {
  change('Dataset type','points')
  change('Observation pairs','0,0\n1,0\n3,0\n9,0\n10,0')
  change('Fitted clusters (k)','2')
  change('Maximum iterations','1')
  change('Initialization seed','0')
}
const click = (name: string) => fireEvent.click(screen.getByRole('button',{name}))

it('posts the inline payload once, synchronizes both charts and counts, and clears invalid results', async () => {
  const fetcher=vi.fn().mockResolvedValue(json(fixture))
  vi.stubGlobal('fetch',fetcher)
  render(<KMeansExperience visible />)
  inline()
  change('Dataset type','synthetic_clusters')
  expect(screen.getByLabelText('Generation seed')).toHaveValue(42)
  change('Dataset type','points')
  expect(screen.queryByLabelText('Generation seed')).toBeNull()
  click('Run experiment')
  await screen.findByText('Run outcome: iteration limit reached')
  const [url,init]=fetcher.mock.calls[0]
  expect(url).toMatch(/\/api\/v1\/experiments\/k-means$/)
  expect(JSON.parse(init.body)).toEqual(request)
  expect(screen.getByText('Inertia at iteration 1: 16.6875')).toBeVisible()
  click('Previous')
  expect(screen.getByText('Inertia at iteration 0: 181')).toBeVisible()
  const row=within(screen.getAllByRole('row')[1])
  expect(row.getByText('4')).toBeVisible()
  expect(row.getByText('(9, 0)')).toBeVisible()
  expect(JSON.parse(screen.getByLabelText('Inertia history with selected iteration').textContent!)[1].x).toEqual([0])
  click('Final result')
  click('Initialization')
  click('Next')
  fireEvent.change(screen.getByRole('slider'),{target:{value:'0'}})
  expect(screen.getByRole('slider')).toHaveValue('0')
  expect(fetcher).toHaveBeenCalledTimes(1)
  change('Observation pairs','0,')
  click('Run experiment')
  expect(screen.getByRole('alert')).toBeVisible()
  expect(screen.queryByRole('slider')).toBeNull()
  expect(fetcher).toHaveBeenCalledTimes(1)
})
it('shows network and structured errors, retries the same submission and resets defaults', async () => {
  const fetcher=vi.fn().mockRejectedValueOnce(new TypeError('offline')).mockResolvedValueOnce({
    ok:false,status:422,json:async()=>({error:{code:'VALIDATION_ERROR',message:'Invalid settings',details:{fields:[{reason:'Check coordinates'}]}}}),
  })
  vi.stubGlobal('fetch',fetcher)
  render(<KMeansExperience visible />)
  inline()
  click('Run experiment')
  await screen.findByText('NETWORK_ERROR')
  click('Retry request')
  await screen.findByText('VALIDATION_ERROR')
  expect(screen.getByRole('alert')).toHaveTextContent('Check coordinates')
  expect(fetcher.mock.calls[1][1].body).toEqual(fetcher.mock.calls[0][1].body)
  click('Reset to defaults')
  expect(screen.queryByRole('alert')).toBeNull()
  expect(screen.getByLabelText('Dataset type')).toHaveValue('synthetic_clusters')
})
it('aborts superseded requests and ignores late completions after reset or unmount', async () => {
  let finish!: (r: unknown) => void
  const fetcher=vi.fn().mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve})).mockResolvedValueOnce(json(fixture))
    .mockImplementation(()=>new Promise(()=>{}))
  vi.stubGlobal('fetch',fetcher)
  const view=render(<KMeansExperience visible />)
  inline()
  click('Run experiment')
  expect(screen.getByText('Clustering observations…')).toBeVisible()
  click('Run experiment')
  await screen.findByRole('slider')
  expect(fetcher.mock.calls[0][1].signal.aborted).toBe(true)
  await act(async()=>{finish(json({}))})
  expect(screen.queryByRole('alert')).toBeNull()
  click('Run experiment')
  click('Reset to defaults')
  expect(fetcher.mock.calls[2][1].signal.aborted).toBe(true)
  click('Run experiment')
  view.unmount()
  expect(fetcher.mock.calls[3][1].signal.aborted).toBe(true)
})
it('displays coincident centers and zero membership', () => {
  const run=structuredClone(fixture)
  const state=run.training.snapshots[1]
  state.centroids=[[0,0],[0,0]]
  state.assignments.fill(0)
  render(<IterationExplorer run={run} visible />)
  expect(screen.getByText('0 (empty)')).toBeVisible()
  expect(screen.getAllByText('(0, 0)')).toHaveLength(2)
})
it('deep links and preserves result, form and selected iteration through all labs and history', async () => {
  window.history.replaceState({},'', '/?lesson=1&algorithm=k-means#home')
  const fetcher=vi.fn((url:string)=>Promise.resolve(json(url.includes('/health')?{status:'ok'}:fixture)))
  vi.stubGlobal('fetch',fetcher)
  const {container}=render(<App />)
  expect(screen.getByRole('heading',{name:'K-Means Lab'})).toBeVisible()
  const linearLink = screen.getByRole('link', {name:'Linear Regression'})
  for (const modifier of [{ctrlKey:true}, {metaKey:true}]) {
    let intercepted: boolean | undefined
    document.addEventListener('click', event => {
      intercepted = event.defaultPrevented
      event.preventDefault() // jsdom has no browser tab to open.
    }, {once:true})
    fireEvent.click(linearLink, modifier)
    expect(intercepted).toBe(false)
    expect(location.search).toContain('algorithm=k-means')
  }
  inline(); click('Run experiment'); await screen.findByRole('slider'); click('Initialization')
  for(const name of ['Linear Regression','Logistic Regression','K-Means']) {
    const link=screen.getByRole('link',{name})
    expect(link.getAttribute('href')).toContain('lesson=1')
    expect(link.getAttribute('href')).toMatch(/#home$/)
    fireEvent.click(link)
  }
  act(()=>window.history.back())
  await screen.findByRole('heading',{name:'Logistic Regression Lab'})
  act(()=>window.history.forward())
  await screen.findByRole('heading',{name:'K-Means Lab'})
  expect(screen.getByRole('slider')).toHaveValue('0')
  expect(screen.getByRole('combobox', {name:'Dataset type'})).toHaveValue('points')
  const ids=[...container.querySelectorAll('[id]')].map(el=>el.id)
  expect(new Set(ids).size).toBe(ids.length)
  expect(fetcher).toHaveBeenCalledTimes(2)
})

it('compares consecutive backend snapshots and keeps training edits separate from iteration selection', async () => {
  const fetcher = vi.fn().mockResolvedValue(json(fixture))
  vi.stubGlobal('fetch', fetcher)
  render(<KMeansExperience visible />)
  inline()
  click('Run experiment')
  expect(await screen.findByText('1 of 5 observations changed cluster since iteration 0.')).toBeVisible()
  const chart = screen.getByLabelText('Observations and centroids at selected iteration')
  expect(JSON.parse(chart.textContent!).at(-1)).toMatchObject({ name: 'Centroid movement', x: [9, 3.25, null], y: [0, 0, null] })
  click('Initialization')
  expect(JSON.parse(chart.textContent!)).toHaveLength(2)
  expect(screen.queryByText(/Showing the previous settings/)).toBeNull()
  change('Initialization seed', '2')
  expect(screen.getByText(/Showing the previous settings/)).toBeVisible()
  change('Initialization seed', '0')
  expect(screen.queryByText(/Showing the previous settings/)).toBeNull()
  expect(fetcher).toHaveBeenCalledTimes(1)
})
