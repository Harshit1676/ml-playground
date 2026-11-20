import { useEffect, useRef, useState } from 'react'
import { ExperimentError, runKMeans } from '../../lib/api/kmeans'
import { IterationExplorer } from './IterationExplorer'
import { buildRequest, defaults } from './request'
import type { Errors, Form } from './request'
import type { KMeansRequest, KMeansResponse } from './types'

export function KMeansExperience({ visible }: { visible: boolean }) {
  const [form, setForm] = useState(defaults)
  const [errors, setErrors] = useState<Errors>({})
  const [failure, setFailure] = useState<ExperimentError>()
  const [pending, setPending] = useState(false)
  const [result, setResult] = useState<KMeansResponse>()
  const active = useRef<AbortController | null>(null)
  const [runNumber, setRunNumber] = useState(0)
  const last = useRef<KMeansRequest | null>(null)
  useEffect(() => () => { active.current?.abort(); active.current = null }, [])

  function clearRun() {
    active.current?.abort()
    active.current = null
    last.current = null
    setPending(false)
    setFailure(undefined)
    setResult(undefined)
  }
  async function execute(request: KMeansRequest) {
    clearRun()
    last.current = request
    const controller = new AbortController()
    active.current = controller
    setPending(true)
    try {
      const data = await runKMeans(request, controller.signal)
      if (active.current === controller) {
        setRunNumber(n => n + 1)
        setResult(data)
      }
    } catch (error) {
      if (active.current === controller) setFailure(error instanceof ExperimentError ? error : new ExperimentError('The experiment failed. Try again.', 'ERROR', true))
    } finally {
      if (active.current === controller) { active.current = null; setPending(false) }
    }
  }

  function field(key: Exclude<keyof Form, 'kind'>, label: string, min: number, max: number, step = '1') {
    const attrs = { id: 'km-' + key, value: form[key], 'aria-invalid': Boolean(errors[key]),
      'aria-describedby': errors[key] ? 'km-error-' + key : undefined,
      onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        setForm({ ...form, [key]: e.target.value })
        setErrors({ ...errors, [key]: undefined })
      } }
    const multiline = key === 'points' || key === 'centers'
    return <div className={'form-field' + (multiline ? ' full-width' : '')}>
      <label className="field-label" htmlFor={attrs.id}>{label}</label>
      {multiline ? <textarea {...attrs} rows={4} spellCheck={false} /> : <input {...attrs} type="number" min={min} max={max} step={step} />}
      {errors[key] && <span className="field-error" id={'km-error-' + key}>{errors[key]}</span>}
    </div>
  }
  return <section className="experiment-section km-lab" aria-labelledby="km-title">
    <header className="experiment-heading"><div><h1 id="km-title">K-Means <em>Lab</em></h1>
      <p className="experiment-intro">Group unlabeled points by distance. Follow each centroid from its starting observation to its final position.</p>
    </div></header>
    <div className="experiment-layout">
      <aside className="controls-panel"><div className="panel-heading"><h2>Choose your data</h2></div>
        <form noValidate onSubmit={e => {
          e.preventDefault()
          const built = buildRequest(form)
          setErrors(built.errors)
          if (built.request) void execute(built.request)
          else clearRun()
        }}>
          <div className="form-field"><label className="field-label" htmlFor="km-kind">Dataset type</label>
            <select id="km-kind" value={form.kind} onChange={e => { setForm({ ...form, kind: e.target.value as Form['kind'] }); setErrors({}) }}>
              <option value="synthetic_clusters">Seeded synthetic clusters</option><option value="points">Inline observations</option>
            </select></div>
          <fieldset className="control-group"><legend>{form.kind === 'points' ? 'Your observations' : 'Source clusters'}</legend>
            <p className="group-description">{form.kind === 'points' ? 'Enter one x,y pair per line. Repeated points are allowed.'
              : 'Enter one x,y source center per line. Each observation chooses a center at random and adds independent uniform offsets on each axis. K-Means learns its own centroids.'}</p>
            <div className="field-grid">{form.kind === 'points' ? field('points', 'Observation pairs', 1, 1000) : <>
              {field('centers', 'Source center pairs', 1, 16)}{field('samples', 'Observations', 1, 1000)}
              {field('spread', 'Spread per axis', 0, 1e6, 'any')}{field('generationSeed', 'Generation seed', 0, 4294967295)}
            </>}</div></fieldset>
          <fieldset className="control-group"><legend>Fit the clusters</legend>
            <p className="group-description">Change the initialization seed to try new starting centroids on the same observations.
              {form.kind === 'synthetic_clusters' && ' Change the generation seed to draw a new dataset.'}</p>
            <div className="field-grid">{field('k', 'Fitted clusters (k)', 1, 16)}
              {field('iterations', 'Maximum iterations', 1, 100)}{field('seed', 'Initialization seed', 0, 4294967295)}</div>
          </fieldset>
          <div className="form-actions"><button className="run-button" type="submit">Run experiment</button>
            <button className="reset-button" type="button" onClick={() => { clearRun(); setForm(defaults); setErrors({}) }}>Reset to defaults</button></div>
        </form>
      </aside>
      <div className="results-column">
        {pending && <p role="status">Clustering observations…</p>}
        {Object.values(errors).some(Boolean) && <p role="alert" className="experiment-error">Check the highlighted experiment settings.</p>}
        {failure && <div className="experiment-error" role="alert"><div><strong>{failure.code}</strong><p>{failure.message}</p>
          {failure.retryable && <button className="retry-button" onClick={() => last.current && void execute(last.current)}>Retry request</button>}</div></div>}
        {result ? <IterationExplorer key={runNumber} run={result} visible={visible} /> : !pending && !failure && <section className="empty-result"><h2>Explore centroid movement</h2>
          <p>Run the experiment to compare the initial grouping with each update and inspect how the distances change.</p></section>}
      </div>
    </div>
  </section>
}
