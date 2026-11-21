import { SettingsFields, SettingsNotice } from '../../components/ExperimentSettings'
import { useEffect, useRef, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { LinearRegressionApiError, LinearRegressionNetworkError, runLinearRegression } from '../../lib/api/linearRegression'
import { DEFAULT_FORM } from './defaults'
import type { ExperimentFormState } from './defaults'
import { buildLinearRegressionRequest } from './request'
import type { FormErrors } from './request'
import type { LinearRegressionRequest, LinearRegressionResponse } from './types'
import { RegressionChart, LossChart } from './RegressionCharts'

// Each recipe changes only the data and its seed. Training settings stay editable.
const datasetPresets = [
  { label: 'Clean rising line', dataset: { datasetKind: 'synthetic_linear', trueSlope: '2', trueIntercept: '1', noiseStd: '0' } },
  { label: 'Noisy downward line', dataset: { datasetKind: 'synthetic_linear', trueSlope: '-1.5', trueIntercept: '2', noiseStd: '2.5' } },
  { label: 'One outlier', dataset: { datasetKind: 'points', inlineX: '-3, -2, -1, 0, 1, 2, 3', inlineY: '-5, -3, -1, 1, 3, 12, 7' } },
] as const

type RunState = 'idle' | 'loading' | 'success' | 'error'

interface CompletedRun {
  request: LinearRegressionRequest
  response: LinearRegressionResponse
}

interface ErrorView {
  title: string
  message: string
  code?: string
  canRetry: boolean
}

interface NumericFieldProps {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  error?: string
  hint?: string
  step?: string
}

function NumericField({ id, label, value, onChange, error, hint, step = 'any' }: NumericFieldProps) {
  const describedBy = [hint ? `${id}-hint` : null, error ? `${id}-error` : null].filter(Boolean).join(' ') || undefined
  return (
    <label className="form-field" htmlFor={id}>
      <span className="field-label">{label}</span>
      <input
        id={id}
        type="number"
        step={step}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-invalid={Boolean(error)}
        aria-describedby={describedBy}
      />
      {hint && <span className="field-hint" id={`${id}-hint`}>{hint}</span>}
      {error && <span className="field-error" id={`${id}-error`}>{error}</span>}
    </label>
  )
}

interface SliderFieldProps extends NumericFieldProps {
  minimum: number
  maximum: number
  rangeStep: number
}

function SliderField({ id, label, value, onChange, error, hint, minimum, maximum, rangeStep, step }: SliderFieldProps) {
  const numericValue = Number(value)
  const sliderValue = Number.isFinite(numericValue) ? Math.min(maximum, Math.max(minimum, numericValue)) : minimum
  return (
    <div className="slider-control">
      <div className="slider-input-line">
        <label className="visually-hidden" htmlFor={`${id}-slider`}>{label} slider</label>
        <input
          id={`${id}-slider`}
          type="range"
          min={minimum}
          max={maximum}
          step={rangeStep}
          value={sliderValue}
          onChange={(event) => onChange(event.target.value)}
          aria-label={`${label} slider`}
        />
        <NumericField id={id} label={label} value={value} onChange={onChange} error={error} hint={hint} step={step} />
      </div>
    </div>
  )
}

function explainError(error: unknown): ErrorView {
  if (error instanceof LinearRegressionNetworkError) {
    return { title: 'The backend could not be reached', message: error.message, code: 'NETWORK_ERROR', canRetry: true }
  }
  if (error instanceof LinearRegressionApiError) {
    let message = error.message
    if (error.code === 'DATASET_VALUE_OUT_OF_RANGE' && error.details) {
      const field = error.details.field
      const index = error.details.index
      const maximum = error.details.max_abs
      const actual = error.details.actual
      message = `${error.message} ${String(field)} at observation ${String(index)} has value ${String(actual)}; the maximum absolute value is ${String(maximum)}.`
    } else if (error.code === 'VALIDATION_ERROR' && Array.isArray(error.details?.fields)) {
      const first = error.details.fields[0]
      if (typeof first === 'object' && first !== null && 'reason' in first && typeof first.reason === 'string') {
        message = `${error.message} ${first.reason}`
      }
    }
    return {
      title: error.code === 'EXPERIMENT_BUSY' ? 'Another experiment is running'
        : error.status >= 500 ? 'The experiment service is unavailable' : 'The experiment could not be run',
      message,
      code: error.code,
      canRetry: error.status >= 500,
    }
  }
  return { title: 'The experiment could not be run', message: 'An unexpected error occurred. Check your settings and try again.', canRetry: true }
}

function formatMetric(value: number, digits = 4): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 })
}

function ResultMetric({ label, value, explanation }: { label: string; value: string; explanation: string }) {
  return (
    <article className="metric-card">
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{explanation}</span>
    </article>
  )
}

function FieldGroup({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return (
    <fieldset className="control-group">
      <legend>{title}</legend>
      <p className="group-description">{description}</p>
      <div className="field-grid">{children}</div>
    </fieldset>
  )
}

export function LinearRegressionExperience() {
  const [form, setForm] = useState<ExperimentFormState>({ ...DEFAULT_FORM })
  const [fieldErrors, setFieldErrors] = useState<FormErrors>({})
  const [runState, setRunState] = useState<RunState>('idle')
  const [errorView, setErrorView] = useState<ErrorView | null>(null)
  const [completedRun, setCompletedRun] = useState<CompletedRun | null>(null)
  const requestVersion = useRef(0)
  const activeController = useRef<AbortController | null>(null)
  const lastRequest = useRef<LinearRegressionRequest | null>(null)

  useEffect(() => () => {
    requestVersion.current += 1
    activeController.current?.abort()
  }, [])

  function changeForm<K extends keyof ExperimentFormState>(key: K, value: ExperimentFormState[K]) {
    setForm((current) => ({ ...current, [key]: value }))
    setFieldErrors((current) => {
      if (!current[key]) return current
      const next = { ...current }
      delete next[key]
      return next
    })
  }

  async function execute(request: LinearRegressionRequest) {
    activeController.current?.abort()
    const controller = new AbortController()
    activeController.current = controller
    const version = ++requestVersion.current
    setRunState('loading')
    setErrorView(null)
    setCompletedRun(null)

    try {
      const response = await runLinearRegression(request, controller.signal)
      if (version !== requestVersion.current) return
      setCompletedRun({ request, response })
      setRunState('success')
    } catch (error) {
      if (version !== requestVersion.current || controller.signal.aborted) return
      setErrorView(explainError(error))
      setRunState('error')
    } finally {
      if (version === requestVersion.current) {
        activeController.current = null
      }
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    requestVersion.current += 1
    activeController.current?.abort()
    activeController.current = null
    const built = buildLinearRegressionRequest(form)
    setFieldErrors(built.errors)
    setErrorView(null)
    if (!built.request) {
      setCompletedRun(null)
      setRunState('error')
      setErrorView({
        title: 'A few settings need attention',
        message: 'Correct the highlighted fields, then run the experiment again.',
        code: 'FORM_VALIDATION',
        canRetry: false,
      })
      return
    }
    lastRequest.current = built.request
    void execute(built.request)
  }

  function retryRequest() {
    if (lastRequest.current) void execute(lastRequest.current)
  }

  function resetDefaults() {
    requestVersion.current += 1
    activeController.current?.abort()
    activeController.current = null
    lastRequest.current = null
    setForm({ ...DEFAULT_FORM })
    setFieldErrors({})
    setRunState('idle')
    setErrorView(null)
    setCompletedRun(null)
  }

  const values = completedRun?.response

  return (
    <section className="experiment-section" aria-labelledby="experiment-title">
      <header className="experiment-heading">
        <div>
          <h1 id="experiment-title">Linear Regression <em>Lab</em></h1>
          <p className="experiment-intro">Fit a line to a dataset, then select a point to explore its prediction error.</p>
        </div>
      </header>

      <div className="experiment-layout">
        <aside className="controls-panel" aria-label="Experiment settings">
          <div className="panel-heading">
            <h2>Experiment settings</h2>
          </div>

          <form onSubmit={handleSubmit} noValidate>
            <div className="form-actions">
              <button className="run-button" type="submit">
                {runState === 'loading' ? <><span className="button-spinner" aria-hidden="true" /> Run updated settings</> : <>Run experiment <span aria-hidden="true">↗</span></>}
              </button>
              <button className="reset-button" type="button" onClick={resetDefaults}>Reset to defaults</button>
            </div>
            <SettingsFields running={runState === 'loading'} errors={fieldErrors}>
              <fieldset className="preset-group">
                <legend>Try a dataset</legend>
                <div className="preset-buttons">{datasetPresets.map(preset => (
                  <button type="button" key={preset.label} onClick={() => {
                    setForm(current => ({ ...current, nSamples: DEFAULT_FORM.nSamples, xMin: DEFAULT_FORM.xMin, xMax: DEFAULT_FORM.xMax, seed: DEFAULT_FORM.seed, ...preset.dataset }))
                    setFieldErrors({})
                  }}>{preset.label}</button>
                ))}</div>
              </fieldset>
              <label className="form-field" htmlFor="dataset-kind">
                <span className="field-label">Dataset type</span>
                <select
                  id="dataset-kind"
                  value={form.datasetKind}
                  onChange={(event) => changeForm('datasetKind', event.target.value as ExperimentFormState['datasetKind'])}
                >
                  <option value="synthetic_linear">Seeded synthetic data</option>
                  <option value="points">Edit points directly</option>
                </select>
              </label>

              {form.datasetKind === 'synthetic_linear' ? (
                <>
                  <FieldGroup title="Data recipe" description="Choose the line that generated the examples. The backend creates the seeded dataset.">
                    <SliderField id="sample-count" label="Sample count" value={form.nSamples} onChange={(value) => changeForm('nSamples', value)} error={fieldErrors.nSamples} minimum={2} maximum={1000} rangeStep={1} step="1" />
                    <NumericField id="true-slope" label="True slope" value={form.trueSlope} onChange={(value) => changeForm('trueSlope', value)} error={fieldErrors.trueSlope} hint="How quickly y changes as x increases." />
                    <NumericField id="true-intercept" label="True intercept" value={form.trueIntercept} onChange={(value) => changeForm('trueIntercept', value)} error={fieldErrors.trueIntercept} hint="The target value when x is zero." />
                    <SliderField id="noise-std" label="Noise level" value={form.noiseStd} onChange={(value) => changeForm('noiseStd', value)} error={fieldErrors.noiseStd} hint="Standard deviation of the added noise." minimum={0} maximum={1000} rangeStep={0.1} step="any" />
                    <NumericField id="feature-min" label="Feature minimum" value={form.xMin} onChange={(value) => changeForm('xMin', value)} error={fieldErrors.xMin} />
                    <NumericField id="feature-max" label="Feature maximum" value={form.xMax} onChange={(value) => changeForm('xMax', value)} error={fieldErrors.xMax} />
                    <NumericField id="random-seed" label="Random seed" value={form.seed} onChange={(value) => changeForm('seed', value)} error={fieldErrors.seed} hint="Reuse a seed to reproduce this dataset." step="1" />
                  </FieldGroup>
                </>
              ) : (
                <FieldGroup title="Your observations" description="Enter comma-separated x values and matching continuous y targets.">
                  <label className="form-field full-width" htmlFor="inline-x">
                    <span className="field-label">Feature values · x</span>
                    <textarea id="inline-x" rows={3} value={form.inlineX} onChange={(event) => changeForm('inlineX', event.target.value)} aria-invalid={Boolean(fieldErrors.inlineX)} aria-describedby={fieldErrors.inlineX ? 'inline-x-error' : undefined} />
                    {fieldErrors.inlineX && <span id="inline-x-error" className="field-error">{fieldErrors.inlineX}</span>}
                  </label>
                  <label className="form-field full-width" htmlFor="inline-y">
                    <span className="field-label">Target values · y</span>
                    <textarea id="inline-y" rows={3} value={form.inlineY} onChange={(event) => changeForm('inlineY', event.target.value)} aria-invalid={Boolean(fieldErrors.inlineY)} aria-describedby={fieldErrors.inlineY ? 'inline-y-error' : undefined} />
                    {fieldErrors.inlineY && <span id="inline-y-error" className="field-error">{fieldErrors.inlineY}</span>}
                  </label>
                </FieldGroup>
              )}

              <FieldGroup title="Training settings" description="Start a line and choose how large each learning step should be.">
                <NumericField id="initial-slope" label="Initial slope" value={form.initialSlope} onChange={(value) => changeForm('initialSlope', value)} error={fieldErrors.initialSlope} />
                <NumericField id="initial-intercept" label="Initial intercept" value={form.initialIntercept} onChange={(value) => changeForm('initialIntercept', value)} error={fieldErrors.initialIntercept} />
                <NumericField id="learning-rate" label="Learning rate" value={form.learningRate} onChange={(value) => changeForm('learningRate', value)} error={fieldErrors.learningRate} hint="Must be greater than 0 and at most 1." />
                <SliderField id="training-steps" label="Training steps" value={form.steps} onChange={(value) => changeForm('steps', value)} error={fieldErrors.steps} hint="Each step is one batch parameter update." minimum={0} maximum={500} rangeStep={1} step="1" />
              </FieldGroup>

            </SettingsFields>
          </form>
        </aside>

        <div className="results-column">
          {errorView && (
            <div className="experiment-error" role="alert">
              <span className="error-symbol" aria-hidden="true">!</span>
              <div className="error-copy">
                <div className="error-title-row"><strong>{errorView.title}</strong>{errorView.code && <code>{errorView.code}</code>}</div>
                <p>{errorView.message}</p>
                {errorView.canRetry && <button className="retry-button" type="button" onClick={retryRequest}>Retry request</button>}
              </div>
            </div>
          )}

          {runState === 'loading' && <p className="loading-note" role="status">Sending the dataset and training settings to the experiment service…</p>}

          {values ? (
            <>
              <SettingsNotice submitted={completedRun.request} draft={buildLinearRegressionRequest(form).request} />
              <section className="results-heading" aria-labelledby="results-title">
                <div>
                  <h2 id="results-title">The model learned a line</h2>
                </div>
                {completedRun.request.dataset.kind === 'synthetic_linear' && (
                  <span className="result-seed">Seed {values.seed}</span>
                )}
              </section>

              <RegressionChart response={values} />
              <div className="metric-grid" aria-label="Final model results">
                <ResultMetric label="Slope" value={formatMetric(values.model.slope)} explanation="Prediction change for a +1 step in x." />
                <ResultMetric label="Intercept" value={formatMetric(values.model.intercept)} explanation="Predicted y when x equals zero." />
                <ResultMetric label="Final MSE" value={formatMetric(values.metrics.mse)} explanation="Average squared vertical error." />
                <ResultMetric label="Updates" value={`${values.training.steps_completed}`} explanation="Batch gradient steps completed." />
              </div>

              <section className="explanation-card" aria-labelledby="explanation-title">
                <span className="explanation-mark" aria-hidden="true">i</span>
                <div>
                  <h3 id="explanation-title">Reading the result</h3>
                  <p>The fitted rule is <strong>ŷ = {formatMetric(values.model.slope, 3)}x {values.model.intercept < 0 ? '−' : '+'} {formatMetric(Math.abs(values.model.intercept), 3)}</strong>. For each one-unit increase in x, this line changes its prediction by about {formatMetric(values.model.slope, 3)}. MSE averages the squared residuals across all observations.</p>
                </div>
              </section>

              <LossChart response={values} />
            </>
          ) : runState === 'loading' ? (
            <section className="empty-result loading-result" aria-label="Experiment is running">
              <div className="empty-icon" aria-hidden="true">↗</div>
              <p className="section-kicker">Working on your experiment</p>
              <h2>Preparing the returned model</h2>
              <p>Your charts will appear after the backend returns the dataset, fitted line, and loss history.</p>
            </section>
          ) : !errorView ? (
            <section className="empty-result" aria-label="Experiment results will appear here">
              <div className="empty-icon" aria-hidden="true">↗</div>
              <p className="section-kicker">Your workspace</p>
              <h2>Ready for your first run</h2>
              <p>Start with the seeded example, or edit a small set of points. The backend will return the model and every recorded loss value.</p>
              <div className="empty-equation">ŷ = slope × x + intercept</div>
            </section>
          ) : null}
        </div>
      </div>
    </section>
  )
}
