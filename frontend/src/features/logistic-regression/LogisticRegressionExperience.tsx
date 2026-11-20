import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { LogisticRegressionApiError, LogisticRegressionNetworkError, runLogisticRegression } from '../../lib/api/logisticRegression'
import { deriveThresholdMetrics } from './classification'
import { DEFAULT_LOGISTIC_FORM } from './defaults'
import type { LogisticExperimentFormState } from './defaults'
import { buildLogisticRegressionRequest } from './request'
import type { LogisticFormErrors } from './request'
import type { LogisticRegressionRequest, LogisticRegressionResponse } from './types'
import { LogisticRegressionCharts } from './LogisticRegressionCharts'

type RunState = 'idle' | 'loading' | 'success' | 'error'

interface CompletedRun {
  request: LogisticRegressionRequest
  response: LogisticRegressionResponse
}

interface ErrorView {
  title: string
  message: string
  code?: string
  canRetry: boolean
}

interface NumberFieldProps {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  error?: string
  hint?: string
  step?: string
  min?: number
  max?: number
}

function NumberField({ id, label, value, onChange, error, hint, step = 'any', min, max }: NumberFieldProps) {
  const describedBy = [hint ? `${id}-hint` : null, error ? `${id}-error` : null].filter(Boolean).join(' ') || undefined
  return (
    <div className="form-field">
      <label className="field-label" htmlFor={id}>{label}</label>
      <input id={id} type="number" step={step} min={min} max={max} value={value} onChange={(event) => onChange(event.target.value)} aria-invalid={Boolean(error)} aria-describedby={describedBy} />
      {hint && <span className="field-hint" id={`${id}-hint`}>{hint}</span>}
      {error && <span className="field-error" id={`${id}-error`}>{error}</span>}
    </div>
  )
}

interface RangeFieldProps extends NumberFieldProps {
  minimum: number
  maximum: number
  rangeStep: number
}

function RangeField({ id, label, value, onChange, error, hint, step, min, max, minimum, maximum, rangeStep }: RangeFieldProps) {
  const numeric = Number(value)
  const sliderValue = Number.isFinite(numeric) ? Math.min(maximum, Math.max(minimum, numeric)) : minimum
  return (
    <div className="slider-control">
      <div className="slider-input-line">
        <label className="visually-hidden" htmlFor={`${id}-slider`}>{label} slider</label>
        <input id={`${id}-slider`} type="range" min={minimum} max={maximum} step={rangeStep} value={sliderValue} onChange={(event) => onChange(event.target.value)} />
        <NumberField id={id} label={label} value={value} onChange={onChange} error={error} hint={hint} step={step} min={min ?? minimum} max={max ?? maximum} />
      </div>
    </div>
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

function MetricCard({ label, value, explanation }: { label: string; value: string; explanation: string }) {
  return <article className="metric-card"><p>{label}</p><strong>{value}</strong><span>{explanation}</span></article>
}

function formatMetric(value: number, digits = 4): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 })
}

function formatPercent(value: number): string {
  return `${formatMetric(value * 100, 1)}%`
}

function explainError(error: unknown): ErrorView {
  if (error instanceof LogisticRegressionNetworkError) {
    return { title: 'The backend could not be reached', message: error.message, code: 'NETWORK_ERROR', canRetry: true }
  }
  if (error instanceof LogisticRegressionApiError) {
    const validationFields = error.details?.fields
    const firstField = Array.isArray(validationFields) ? validationFields[0] : undefined
    const message = error.code === 'VALIDATION_ERROR' && typeof firstField === 'object' && firstField !== null && 'reason' in firstField && typeof firstField.reason === 'string'
      ? `${error.message} ${firstField.reason}`
      : error.message
    const singleClass = error.code === 'SINGLE_CLASS_DATASET'
    return {
      title: singleClass ? 'The generated sample contains one class' : error.status >= 500 ? 'The experiment service is unavailable' : 'The experiment could not be run',
      message,
      code: error.code,
      canRetry: error.status >= 500,
    }
  }
  return { title: 'The experiment could not be run', message: 'An unexpected error occurred. Check your settings and try again.', canRetry: true }
}

function confusionCell(value: number, abbreviation: string) {
  return <td><strong>{value}</strong><span className="confusion-abbreviation">{abbreviation}</span></td>
}

export function LogisticRegressionExperience() {
  const [form, setForm] = useState<LogisticExperimentFormState>({ ...DEFAULT_LOGISTIC_FORM })
  const [fieldErrors, setFieldErrors] = useState<LogisticFormErrors>({})
  const [runState, setRunState] = useState<RunState>('idle')
  const [errorView, setErrorView] = useState<ErrorView | null>(null)
  const [completedRun, setCompletedRun] = useState<CompletedRun | null>(null)
  const [explorationThreshold, setExplorationThreshold] = useState(0.5)
  const [explorationInput, setExplorationInput] = useState('0.5')
  const [explorationError, setExplorationError] = useState<string | null>(null)
  const requestVersion = useRef(0)
  const activeController = useRef<AbortController | null>(null)
  const lastRequest = useRef<LogisticRegressionRequest | null>(null)

  useEffect(() => () => {
    requestVersion.current += 1
    activeController.current?.abort()
  }, [])

  function changeForm<K extends keyof LogisticExperimentFormState>(key: K, value: LogisticExperimentFormState[K]) {
    setForm((current) => ({ ...current, [key]: value }))
    setFieldErrors((current) => {
      if (!current[key]) return current
      const next = { ...current }
      delete next[key]
      return next
    })
  }

  function setExplorationValue(value: string) {
    setExplorationInput(value)
    const threshold = Number(value)
    if (value.trim() === '' || !Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
      setExplorationError('Enter a threshold from 0 to 1.')
      return
    }
    setExplorationError(null)
    setExplorationThreshold(threshold)
  }

  async function execute(request: LogisticRegressionRequest) {
    activeController.current?.abort()
    const controller = new AbortController()
    activeController.current = controller
    const version = ++requestVersion.current
    setRunState('loading')
    setErrorView(null)
    setCompletedRun(null)

    try {
      const response = await runLogisticRegression(request, controller.signal)
      if (version !== requestVersion.current) return
      setCompletedRun({ request, response })
      setExplorationThreshold(response.threshold)
      setExplorationInput(String(response.threshold))
      setExplorationError(null)
      setRunState('success')
    } catch (error) {
      if (version !== requestVersion.current || controller.signal.aborted) return
      setErrorView(explainError(error))
      setRunState('error')
    } finally {
      if (version === requestVersion.current) activeController.current = null
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    requestVersion.current += 1
    activeController.current?.abort()
    activeController.current = null
    lastRequest.current = null
    const built = buildLogisticRegressionRequest(form)
    setFieldErrors(built.errors)
    setErrorView(null)
    if (!built.request) {
      setCompletedRun(null)
      setRunState('error')
      setErrorView({ title: 'A few settings need attention', message: 'Correct the highlighted fields, then run the experiment again.', code: 'FORM_VALIDATION', canRetry: false })
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
    setForm({ ...DEFAULT_LOGISTIC_FORM })
    setFieldErrors({})
    setRunState('idle')
    setErrorView(null)
    setCompletedRun(null)
    setExplorationThreshold(0.5)
    setExplorationInput('0.5')
    setExplorationError(null)
  }

  const response = completedRun?.response
  const thresholdMetrics = useMemo(() => response
    ? deriveThresholdMetrics(response.dataset.y, response.predictions.probabilities, explorationThreshold)
    : null, [response, explorationThreshold])
  const actualClasses = response ? new Set(response.dataset.y) : new Set<number>()

  return (
    <section className="experiment-section logistic-section" aria-labelledby="logistic-title">
      <header className="experiment-heading">
        <div>
          <p className="eyebrow"><span className="eyebrow-line" /> Your next model</p>
          <h1 id="logistic-title">Logistic Regression <em>Lab</em></h1>
          <p className="experiment-intro">Learn how a model turns a feature into a probability, then explore how a threshold turns that probability into a class.</p>
        </div>
        <div className="lesson-pill"><span aria-hidden="true">02</span><span>One feature · Two classes</span></div>
      </header>

      <div className="experiment-layout">
        <aside className="controls-panel" aria-label="Logistic Regression experiment settings">
          <div className="panel-heading">
            <div><p className="section-kicker">Set up your experiment</p><h2>Choose your data</h2></div>
            <span className="step-index">B</span>
          </div>

          <form onSubmit={handleSubmit} noValidate>
            <label className="form-field" htmlFor="logistic-dataset-kind">
              <span className="field-label">Dataset type</span>
              <select id="logistic-dataset-kind" value={form.datasetKind} onChange={(event) => changeForm('datasetKind', event.target.value as LogisticExperimentFormState['datasetKind'])}>
                <option value="synthetic_logistic">Seeded synthetic data</option>
                <option value="points">Edit binary observations</option>
              </select>
            </label>

            {form.datasetKind === 'synthetic_logistic' ? (
              <FieldGroup title="Data recipe" description="The service samples features and draws labels from the configured sigmoid probabilities.">
                <RangeField id="logistic-sample-count" label="Sample count" value={form.nSamples} onChange={(value) => changeForm('nSamples', value)} error={fieldErrors.nSamples} minimum={2} maximum={1000} rangeStep={1} step="1" min={2} max={1000} />
                <NumberField id="true-coefficient" label="True coefficient" value={form.trueCoefficient} onChange={(value) => changeForm('trueCoefficient', value)} error={fieldErrors.trueCoefficient} hint="Controls how the true probability changes with x." min={-1_000_000} max={1_000_000} />
                <NumberField id="logistic-true-intercept" label="True intercept" value={form.trueIntercept} onChange={(value) => changeForm('trueIntercept', value)} error={fieldErrors.trueIntercept} hint="Sets the true log-odds when x is zero." min={-1_000_000} max={1_000_000} />
                <NumberField id="logistic-feature-min" label="Feature minimum" value={form.xMin} onChange={(value) => changeForm('xMin', value)} error={fieldErrors.xMin} min={-1_000_000} max={1_000_000} />
                <NumberField id="logistic-feature-max" label="Feature maximum" value={form.xMax} onChange={(value) => changeForm('xMax', value)} error={fieldErrors.xMax} min={-1_000_000} max={1_000_000} />
                <NumberField id="logistic-random-seed" label="Random seed" value={form.seed} onChange={(value) => changeForm('seed', value)} error={fieldErrors.seed} hint="Reuse a seed to reproduce the generated sample." step="1" min={0} max={4_294_967_295} />
              </FieldGroup>
            ) : (
              <FieldGroup title="Your observations" description="Enter one feature value and one binary label (0 or 1) per observation.">
                <label className="form-field full-width" htmlFor="logistic-inline-x">
                  <span className="field-label">Feature values · x</span>
                  <textarea id="logistic-inline-x" rows={3} value={form.inlineX} onChange={(event) => changeForm('inlineX', event.target.value)} aria-invalid={Boolean(fieldErrors.inlineX)} aria-describedby={fieldErrors.inlineX ? 'logistic-inline-x-error' : undefined} />
                  {fieldErrors.inlineX && <span id="logistic-inline-x-error" className="field-error">{fieldErrors.inlineX}</span>}
                </label>
                <label className="form-field full-width" htmlFor="logistic-inline-y">
                  <span className="field-label">Binary labels · y</span>
                  <textarea id="logistic-inline-y" rows={3} value={form.inlineY} onChange={(event) => changeForm('inlineY', event.target.value)} aria-invalid={Boolean(fieldErrors.inlineY)} aria-describedby={fieldErrors.inlineY ? 'logistic-inline-y-error' : undefined} />
                  {fieldErrors.inlineY && <span id="logistic-inline-y-error" className="field-error">{fieldErrors.inlineY}</span>}
                </label>
              </FieldGroup>
            )}

            <FieldGroup title="Training settings" description="Choose a starting boundary and how many batch updates to run.">
              <NumberField id="initial-coefficient" label="Initial coefficient" value={form.initialCoefficient} onChange={(value) => changeForm('initialCoefficient', value)} error={fieldErrors.initialCoefficient} min={-1_000_000} max={1_000_000} />
              <NumberField id="initial-logistic-intercept" label="Initial intercept" value={form.initialIntercept} onChange={(value) => changeForm('initialIntercept', value)} error={fieldErrors.initialIntercept} min={-1_000_000} max={1_000_000} />
              <NumberField id="logistic-learning-rate" label="Learning rate" value={form.learningRate} onChange={(value) => changeForm('learningRate', value)} error={fieldErrors.learningRate} hint="Must be greater than 0 and at most 1." min={0} max={1} />
              <RangeField id="logistic-training-steps" label="Training updates" value={form.steps} onChange={(value) => changeForm('steps', value)} error={fieldErrors.steps} hint="Loss history includes step 0 plus each update." minimum={0} maximum={500} rangeStep={1} step="1" min={0} max={500} />
              <RangeField id="submitted-threshold" label="Threshold sent to backend" value={form.threshold} onChange={(value) => changeForm('threshold', value)} error={fieldErrors.threshold} hint="Class 1 when probability is greater than or equal to this value." minimum={0} maximum={1} rangeStep={0.01} step="any" min={0} max={1} />
            </FieldGroup>

            <div className="form-actions">
              <button className="run-button" type="submit">{runState === 'loading' ? <><span className="button-spinner" aria-hidden="true" /> Running experiment</> : <>Run experiment <span aria-hidden="true">↗</span></>}</button>
              <button className="reset-button" type="button" onClick={resetDefaults}>Reset to defaults</button>
            </div>
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

          {runState === 'loading' && <p className="loading-note" role="status">Sending this dataset and its training settings to FastAPI…</p>}

          {response && thresholdMetrics ? (
            <>
              <section className="results-heading" aria-labelledby="logistic-results-title">
                <div>
                  <p className="section-kicker">Your result</p>
                  <h2 id="logistic-results-title">Probabilities become classes</h2>
                  <p>The model probabilities and BCE remain fixed while you explore a different classification threshold.</p>
                  {completedRun?.request.dataset.kind === 'points' && <p className="result-context">Manually entered observations</p>}
                </div>
                {completedRun?.request.dataset.kind === 'synthetic_logistic' && <span className="result-seed">Seed {response.seed}</span>}
              </section>

              <div className="threshold-explorer" aria-labelledby="threshold-explorer-title">
                <div className="threshold-explorer-copy">
                  <p className="section-kicker">Explore without retraining</p>
                  <h3 id="threshold-explorer-title">Move the classification threshold</h3>
                  <p>Submitted threshold: {response.threshold} · Exploring now: <strong aria-live="polite">{explorationThreshold}</strong></p>
                </div>
                <div className="threshold-explorer-controls">
                  <label className="visually-hidden" htmlFor="exploration-threshold-slider">Explore classification threshold slider</label>
                  <input id="exploration-threshold-slider" type="range" min="0" max="1" step="0.01" value={explorationThreshold} onChange={(event) => setExplorationValue(event.target.value)} aria-describedby={explorationError ? 'exploration-threshold-error' : undefined} />
                  <label className="form-field" htmlFor="exploration-threshold">
                    <span className="field-label">Explore threshold</span>
                    <input id="exploration-threshold" type="number" min="0" max="1" step="any" value={explorationInput} onChange={(event) => setExplorationValue(event.target.value)} aria-invalid={Boolean(explorationError)} aria-describedby={explorationError ? 'exploration-threshold-error' : undefined} />
                  </label>
                  {explorationError && <span className="field-error" id="exploration-threshold-error">{explorationError}</span>}
                </div>
              </div>

              <section className="metric-grid logistic-metric-grid" aria-label="Classification and training results" aria-live="polite">
                <MetricCard label="Coefficient" value={formatMetric(response.model.coefficient)} explanation="How x changes the model's log-odds." />
                <MetricCard label="Intercept" value={formatMetric(response.model.intercept)} explanation="The model's log-odds when x is zero." />
                <MetricCard label="Binary cross-entropy" value={formatMetric(response.metrics.binary_cross_entropy)} explanation="Training loss from the backend response." />
                <MetricCard label="Accuracy" value={formatPercent(thresholdMetrics.accuracy)} explanation="Share of observations classified correctly." />
                <MetricCard label="Precision" value={formatPercent(thresholdMetrics.precision)} explanation="Of predicted positives, the share that are positive." />
                <MetricCard label="Recall" value={formatPercent(thresholdMetrics.recall)} explanation="Of actual positives, the share found by the model." />
                <MetricCard label="F1" value={formatPercent(thresholdMetrics.f1)} explanation="Balance between precision and recall." />
                <MetricCard label="Training updates" value={String(response.training.steps_completed)} explanation="Batch gradient steps completed by the backend." />
              </section>

              <section className="confusion-card" aria-labelledby="confusion-title">
                <div className="confusion-heading">
                  <div><p className="section-kicker">Current threshold: {explorationThreshold}</p><h3 id="confusion-title">Confusion matrix</h3></div>
                  <span className="chart-unit">Rows: actual · Columns: predicted</span>
                </div>
                <div className="confusion-table-scroll">
                  <table className="confusion-table">
                    <caption className="visually-hidden">Counts of actual versus predicted classes at the explored threshold</caption>
                    <thead><tr><th scope="col">Actual / predicted</th><th scope="col">Predicted 0</th><th scope="col">Predicted 1</th></tr></thead>
                    <tbody>
                      <tr><th scope="row">Actual 0</th>{confusionCell(thresholdMetrics.confusionMatrix.true_negative, 'TN')}{confusionCell(thresholdMetrics.confusionMatrix.false_positive, 'FP')}</tr>
                      <tr><th scope="row">Actual 1</th>{confusionCell(thresholdMetrics.confusionMatrix.false_negative, 'FN')}{confusionCell(thresholdMetrics.confusionMatrix.true_positive, 'TP')}</tr>
                    </tbody>
                  </table>
                </div>
                <ul className="confusion-definitions">
                  <li><strong>TP</strong> — actual 1, predicted 1.</li>
                  <li><strong>TN</strong> — actual 0, predicted 0.</li>
                  <li><strong>FP</strong> — actual 0, predicted 1.</li>
                  <li><strong>FN</strong> — actual 1, predicted 0.</li>
                </ul>
                {actualClasses.size === 1 && (
                  <p className="single-class-note">These observations contain only one actual class. If precision, recall, or F1 has a zero denominator, this lab follows the backend convention and displays 0% for that metric.</p>
                )}
              </section>

              <section className="explanation-card" aria-labelledby="logistic-reading-title">
                <span className="explanation-mark" aria-hidden="true">i</span>
                <div>
                  <h3 id="logistic-reading-title">Probability is not the predicted class</h3>
                  <p>The sigmoid maps the score <strong>z = coefficient × x + intercept</strong> to a probability between 0 and 1. The threshold then turns that probability into class 0 or 1. Moving the threshold changes these labels and their metrics; it does not change the learned model, probabilities, BCE, or training history. Accuracy is only one view of classification quality, so compare it with precision and recall.</p>
                </div>
              </section>

              <LogisticRegressionCharts response={response} threshold={explorationThreshold} predictedClasses={thresholdMetrics.predictedClasses} />

              <details className="classification-details">
                <summary>Show predicted labels for {response.dataset.x.length} observations</summary>
                <div className="classification-table-scroll">
                  <table className="classification-table">
                    <caption className="visually-hidden">Feature values, backend probabilities, actual labels, and classes predicted at the explored threshold</caption>
                    <thead><tr><th scope="col">Feature x</th><th scope="col">Probability</th><th scope="col">Actual label</th><th scope="col">Predicted class</th></tr></thead>
                    <tbody>{response.dataset.x.map((feature, index) => (
                      <tr key={`${feature}-${index}`}>
                        <td>{formatMetric(feature, 4)}</td>
                        <td>{String(response.predictions.probabilities[index])}</td>
                        <td>{response.dataset.y[index]}</td>
                        <td>{thresholdMetrics.predictedClasses[index]}</td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
              </details>
            </>
          ) : runState === 'loading' ? (
            <section className="empty-result loading-result" aria-label="Experiment is running">
              <div className="empty-icon" aria-hidden="true">↗</div><p className="section-kicker">Training the classifier</p><h2>Preparing your probabilities</h2>
              <p>The charts and metrics will appear when the backend returns the fitted model and loss history.</p>
            </section>
          ) : !errorView ? (
            <section className="empty-result" aria-label="Experiment results will appear here">
              <div className="empty-icon" aria-hidden="true">↗</div><p className="section-kicker">Your workspace</p><h2>Ready to classify examples</h2>
              <p>Run the seeded dataset or enter binary observations. The response includes probabilities, thresholded classes, metrics, and each recorded BCE value.</p>
              <div className="empty-equation">p(class 1) = sigmoid(coefficient × x + intercept)</div>
            </section>
          ) : null}
        </div>
      </div>
    </section>
  )
}
