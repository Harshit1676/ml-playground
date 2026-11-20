import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { deriveThresholdMetrics } from './classification'
import { DEFAULT_LOGISTIC_REQUEST } from './defaults'
import type { LogisticRegressionRequest } from './types'
import { LogisticRegressionExperience } from './LogisticRegressionExperience'
import { jsonResponse, requestFromFetch, responseForLogisticRequest } from '../../test/logisticRegressionFixtures'

vi.mock('../../components/PlotlyFigure', () => ({
  PlotlyFigure: ({ label, data }: { label: string; data: unknown[] }) => (
    <output data-testid={label.startsWith('Observed binary') ? 'probability-chart' : 'bce-chart'}>{JSON.stringify(data)}</output>
  ),
}))

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function inputById(id: string): HTMLInputElement {
  const input = document.getElementById(id)
  if (!(input instanceof HTMLInputElement)) throw new Error(`Expected input #${id}`)
  return input
}

function metricValue(label: string): string | null {
  const card = screen.getByText(label, { exact: true }).closest('.metric-card')
  return card?.querySelector('strong')?.textContent ?? null
}

function displayPercent(value: number): string {
  return `${(value * 100).toLocaleString(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 0 })}%`
}

function paragraphContaining(text: string): HTMLElement {
  return screen.getByText((_, element) => element?.tagName === 'P' && (element.textContent ?? '').includes(text))
}

function successfulFetch() {
  const fetchMock = vi.fn((_url: string, init: RequestInit) => {
    const request = JSON.parse(String(init.body)) as LogisticRegressionRequest
    return Promise.resolve(jsonResponse(responseForLogisticRequest(request)))
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('LogisticRegressionExperience', () => {
  it('shows the seeded default and submits the exact backend request', async () => {
    const fetchMock = successfulFetch()
    const user = userEvent.setup()
    render(<LogisticRegressionExperience />)

    expect(screen.getByRole('heading', { name: 'Logistic Regression Lab' })).toBeInTheDocument()
    expect(screen.getByLabelText('Dataset type')).toHaveValue('synthetic_logistic')
    expect(screen.getByLabelText('Sample count')).toHaveValue(100)
    expect(screen.getByLabelText('Random seed')).toHaveValue(42)
    expect(screen.getByLabelText('Threshold sent to backend')).toHaveValue(0.5)
    expect(screen.getByRole('heading', { name: 'Ready to classify examples' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /run experiment/i }))
    expect(await screen.findByRole('heading', { name: 'Probabilities become classes' })).toBeInTheDocument()
    expect(requestFromFetch(fetchMock)).toEqual(DEFAULT_LOGISTIC_REQUEST)
    expect(screen.getByText(`Seed ${DEFAULT_LOGISTIC_REQUEST.seed}`)).toBeInTheDocument()
    expect(screen.getByTestId('probability-chart')).toBeInTheDocument()
    expect(screen.getByTestId('bce-chart')).toBeInTheDocument()
  })

  it('preserves dataset inputs when switching modes and hides the inline seed control', async () => {
    const user = userEvent.setup()
    const fetchMock = successfulFetch()
    render(<LogisticRegressionExperience />)

    fireEvent.change(inputById('logistic-random-seed'), { target: { value: '77' } })
    await user.selectOptions(screen.getByLabelText('Dataset type'), 'points')
    expect(screen.queryByLabelText('Random seed')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Feature values · x')).toHaveValue('-3, -2, -1, 1, 2, 3')
    expect(screen.getByLabelText('Binary labels · y')).toHaveValue('0, 0, 0, 1, 1, 1')

    fireEvent.change(screen.getByLabelText('Feature values · x'), { target: { value: '-2, 2' } })
    fireEvent.change(screen.getByLabelText('Binary labels · y'), { target: { value: '0, 1' } })
    await user.selectOptions(screen.getByLabelText('Dataset type'), 'synthetic_logistic')
    expect(screen.getByLabelText('Random seed')).toHaveValue(77)
    await user.selectOptions(screen.getByLabelText('Dataset type'), 'points')
    expect(screen.getByLabelText('Feature values · x')).toHaveValue('-2, 2')
    expect(screen.getByLabelText('Binary labels · y')).toHaveValue('0, 1')

    await user.click(screen.getByRole('button', { name: /run experiment/i }))
    expect(await screen.findByRole('heading', { name: 'Probabilities become classes' })).toBeInTheDocument()
    expect(requestFromFetch(fetchMock)).toMatchObject({ dataset: { kind: 'points', x: [-2, 2], y: [0, 1] }, seed: 42 })
  })

  it('rejects inline non-binary labels without sending a request', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<LogisticRegressionExperience />)

    await user.selectOptions(screen.getByLabelText('Dataset type'), 'points')
    fireEvent.change(screen.getByLabelText('Binary labels · y'), { target: { value: '0, 0.5, 1' } })
    await user.click(screen.getByRole('button', { name: /run experiment/i }))

    expect(screen.getByRole('alert')).toHaveTextContent('FORM_VALIDATION')
    expect(screen.getByText('Each label must be exactly 0 or 1.')).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('clears old metrics and charts after a newer invalid submission', async () => {
    const fetchMock = successfulFetch()
    const user = userEvent.setup()
    render(<LogisticRegressionExperience />)

    await user.click(screen.getByRole('button', { name: /run experiment/i }))
    expect(await screen.findByRole('heading', { name: 'Probabilities become classes' })).toBeInTheDocument()
    expect(screen.getByTestId('probability-chart')).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Dataset type'), 'points')
    fireEvent.change(screen.getByLabelText('Binary labels · y'), { target: { value: '0, 0.5, 1' } })
    await user.click(screen.getByRole('button', { name: /run experiment/i }))

    expect(screen.getByRole('alert')).toHaveTextContent('FORM_VALIDATION')
    expect(screen.queryByText('Binary cross-entropy', { selector: '.metric-card p' })).not.toBeInTheDocument()
    expect(screen.queryByTestId('probability-chart')).not.toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('reports a structured generated single-class error without retrying the same seed', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      error: { code: 'SINGLE_CLASS_DATASET', message: 'Seeded data produced only class 0; change the seed or generator settings.', details: { class_label: 0, n_samples: 100 } },
    }, 422)))
    const user = userEvent.setup()
    render(<LogisticRegressionExperience />)

    await user.click(screen.getByRole('button', { name: /run experiment/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent('SINGLE_CLASS_DATASET')
    expect(screen.getByRole('alert')).toHaveTextContent('change the seed or generator settings')
    expect(screen.queryByRole('button', { name: 'Retry request' })).not.toBeInTheDocument()
  })

  it('retries network failures using the previous valid request', async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError('offline'))
      .mockImplementationOnce((_url: string, init: RequestInit) => {
        const request = JSON.parse(String(init.body)) as LogisticRegressionRequest
        return Promise.resolve(jsonResponse(responseForLogisticRequest(request)))
      })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<LogisticRegressionExperience />)

    await user.click(screen.getByRole('button', { name: /run experiment/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent('The backend could not be reached')
    await user.click(screen.getByRole('button', { name: 'Retry request' }))
    expect(await screen.findByRole('heading', { name: 'Probabilities become classes' })).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ body: fetchMock.mock.calls[1][1].body })
  })

  it('updates threshold metrics, labels, and guide without retraining or changing model results', async () => {
    const fetchMock = successfulFetch()
    const user = userEvent.setup()
    render(<LogisticRegressionExperience />)
    await user.click(screen.getByRole('button', { name: /run experiment/i }))
    expect(await screen.findByRole('heading', { name: 'Probabilities become classes' })).toBeInTheDocument()

    const response = responseForLogisticRequest(DEFAULT_LOGISTIC_REQUEST)
    const originalCoefficient = metricValue('Coefficient')
    const originalBce = metricValue('Binary cross-entropy')
    const originalLoss = response.training.loss_history
    const expectedAtBackendThreshold = deriveThresholdMetrics(response.dataset.y, response.predictions.probabilities, response.threshold)
    expect(metricValue('Accuracy')).toBe(displayPercent(expectedAtBackendThreshold.accuracy))
    expect(metricValue('Precision')).toBe(displayPercent(expectedAtBackendThreshold.precision))
    expect(metricValue('Recall')).toBe(displayPercent(expectedAtBackendThreshold.recall))
    expect(metricValue('F1')).toBe(displayPercent(expectedAtBackendThreshold.f1))

    fireEvent.change(screen.getByLabelText('Explore classification threshold slider'), { target: { value: '0.8' } })
    await waitFor(() => expect(inputById('exploration-threshold').value).toBe('0.8'))
    const expected = deriveThresholdMetrics(response.dataset.y, response.predictions.probabilities, 0.8)
    expect(metricValue('Accuracy')).toBe(displayPercent(expected.accuracy))
    expect(metricValue('Precision')).toBe(displayPercent(expected.precision))
    expect(metricValue('Recall')).toBe(displayPercent(expected.recall))
    expect(metricValue('F1')).toBe(displayPercent(expected.f1))
    expect(paragraphContaining('Submitted threshold: 0.5')).toBeInTheDocument()
    expect(paragraphContaining('Exploring now: 0.8')).toBeInTheDocument()
    expect(screen.getByTestId('probability-chart').textContent).toContain('Threshold 0.8')
    expect(metricValue('Coefficient')).toBe(originalCoefficient)
    expect(metricValue('Binary cross-entropy')).toBe(originalBce)
    expect(screen.getByTestId('bce-chart').textContent).toContain(JSON.stringify(originalLoss))
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('reclassifies four known probabilities while preserving every fitted result', async () => {
    const probabilities = [0.2, 0.4, 0.6, 0.8]
    const x = probabilities.map(p => Math.log(p / (1 - p)))
    const fetchMock = vi.fn((_url: string, init: RequestInit) => {
      const result = responseForLogisticRequest(JSON.parse(String(init.body)))
      result.predictions.probabilities = [...probabilities]
      return Promise.resolve(jsonResponse(result))
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    const { container } = render(<LogisticRegressionExperience />)
    await user.selectOptions(screen.getByLabelText('Dataset type'), 'points')
    fireEvent.change(container.querySelector('#logistic-inline-x')!, { target: { value: x.join(',') } })
    fireEvent.change(container.querySelector('#logistic-inline-y')!, { target: { value: '0,1,0,1' } })
    fireEvent.change(inputById('initial-coefficient'), { target: { value: '1' } })
    fireEvent.change(inputById('logistic-training-steps'), { target: { value: '0' } })
    await user.click(screen.getByRole('button', { name: /run experiment/i }))
    await screen.findByRole('heading', { name: 'Probabilities become classes' })

    const rows = () => [...container.querySelectorAll<HTMLTableRowElement>('.classification-table tbody tr')]
    const predictions = () => rows().map(row => Number(row.cells[3].textContent))
    const matrix = () => [...container.querySelectorAll('.confusion-table td strong')].map(cell => Number(cell.textContent))
    const metrics = () => ['Accuracy', 'Precision', 'Recall', 'F1'].map(metricValue)
    const fitted = () => ({
      model: ['Coefficient', 'Intercept', 'Binary cross-entropy'].map(metricValue),
      probabilities: rows().map(row => Number(row.cells[1].textContent)),
      curve: JSON.parse(screen.getByTestId('probability-chart').textContent!)[2],
      history: screen.getByTestId('bce-chart').textContent,
    })
    const original = fitted()
    expect(original.probabilities).toEqual(probabilities)
    expect(predictions()).toEqual([0, 0, 1, 1])
    expect(matrix()).toEqual([1, 1, 1, 1]) // TN, FP, FN, TP
    expect(metrics()).toEqual(['50%', '50%', '50%', '50%'])

    fireEvent.change(inputById('exploration-threshold-slider'), { target: { value: '0.7' } })
    expect(screen.getByText('1 of 4 predicted positive · Predictions changed from the submitted run: 1.')).toBeVisible()
    expect(screen.queryByText(/Showing the previous settings/)).toBeNull()
    expect(predictions()).toEqual([0, 0, 0, 1])
    expect(matrix()).toEqual([2, 0, 1, 1])
    expect(metrics()).toEqual(['75%', '100%', '50%', displayPercent(2 / 3)])
    expect(JSON.parse(screen.getByTestId('probability-chart').textContent!)[3].y).toEqual([0.7, 0.7])
    expect(fitted()).toEqual(original)

    fireEvent.change(inputById('exploration-threshold'), { target: { value: '0.800001' } })
    expect(paragraphContaining('Exploring now: 0.800001')).toBeInTheDocument()
    expect(predictions()).toEqual([0, 0, 0, 0])
    expect(screen.getByTestId('probability-chart')).toHaveTextContent('Threshold 0.800001')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('resets controls and removes prior results', async () => {
    const fetchMock = successfulFetch()
    const user = userEvent.setup()
    render(<LogisticRegressionExperience />)
    await user.click(screen.getByRole('button', { name: /run experiment/i }))
    expect(await screen.findByRole('heading', { name: 'Probabilities become classes' })).toBeInTheDocument()
    await user.clear(screen.getByLabelText('True coefficient'))
    await user.type(screen.getByLabelText('True coefficient'), '-2')
    await user.click(screen.getByRole('button', { name: 'Reset to defaults' }))

    expect(screen.getByLabelText('True coefficient')).toHaveValue(1.5)
    expect(screen.getByLabelText('Random seed')).toHaveValue(42)
    expect(screen.queryByRole('heading', { name: 'Probabilities become classes' })).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Ready to classify examples' })).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('ignores a stale response when a newer run completes first', async () => {
    const pending: Array<(value: ReturnType<typeof jsonResponse>) => void> = []
    const fetchMock = vi.fn((_url: string, _init: RequestInit) => new Promise((resolve) => pending.push(resolve)))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<LogisticRegressionExperience />)

    await user.click(screen.getByRole('button', { name: /run experiment/i }))
    fireEvent.change(inputById('initial-coefficient'), { target: { value: '2' } })
    await user.click(screen.getByRole('button', { name: /running experiment/i }))
    expect(pending).toHaveLength(2)
    expect((fetchMock.mock.calls[0][1] as RequestInit).signal?.aborted).toBe(true)
    expect((fetchMock.mock.calls[1][1] as RequestInit).signal?.aborted).toBe(false)

    const oldRequest = requestFromFetch(fetchMock, 0)
    const currentRequest = requestFromFetch(fetchMock, 1)
    const oldResponse = responseForLogisticRequest(oldRequest)
    const currentResponse = responseForLogisticRequest(currentRequest)
    await act(async () => pending[1](jsonResponse(currentResponse)))
    expect(await screen.findByText(currentResponse.model.coefficient.toLocaleString(undefined, { maximumFractionDigits: 4, minimumFractionDigits: 0 }))).toBeInTheDocument()
    await act(async () => pending[0](jsonResponse(oldResponse)))

    expect(metricValue('Coefficient')).toBe(currentResponse.model.coefficient.toLocaleString(undefined, { maximumFractionDigits: 4, minimumFractionDigits: 0 }))
    expect(metricValue('Coefficient')).not.toBe(oldResponse.model.coefficient.toLocaleString(undefined, { maximumFractionDigits: 4, minimumFractionDigits: 0 }))
  })

  it('aborts an active experiment when the lab unmounts', async () => {
    const fetchMock = vi.fn((_url: string, _init: RequestInit) => new Promise(() => {}))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    const { unmount } = render(<LogisticRegressionExperience />)

    await user.click(screen.getByRole('button', { name: /run experiment/i }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    const signal = (fetchMock.mock.calls[0][1] as RequestInit).signal

    unmount()
    expect(signal?.aborted).toBe(true)
  })
})

it('keeps the trained chart visible and identifies unsent training edits', async () => {
  const fetcher = successfulFetch()
  render(<LogisticRegressionExperience />)
  fireEvent.click(screen.getByRole('button', { name: /run experiment/i }))
  await screen.findByTestId('probability-chart')
  fireEvent.change(inputById('logistic-learning-rate'), { target: { value: '' } })
  expect(screen.getByText(/Showing the previous settings/)).toBeVisible()
  expect(screen.getByTestId('probability-chart')).toBeVisible()
  expect(fetcher).toHaveBeenCalledTimes(1)
})
