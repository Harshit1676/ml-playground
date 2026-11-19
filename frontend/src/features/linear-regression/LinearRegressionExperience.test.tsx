import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { LinearRegressionRequest, LinearRegressionResponse } from './types'
import { LinearRegressionExperience } from './LinearRegressionExperience'

vi.mock('../../components/PlotlyFigure', () => ({
  PlotlyFigure: ({ label, data }: { label: string; data: unknown[] }) => (
    <output data-testid={label.startsWith('Scatterplot') ? 'regression-chart' : 'loss-chart'}>{JSON.stringify(data)}</output>
  ),
}))

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function responseFor(request: LinearRegressionRequest, slope = 1.25): LinearRegressionResponse {
  const x = request.dataset.kind === 'points' ? request.dataset.x : [-1, 0, 1]
  const y = request.dataset.kind === 'points' ? request.dataset.y : [-1.1, 0.8, 2.4]
  return {
    algorithm: 'linear_regression',
    seed: request.seed,
    dataset: { x, y },
    parameters: request.parameters,
    model: { slope, intercept: 0.4 },
    predictions: { y_hat: x.map((value) => slope * value + 0.4) },
    metrics: { mse: 0.72 },
    training: {
      steps_completed: request.parameters.steps,
      loss_history: Array.from({ length: request.parameters.steps + 1 }, (_, index) => 2 - index * 0.01),
    },
  }
}

function jsonResponse(payload: unknown, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => payload }
}

function requestFromFetch(fetchMock: ReturnType<typeof vi.fn>, index = 0): LinearRegressionRequest {
  const init = fetchMock.mock.calls[index][1] as RequestInit
  return JSON.parse(String(init.body)) as LinearRegressionRequest
}

function inputById(id: string): HTMLInputElement {
  const input = document.getElementById(id)
  if (!(input instanceof HTMLInputElement)) throw new Error(`Expected input #${id}`)
  return input
}

describe('LinearRegressionExperience', () => {
  it('shows a runnable seeded default experiment', () => {
    render(<LinearRegressionExperience />)

    expect(screen.getByRole('heading', { name: 'Linear Regression Lab' })).toBeInTheDocument()
    expect(screen.getByLabelText('Dataset type')).toHaveValue('synthetic_linear')
    expect(screen.getByLabelText('Sample count')).toHaveValue(60)
    expect(screen.getByLabelText(/True slope/)).toHaveValue(1.8)
    expect(screen.getByLabelText(/Learning rate/)).toHaveValue(0.01)
    expect(inputById('training-steps')).toHaveValue(100)
    expect(screen.getByRole('heading', { name: 'Ready for your first run' })).toBeInTheDocument()
  })

  it('keeps the noise slider and numeric input synchronized in both directions', () => {
    render(<LinearRegressionExperience />)

    expect(inputById('noise-std')).toHaveValue(1.2)
    expect(inputById('noise-std-slider').value).toBe('1.2')

    fireEvent.change(inputById('noise-std'), { target: { value: '2.3' } })
    expect(inputById('noise-std-slider').value).toBe('2.3')

    fireEvent.change(inputById('noise-std-slider'), { target: { value: '3.4' } })
    expect(inputById('noise-std')).toHaveValue(3.4)
  })

  it('shows the random seed only for seeded synthetic datasets', async () => {
    const user = userEvent.setup()
    render(<LinearRegressionExperience />)

    expect(screen.getByLabelText(/Random seed/)).toHaveValue(42)
    fireEvent.change(inputById('random-seed'), { target: { value: '77' } })
    await user.selectOptions(screen.getByLabelText('Dataset type'), 'points')
    expect(screen.queryByLabelText(/Random seed/)).not.toBeInTheDocument()
    expect(screen.getByLabelText('Feature values · x')).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Dataset type'), 'synthetic_linear')
    expect(screen.getByLabelText(/Random seed/)).toHaveValue(77)
  })

  it('validates form data before making an API request', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<LinearRegressionExperience />)

    await user.clear(screen.getByLabelText('Initial slope'))
    await user.click(screen.getByRole('button', { name: /run experiment/i }))

    expect(screen.getByText('Enter a finite number.')).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('sends the constructed request and shows API results and response-derived charts', async () => {
    const fetchMock = vi.fn().mockImplementation((_url: string, init: RequestInit) => {
      const request = JSON.parse(String(init.body)) as LinearRegressionRequest
      return Promise.resolve(jsonResponse(responseFor(request)))
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<LinearRegressionExperience />)

    await user.click(screen.getByRole('button', { name: /run experiment/i }))

    expect(await screen.findByRole('heading', { name: 'The model learned a line' })).toBeInTheDocument()
    expect(requestFromFetch(fetchMock)).toEqual({
      dataset: { kind: 'synthetic_linear', n_samples: 60, slope: 1.8, intercept: 0.7, noise_std: 1.2, x_min: -5, x_max: 5 },
      parameters: { initial_slope: 0, initial_intercept: 0, learning_rate: 0.01, steps: 100 },
      seed: 42,
    })
    expect(screen.getByText('1.25')).toBeInTheDocument()
    expect(screen.getByText('0.72')).toBeInTheDocument()
    expect(screen.getByTestId('regression-chart').textContent).toContain('"y":[-1.1,0.8,2.4]')
    expect(screen.getByTestId('regression-chart').textContent).toContain('"y":[-0.85,1.65]')
    expect(screen.getByTestId('loss-chart').textContent).toContain('"x":[0,1,2,3')
  })

  it('displays structured API validation errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      error: { code: 'VALIDATION_ERROR', message: 'Request validation failed.', details: { fields: [{ field: 'dataset.x', reason: 'The x and y arrays must match.' }] } },
    }, 422)))
    const user = userEvent.setup()
    render(<LinearRegressionExperience />)

    await user.click(screen.getByRole('button', { name: /run experiment/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent('VALIDATION_ERROR')
    expect(screen.getByRole('alert')).toHaveTextContent('The x and y arrays must match.')
  })

  it('shows loading state while keeping a new submission available', async () => {
    let resolveRequest!: (value: ReturnType<typeof jsonResponse>) => void
    const fetchMock = vi.fn((_url: string, init: RequestInit) => new Promise<ReturnType<typeof jsonResponse>>((resolve) => {
      resolveRequest = resolve
      void init
    }))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<LinearRegressionExperience />)

    await user.click(screen.getByRole('button', { name: /run experiment/i }))

    expect(screen.getByRole('status')).toHaveTextContent('Sending the dataset')
    expect(screen.getByRole('button', { name: /run updated settings/i })).toBeEnabled()
    await act(async () => resolveRequest(jsonResponse(responseFor(requestFromFetch(fetchMock)))))
    expect(await screen.findByRole('heading', { name: 'The model learned a line' })).toBeInTheDocument()
  })

  it('retries a network failure with the same request', async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError('offline'))
      .mockImplementationOnce((_url: string, init: RequestInit) => Promise.resolve(jsonResponse(responseFor(JSON.parse(String(init.body)) as LinearRegressionRequest))))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<LinearRegressionExperience />)

    await user.click(screen.getByRole('button', { name: /run experiment/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent('The backend could not be reached')
    await user.click(screen.getByRole('button', { name: 'Retry request' }))

    expect(await screen.findByRole('heading', { name: 'The model learned a line' })).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ body: fetchMock.mock.calls[1][1].body })
  })

  it('resets controls and result state to the defaults', async () => {
    const fetchMock = vi.fn().mockImplementation((_url: string, init: RequestInit) => Promise.resolve(jsonResponse(responseFor(JSON.parse(String(init.body)) as LinearRegressionRequest))))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<LinearRegressionExperience />)

    await user.click(screen.getByRole('button', { name: /run experiment/i }))
    expect(await screen.findByRole('heading', { name: 'The model learned a line' })).toBeInTheDocument()
    await user.clear(inputById('true-slope'))
    await user.type(inputById('true-slope'), '3.5')
    await user.click(screen.getByRole('button', { name: 'Reset to defaults' }))

    expect(inputById('true-slope')).toHaveValue(1.8)
    expect(screen.queryByRole('heading', { name: 'The model learned a line' })).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Ready for your first run' })).toBeInTheDocument()
  })

  it('ignores an older response when a newer request finishes first', async () => {
    const pending: Array<(value: ReturnType<typeof jsonResponse>) => void> = []
    const fetchMock = vi.fn((_url: string, _init: RequestInit) => new Promise((resolve) => pending.push(resolve)))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<LinearRegressionExperience />)

    await user.click(screen.getByRole('button', { name: /run experiment/i }))
    fireEvent.change(screen.getByLabelText('Initial slope'), { target: { value: '5' } })
    await user.click(screen.getByRole('button', { name: /run updated settings/i }))
    expect(pending).toHaveLength(2)

    const firstRequest = requestFromFetch(fetchMock, 0)
    const secondRequest = requestFromFetch(fetchMock, 1)
    await act(async () => pending[1](jsonResponse(responseFor(secondRequest, 9))))
    expect(await screen.findByText('9')).toBeInTheDocument()
    await act(async () => pending[0](jsonResponse(responseFor(firstRequest, 2))))

    expect(screen.getByText('9')).toBeInTheDocument()
    expect(screen.queryByText('2')).not.toBeInTheDocument()
  })

  it('does not accept a pending response after a newer invalid submission', async () => {
    let resolveRequest!: (value: ReturnType<typeof jsonResponse>) => void
    const fetchMock = vi.fn((_url: string, _init: RequestInit) => new Promise<ReturnType<typeof jsonResponse>>((resolve) => {
      resolveRequest = resolve
    }))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<LinearRegressionExperience />)

    await user.click(screen.getByRole('button', { name: /run experiment/i }))
    fireEvent.change(inputById('initial-slope'), { target: { value: '' } })
    await user.click(screen.getByRole('button', { name: /run updated settings/i }))
    expect(screen.getByRole('alert')).toHaveTextContent('FORM_VALIDATION')

    await act(async () => resolveRequest(jsonResponse(responseFor(requestFromFetch(fetchMock)))))

    expect(screen.getByRole('alert')).toHaveTextContent('FORM_VALIDATION')
    expect(screen.queryByRole('heading', { name: 'The model learned a line' })).not.toBeInTheDocument()
  })
})
