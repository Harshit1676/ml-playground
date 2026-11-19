import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('ML Playground landing page', () => {
  it('shows a loading state before the health check completes', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
    render(<App />)

    expect(screen.getByRole('status')).toHaveTextContent('Checking backend connection')
  })

  it('shows connected when the backend health check succeeds', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'ok' }),
    }))
    render(<App />)

    expect(await screen.findByText('Backend connected')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Make machine learning make sense.' })).toBeInTheDocument()
  })

  it('shows failure and retries after a failed request', async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce({ ok: true, json: async () => ({ status: 'ok' }) })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<App />)

    expect(await screen.findByRole('alert')).toHaveTextContent('Backend unavailable')
    await user.click(screen.getByRole('button', { name: /try again/i }))
    expect(await screen.findByText('Backend connected')).toBeInTheDocument()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
  })
})
