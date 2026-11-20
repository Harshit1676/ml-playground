import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  window.history.replaceState({}, '', '/')
})

describe('ML Playground application shell', () => {
  it('shows backend loading while rendering the Linear Regression lab', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
    render(<App />)

    expect(screen.getByRole('status')).toHaveTextContent('Checking backend')
    expect(screen.getByRole('heading', { name: 'Linear Regression Lab' })).toBeInTheDocument()
  })

  it('shows connected when the health endpoint succeeds', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'ok' }) }))
    render(<App />)

    expect(await screen.findByText('Backend connected')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Linear Regression Lab' })).toBeInTheDocument()
  })

  it('shows backend failure and retries the health check', async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce({ ok: true, json: async () => ({ status: 'ok' }) })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<App />)

    expect(await screen.findByRole('alert')).toHaveTextContent('Backend unavailable')
    await user.click(screen.getByRole('button', { name: 'Retry' }))
    expect(await screen.findByText('Backend connected')).toBeInTheDocument()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
  })

  it('navigates between algorithm labs with keyboard-accessible deep links', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'ok' }) }))
    const user = userEvent.setup()
    render(<App />)

    const navigation = screen.getByRole('navigation', { name: 'Choose algorithm' })
    const linearLink = screen.getByRole('link', { name: 'Linear Regression' })
    const logisticLink = screen.getByRole('link', { name: 'Logistic Regression' })
    expect(linearLink).toHaveAttribute('aria-current', 'page')
    expect(logisticLink).toHaveAttribute('href', '/?algorithm=logistic-regression')
    expect(navigation).toBeInTheDocument()

    logisticLink.focus()
    await user.keyboard('{Enter}')
    expect(window.location.search).toContain('algorithm=logistic-regression')
    expect(logisticLink).toHaveAttribute('aria-current', 'page')
    expect(screen.getByRole('heading', { name: 'Logistic Regression Lab' })).toBeInTheDocument()

    await user.click(linearLink)
    expect(window.location.search).toContain('algorithm=linear-regression')
    expect(linearLink).toHaveAttribute('aria-current', 'page')
    expect(screen.getByRole('heading', { name: 'Linear Regression Lab' })).toBeInTheDocument()
  })

  it('links each label to its own lab and excludes inactive controls from keyboard navigation', async () => {
    window.history.replaceState({}, '', '/?algorithm=unknown')
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
    const user = userEvent.setup()
    const { container } = render(<App />)
    expect(screen.getByRole('heading', { name: 'Linear Regression Lab' })).toBeVisible()
    const ids = [...container.querySelectorAll('[id]')].map(element => element.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const label of container.querySelectorAll('label[for]')) {
      expect((label as HTMLLabelElement).control?.closest('form')).toBe(label.closest('form'))
    }

    const link = screen.getByRole('link', { name: 'Logistic Regression' })
    await user.click(link)
    const activeSelect = screen.getByRole('combobox')
    expect(activeSelect).toHaveAttribute('id', 'logistic-dataset-kind')
    expect(container.querySelector('#dataset-kind')).not.toBeVisible()
    link.focus()
    await user.tab()
    expect(screen.getByRole('link', { name: 'K-Means' })).toHaveFocus()
    await user.tab()
    expect(activeSelect).toHaveFocus()
  })

  it('avoids duplicate history entries and handles history traversal', async () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
    const user = userEvent.setup()
    render(<App />)
    const link = screen.getByRole('link', { name: 'Logistic Regression' })
    await user.click(link)
    const historyLength = window.history.length
    await user.click(link)
    expect(window.history.length).toBe(historyLength)
    act(() => window.history.back())
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Linear Regression Lab' })).toBeVisible())
    act(() => window.history.forward())
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Logistic Regression Lab' })).toBeVisible())
  })

  it('opens the selected algorithm from its query-string deep link', () => {
    window.history.replaceState({}, '', '/?algorithm=logistic-regression')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'ok' }) }))
    render(<App />)

    expect(screen.getByRole('link', { name: 'Logistic Regression' })).toHaveAttribute('aria-current', 'page')
    expect(screen.getByRole('heading', { name: 'Logistic Regression Lab' })).toBeInTheDocument()
  })
})
