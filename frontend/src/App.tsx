import { useEffect, useState } from 'react'
import type { MouseEvent } from 'react'
import { LinearRegressionExperience } from './features/linear-regression/LinearRegressionExperience'
import { LogisticRegressionExperience } from './features/logistic-regression/LogisticRegressionExperience'
import { getHealth } from './lib/api/health'
import { KMeansExperience } from './features/k-means/KMeansExperience'

type ConnectionState = 'loading' | 'connected' | 'failed'
type Algorithm = 'linear-regression' | 'logistic-regression' | 'k-means'

function algorithmFromLocation(): Algorithm {
  const value = new URLSearchParams(window.location.search).get('algorithm')
  return value === 'logistic-regression' || value === 'k-means' ? value : 'linear-regression'
}

function algorithmHref(algorithm: Algorithm): string {
  const url = new URL(window.location.href)
  url.searchParams.set('algorithm', algorithm)
  return url.pathname + url.search + url.hash
}

function App() {
  const [connection, setConnection] = useState<ConnectionState>('loading')
  const [attempt, setAttempt] = useState(0)
  const [algorithm, setAlgorithm] = useState<Algorithm>(algorithmFromLocation)

  useEffect(() => {
    const handlePopState = () => setAlgorithm(algorithmFromLocation())
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  function navigateTo(event: MouseEvent<HTMLAnchorElement>, nextAlgorithm: Algorithm) {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return
    event.preventDefault()
    if (nextAlgorithm === algorithm) return
    window.history.pushState(null, '', event.currentTarget.href)
    setAlgorithm(nextAlgorithm)
  }

  useEffect(() => {
    const controller = new AbortController()
    let current = true

    setConnection('loading')
    getHealth(controller.signal)
      .then(() => {
        if (current) setConnection('connected')
      })
      .catch(() => {
        if (current) setConnection('failed')
      })

    return () => {
      current = false
      controller.abort()
    }
  }, [attempt])

  const connectionCopy = {
    loading: 'Checking backend',
    connected: 'Backend connected',
    failed: 'Backend unavailable',
  }[connection]

  return (
    <main id="home" className="page-shell">
      <nav className="topbar" aria-label="Main navigation">
        <a className="brand" href="#home" aria-label="ML Playground home">
          <span className="brand-mark" aria-hidden="true">M</span>
          <span>ML Playground</span>
        </a>
        <div className={`topbar-connection connection-${connection}`} aria-live="polite">
          <span className="connection-indicator" aria-hidden="true" />
          <span role={connection === 'failed' ? 'alert' : 'status'}>{connectionCopy}</span>
          {connection === 'failed' && (
            <button className="header-retry" onClick={() => setAttempt((value) => value + 1)}>Retry</button>
          )}
        </div>
      </nav>

      <nav className="algorithm-nav" aria-label="Choose algorithm">
        <a
          href={algorithmHref('linear-regression')}
          onClick={(event) => navigateTo(event, 'linear-regression')}
          aria-current={algorithm === 'linear-regression' ? 'page' : undefined}
        >
          Linear Regression
        </a>
        <a
          href={algorithmHref('logistic-regression')}
          onClick={(event) => navigateTo(event, 'logistic-regression')}
          aria-current={algorithm === 'logistic-regression' ? 'page' : undefined}
        >
          Logistic Regression
        </a>
        <a href={algorithmHref('k-means')} onClick={event => navigateTo(event, 'k-means')}
          aria-current={algorithm === 'k-means' ? 'page' : undefined}>K-Means</a>
      </nav>

      <div hidden={algorithm !== 'linear-regression'}>
        <LinearRegressionExperience />
      </div>
      <div hidden={algorithm !== 'logistic-regression'}>
        <LogisticRegressionExperience />
      </div>
      <div hidden={algorithm !== 'k-means'}>
        <KMeansExperience visible={algorithm === 'k-means'} />
      </div>

      <footer className="footer">
        <span>ML Playground <span aria-hidden="true">·</span> Learn through experimentation</span>
        <span>One experiment at a time</span>
      </footer>
    </main>
  )
}

export default App
