import { useEffect, useState } from 'react'
import { LinearRegressionExperience } from './features/linear-regression/LinearRegressionExperience'
import { getHealth } from './lib/api/health'

type ConnectionState = 'loading' | 'connected' | 'failed'

function App() {
  const [connection, setConnection] = useState<ConnectionState>('loading')
  const [attempt, setAttempt] = useState(0)

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
    <main className="page-shell">
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

      <LinearRegressionExperience />

      <footer className="footer">
        <span>ML Playground <span aria-hidden="true">·</span> Learn through experimentation</span>
        <span>One experiment at a time</span>
      </footer>
    </main>
  )
}

export default App
