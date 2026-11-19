import { useEffect, useState } from 'react'
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
    loading: 'Checking backend connection',
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
        <span className="topbar-note">A hands-on learning space</span>
      </nav>

      <section className="hero" id="home" aria-labelledby="page-title">
        <div className="hero-copy">
          <p className="eyebrow"><span className="eyebrow-line" /> Learn by exploring</p>
          <h1 id="page-title" aria-label="Make machine learning make sense.">Make machine learning<br /><em>make sense.</em></h1>
          <p className="hero-description">
            Change the data. Tune the model. See what happens. Build intuition for the ideas behind machine learning, one experiment at a time.
          </p>
          <div className="hero-meta">
            <span className="meta-dot" aria-hidden="true" />
            <span>Built for curious minds, from first steps to deeper questions</span>
          </div>
        </div>

        <aside className={`connection-card connection-${connection}`} aria-live="polite">
          <div className="connection-heading">
            <span className="connection-indicator" aria-hidden="true" />
            <span className="connection-label">System status</span>
          </div>
          <p className="connection-title" role={connection === 'failed' ? 'alert' : 'status'}>
            {connectionCopy}
          </p>
          <p className="connection-detail">
            {connection === 'connected'
              ? 'The experiment service is ready when you are.'
              : connection === 'failed'
                ? 'We could not reach the experiment service. Check that the backend is running.'
                : 'Connecting to the experiment service…'}
          </p>
          {connection === 'failed' && (
            <button className="retry-button" onClick={() => setAttempt((value) => value + 1)}>
              Try again <span aria-hidden="true">↗</span>
            </button>
          )}
        </aside>
      </section>

      <section className="next-section" aria-labelledby="next-title">
        <div>
          <p className="eyebrow"><span className="eyebrow-line" /> Your lab starts here</p>
          <h2 id="next-title">Big ideas, made tangible.</h2>
        </div>
        <p className="next-description">
          Start with a simple model, explore how its parts work together, and let the patterns reveal themselves.
        </p>
      </section>

      <footer className="footer">
        <span>ML Playground <span aria-hidden="true">·</span> Learn through experimentation</span>
        <span>Made for the wonderfully curious</span>
      </footer>
    </main>
  )
}

export default App
