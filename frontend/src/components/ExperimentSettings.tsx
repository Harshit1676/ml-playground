import { useEffect, useId, useRef, useState } from 'react'
import type { ReactNode } from 'react'

export function SettingsFields({ children, errors, running }: { children: ReactNode; errors: object; running: boolean }) {
  const [expanded, setExpanded] = useState(false)
  const id = useId()
  const fields = useRef<HTMLDivElement>(null)
  const toggle = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    if (running) {
      if (toggle.current?.offsetWidth && fields.current?.contains(document.activeElement)) {
        toggle.current.focus()
      }
      setExpanded(false)
    }
  }, [running])
  useEffect(() => {
    if (Object.values(errors).some(Boolean)) setExpanded(true)
  }, [errors])

  return <>
    <button ref={toggle} className="settings-toggle" type="button" aria-expanded={expanded} aria-controls={id}
      onClick={() => setExpanded(!expanded)}>{expanded ? 'Hide settings' : 'Edit experiment settings'}</button>
    <div ref={fields} className={expanded ? 'settings-fields expanded' : 'settings-fields'} id={id}>{children}</div>
  </>
}

// Both values come from the same lab's request builder, in canonical field order.
// Compare the effective payload so editing a number's formatting keeps the run current.
export function SettingsNotice({ submitted, draft }: { submitted: object; draft?: object | null }) {
  return <div role="status">{JSON.stringify(submitted) !== JSON.stringify(draft) &&
    <p className="stale-result">Showing the previous settings. Run the experiment to update these results.</p>}</div>
}
