export function releaseApiUrl(value: string | undefined): string {
  const raw = value?.trim() ?? ''
  const invalid = () => new Error('VITE_API_BASE_URL must be an explicit HTTPS API URL with a non-loopback host and no credentials, query or fragment.')
  if (!/^https:\/\/[^/]/i.test(raw) || /[\\\s?#]/.test(raw)) throw invalid()
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    throw invalid()
  }

  // URL canonicalizes shorthand IPv4 and expands dotted IPv4 tails into hexadecimal.
  const host = url.hostname.replace(/\.$/, '')
  const localName = host === 'localhost' || host.endsWith('.localhost')
  const localV4 = host.startsWith('127.') && /^\d+\.\d+\.\d+\.\d+$/.test(host)
  const localV6 = /^\[::(?:1|(?:ffff:)?7f[\da-f]{2}:[\da-f]+)?\]$/i.test(host)
  const validDns = host.startsWith('[') || host.split('.').every(label =>
    /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/i.test(label))
  if (!validDns || localName || localV4 || localV6 || host === '0.0.0.0' || url.username || url.password) throw invalid()
  return url.href.replace(/\/+$/, '')
}
