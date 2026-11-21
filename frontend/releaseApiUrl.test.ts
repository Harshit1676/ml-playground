import { describe, expect, it } from 'vitest'
import { releaseApiUrl } from './releaseApiUrl'

describe('release API configuration', () => {
  it.each([undefined, '', '  ', '/api', 'https:///example.org', 'http://api.example.org',
    'https://localhost', 'https://dev.localhost.:8080', 'https://127.0.0.1', 'https://127.255.0.2',
    'https://127.1', 'https://2130706433', 'https://0x7f000001', 'https://[::1]',
    'https://[0:0:0:0:0:0:0:1]', 'https://[::ffff:127.0.0.2]', 'https://0.0.0.0',
    'https://[::ffff:0:0]', 'https://[::ffff:0.0.0.0]',
    'https://bad host', 'https://user:pass@example.org', 'https://api.example.org?key=x',
    'https://api.example.org#section'])('rejects %s', value => {
    expect(() => releaseApiUrl(value)).toThrow(/VITE_API_BASE_URL/)
  })

  it.each([
    ['https://api.example.org/', 'https://api.example.org'],
    ['https://api.example.org:8443/service/', 'https://api.example.org:8443/service'],
    ['https://[2001:db8::2]', 'https://[2001:db8::2]'],
  ])('accepts %s', (input, expected) => expect(releaseApiUrl(input)).toBe(expected))
})
