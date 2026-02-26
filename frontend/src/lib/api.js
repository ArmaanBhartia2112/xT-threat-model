const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

async function fetchJson(path) {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) {
    throw new Error(`API error ${res.status}`)
  }
  return res.json()
}

export const api = {
  surface: () => fetchJson('/surface'),
  players: () => fetchJson('/players'),
  playerActions: (name) => fetchJson(`/player-actions?player_name=${encodeURIComponent(name)}`),
  playerStats: (name) => fetchJson(`/player-stats?player_name=${encodeURIComponent(name)}`),
}
