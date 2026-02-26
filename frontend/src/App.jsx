import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from './lib/api'
import PitchPanel from './components/PitchPanel'
import StatCards from './components/StatCards'
import ValidationChart from './components/ValidationChart'

export default function App() {
  const [selectedPlayer, setSelectedPlayer] = useState('')
  const [actionType, setActionType] = useState('all')
  const [progressiveOnly, setProgressiveOnly] = useState(false)

  const playersQuery = useQuery({ queryKey: ['players'], queryFn: api.players })
  const surfaceQuery = useQuery({ queryKey: ['surface'], queryFn: api.surface })

  const actionsQuery = useQuery({
    queryKey: ['actions', selectedPlayer],
    queryFn: () => api.playerActions(selectedPlayer),
    enabled: Boolean(selectedPlayer),
  })

  const statsQuery = useQuery({
    queryKey: ['stats', selectedPlayer],
    queryFn: () => api.playerStats(selectedPlayer),
    enabled: Boolean(selectedPlayer),
  })

  const playerOptions = playersQuery.data ?? []

  const filteredActions = useMemo(() => {
    const arr = actionsQuery.data ?? []
    return arr.filter((a) => {
      if (actionType !== 'all' && a.type?.toLowerCase() !== actionType) return false
      if (progressiveOnly && !a.progressive_flag) return false
      return true
    })
  }, [actionsQuery.data, actionType, progressiveOnly])

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>Hybrid xT</h1>

        <label>Player</label>
        <input
          list="players"
          placeholder="Type player name"
          value={selectedPlayer}
          onChange={(e) => setSelectedPlayer(e.target.value)}
        />
        <datalist id="players">
          {playerOptions.map((p) => (
            <option key={p.player} value={p.player} />
          ))}
        </datalist>

        <label>Action Filter</label>
        <select value={actionType} onChange={(e) => setActionType(e.target.value)}>
          <option value="all">All actions</option>
          <option value="pass">Passes only</option>
          <option value="carry">Carries only</option>
        </select>

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={progressiveOnly}
            onChange={(e) => setProgressiveOnly(e.target.checked)}
          />
          Progressive only
        </label>

        <div className="panel-title">Player Stats</div>
        <StatCards stats={statsQuery.data} />
      </aside>

      <main className="main">
        <div className="panel">
          <PitchPanel surface={surfaceQuery.data?.grid ?? []} actions={filteredActions} />
        </div>

        <div className="panel">
          <h2>Validation View</h2>
          <ValidationChart players={playersQuery.data ?? []} selectedPlayer={selectedPlayer} />
        </div>
      </main>
    </div>
  )
}
