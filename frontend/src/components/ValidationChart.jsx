import * as d3 from 'd3'

export default function ValidationChart({ players, selectedPlayer }) {
  const w = 520
  const h = 300
  const pad = 36

  if (!players?.length) return null

  const x = d3.scaleLinear().domain(d3.extent(players, (d) => d.xt_per_90)).nice().range([pad, w - pad])
  const y = d3.scaleLinear().domain(d3.extent(players, (d) => d.goals_per_90)).nice().range([h - pad, pad])

  return (
    <svg className="validation" viewBox={`0 0 ${w} ${h}`}>
      <rect x="0" y="0" width={w} height={h} fill="#101f1f" rx="12" />
      <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="#d1d5db" />
      <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke="#d1d5db" />

      {players.map((p) => {
        const selected = p.player === selectedPlayer
        return (
          <circle
            key={p.player}
            cx={x(p.xt_per_90)}
            cy={y(p.goals_per_90)}
            r={selected ? 5 : 2.8}
            fill={selected ? '#f97316' : '#34d399'}
            opacity={selected ? 1 : 0.65}
          />
        )
      })}

      <text x={w / 2} y={h - 8} textAnchor="middle" fill="#e5e7eb" fontSize="12">xT per 90</text>
      <text x={14} y={h / 2} textAnchor="middle" fill="#e5e7eb" fontSize="12" transform={`rotate(-90 14 ${h / 2})`}>
        Goals per 90
      </text>
    </svg>
  )
}
