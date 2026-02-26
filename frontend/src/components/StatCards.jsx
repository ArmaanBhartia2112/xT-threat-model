export default function StatCards({ stats }) {
  if (!stats) return <div className="panel">Select a player</div>

  const cards = [
    ['xT / 90', Number(stats.xt_per_90 ?? 0).toFixed(3)],
    ['Total xT', Number(stats.total_xt ?? 0).toFixed(2)],
    ['Progressive', stats.progressive_actions ?? 0],
    ['Pressure %', `${((stats.pressure_action_rate ?? 0) * 100).toFixed(1)}%`],
  ]

  return (
    <div className="cards-grid">
      {cards.map(([label, value]) => (
        <div className="card" key={label}>
          <div className="card-label">{label}</div>
          <div className="card-value">{value}</div>
        </div>
      ))}
    </div>
  )
}
