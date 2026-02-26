import * as d3 from 'd3'

const W = 800
const H = 540

function xtColor(v) {
  return d3.scaleSequential([0, 0.2], d3.interpolateYlOrRd)(v)
}

function actionColor(v) {
  return d3.scaleDiverging([-0.08, 0, 0.08], d3.interpolateRdYlGn)(v)
}

export default function PitchPanel({ surface, actions }) {
  const cellW = W / 16
  const cellH = H / 12

  return (
    <svg className="pitch" viewBox={`0 0 ${W} ${H}`}>
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="#f9fafb" />
        </marker>
      </defs>

      {surface?.map((row, r) =>
        row.map((v, c) => (
          <rect key={`${r}-${c}`} x={c * cellW} y={r * cellH} width={cellW} height={cellH} fill={xtColor(v)} opacity={0.85} />
        )),
      )}

      <rect x="1" y="1" width={W - 2} height={H - 2} fill="none" stroke="#fff" strokeWidth="2" />
      <line x1={W / 2} y1={0} x2={W / 2} y2={H} stroke="#fff" strokeWidth="2" />
      <circle cx={W / 2} cy={H / 2} r="60" fill="none" stroke="#fff" strokeWidth="2" />
      <rect x={0} y={H * 0.2} width={130} height={H * 0.6} fill="none" stroke="#fff" strokeWidth="2" />
      <rect x={W - 130} y={H * 0.2} width={130} height={H * 0.6} fill="none" stroke="#fff" strokeWidth="2" />

      {actions.map((a) => {
        const x1 = (a.start_x / 120) * W
        const y1 = (a.start_y / 80) * H
        const x2 = (a.end_x / 120) * W
        const y2 = (a.end_y / 80) * H
        return (
          <line
            key={a.id}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke={actionColor(a.xt_value)}
            strokeWidth="2"
            markerEnd="url(#arrow)"
            opacity="0.9"
          />
        )
      })}
    </svg>
  )
}
