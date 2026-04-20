"use client"

import type { AirfoilCoordinates } from "@/lib/airfoil"

type Props = {
  data: AirfoilCoordinates
  chord: number
}

export function AirfoilSvg({ data, chord }: Props) {
  const width = 600
  const height = 240
  const padX = 24
  const padY = 24

  // Determine bounds from loop
  let minY = Infinity
  let maxY = -Infinity
  for (const p of data.loop) {
    if (p.y < minY) minY = p.y
    if (p.y > maxY) maxY = p.y
  }
  // Symmetric margin around zero
  const yRange = Math.max(Math.abs(minY), Math.abs(maxY)) * 2 * 1.1 || 0.2

  const sx = (width - 2 * padX) / 1 // x in [0,1]
  const sy = (height - 2 * padY) / yRange

  const tx = (x: number) => padX + x * sx
  const ty = (y: number) => height / 2 - y * sy

  const loopPath =
    data.loop
      .map((p, i) => `${i === 0 ? "M" : "L"} ${tx(p.x).toFixed(2)} ${ty(p.y).toFixed(2)}`)
      .join(" ") + " Z"

  const camberPath = data.camber
    .map((p, i) => `${i === 0 ? "M" : "L"} ${tx(p.x).toFixed(2)} ${ty(p.y).toFixed(2)}`)
    .join(" ")

  // Chord line
  const chordPath = `M ${tx(0)} ${ty(0)} L ${tx(1)} ${ty(0)}`

  // Gridlines at 0, 0.25, 0.5, 0.75, 1.0
  const gridX = [0, 0.25, 0.5, 0.75, 1].map((v) => tx(v))

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-full"
      role="img"
      aria-label={`NACA ${data.params.code} airfoil 2D profile`}
    >
      {/* Grid */}
      {gridX.map((gx, i) => (
        <line
          key={i}
          x1={gx}
          x2={gx}
          y1={padY}
          y2={height - padY}
          stroke="var(--color-border)"
          strokeWidth={1}
          strokeDasharray="2 4"
        />
      ))}
      <line
        x1={padX}
        x2={width - padX}
        y1={height / 2}
        y2={height / 2}
        stroke="var(--color-border)"
        strokeWidth={1}
      />

      {/* Chord line */}
      <path d={chordPath} stroke="var(--color-muted-foreground)" strokeDasharray="4 4" strokeWidth={1} fill="none" />

      {/* Camber line */}
      <path d={camberPath} stroke="var(--color-chart-1)" strokeWidth={1.25} fill="none" opacity={0.85} />

      {/* Airfoil profile */}
      <path
        d={loopPath}
        fill="var(--color-chart-1)"
        fillOpacity={0.08}
        stroke="var(--color-foreground)"
        strokeWidth={1.75}
        strokeLinejoin="round"
      />

      {/* Labels */}
      <text x={padX} y={height - 6} fontSize={10} fill="var(--color-muted-foreground)" fontFamily="var(--font-mono)">
        x / c = 0
      </text>
      <text
        x={width - padX}
        y={height - 6}
        fontSize={10}
        textAnchor="end"
        fill="var(--color-muted-foreground)"
        fontFamily="var(--font-mono)"
      >
        x / c = 1
      </text>
      <text x={padX} y={padY - 8} fontSize={10} fill="var(--color-muted-foreground)" fontFamily="var(--font-mono)">
        NACA {data.params.code} · chord {chord} mm · {data.loop.length} pts
      </text>
    </svg>
  )
}
