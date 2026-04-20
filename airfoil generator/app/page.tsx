"use client"

import { useMemo, useState } from "react"
import dynamic from "next/dynamic"
import { AirfoilSvg } from "@/components/airfoil-svg"
import { AirfoilSidebar } from "@/components/airfoil-sidebar"
import {
  downloadTextFile,
  extrudeAirfoil,
  generateNaca4,
  toAsciiStl,
  toCsvFile,
  toDatFile,
} from "@/lib/airfoil"
import { Move3d, Ruler, Sparkles } from "lucide-react"

const Airfoil3D = dynamic(() => import("@/components/airfoil-3d").then((m) => m.Airfoil3D), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center text-xs text-muted-foreground">
      Initializing 3D viewer…
    </div>
  ),
})

export default function Page() {
  const [modelType, setModelType] = useState("naca4")
  const [code, setCode] = useState("2412")
  const [chord, setChord] = useState(200)
  const [span, setSpan] = useState(400)
  const [view, setView] = useState<"3d" | "2d">("3d")

  const { data, error } = useMemo(() => {
    if (code.length !== 4) {
      return { data: null, error: "NACA code must be exactly 4 digits." }
    }
    const d = generateNaca4(code, 140)
    if (!d) return { data: null, error: "Invalid NACA 4-digit code." }
    return { data: d, error: null }
  }, [code])

  const handleDownloadDat = () => {
    if (!data) return
    const scaled = data.loop.map((p) => ({ x: p.x * chord, y: p.y * chord }))
    const content = toDatFile(`NACA ${data.params.code} (chord=${chord}mm)`, scaled)
    downloadTextFile(`naca${data.params.code}_chord${chord}.dat`, content)
  }

  const handleDownloadCsv = () => {
    if (!data) return
    const scaled = data.loop.map((p) => ({ x: p.x * chord, y: p.y * chord }))
    const content = toCsvFile(scaled)
    downloadTextFile(`naca${data.params.code}_chord${chord}.csv`, content, "text/csv")
  }

  const handleDownloadStl = () => {
    if (!data) return
    const mesh = extrudeAirfoil(data.loop, chord, span)
    const content = toAsciiStl(`NACA_${data.params.code}`, mesh)
    downloadTextFile(
      `naca${data.params.code}_chord${chord}_span${span}.stl`,
      content,
      "model/stl",
    )
  }

  return (
    <main className="grid h-dvh w-full grid-cols-1 bg-background text-foreground md:grid-cols-[320px_1fr]">
      <AirfoilSidebar
        modelType={modelType}
        onModelTypeChange={setModelType}
        code={code}
        onCodeChange={setCode}
        chord={chord}
        onChordChange={setChord}
        span={span}
        onSpanChange={setSpan}
        data={data}
        error={error}
        onDownloadDat={handleDownloadDat}
        onDownloadCsv={handleDownloadCsv}
        onDownloadStl={handleDownloadStl}
      />

      <section className="flex min-h-0 flex-col">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border bg-card px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-md border border-border bg-background">
              <Sparkles className="h-4 w-4" aria-hidden />
            </div>
            <div className="flex flex-col">
              <h1 className="text-base font-semibold leading-tight">
                NACA {data ? data.params.code : code || "----"} · Airfoil Profile
              </h1>
              <p className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
                {modelType === "naca4" ? "4-digit series" : modelType} · chord {chord} mm · span {span} mm
              </p>
            </div>
          </div>

          <div className="inline-flex rounded-md border border-border bg-background p-1 text-xs">
            <button
              type="button"
              onClick={() => setView("3d")}
              className={`inline-flex items-center gap-1.5 rounded px-3 py-1.5 font-medium transition-colors ${
                view === "3d"
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground"
              }`}
              aria-pressed={view === "3d"}
            >
              <Move3d className="h-3.5 w-3.5" aria-hidden />
              3D View
            </button>
            <button
              type="button"
              onClick={() => setView("2d")}
              className={`inline-flex items-center gap-1.5 rounded px-3 py-1.5 font-medium transition-colors ${
                view === "2d"
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground"
              }`}
              aria-pressed={view === "2d"}
            >
              <Ruler className="h-3.5 w-3.5" aria-hidden />
              2D Profile
            </button>
          </div>
        </header>

        <div className="relative min-h-0 flex-1 bg-[linear-gradient(var(--color-muted)_1px,transparent_1px),linear-gradient(90deg,var(--color-muted)_1px,transparent_1px)] bg-[size:32px_32px]">
          {/* Subtle scrim so the grid reads as an industrial backdrop */}
          <div className="absolute inset-0 bg-background/60" aria-hidden />

          <div className="relative h-full w-full">
            {!data ? (
              <div className="flex h-full items-center justify-center p-8 text-sm text-muted-foreground">
                Enter a valid 4-digit NACA code to generate geometry.
              </div>
            ) : view === "3d" ? (
              <Airfoil3D data={data} chord={chord} span={Math.max(span, 1)} />
            ) : (
              <div className="flex h-full items-center justify-center p-8">
                <div className="w-full max-w-4xl rounded-lg border border-border bg-card p-6 shadow-sm">
                  <AirfoilSvg data={data} chord={chord} />
                </div>
              </div>
            )}
          </div>
        </div>

        <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-border bg-card px-6 py-3 font-mono text-[11px] text-muted-foreground">
          <span>
            Loop points: <span className="text-foreground">{data?.loop.length ?? 0}</span>
          </span>
          <span>
            Max thickness: <span className="text-foreground">{data ? (data.params.t * 100).toFixed(1) : "--"}%</span>
          </span>
          <span>
            Camber: <span className="text-foreground">{data ? (data.params.m * 100).toFixed(1) : "--"}%</span>
          </span>
          <span className="uppercase tracking-widest">ready · exports scaled to mm</span>
        </footer>
      </section>
    </main>
  )
}
