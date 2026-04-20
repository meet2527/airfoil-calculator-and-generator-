"use client"

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { FileDown, FileText, Boxes, Info, Gauge } from "lucide-react"
import type { AirfoilCoordinates } from "@/lib/airfoil"

type Props = {
  modelType: string
  onModelTypeChange: (v: string) => void
  code: string
  onCodeChange: (v: string) => void
  chord: number
  onChordChange: (v: number) => void
  span: number
  onSpanChange: (v: number) => void
  data: AirfoilCoordinates | null
  error: string | null
  onDownloadDat: () => void
  onDownloadCsv: () => void
  onDownloadStl: () => void
}

export function AirfoilSidebar({
  modelType,
  onModelTypeChange,
  code,
  onCodeChange,
  chord,
  onChordChange,
  span,
  onSpanChange,
  data,
  error,
  onDownloadDat,
  onDownloadCsv,
  onDownloadStl,
}: Props) {
  const canExport = !!data && !error

  return (
    <aside className="flex h-full w-full flex-col border-r border-border bg-sidebar text-sidebar-foreground">
      <div className="flex items-center gap-3 border-b border-sidebar-border px-5 py-4">
        <div className="flex h-9 w-9 items-center justify-center rounded-md bg-foreground text-background">
          <Gauge className="h-5 w-5" aria-hidden />
        </div>
        <div className="flex flex-col">
          <span className="text-sm font-semibold leading-tight">Airfoil Studio</span>
          <span className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
            Geometry Generator
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-5">
        <div className="flex flex-col gap-5">
          <section className="flex flex-col gap-2">
            <h2 className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">Parameters</h2>

            <div className="flex flex-col gap-2">
              <Label htmlFor="model-type" className="text-xs">
                Airfoil Model Type
              </Label>
              <Select value={modelType} onValueChange={onModelTypeChange}>
                <SelectTrigger id="model-type">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="naca4">NACA 4-digit</SelectItem>
                  <SelectItem value="naca5" disabled>
                    NACA 5-digit (coming soon)
                  </SelectItem>
                  <SelectItem value="selig" disabled>
                    Selig / UIUC (coming soon)
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="naca-code" className="text-xs">
                NACA Code
              </Label>
              <Input
                id="naca-code"
                value={code}
                onChange={(e) => onCodeChange(e.target.value.replace(/[^0-9]/g, "").slice(0, 4))}
                placeholder="2412"
                inputMode="numeric"
                maxLength={4}
                className="font-mono tracking-widest"
              />
              <p className="text-[11px] leading-snug text-muted-foreground">
                4 digits: max camber %, position /10, thickness %
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="chord" className="text-xs">
                Chord Length (mm)
              </Label>
              <Input
                id="chord"
                type="number"
                min={1}
                step={1}
                value={Number.isFinite(chord) ? chord : 0}
                onChange={(e) => onChordChange(Number.parseFloat(e.target.value) || 0)}
                className="font-mono"
              />
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="span" className="text-xs">
                Span (mm)
              </Label>
              <Input
                id="span"
                type="number"
                min={1}
                step={1}
                value={Number.isFinite(span) ? span : 0}
                onChange={(e) => onSpanChange(Number.parseFloat(e.target.value) || 0)}
                className="font-mono"
              />
            </div>
          </section>

          <Separator />

          <section className="flex flex-col gap-2">
            <h2 className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">Geometry</h2>
            {data ? (
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 rounded-md border border-sidebar-border bg-background/60 p-3 font-mono text-[11px]">
                <dt className="text-muted-foreground">Max camber</dt>
                <dd className="text-right">{(data.params.m * 100).toFixed(2)}%</dd>
                <dt className="text-muted-foreground">Camber pos.</dt>
                <dd className="text-right">{(data.params.p * 100).toFixed(1)}%</dd>
                <dt className="text-muted-foreground">Thickness</dt>
                <dd className="text-right">{(data.params.t * 100).toFixed(2)}%</dd>
                <dt className="text-muted-foreground">Points / surface</dt>
                <dd className="text-right">{data.upper.length}</dd>
                <dt className="text-muted-foreground">Chord</dt>
                <dd className="text-right">{chord} mm</dd>
                <dt className="text-muted-foreground">Span</dt>
                <dd className="text-right">{span} mm</dd>
              </dl>
            ) : (
              <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-xs text-destructive">
                <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" aria-hidden />
                <span>{error ?? "Enter a valid 4-digit NACA code."}</span>
              </div>
            )}
          </section>

          <Separator />

          <section className="flex flex-col gap-2">
            <h2 className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">Export</h2>
            <div className="flex flex-col gap-2">
              <Button variant="outline" size="sm" onClick={onDownloadDat} disabled={!canExport}>
                <FileText className="h-4 w-4" aria-hidden />
                Download .dat (Selig)
              </Button>
              <Button variant="outline" size="sm" onClick={onDownloadCsv} disabled={!canExport}>
                <FileDown className="h-4 w-4" aria-hidden />
                Download .csv (x, y)
              </Button>
              <Button size="sm" onClick={onDownloadStl} disabled={!canExport}>
                <Boxes className="h-4 w-4" aria-hidden />
                Download .stl (3D mesh)
              </Button>
            </div>
            <p className="text-[11px] leading-snug text-muted-foreground">
              Coordinates are scaled to the chord length. STL is extruded along Z.
            </p>
          </section>
        </div>
      </div>

      <div className="border-t border-sidebar-border px-5 py-3">
        <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          v1.0 · naca 4-series
        </p>
      </div>
    </aside>
  )
}
