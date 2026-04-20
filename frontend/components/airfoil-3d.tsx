"use client"

import { Canvas } from "@react-three/fiber"
import { OrbitControls, GizmoHelper, GizmoViewport } from "@react-three/drei"
import { useMemo } from "react"
import * as THREE from "three"
import type { AirfoilCoordinates } from "@/lib/airfoil"
import { extrudeAirfoil } from "@/lib/airfoil"

type Props = {
  data: AirfoilCoordinates
  chord: number
  span: number
}

function AirfoilMesh({ data, chord, span }: Props) {
  const geometry = useMemo(() => {
    const mesh = extrudeAirfoil(data.loop, chord, span)
    const geom = new THREE.BufferGeometry()

    const positions = new Float32Array(mesh.faces.length * 3 * 3)
    let p = 0
    for (const [ia, ib, ic] of mesh.faces) {
      const a = mesh.vertices[ia]
      const b = mesh.vertices[ib]
      const c = mesh.vertices[ic]
      positions[p++] = a[0]
      positions[p++] = a[1]
      positions[p++] = a[2]
      positions[p++] = b[0]
      positions[p++] = b[1]
      positions[p++] = b[2]
      positions[p++] = c[0]
      positions[p++] = c[1]
      positions[p++] = c[2]
    }
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3))
    geom.computeVertexNormals()
    return geom
  }, [data, chord, span])

  const edges = useMemo(() => new THREE.EdgesGeometry(geometry, 25), [geometry])

  return (
    <group>
      <mesh geometry={geometry} castShadow receiveShadow>
        <meshStandardMaterial color="#d8dadc" metalness={0.35} roughness={0.55} side={THREE.DoubleSide} />
      </mesh>
      <lineSegments geometry={edges}>
        <lineBasicMaterial color="#111418" linewidth={1} />
      </lineSegments>
    </group>
  )
}

export function Airfoil3D({ data, chord, span }: Props) {
  const maxDim = Math.max(chord, span)
  const camDist = maxDim * 1.6

  return (
    <Canvas
      shadows
      camera={{ position: [camDist * 0.8, camDist * 0.5, camDist], fov: 35, near: 0.1, far: maxDim * 20 }}
      style={{ background: "transparent" }}
    >
      <ambientLight intensity={0.55} />
      <directionalLight
        position={[chord * 2, span * 2, span * 2]}
        intensity={1.1}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <directionalLight position={[-chord, -span, -span]} intensity={0.35} />

      <AirfoilMesh data={data} chord={chord} span={span} />



      <OrbitControls
        makeDefault
        target={[chord / 2, 0, 0]}
        enableDamping
        dampingFactor={0.1}
        maxDistance={maxDim * 8}
      />

      <GizmoHelper alignment="bottom-right" margin={[64, 64]}>
        <GizmoViewport axisColors={["#c1272d", "#2f855a", "#2b4b8a"]} labelColor="#111418" />
      </GizmoHelper>
    </Canvas>
  )
}
