import math
from typing import Dict, List, Optional, Tuple

def parse_naca4(code: str) -> Optional[Dict[str, float]]:
    clean = code.strip()
    if len(clean) != 4 or not clean.isdigit():
        return None
    m = int(clean[0]) / 100.0  # max camber
    p = int(clean[1]) / 10.0   # position of max camber
    t = int(clean[2:]) / 100.0 # thickness
    return {"m": m, "p": p, "t": t, "code": clean}

def _thickness(x: float, t: float, closed_te: bool = True) -> float:
    a4 = -0.1036 if closed_te else -0.1015
    return (t / 0.2) * (
        0.2969 * math.sqrt(x) -
        0.1260 * x -
        0.3516 * (x ** 2) +
        0.2843 * (x ** 3) +
        a4 * (x ** 4)
    )

def _camber_line(x: float, m: float, p: float) -> Tuple[float, float]:
    if m == 0 or p == 0:
        return 0.0, 0.0
    if x < p:
        yc = (m / (p * p)) * (2 * p * x - (x ** 2))
        dyc = ((2 * m) / (p * p)) * (p - x)
        return yc, dyc
    else:
        yc = (m / ((1 - p) ** 2)) * (1 - 2 * p + 2 * p * x - (x ** 2))
        dyc = ((2 * m) / ((1 - p) ** 2)) * (p - x)
        return yc, dyc

def generate_naca4(code: str, n_points: int = 120) -> Optional[Dict]:
    parsed = parse_naca4(code)
    if not parsed:
        return None

    m, p, t = parsed["m"], parsed["p"], parsed["t"]
    upper = []
    lower = []
    camber = []

    # Cosine spacing concentrates points near the leading edge for a cleaner profile.
    for i in range(n_points + 1):
        beta = (math.pi * i) / n_points
        x = 0.5 * (1.0 - math.cos(beta))
        yt = _thickness(x, t, True)
        yc, dyc = _camber_line(x, m, p)
        theta = math.atan(dyc)

        xu = x - yt * math.sin(theta)
        yu = yc + yt * math.cos(theta)
        xl = x + yt * math.sin(theta)
        yl = yc - yt * math.cos(theta)

        # Truncate floating values slightly to avoid JSON bloat and ensure clean UI lines
        upper.append([round(xu, 6), round(yu, 6)])
        lower.append([round(xl, 6), round(yl, 6)])
        camber.append([round(x, 6), round(yc, 6)])

    # Build a closed loop: upper from TE -> LE, then lower LE -> TE.
    upper_rev = upper[::-1]
    lower_fwd = lower[1:]
    loop = upper_rev + lower_fwd

    return {
        "upper": upper,
        "lower": lower,
        "loop": loop,
        "camber": camber,
        "params": parsed
    }

def scale_coordinates(coords: List[List[float]], chord: float) -> List[List[float]]:
    return [[pt[0] * chord, pt[1] * chord] for pt in coords]

def to_dat_file(name: str, loop: List[List[float]]) -> str:
    lines = [name]
    for pt in loop:
        lines.append(f"  {pt[0]:.6f}  {pt[1]:.6f}")
    return "\n".join(lines) + "\n"

def to_csv_file(loop: List[List[float]]) -> str:
    lines = ["x,y"]
    for pt in loop:
        lines.append(f"{pt[0]:.6f},{pt[1]:.6f}")
    return "\n".join(lines) + "\n"

def extrude_airfoil(loop: List[List[float]], chord_mm: float, span_mm: float) -> Dict:
    # Deduplicate closing point if loop[0] == loop[-1]
    pts = list(loop)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()

    n = len(pts)
    z0 = -span_mm / 2.0
    z1 = span_mm / 2.0

    vertices = []
    for pt in pts:
        vertices.append([pt[0] * chord_mm, pt[1] * chord_mm, z0])
    for pt in pts:
        vertices.append([pt[0] * chord_mm, pt[1] * chord_mm, z1])

    faces = []
    # Side walls
    for i in range(n):
        a = i
        b = (i + 1) % n
        c = i + n
        d = ((i + 1) % n) + n
        faces.append([a, b, d])
        faces.append([a, d, c])

    # End caps via fan triangulation
    for i in range(1, n - 1):
        # Back cap (z = z0) — wind so outward normal points -Z.
        faces.append([0, i + 1, i])
        # Front cap (z = z1) — outward normal +Z.
        faces.append([n, n + i, n + i + 1])

    return {"vertices": vertices, "faces": faces}

def to_ascii_stl(name: str, vertices: List[List[float]], faces: List[List[int]]) -> str:
    lines = [f"solid {name}"]
    
    for ia, ib, ic in faces:
        a, b, c = vertices[ia], vertices[ib], vertices[ic]
        # Compute face normal
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        
        len_n = math.hypot(nx, ny, nz)
        if len_n == 0:
            len_n = 1.0
        nx, ny, nz = nx / len_n, ny / len_n, nz / len_n
        
        lines.append(f"  facet normal {nx:e} {ny:e} {nz:e}")
        lines.append("    outer loop")
        lines.append(f"      vertex {a[0]:e} {a[1]:e} {a[2]:e}")
        lines.append(f"      vertex {b[0]:e} {b[1]:e} {b[2]:e}")
        lines.append(f"      vertex {c[0]:e} {c[1]:e} {c[2]:e}")
        lines.append("    endloop")
        lines.append("  endfacet")
        
    lines.append(f"endsolid {name}")
    return "\n".join(lines) + "\n"
