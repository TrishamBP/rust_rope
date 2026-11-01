"""Generates the SVG diagrams of ROPE_Engineering (docs/diagrams/*.svg).

Pure Python, no dependencies. Run: python scripts/make_diagrams.py
Every numeric plot is calculated from the same formulas as the Rust code
(theta_j = b^(-2j/d), the scaling formulas of src/rope/scaling.rs).
"""

from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "diagrams"

BLUE, GREEN, ORANGE, GREY, RED, PURPLE = "#2563eb", "#15803d", "#c2410c", "#6b7280", "#b91c1c", "#7c3aed"
LIGHT = {"blue": "#dbeafe", "green": "#dcfce7", "orange": "#ffedd5", "grey": "#f3f4f6", "purple": "#ede9fe", "red": "#fee2e2"}
FONT = "Segoe UI, Arial, sans-serif"


class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h, self.parts = w, h, []

    def add(self, s: str):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill="white", stroke="#111827", sw=1.2, rx=6, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=13, anchor="middle", color="#111827", weight="normal", italic=False):
        st = ' font-style="italic"' if italic else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}"{st}>{s}</text>')

    def line(self, x1, y1, x2, y2, color="#111827", sw=1.5, arrow=False, dash=None):
        m = ' marker-end="url(#arrow)"' if arrow else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{m}{d}/>')

    def poly(self, pts, color, sw=2, fill="none", dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.add(f'<polyline points="{p}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{d}/>')

    def circle(self, x, y, r, fill="none", stroke="#111827", sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def path(self, d, color, sw=1.5, fill="none", arrow=False, dash=None):
        m = ' marker-end="url(#arrow)"' if arrow else ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{m}{da}/>')

    def box(self, x, y, w, h, title, sub=None, fill="white", stroke="#111827", dash=None, size=13):
        self.rect(x, y, w, h, fill=fill, stroke=stroke, dash=dash)
        if sub:
            self.text(x + w / 2, y + h / 2 - 4, title, size=size, weight="bold")
            self.text(x + w / 2, y + h / 2 + 13, sub, size=size - 2, color="#374151")
        else:
            self.text(x + w / 2, y + h / 2 + 5, title, size=size, weight="bold")

    def save(self, name: str):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">'
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#111827"/>'
                '</marker></defs><rect width="100%" height="100%" fill="white"/>')
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(head + "".join(self.parts) + "</svg>", encoding="utf-8")


def legend(s: Svg, x, y, items):
    for i, (color, label, dash) in enumerate(items):
        yy = y + i * 20
        s.line(x, yy, x + 28, yy, color=color, sw=3, dash=dash)
        s.text(x + 36, yy + 4, label, size=12, anchor="start")


def plot_axes(s, x0, y0, w, h, xlabel, ylabel, xticks, yticks, xmap, ymap):
    s.line(x0, y0 + h, x0 + w, y0 + h)
    s.line(x0, y0, x0, y0 + h)
    for v, lab in xticks:
        x = xmap(v)
        s.line(x, y0 + h, x, y0 + h + 5)
        s.text(x, y0 + h + 19, lab, size=11)
    for v, lab in yticks:
        y = ymap(v)
        s.line(x0 - 5, y, x0, y)
        s.line(x0, y, x0 + w, y, color="#e5e7eb", sw=1)
        s.text(x0 - 8, y + 4, lab, size=11, anchor="end")
    s.text(x0 + w / 2, y0 + h + 40, xlabel, size=12)
    s.add(f'<text x="{x0 - 52}" y="{y0 + h / 2}" font-family="{FONT}" font-size="12" text-anchor="middle" '
          f'transform="rotate(-90 {x0 - 52} {y0 + h / 2})">{ylabel}</text>')


# 1 ---------------------------------------------------------------------------
def attention_flow():
    s = Svg(900, 330)
    s.text(450, 24, "Attention with RoPE: where this project works", size=15, weight="bold")
    s.box(20, 140, 120, 56, "Hidden states X", "[B, S, D_model]", fill=LIGHT["grey"], dash="5,4")
    ys = [70, 150, 230]
    for y, name in zip(ys, ["Q = X·W_Q", "K = X·W_K", "V = X·W_V"]):
        s.box(190, y, 130, 46, name, "projection", fill=LIGHT["grey"], dash="5,4", size=12)
        s.line(140, 168, 190, y + 23, arrow=True)
    s.box(370, 66, 150, 54, "RoPE(Q, m)", "rotate_query_key", fill=LIGHT["green"], stroke=GREEN)
    s.box(370, 146, 150, 54, "RoPE(K, n)", "same R(·) per head", fill=LIGHT["green"], stroke=GREEN)
    s.line(320, 93, 370, 93, arrow=True)
    s.line(320, 173, 370, 173, arrow=True)
    s.box(570, 100, 150, 60, "Scores", "Q'·K'ᵀ / √D", fill=LIGHT["grey"], dash="5,4")
    s.line(520, 93, 570, 120, arrow=True)
    s.line(520, 173, 570, 142, arrow=True)
    s.box(570, 200, 150, 50, "softmax · V", "weighted sum", fill=LIGHT["grey"], dash="5,4")
    s.line(645, 160, 645, 200, arrow=True)
    s.line(320, 253, 570, 230, arrow=True)
    s.text(445, 262, "V is not rotated", size=11, color=RED, italic=True)
    s.box(760, 200, 120, 50, "Output", "[B, S, D_model]", fill=LIGHT["grey"], dash="5,4")
    s.line(720, 225, 760, 225, arrow=True)
    s.rect(30, 290, 18, 14, fill=LIGHT["green"], stroke=GREEN, rx=2)
    s.text(55, 302, "implemented in rust_rope", size=12, anchor="start")
    s.rect(260, 290, 18, 14, fill=LIGHT["grey"], dash="4,3", rx=2)
    s.text(285, 302, "context only — not implemented (projections, softmax, attention)", size=12, anchor="start")
    s.save("01_attention_flow.svg")


# 2 ---------------------------------------------------------------------------
def qk_integration():
    s = Svg(900, 360)
    s.text(450, 24, "RoPE on query and key tensors [B, S, H, D] (start = 0)", size=15, weight="bold")
    for b in range(2):
        y0 = 50 + b * 140
        s.text(30, y0 + 60, f"b = {b}", size=13, weight="bold", anchor="start")
        for sidx in range(4):
            x0 = 90 + sidx * 200
            s.rect(x0, y0, 180, 120, fill="#fafafa", stroke=GREY, rx=8)
            s.text(x0 + 90, y0 + 17, f"token s = {sidx}  →  m = {sidx}", size=12, weight="bold")
            for h in range(2):
                yy = y0 + 28 + h * 44
                s.rect(x0 + 10, yy, 160, 36, fill=LIGHT["blue"] if h == 0 else LIGHT["purple"], stroke=GREY, rx=4)
                s.text(x0 + 90, yy + 15, f"head h = {h}: q, k ∈ R^D", size=11)
                s.text(x0 + 90, yy + 30, f"q' = R_{sidx}·q,  k' = R_{sidx}·k", size=11, color=GREEN)
    s.text(450, 345, "Every head and batch element at sequence index s uses the same R_m (m = start + s). V is never passed to RoPE.",
           size=12, italic=True)
    s.save("02_qk_integration.svg")


# 3 ---------------------------------------------------------------------------
def rotation_2d():
    s = Svg(560, 420)
    cx, cy, sc = 230, 300, 70
    s.text(280, 24, "Two-dimensional rotation x' = R(θ)·x", size=15, weight="bold")
    s.line(cx - 200, cy, cx + 300, cy, color=GREY, arrow=True)
    s.line(cx, cy + 90, cx, 45, color=GREY, arrow=True)
    s.text(cx + 300, cy + 18, "x₀", size=13)
    s.text(cx - 14, 52, "x₁", size=13)
    x = (3.0, 1.0)
    th = 0.9
    xr = (x[0] * math.cos(th) - x[1] * math.sin(th), x[0] * math.sin(th) + x[1] * math.cos(th))
    s.line(cx, cy, cx + x[0] * sc, cy - x[1] * sc, color=BLUE, sw=3, arrow=True)
    s.line(cx, cy, cx + xr[0] * sc, cy - xr[1] * sc, color=ORANGE, sw=3, arrow=True)
    r = math.hypot(*x) * sc
    s.circle(cx, cy, r, stroke="#d1d5db", dash="4,4")
    a0, a1 = math.atan2(x[1], x[0]), math.atan2(xr[1], xr[0])
    rr = 60
    s.path(f"M {cx + rr * math.cos(a0):.1f} {cy - rr * math.sin(a0):.1f} A {rr} {rr} 0 0 0 "
           f"{cx + rr * math.cos(a1):.1f} {cy - rr * math.sin(a1):.1f}", PURPLE, sw=2, arrow=True)
    s.text(cx + 78 * math.cos((a0 + a1) / 2), cy - 78 * math.sin((a0 + a1) / 2), "θ = 0.9", size=13, color=PURPLE, weight="bold")
    s.text(cx + x[0] * sc + 10, cy - x[1] * sc + 4, "x = (3, 1)", size=13, color=BLUE, anchor="start")
    s.text(cx + xr[0] * sc + 8, cy - xr[1] * sc - 6, f"x' = ({xr[0]:.3f}, {xr[1]:.3f})", size=13, color=ORANGE, anchor="start")
    s.text(280, 400, "x'₀ = x₀cos θ − x₁sin θ,  x'₁ = x₀sin θ + x₁cos θ;  ‖x'‖ = ‖x‖ = √10 (dashed circle)", size=12)
    s.save("03_rotation_2d.svg")


# 4 ---------------------------------------------------------------------------
def unit_circle():
    s = Svg(520, 440)
    cx, cy, r = 250, 225, 150
    s.text(260, 24, "The unit circle: (cos θ, sin θ)", size=15, weight="bold")
    s.line(cx - 200, cy, cx + 210, cy, color=GREY, arrow=True)
    s.line(cx, cy + 190, cx, cy - 195, color=GREY, arrow=True)
    s.circle(cx, cy, r, stroke=BLUE, sw=2)
    th = 0.8
    px, py = cx + r * math.cos(th), cy - r * math.sin(th)
    s.line(cx, cy, px, py, color=ORANGE, sw=2.5)
    s.line(px, py, px, cy, color=GREEN, sw=2, dash="5,3")
    s.line(cx, cy, px, cy, color=RED, sw=3)
    s.circle(px, py, 4, fill=ORANGE, stroke=ORANGE)
    s.path(f"M {cx + 40} {cy} A 40 40 0 0 0 {cx + 40 * math.cos(th):.1f} {cy - 40 * math.sin(th):.1f}", PURPLE, sw=2)
    s.text(cx + 55, cy - 16, "θ", size=14, color=PURPLE, weight="bold")
    s.text(px + 8, py - 8, "(cos θ, sin θ)", size=13, anchor="start", color=ORANGE)
    s.text((cx + px) / 2, cy + 18, "cos θ", size=12, color=RED)
    s.text(px + 8, (py + cy) / 2, "sin θ", size=12, color=GREEN, anchor="start")
    for ang, lab, dx, dy in [(0, "θ = 0: (1, 0)", 10, -8), (math.pi / 2, "π/2: (0, 1)", 8, -6),
                             (math.pi, "π: (−1, 0)", -8, -8), (3 * math.pi / 2, "3π/2: (0, −1)", 8, 18)]:
        x, y = cx + r * math.cos(ang), cy - r * math.sin(ang)
        s.circle(x, y, 3.5, fill="#111827")
        s.text(x + dx, y + dy, lab, size=11, anchor="end" if ang == math.pi else "start")
    s.text(260, 425, "radius 1, so cos²θ + sin²θ = 1; one full turn = 2π rad", size=12)
    s.save("04_unit_circle.svg")


# 5 ---------------------------------------------------------------------------
def complex_rotation():
    s = Svg(560, 420)
    cx, cy, sc = 260, 300, 85
    s.text(280, 24, "Complex multiplication by e^(iθ) is a rotation", size=15, weight="bold")
    s.line(cx - 230, cy, cx + 250, cy, color=GREY, arrow=True)
    s.line(cx, cy + 80, cx, 40, color=GREY, arrow=True)
    s.text(cx + 250, cy + 18, "Re", size=13)
    s.text(cx - 18, 48, "Im", size=13)
    z = complex(1, 2)
    w = z * complex(math.cos(1.0), math.sin(1.0))
    for v, col, lab, anc in [(z, BLUE, "z = 1 + 2i", "start"), (w, ORANGE, "z·e^(i·1) = −1.1426 + 1.9221i", "end")]:
        s.line(cx, cy, cx + v.real * sc, cy - v.imag * sc, color=col, sw=3, arrow=True)
        s.text(cx + v.real * sc + (8 if anc == "start" else -8), cy - v.imag * sc - 6, lab, size=13, color=col, anchor=anc)
    a0, a1 = math.atan2(z.imag, z.real), math.atan2(w.imag, w.real)
    s.path(f"M {cx + 70 * math.cos(a0):.1f} {cy - 70 * math.sin(a0):.1f} A 70 70 0 0 0 "
           f"{cx + 70 * math.cos(a1):.1f} {cy - 70 * math.sin(a1):.1f}", PURPLE, sw=2, arrow=True)
    s.text(cx - 6, cy - 82, "+θ = 1 rad", size=12, color=PURPLE, weight="bold")
    s.circle(cx, cy, abs(z) * sc, stroke="#d1d5db", dash="4,4")
    s.text(280, 390, "|z·e^(iθ)| = |z|·1 = √5;  arg(z·e^(iθ)) = arg(z) + θ = 1.1071 + 1 = 2.1071", size=12)
    s.save("05_complex_rotation.svg")


def cells(s, x0, y, labels, fills, w=58, h=40):
    for i, (lab, f) in enumerate(zip(labels, fills)):
        s.rect(x0 + i * w, y, w - 4, h, fill=f, stroke="#374151", rx=4)
        s.text(x0 + i * w + (w - 4) / 2, y + h / 2 + 5, lab, size=13, weight="bold")


PAIR_COLORS = [LIGHT["blue"], LIGHT["green"], LIGHT["orange"], LIGHT["purple"]]
PAIR_STROKES = [BLUE, GREEN, ORANGE, PURPLE]


# 6 / 7 -----------------------------------------------------------------------
def pairing(name, title, pairs, note):
    s = Svg(620, 230)
    s.text(310, 24, title, size=15, weight="bold")
    x0, y = 80, 120
    fills = ["white"] * 8
    for j, (a, b) in enumerate(pairs):
        fills[a] = fills[b] = PAIR_COLORS[j]
    cells(s, x0, y, [f"x{k}" for k in range(8)], fills)
    for j, (a, b) in enumerate(pairs):
        xa, xb = x0 + a * 58 + 27, x0 + b * 58 + 27
        top = y - 12 - 14 * (b - a > 1) * (1 + j * 0.6)
        s.path(f"M {xa} {y} C {xa} {top - 30}, {xb} {top - 30}, {xb} {y}", PAIR_STROKES[j], sw=2.2)
        s.text((xa + xb) / 2, top - 18 - (8 if b - a > 1 else 0), f"pair {j}: θ{j}", size=11, color=PAIR_STROKES[j], weight="bold")
    s.text(310, 195, note, size=12)
    s.text(310, 214, "Each pair j rotates by φ = m·θⱼ with θⱼ = b^(−2j/d); d = 8", size=12, color="#374151")
    s.save(name)


# 8 ---------------------------------------------------------------------------
def partial_rope():
    s = Svg(760, 230)
    s.text(380, 24, "Partial RoPE: head dimension D = 16, rotary dimension d = 8", size=15, weight="bold")
    x0, y = 30, 90
    fills = [PAIR_COLORS[k // 2] for k in range(8)] + [LIGHT["grey"]] * 8
    cells(s, x0, y, [f"x{k}" for k in range(16)], fills, w=44, h=40)
    s.path(f"M {x0} {y - 10} L {x0} {y - 22} L {x0 + 8 * 44 - 4} {y - 22} L {x0 + 8 * 44 - 4} {y - 10}", GREEN, sw=2)
    s.text(x0 + 4 * 44, y - 30, "rotary part [0, d): split_at_mut(d) → rotated (either layout)", size=12, color=GREEN, weight="bold")
    s.path(f"M {x0 + 8 * 44} {y + 52} L {x0 + 8 * 44} {y + 64} L {x0 + 16 * 44 - 4} {y + 64} L {x0 + 16 * 44 - 4} {y + 52}", GREY, sw=2)
    s.text(x0 + 12 * 44, y + 82, "pass-through [d, D): never touched (bit-for-bit unchanged)", size=12, color=GREY, weight="bold")
    s.text(380, 205, "x' = [RoPE(x[0..d]), x[d..D]]   with   θⱼ = b^(−2j/d)  (d, not D, in the exponent)", size=12)
    s.save("08_partial_rope.svg")


# 9 ---------------------------------------------------------------------------
def frequency_distribution():
    s = Svg(720, 420)
    s.text(360, 24, "Inverse frequencies θⱼ = b^(−2j/d) (logarithmic axis)", size=15, weight="bold")
    x0, y0, w, h = 90, 50, 580, 290
    xmap = lambda j: x0 + j / 63 * w
    ymap = lambda v: y0 + (-math.log10(v)) / 6 * h  # 1 → top, 1e-6 → bottom
    plot_axes(s, x0, y0, w, h, "pair index j (d = 128: j = 0 … 63)", "θⱼ (rad / position)",
              [(j, str(j)) for j in (0, 16, 32, 48, 63)],
              [(10.0 ** -k, f"1e-{k}" if k else "1") for k in range(7)], xmap, ymap)
    for b, col in [(10_000.0, BLUE), (500_000.0, ORANGE)]:
        s.poly([(xmap(j), ymap(b ** (-2 * j / 128))) for j in range(64)], col, sw=2.5)
    pts8 = [(xmap(j * 63 / 3), ymap(10_000.0 ** (-2 * j / 8))) for j in range(4)]
    for x, y in pts8:
        s.circle(x, y, 4.5, fill=GREEN, stroke=GREEN)
    legend(s, 420, 70, [(BLUE, "d = 128, b = 10 000", None), (ORANGE, "d = 128, b = 500 000", None),
                        (GREEN, "d = 8, b = 10 000 (j = 0…3 at relative positions)", None)])
    s.text(360, 405, "A straight line on a log axis = a geometric sequence (constant ratio b^(−2/d)). Fast pairs left, slow pairs right.", size=12)
    s.save("09_frequency_distribution.svg")


# 10 --------------------------------------------------------------------------
def position_interpolation():
    s = Svg(720, 420)
    s.text(360, 24, "Angle of the slowest pair (d = 8, θ₃ = 0.001): extrapolation vs interpolation", size=15, weight="bold")
    x0, y0, w, h = 90, 50, 580, 290
    xmap = lambda m: x0 + m / 8192 * w
    ymap = lambda a: y0 + h - a / 8.5 * h
    s.rect(xmap(0), ymap(2.048), xmap(2048) - xmap(0), ymap(0) - ymap(2.048), fill=LIGHT["green"], stroke="none", rx=0)
    plot_axes(s, x0, y0, w, h, "token position m", "angle m·θ'₃ (rad)",
              [(m, str(m)) for m in (0, 2048, 4096, 6144, 8192)], [(a, str(a)) for a in range(0, 9, 2)], xmap, ymap)
    s.poly([(xmap(m), ymap(m * 0.001)) for m in (0, 8191)], RED, sw=2.5)
    s.poly([(xmap(m), ymap(m * 0.001 / 4)) for m in (0, 8191)], BLUE, sw=2.5)
    s.text(xmap(1024), ymap(2.048) - 8, "trained range (m < L = 2048)", size=11, color=GREEN, weight="bold")
    s.circle(xmap(4096), ymap(1.024), 5, fill=BLUE, stroke=BLUE)
    s.text(xmap(4096) + 8, ymap(1.024) + 18, "m = 4096 → 1.024 rad (as m = 1024 untrained)", size=11, color=BLUE, anchor="start")
    s.circle(xmap(4096), ymap(4.096), 5, fill=RED, stroke=RED)
    s.text(xmap(4096) - 8, ymap(4.096) - 8, "4.096 rad: never seen", size=11, color=RED, anchor="end")
    legend(s, 110, 70, [(RED, "standard RoPE (extrapolation)", None), (BLUE, "position interpolation, L = 2048 → L' = 8192", None)])
    s.save("10_position_interpolation.svg")


# 10b -------------------------------------------------------------------------
def scaling_ratios():
    s = Svg(720, 420)
    s.text(360, 24, "How each method changes θⱼ: ratio θ'ⱼ / θⱼ (d = 128, b = 10 000, s = 4, L = 2048)", size=14, weight="bold")
    x0, y0, w, h = 90, 50, 580, 290
    d, b, sc, L = 128, 10_000.0, 4.0, 2048
    xmap = lambda j: x0 + j / 63 * w
    ymap = lambda v: y0 + h - (v - 0.2) / 0.85 * h
    plot_axes(s, x0, y0, w, h, "pair index j", "θ'ⱼ / θⱼ", [(j, str(j)) for j in (0, 16, 32, 48, 63)],
              [(v, f"{v:.2f}") for v in (0.25, 0.5, 0.75, 1.0)], xmap, ymap)
    theta = [b ** (-2 * j / d) for j in range(64)]
    s.poly([(xmap(j), ymap(1 / sc)) for j in (0, 63)], BLUE, sw=2.5)
    bp = b * sc ** (d / (d - 2))
    s.poly([(xmap(j), ymap(bp ** (-2 * j / d) / theta[j])) for j in range(64)], ORANGE, sw=2.5)
    pts = []
    for j, t in enumerate(theta):
        r = L * t / (2 * math.pi)
        g = min(max((r - 1) / 31, 0.0), 1.0)
        pts.append((xmap(j), ymap(((1 - g) * t / sc + g * t) / t)))
    s.poly(pts, GREEN, sw=2.5)
    legend(s, 400, 80, [(BLUE, "position interpolation / linear: 1/s", None), (ORANGE, "NTK-aware: s^(−2j/(d−2))", None),
                        (GREEN, "YaRN (paper ramp, α = 1, β = 32)", None)])
    s.text(360, 405, "1 = unchanged (high frequencies keep local resolution); 0.25 = fully interpolated.", size=12)
    s.save("16_scaling_ratios.svg")


# 11 --------------------------------------------------------------------------
def tensor_memory():
    s = Svg(900, 300)
    s.text(450, 24, "Flat row-major storage of a [B, S, H, D] = [2, 2, 2, 2] tensor", size=15, weight="bold")
    x0, y, w = 30, 110, 52
    fills = []
    labels = []
    for b in range(2):
        for sq in range(2):
            for hh in range(2):
                for dd in range(2):
                    labels.append(f"{b}{sq}{hh}{dd}")
                    fills.append(LIGHT["blue"] if b == 0 else LIGHT["orange"])
    for i, (lab, f) in enumerate(zip(labels, fills)):
        s.rect(x0 + i * w, y, w - 3, 40, fill=f, stroke="#374151", rx=3)
        s.text(x0 + i * w + 24, y + 25, lab, size=12, weight="bold")
        s.text(x0 + i * w + 24, y + 58, str(i), size=11, color=GREY)
    s.text(x0 - 4, y + 25, "", size=11)
    s.text(450, y - 50, "label = b s h d;   index = ((b·S + s)·H + h)·D + d;   strides = [8, 4, 2, 1]", size=13)
    for i in range(0, 16, 2):
        s.path(f"M {x0 + i * w} {y - 6} L {x0 + i * w} {y - 14} L {x0 + (i + 2) * w - 3} {y - 14} L {x0 + (i + 2) * w - 3} {y - 6}", GREY, sw=1.2)
    s.text(x0 + w, y - 20, "head (D = 2)", size=10, color=GREY)
    for i in range(0, 16, 4):
        s.path(f"M {x0 + i * w} {y + 70} L {x0 + i * w} {y + 80} L {x0 + (i + 4) * w - 3} {y + 80} L {x0 + (i + 4) * w - 3} {y + 70}", PURPLE, sw=1.5)
        s.text(x0 + i * w + 2 * w, y + 96, f"token (b={i // 8}, s={(i // 4) % 2}): H·D = 4", size=11, color=PURPLE)
    for i in range(0, 16, 8):
        s.path(f"M {x0 + i * w} {y + 104} L {x0 + i * w} {y + 116} L {x0 + (i + 8) * w - 3} {y + 116} L {x0 + (i + 8) * w - 3} {y + 104}", RED, sw=1.8)
        s.text(x0 + i * w + 4 * w, y + 134, f"batch element b = {i // 8}: S·H·D = 8 contiguous values", size=12, color=RED, weight="bold")
    s.text(450, 290, "One Vec<f64>: 16 × 8 bytes = 128 contiguous bytes = 2 cache lines of 64 bytes.", size=12, italic=True)
    s.save("11_tensor_memory.svg")


# 12 --------------------------------------------------------------------------
def architecture():
    s = Svg(900, 470)
    s.text(450, 24, "rust_rope crate architecture (arrows: \"uses\")", size=15, weight="bold")
    s.rect(20, 50, 560, 250, fill="#f0fdf4", stroke=GREEN, rx=10)
    s.text(40, 72, "rope (RoPE-specific)", size=13, anchor="start", weight="bold", color=GREEN)
    mods = [("config", "RopeConfig, RopeLayout", 40, 90), ("position", "Position, PositionRange", 220, 90),
            ("frequencies", "θⱼ, angle, AngleTable", 400, 90), ("rotations", "rotate_pair", 40, 210),
            ("embeddings", "RotaryEmbedding", 220, 210), ("scaling", "RopeScaling", 400, 210)]
    for name, sub, x, y in mods:
        s.box(x, y, 160, 60, name, sub, fill="white", stroke=GREEN)
    s.rect(620, 50, 260, 250, fill="#eff6ff", stroke=BLUE, rx=10)
    s.text(640, 72, "math (general, no RoPE)", size=13, anchor="start", weight="bold", color=BLUE)
    for i, (name, sub) in enumerate([("trigonometry", "sin_cos"), ("matrix", "Matrix2, rotate_vector"),
                                     ("complex", "Complex"), ("tensor", "Tensor4 [B,S,H,D]")]):
        s.box(640, 82 + i * 54, 220, 46, name, sub, fill="white", stroke=BLUE, size=12)
    s.box(220, 330, 200, 56, "utils::validation", "RopeError, validate_*", fill=LIGHT["orange"], stroke=ORANGE)
    s.line(300, 270, 300, 330, arrow=True)
    s.line(300, 210, 300, 150, arrow=True)
    s.line(380, 240, 400, 240, arrow=True)
    s.line(220, 240, 200, 240, arrow=True)
    s.line(380, 225, 480, 150, arrow=True)
    s.line(580, 240, 640, 150, arrow=True, color=BLUE)
    s.line(580, 270, 640, 270, arrow=True, color=BLUE)
    s.box(20, 400, 270, 56, "tests/, examples/, benches/", "public API only", fill=LIGHT["grey"])
    s.box(320, 400, 270, 56, "reference/ (Python)", "independent PyTorch validation", fill=LIGHT["purple"], stroke=PURPLE)
    s.box(620, 400, 260, 56, "src/main.rs", "thin demo binary", fill=LIGHT["grey"])
    s.text(450, 316, "math never uses rope or utils (one-way dependency)", size=11, italic=True, color=BLUE)
    s.save("12_architecture.svg")


# 13 --------------------------------------------------------------------------
def end_to_end():
    s = Svg(820, 600)
    s.text(410, 24, "End-to-end RoPE execution flow", size=15, weight="bold")
    steps = [("Input embeddings / hidden states", "not implemented (context)", True),
             ("Query and key projections Q = XW_Q, K = XW_K", "not implemented (caller provides Q, K)", True),
             ("RopeConfig::new(d, b)?.with_layout(..).with_head_dimension(D)?", "src/rope/config.rs", False),
             ("inverse_frequencies(&config)  →  θⱼ = b^(−2j/d)", "src/rope/frequencies.rs", False),
             ("optional scaling: scaled_inverse_frequencies → θ'ⱼ, attention factor", "src/rope/scaling.rs (RotaryEmbedding::with_scaling)", False),
             ("positions m = start + s  (PositionRange)", "src/rope/position.rs", False),
             ("angles φ = m·θⱼ (angle_unchecked), sin_cos(φ)", "frequencies.rs, math/trigonometry.rs", False),
             ("rotate each pair (rotate_pair → rotate_vector) for Q and K", "rotations.rs, math/matrix.rs, embeddings.rs", False),
             ("attention scores Q'·K'ᵀ (only as a test property)", "not implemented as a feature", True)]
    y = 50
    for i, (title, sub, ctx) in enumerate(steps):
        s.box(90, y, 640, 50, title, sub, fill=LIGHT["grey"] if ctx else LIGHT["green"],
              stroke="#111827" if ctx else GREEN, dash="5,4" if ctx else None, size=13)
        if i < len(steps) - 1:
            s.line(410, y + 50, 410, y + 62, arrow=True)
        y += 62
    s.save("13_end_to_end.svg")


# 14 --------------------------------------------------------------------------
def cache_hierarchy():
    s = Svg(820, 380)
    s.text(410, 24, "CPU memory hierarchy (this computer: AMD Ryzen 7 6800H, 8 cores / 16 threads)", size=14, weight="bold")
    levels = [("Registers", "16 general + 16 vector registers per core", 160),
              ("L1 data cache", "32 KiB per core (Zen 3+ specification)", 300),
              ("L2 cache", "512 KiB per core (4 MiB total, reported by Windows)", 380),
              ("L3 cache", "16 MiB shared (reported by Windows)", 520),
              ("Main memory (DRAM)", "15.2 GiB installed", 700)]
    y = 50
    for name, sub, w in levels:
        s.box(410 - w / 2, y, w, 52, name, sub, fill=LIGHT["blue"], stroke=BLUE, size=13)
        y += 62
    s.text(410, 368, "Higher levels are smaller and faster. Data moves in cache lines of 64 bytes (8 × f64).", size=12, italic=True)
    s.save("14_cache_hierarchy.svg")


# 15 --------------------------------------------------------------------------
def simd():
    s = Svg(820, 300)
    s.text(410, 24, "Scalar versus SIMD processing (concept — SIMD is NOT implemented in rust_rope)", size=14, weight="bold")
    s.text(200, 60, "Scalar: 4 instructions", size=13, weight="bold")
    for i in range(4):
        y = 75 + i * 46
        s.rect(60, y, 60, 34, fill=LIGHT["blue"], stroke=BLUE, rx=4)
        s.text(90, y + 22, f"a{i}", size=13)
        s.text(140, y + 22, "×", size=15)
        s.rect(160, y, 60, 34, fill=LIGHT["orange"], stroke=ORANGE, rx=4)
        s.text(190, y + 22, f"c{i}", size=13)
        s.text(240, y + 22, "=", size=15)
        s.rect(260, y, 70, 34, fill=LIGHT["green"], stroke=GREEN, rx=4)
        s.text(295, y + 22, f"a{i}c{i}", size=13)
    s.text(610, 60, "SIMD (AVX2, 256-bit register = 4 × f64): 1 instruction", size=13, weight="bold")
    for k, (lab, fill, stroke, y) in enumerate([("a", LIGHT["blue"], BLUE, 90), ("c", LIGHT["orange"], ORANGE, 150), ("ac", LIGHT["green"], GREEN, 220)]):
        s.rect(450, y, 320, 40, fill="white", stroke=stroke, rx=6)
        for i in range(4):
            s.rect(455 + i * 79, y + 4, 73, 32, fill=fill, stroke=stroke, rx=3)
            s.text(491 + i * 79, y + 25, f"{lab}{i}" if lab != "ac" else f"a{i}c{i}", size=13)
    s.text(610, 205, "vmulpd ymm2, ymm0, ymm1", size=12, color=PURPLE, weight="bold")
    s.text(410, 288, "AVX-512 registers hold 8 × f64. The compiler may auto-vectorize simple loops; this project has not measured it (Stage 9).", size=11, italic=True)
    s.save("15_simd.svg")


def main():
    attention_flow()
    qk_integration()
    rotation_2d()
    unit_circle()
    complex_rotation()
    pairing("06_interleaved.svg", "Interleaved layout: adjacent pairs (x₂ⱼ, x₂ⱼ₊₁)",
            [(0, 1), (2, 3), (4, 5), (6, 7)], "RopeLayout::Interleaved — rotary.chunks_exact_mut(2)")
    pairing("07_split_half.svg", "Split-half layout: pairs (xⱼ, x_(j + d/2))",
            [(0, 4), (1, 5), (2, 6), (3, 7)], "RopeLayout::SplitHalf — rotary.split_at_mut(d/2)")
    partial_rope()
    frequency_distribution()
    position_interpolation()
    scaling_ratios()
    tensor_memory()
    architecture()
    end_to_end()
    cache_hierarchy()
    simd()
    print("diagrams:", len(list(OUT.glob("*.svg"))))


if __name__ == "__main__":
    main()
