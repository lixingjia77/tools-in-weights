"""把 4B 的并发-吞吐测量画成 SVG（不依赖 matplotlib）。

输入：results/cost/4b_tput_apc{0,1}.json
输出：assets/throughput.svg

用法：python3 -I src/plot_throughput.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "throughput.svg")

W, H = 960, 420
PAD_L, PAD_R, PAD_T, PAD_B = 70, 30, 78, 75
GAP = 70
PANEL_W = (W - PAD_L - PAD_R - GAP) // 2
PANEL_H = H - PAD_T - PAD_B
Y_MAX = 6000
Y_STEP = 1000
COLORS = {"P1": "#d9480f", "P2": "#1971c2"}
LABELS = {"P1": "P1 — schema in prompt", "P2": "P2 — no schema"}


def load(apc):
    with open(os.path.join(ROOT, "results", "cost", f"4b_tput_apc{apc}.json"),
              encoding="utf-8") as f:
        return json.load(f)


def x_of(c, x0):
    # 并发 1/8/32/100 在对数轴上等距
    import math
    t = math.log2(c) / math.log2(100)
    return x0 + t * PANEL_W


def y_of(v, y0):
    return y0 + PANEL_H - (v / Y_MAX) * PANEL_H


def panel(data, x0, title):
    y0 = PAD_T
    out = []
    # 边框与水平网格
    out.append(f'<rect x="{x0}" y="{y0}" width="{PANEL_W}" height="{PANEL_H}" '
               f'fill="#fbfbfb" stroke="#dee2e6"/>')
    for v in range(0, Y_MAX + 1, Y_STEP):
        y = y_of(v, y0)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + PANEL_W}" y2="{y:.1f}" '
                   f'stroke="#e9ecef"/>')
        out.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" '
                   f'font-size="12" fill="#868e96">{v}</text>')
    # x 轴刻度
    for c in (1, 8, 32, 100):
        x = x_of(c, x0)
        out.append(f'<text x="{x:.1f}" y="{y0 + PANEL_H + 34}" text-anchor="middle" '
                   f'font-size="12" fill="#495057">{c}</text>')
    out.append(f'<text x="{x0 + PANEL_W / 2:.1f}" y="{y0 + PANEL_H + 58}" '
               f'text-anchor="middle" font-size="13" fill="#212529">concurrency</text>')
    if x0 == PAD_L:
        out.append(f'<text x="18" y="{y0 + PANEL_H / 2:.1f}" text-anchor="middle" '
                   f'font-size="13" fill="#212529" '
                   f'transform="rotate(-90 18 {y0 + PANEL_H / 2:.1f})">output tokens / s</text>')

    out.append(f'<text x="{x0 + PANEL_W / 2:.1f}" y="{y0 - 22}" text-anchor="middle" '
               f'font-size="14" font-weight="600" fill="#212529">{title}</text>')

    by = {}
    for r in data["rows"]:
        by.setdefault(r["paradigm"], []).append((r["concurrency"], r["out_tok_s"]))

    for p in ("P2", "P1"):
        pts = sorted(by[p])
        d = " ".join(f"{x_of(c, x0):.1f},{y_of(v, y0):.1f}" for c, v in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="{COLORS[p]}" '
                   f'stroke-width="2.5" stroke-linejoin="round"/>')
        for c, v in pts:
            x, y = x_of(c, x0), y_of(v, y0)
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{COLORS[p]}"/>')
            # P2 标在点上方，P1 标在点下方：两条线不会互相压住
            dy = -11 if p == "P2" else 17
            # P2 的线向右上走：标在点左上方，避开线；第一个点没有左侧空间，改标右上
            # P1 的线较平：标在点右下方；最右一个点改右对齐，避免出框
            if p == "P2" and x > x0 + 40:
                tx, anchor = x - 7, "end"
            elif x > x0 + PANEL_W - 46:
                tx, anchor = x - 7, "end"
            else:
                tx, anchor = x + 7, "start"
            out.append(f'<text x="{tx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" '
                       f'font-size="11" font-weight="600" fill="{COLORS[p]}">'
                       f'{v:.0f}</text>')
    return "\n".join(out)


def main():
    with_apc = panel(load(1), PAD_L, "prefix caching ON")
    without = panel(load(0), PAD_L + PANEL_W + GAP, "prefix caching OFF")

    legend = (
        f'<rect x="{W - 330}" y="12" width="14" height="4" fill="{COLORS["P1"]}"/>'
        f'<text x="{W - 310}" y="19" font-size="12" fill="#212529">{LABELS["P1"]}</text>'
        f'<rect x="{W - 175}" y="12" width="14" height="4" fill="{COLORS["P2"]}"/>'
        f'<text x="{W - 155}" y="19" font-size="12" fill="#212529">{LABELS["P2"]}</text>'
    )

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="Inter, -apple-system, '
           f'Segoe UI, Helvetica, Arial, sans-serif">'
           f'<rect width="{W}" height="{H}" fill="white"/>'
           f'{legend}\n{with_apc}\n{without}\n</svg>\n')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"→ {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
