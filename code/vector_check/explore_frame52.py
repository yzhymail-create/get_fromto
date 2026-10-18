#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""矢量校验工具 - 阶段A: 框#52 内图元普查 + 落盘 pickle (独立目录, 不混入视觉管线)

注意: ezdxf 1.4.5 的 GraphicEntity 没有 .bbox() 方法, 这里直接用几何坐标做框选。
"""
import json, pickle, re
from pathlib import Path
from collections import Counter

import ezdxf

HERE = Path(__file__).resolve().parent
WS = HERE.parent  # ~/workspace/visual_tracer
PKL = HERE / "frame52_entities.pkl"

def load_frame_bbox():
    d = json.load(open(WS / "qa" / "frames" / "subframes.json"))
    f52 = [f for f in d["frames"] if f.get("id") == 52][0]
    return f52["units"]  # [xmin, ymin, xmax, ymax]

def seg_intersects_bb(x1, y1, x2, y2, bb, m):
    # 线段包围盒与框相交(含 margin)即保留; 端点在框内是其子集
    sx0, sx1 = min(x1, x2), max(x1, x2)
    sy0, sy1 = min(y1, y2), max(y1, y2)
    return not (sx1 < bb[0]-m or sx0 > bb[2]+m or sy1 < bb[1]-m or sy0 > bb[3]+m)

def main():
    dxf_path = next((WS / "incoming" / "v2").glob("*.dxf"))
    print("loading", dxf_path, flush=True)
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    bb = load_frame_bbox()
    print("frame52 bbox units:", bb, flush=True)
    M = 10.0  # margin, drawing units

    type_layer = Counter()
    texts, circles, rects, lines = [], [], [], []
    arcs, plines = [], []
    n = 0
    for e in msp:
        n += 1
        if n % 100000 == 0:
            print(f"  scanned {n}", flush=True)
        t = e.dxftype()
        try:
            if t == "LINE":
                s, en = e.dxf.start, e.dxf.end
                if not seg_intersects_bb(s.x, s.y, en.x, en.y, bb, M):
                    continue
                type_layer[(t, e.dxf.layer)] += 1
                lines.append((e.dxf.layer, s.x, s.y, en.x, en.y))
            elif t == "CIRCLE":
                c, r = e.dxf.center, e.dxf.radius
                if not (bb[0]-M-r <= c.x <= bb[2]+M+r and bb[1]-M-r <= c.y <= bb[3]+M+r):
                    continue
                type_layer[(t, e.dxf.layer)] += 1
                circles.append((e.dxf.layer, c.x, c.y, r))
            elif t == "LWPOLYLINE":
                try:
                    pts = [(p[0], p[1]) for p in e.get_points("xy")]
                except Exception:
                    continue
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                if not seg_intersects_bb(min(xs), min(ys), max(xs), max(ys), bb, M):
                    continue
                type_layer[(t, e.dxf.layer)] += 1
                plines.append((e.dxf.layer, e.closed, pts))
                is_rect = e.closed and len(pts) == 5
                w, h = max(xs)-min(xs), max(ys)-min(ys)
                if is_rect:
                    rects.append((e.dxf.layer, min(xs), min(ys), w, h))
            elif t == "TEXT":
                ix, iy = e.dxf.insert.x, e.dxf.insert.y
                if not (bb[0]-M <= ix <= bb[2]+M and bb[1]-M <= iy <= bb[3]+M):
                    continue
                type_layer[(t, e.dxf.layer)] += 1
                texts.append((e.dxf.layer, ix, iy, e.dxf.text, e.dxf.height))
            elif t == "ARC":
                c = e.dxf.center
                if not (bb[0]-M <= c.x <= bb[2]+M and bb[1]-M <= c.y <= bb[3]+M):
                    continue
                type_layer[(t, e.dxf.layer)] += 1
                import math as _m
                r = e.dxf.radius
                a1 = _m.radians(e.dxf.start_angle); a2 = _m.radians(e.dxf.end_angle)
                arcs.append((e.dxf.layer, c.x, c.y, r,
                             c.x + r*_m.cos(a1), c.y + r*_m.sin(a1),
                             c.x + r*_m.cos(a2), c.y + r*_m.sin(a2)))
        except Exception:
            continue

    with open(PKL, "wb") as f:
        pickle.dump({"bbox": bb, "lines": lines, "circles": circles,
                     "rects": rects, "texts": texts, "arcs": arcs,
                     "plines": plines}, f)
    print(f"\nsaved {PKL}", flush=True)

    print("\n=== (type, layer) counts inside frame52 ===")
    for (t, l), c in type_layer.most_common(40):
        print(f"{t:12s} {l:20s} {c}")

    print(f"\nLINE: {len(lines)}  CIRCLE: {len(circles)}  rects: {len(rects)}  TEXT: {len(texts)}  ARCs: {len(arcs)}  PLINEs: {len(plines)}")
    if circles:
        rs = sorted(c[3] for c in circles)
        print(f"CIRCLE radius min={rs[0]:.2f} max={rs[-1]:.2f} median={rs[len(rs)//2]:.2f}")
        rl = Counter(c[0] for c in circles)
        print("CIRCLE layers:", dict(rl.most_common(10)))
    if rects:
        print("--- largest 25 rects ---")
        for r in sorted(rects, key=lambda r: -(r[3]*r[4]))[:25]:
            print(f"  layer={r[0]:16s} x={r[1]:.1f} y={r[2]:.1f} w={r[3]:.1f} h={r[4]:.1f}")
        rlc = Counter(r[0] for r in rects)
        print("rect layers:", dict(rlc.most_common(10)))

    print("\n--- sample TEXTs (80) ---")
    for l, x, y, s, h in texts[:80]:
        print(f"  [{l}] h={h:.1f} ({x:.1f},{y:.1f}) {s!r}")
    pin_like = [s for _,_,_,s,_ in texts if re.match(r"^[A-Z0-9]+-\d+$", s.strip())]
    wire_like = [s for _,_,_,s,_ in texts if re.match(r"^[A-Z]{2,}\d+_\d+", s.strip())]
    print(f"\npin-like: {len(pin_like)} e.g. {pin_like[:12]}")
    print(f"wire-like: {len(wire_like)} e.g. {wire_like[:12]}")
    print("\n--- PLINE detail (closed, npts, layer, bbox) ---")
    for layer, closed, pts in plines:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        print(f"  layer={layer} closed={closed} npts={len(pts)} "
              f"x=[{min(xs):.1f},{max(xs):.1f}] y=[{min(ys):.1f},{max(ys):.1f}]")
    print("\n--- ARC detail (layer, center, r) first 15 ---")
    for a in arcs[:15]:
        print(f"  layer={a[0]} c=({a[1]:.1f},{a[2]:.1f}) r={a[3]:.2f}")

if __name__ == "__main__":
    main()
