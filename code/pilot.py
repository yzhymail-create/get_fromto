# -*- coding: utf-8 -*-
"""试点：渲染指定 bbox -> 纯视觉追踪 -> 叠加图 + 统计。

用法: python3 pilot.py --bbox x1,y1,x2,y2 --dpi-px 8 --out qa/pilot1
"""
import argparse, os, sys, time, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import render_dxf
from trace import binarize_otsu, skeleton_of, extract_nets, filter_nets, overlay_nets, estimate_wire_width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bbox', required=True, help='x1,y1,x2,y2 图纸单位')
    ap.add_argument('--px', type=float, default=8.0, help='每图纸单位像素数')
    ap.add_argument('--out', default='qa/pilot1', help='输出目录')
    ap.add_argument('--dxf', default=None)
    args = ap.parse_args()

    bbox = tuple(float(v) for v in args.bbox.split(','))
    dxf = args.dxf or glob.glob('/home/hatch/workspace/visual_tracer/incoming/v2/*.dxf')[0]
    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, 'render.png')

    t0 = time.time()
    w, h = render_dxf(dxf, png, bbox=bbox, px_per_unit=args.px)
    print('render done: %.1fs' % (time.time() - t0), flush=True)

    t0 = time.time()
    gray = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    print('image shape:', gray.shape, flush=True)
    fg = binarize_otsu(gray)
    print('foreground ratio: %.4f (%.1fs)' % (fg.mean(), time.time() - t0), flush=True)

    t0 = time.time()
    skel = skeleton_of(fg)
    print('skeleton done: %.1fs, skel pixels: %d' % (time.time() - t0, skel.sum()), flush=True)

    t0 = time.time()
    nets, wire_width, dots = extract_nets(fg, skel)
    print('extract_nets: %.1fs, raw nets: %d, wire_width≈%.1fpx, dot junctions: %d'
          % (time.time() - t0, len(nets), wire_width, len(dots)), flush=True)

    wire_nets, small_nets = filter_nets(nets, min_pixels=30)
    print('wire nets: %d, small/filtered: %d' % (len(wire_nets), len(small_nets)))

    # 统计
    lens = [len(n['pixels']) for n in wire_nets]
    eps = [n['n_endpoints'] for n in wire_nets]
    import collections
    ec = collections.Counter(eps)
    print('net pixel-length: min=%d p50=%d p90=%d max=%d' % (
        min(lens), int(np.percentile(lens, 50)), int(np.percentile(lens, 90)), max(lens)) if lens else 'none')
    print('endpoints per net:', dict(sorted(ec.items())))
    dangling = sum(1 for n in wire_nets for _ in n['endpoints'])
    print('total endpoints: %d' % dangling)

    t0 = time.time()
    overlay_nets(gray, wire_nets, os.path.join(args.out, 'overlay.png'))
    print('overlay done: %.1fs' % (time.time() - t0), flush=True)
    # 同时存一份原渲染小图便于对照
    small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(args.out, 'render_small.png'), small)
    print('ALL DONE ->', args.out, flush=True)


if __name__ == '__main__':
    main()
