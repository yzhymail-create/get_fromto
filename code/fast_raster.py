# -*- coding: utf-8 -*-
"""轻量光栅化器：DXF -> PNG（纯渲染前端，追踪仍只看像素）。

matplotlib 后端在 53 万图元下 OOM，改用 cv2 直接画线，
内存占用恒定（仅一张 uint8 图像）。
"""
import cv2
import numpy as np


def rasterize(msp, out_png, bbox, ppu, off_layers=('00_BACKGROUND',), thickness=2):
    """msp: 已读取的 modelspace；bbox=(x1,y1,x2,y2) 图纸单位；ppu: 每单位像素数。"""
    x1, y1, x2, y2 = bbox
    W = int((x2 - x1) * ppu)
    H = int((y2 - y1) * ppu)
    print('raster %d x %d px' % (W, H), flush=True)
    img = np.full((H, W), 255, np.uint8)

    def X(x):
        return (x - x1) * ppu

    def Y(y):
        return (y2 - y) * ppu

    n = 0
    for e in msp:
        try:
            if e.dxf.layer in off_layers:
                continue
            t = e.dxftype()
        except Exception:
            continue
        try:
            if t == 'LINE':
                s, en = e.dxf.start, e.dxf.end
                cv2.line(img, (int(X(s.x)), int(Y(s.y))),
                         (int(X(en.x)), int(Y(en.y))), 0, thickness, cv2.LINE_8)
            elif t == 'LWPOLYLINE':
                pts = [(int(X(p[0])), int(Y(p[1]))) for p in e.get_points('xy')]
                if len(pts) >= 2:
                    cv2.polylines(img, [np.array(pts, np.int32)], e.is_closed, 0, thickness, cv2.LINE_8)
            elif t == 'POLYLINE':
                pts = [(int(X(v.dxf.location.x)), int(Y(v.dxf.location.y))) for v in e.vertices]
                if len(pts) >= 2:
                    cv2.polylines(img, [np.array(pts, np.int32)], e.is_closed, 0, thickness, cv2.LINE_8)
            elif t == 'CIRCLE':
                c, r = e.dxf.center, e.dxf.radius * ppu
                if r >= 0.75:
                    cv2.circle(img, (int(X(c.x)), int(Y(c.y))), int(round(r)), 0, thickness, cv2.LINE_8)
            elif t == 'ARC':
                c, r = e.dxf.center, e.dxf.radius * ppu
                if r >= 0.75:
                    # 图像 y 翻转，角度取反
                    cv2.ellipse(img, (int(X(c.x)), int(Y(c.y))),
                                (int(round(r)), int(round(r))), 0,
                                -e.dxf.end_angle, -e.dxf.start_angle, 0, thickness, cv2.LINE_8)
        except Exception:
            pass
        n += 1
        if n % 200000 == 0:
            print('  ..%d entities' % n, flush=True)
    cv2.imwrite(out_png, img)
    print('saved', out_png, flush=True)
    return W, H
