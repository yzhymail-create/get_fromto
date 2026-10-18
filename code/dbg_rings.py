#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证：HoughCircles 在管脚文本窗口内检测端子圆环。"""
import os
import sys
import json
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/hatch/workspace/.pylibs')

OUT = os.path.join(HERE, 'qa', 'frames', 'pilot_112')
gray = cv2.imread(os.path.join(OUT, 'render.png'), cv2.IMREAD_GRAYSCALE)
texts = json.load(open(os.path.join(OUT, 'ocr_texts.json'), encoding='utf-8'))

import re
PIN_RE = re.compile(r'^[A-Z]{2,}\d+-\d+[A-Z]?$')
def norm(s): return re.sub(r'\s+', '', s or '').upper()
def cx(a): return (a[0] + a[2]) / 2
def cy(a): return (a[1] + a[3]) / 2

pins = [(t['text'].strip(), t['aabb']) for t in texts
        if PIN_RE.match(norm(t['text']))]
print('pin texts:', len(pins))

H, W = gray.shape
found = 0
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
for name, a in pins:
    px, py = cx(a), cy(a)
    # 窗口：文本周围 120x120（圆环紧贴文本）
    x0, x1 = max(0, int(px - 70)), min(W, int(px + 70))
    y0, y1 = max(0, int(py - 70)), min(H, int(py + 70))
    win = gray[y0:y1, x0:x1]
    # 高斯模糊 + Hough 找圆（端子圆环内圈半径约 5~12px）
    blur = cv2.GaussianBlur(win, (5, 5), 1.0)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=80, param2=14,
                               minRadius=5, maxRadius=16)
    best = None
    if circles is not None:
        # 选离文本中心最近的圆
        for (ccx, ccy, r) in circles[0]:
            d = ((ccx + x0 - px) ** 2 + (ccy + y0 - py) ** 2) ** 0.5
            if d < 60 and (best is None or d < best[3]):
                best = (ccx + x0, ccy + y0, r, d)
    if best is not None:
        found += 1
        cv2.circle(vis, (int(best[0]), int(best[1])), int(best[2]),
                   (0, 0, 255), 3)
        cv2.putText(vis, name, (int(best[0]) + 18, int(best[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # 没找到：标黄框
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(vis, name + '?', (x0, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

print('rings found: %d/%d' % (found, len(pins)))
# 存全图缩小版
scale = min(1.0, 2800.0 / max(H, W))
small = cv2.resize(vis, (int(W * scale), int(H * scale)))
cv2.imwrite('/tmp/dbg_rings.png', small)
print('wrote /tmp/dbg_rings.png')
