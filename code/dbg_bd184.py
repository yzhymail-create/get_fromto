#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调试：BD184-1 vs BD184-2 区域的骨架/端点/管脚匹配。"""
import os
import sys
import pickle
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('PYTHONPATH', '/home/hatch/workspace/.pylibs')
sys.path.insert(0, '/home/hatch/workspace/.pylibs')
from trace import binarize_otsu, skeleton_of, neighbor_degree

OUT = os.path.join(HERE, 'qa', 'frames', 'pilot_112')
gray = cv2.imread(os.path.join(OUT, 'render.png'), cv2.IMREAD_GRAYSCALE)

# 区域：BD184 器件右端管脚区
X0, X1, Y0, Y1 = 1300, 1750, 1600, 1900
crop = gray[Y0:Y1, X0:X1]
fg = binarize_otsu(crop)
skel = skeleton_of(fg)
deg = neighbor_degree(skel)
ey, ex = np.nonzero((deg == 1) & skel)
print('crop 内端点数:', len(ex))
for x, y in zip(ex, ey):
    print('  endpoint (%d,%d) -> 全图 (%d,%d)' % (x, y, x + X0, y + Y0))

# 可视化：灰底 + 骨架红 + 端点绿圈 + 管脚文本框蓝
vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
vis[skel] = (0, 0, 255)
for x, y in zip(ex, ey):
    cv2.circle(vis, (int(x), int(y)), 6, (0, 255, 0), 2)
# 管脚文本框
pins = {'BD184-2': [1442, 1728, 1491, 1745], 'BD184-1': [1442, 1788, 1491, 1805]}
for name, a in pins.items():
    cv2.rectangle(vis, (int(a[0] - X0), int(a[1] - Y0)),
                  (int(a[2] - X0), int(a[3] - Y0)), (255, 0, 0), 2)
    cv2.putText(vis, name, (int(a[0] - X0), int(a[1] - Y0) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
# 已匹配的端点（pin_assoc）
pa = pickle.load(open(os.path.join(OUT, 'pin_assoc.pkl'), 'rb'))
for pname, (ni, px, py, d) in pa['pin_match'].items():
    if pname in pins:
        cv2.circle(vis, (int(px - X0), int(py - Y0)), 10, (255, 255, 0), 3)
        cv2.putText(vis, '%s->net%d' % (pname, ni),
                    (int(px - X0) + 12, int(py - Y0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
cv2.imwrite('/tmp/dbg_bd184.png', vis)
print('wrote /tmp/dbg_bd184.png')

# 再看：BD184-1 文本中心 150px 内所有端点（全图坐标）
cx1, cy1 = 1466, 1796
print('BD184-1 中心 150px 内端点:')
for x, y in zip(ex, ey):
    gx, gy = x + X0, y + Y0
    d = ((gx - cx1) ** 2 + (gy - cy1) ** 2) ** 0.5
    if d < 150:
        print('  (%d,%d) d=%.1f' % (gx, gy, d))
