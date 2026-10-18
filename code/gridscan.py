# -*- coding: utf-8 -*-
"""渲染区域选择：在 6x6 网格上统计每格图元数，选最密的格做试点。
纯属渲染预处理（决定"看哪里"），追踪本身只看像素。
"""
import ezdxf, glob, sys
sys.path.insert(0, '/home/hatch/workspace/visual_tracer')
from render import _entity_in_bbox

dxf = glob.glob('/home/hatch/workspace/visual_tracer/incoming/v2/*.dxf')[0]
doc = ezdxf.readfile(dxf)
msp = doc.modelspace()

X0, Y0, X1, Y1 = 108117.0, -12872.0, 121985.0, 1145.0
N = 6
cells = []
for i in range(N):
    for j in range(N):
        cx1 = X0 + (X1 - X0) * i / N
        cx2 = X0 + (X1 - X0) * (i + 1) / N
        cy1 = Y0 + (Y1 - Y0) * j / N
        cy2 = Y0 + (Y1 - Y0) * (j + 1) / N
        cells.append((cx1, cy1, cx2, cy2))

import time
t0 = time.time()
counts = []
for (cx1, cy1, cx2, cy2) in cells:
    c = 0
    for e in msp:
        try:
            if e.dxf.layer == '00_BACKGROUND':
                continue
        except Exception:
            pass
        if _entity_in_bbox(e, cx1, cy1, cx2, cy2, 200.0):
            c += 1
    counts.append(c)
    print('cell (%.0f,%.0f)-(%.0f,%.0f): %d entities' % (cx1, cy1, cx2, cy2, c), flush=True)
print('total %.1fs' % (time.time() - t0))
order = sorted(range(len(cells)), key=lambda i: -counts[i])
print('top3 cells:', [(cells[i], counts[i]) for i in order[:3]])
