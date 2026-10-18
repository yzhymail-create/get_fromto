#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单框试点：分图框#52 的完整纯视觉分析。

流程（只看渲染像素）：
  渲染框区域 -> 二值化/细化 -> 导线网络提取
  -> 器件框检测（含标题栏过滤） -> 文字检测+OCR识别
  -> From-To（含器件关联）+ 叠加 QA 图
"""
import json, os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import render_dxf
from trace import (binarize_otsu, skeleton_of, extract_nets, filter_nets,
                   neighbor_degree)

OUT = '/home/hatch/workspace/visual_tracer/qa/frames/pilot_52'
DXF = '/home/hatch/workspace/visual_tracer/incoming/v2/' + \
    [f for f in os.listdir('/home/hatch/workspace/visual_tracer/incoming/v2/') if f.endswith('.dxf')][0]
# 框#52 (id=52) bbox（图纸单位）
BBOX = (109613.9, -6434.9, 110451.5, -5860.1)
PX_PER_UNIT = 8.0
os.makedirs(OUT, exist_ok=True)
log = []


def say(s):
    print(s, flush=True)
    log.append(s)


def find_rectangles(fg, min_area=800, max_area_ratio=0.5):
    """找矩形轮廓。返回 [(x,y,w,h), ...]（像素坐标）。"""
    H, W = fg.shape
    img = (fg.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > H * W * max_area_ratio:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # 检查四角接近直角
            pts = approx.reshape(4, 2).astype(float)
            ok = True
            for i in range(4):
                v1 = pts[(i - 1) % 4] - pts[i]
                v2 = pts[(i + 1) % 4] - pts[i]
                cosang = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
                if cosang > 0.25:  # 偏离直角太多
                    ok = False
                    break
            if ok:
                x, y, w, h = cv2.boundingRect(approx)
                rects.append((x, y, w, h))
    return rects


def main():
    cache_path = os.path.join(OUT, 'trace_cache.pkl')
    # 1. 渲染
    t0 = time.time()
    png = os.path.join(OUT, 'render.png')
    w, h = render_dxf(DXF, png, bbox=BBOX, px_per_unit=PX_PER_UNIT)
    say('render: %dx%d px (%.1fs)' % (w, h, time.time() - t0))
    gray = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    H, W = gray.shape

    # 2-3. 二值化 + 细化 + 导线提取
    fg = binarize_otsu(gray)
    say('binarize fg_ratio=%.4f' % fg.mean())
    skel = skeleton_of(fg)
    say('skeleton px=%d' % skel.sum())
    nets, wire_width, dots = extract_nets(fg, skel)
    say('raw nets=%d wire_width=%.1fpx dots=%d' % (len(nets), wire_width, len(dots)))
    wire_nets, small_nets = filter_nets(nets, min_pixels=50)
    say('wire nets=%d small=%d' % (len(wire_nets), len(small_nets)))
    degree = neighbor_degree(skel)

    # 4. 器件框检测
    rects = find_rectangles(fg, min_area=1500, max_area_ratio=0.4)
    say('rectangle candidates=%d' % len(rects))
    # 4a. 标题栏过滤：底部条带内的网格格子剔除
    title_top = int(H * 0.80)   # 标题栏大致从 80% 高度开始
    header_bottom = int(H * 0.10)  # 顶部表头
    # 底部条带内找密集小格：尺寸相近且数量多 -> 视为标题栏格子
    bottom_cells = [r for r in rects if r[1] > title_top]
    say('bottom strip candidates=%d' % len(bottom_cells))
    # 网格判定：底部候选若宽高相近者>=4 个，整批视为标题栏格子
    def similar(a, b):
        return abs(a[2] - b[2]) < 40 and abs(a[3] - b[3]) < 40
    grid_flags = set()
    for i, a in enumerate(bottom_cells):
        cnt = sum(1 for b in bottom_cells if similar(a, b))
        if cnt >= 4:
            grid_flags.add(i)
    say('bottom grid cells excluded=%d' % len(grid_flags))

    devices = []
    for (x, y, bw, bh) in rects:
        # 剔除：底部标题栏网格、顶部表头小格
        if y > title_top:
            continue
        if y + bh < header_bottom and bw * bh < 60000:
            continue
        # 要求：边框上有导线接入（border 像素度数>=3）或内部有文字
        # 取 border 附近的骨架度数
        ring = np.zeros_like(skel, dtype=bool)
        cv2.rectangle(ring, (x - 6, y - 6), (x + bw + 6, y + bh + 6), True, 12)
        inner = np.zeros_like(skel, dtype=bool)
        cv2.rectangle(inner, (x + 4, y + 4), (x + bw - 4, y + bh - 4), True, -1)
        ring = ring & ~inner
        border_hit = bool(np.any(ring & skel))
        # 边框上的 junction（度数>=3）数量
        junc_on_border = int(np.sum((degree >= 3) & ring))
        if not border_hit:
            continue
        devices.append({'bbox_xywh': [int(x), int(y), int(bw), int(bh)],
                        'border_junctions': junc_on_border})
    # 按 x 排序编号
    devices.sort(key=lambda d: (d['bbox_xywh'][1] // 200, d['bbox_xywh'][0]))
    for i, d in enumerate(devices):
        d['id'] = i
    say('device boxes kept=%d' % len(devices))

    # 5. 文字检测 + OCR（tesseract 可用则识别；结果缓存）
    from ocr import find_text_regions, ocr_image, attach_text_to_nets
    import pickle
    ocr_cache = os.path.join(OUT, 'ocr_cache.pkl')
    regions = find_text_regions(fg)
    say('text regions detected=%d' % len(regions))
    texts = []
    if os.path.exists(ocr_cache):
        with open(ocr_cache, 'rb') as f:
            texts = pickle.load(f)
        say('ocr loaded from cache=%d' % len(texts))
    else:
        try:
            texts = ocr_image(gray, fg, max_regions=300)
            say('ocr recognized=%d' % len(texts))
            with open(ocr_cache, 'wb') as f:
                pickle.dump(texts, f)
        except Exception as e:
            say('ocr failed: %s' % e)
    # 文字绑到器件框（中心点在框内）
    for d in devices:
        x, y, bw, bh = d['bbox_xywh']
        inside = []
        for (bx, by, bw2, bh2), t, conf in texts:
            cx, cy = bx + bw2 / 2, by + bh2 / 2
            if x < cx < x + bw and y < cy < y + bh:
                inside.append(t)
        d['texts'] = inside
    attached = attach_text_to_nets([(b, t, c) for b, t, c in texts], wire_nets) \
        if texts else {}

    # 6b. 端子圆环检测 + 导线-端子-器件关联
    # （导线在端子圆环处终止，圆环旁有引脚标注；OCR 不可用时按几何最近器件关联）
    def find_terminal_circles(gray):
        blur = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 30,
                                   param1=100, param2=18,
                                   minRadius=10, maxRadius=22)
        if circles is None:
            return []
        return [(int(x), int(y), int(r)) for x, y, r in np.uint16(np.around(circles))[0]]

    terminals = find_terminal_circles(gray)
    say('terminal circles=%d' % len(terminals))

    # 端子圆 -> 最近导线端点（50px内）-> 最近器件框（400px内）
    def nearest_endpoint(cx, cy, max_d=50):
        best, best_d = None, max_d
        for i, n in enumerate(wire_nets):
            for (ey, ex) in n['endpoints']:
                d = ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5
                if d < best_d:
                    best_d, best = d, i
        return best

    def nearest_device(cx, cy, max_d=400):
        best, best_d = None, max_d
        for d in devices:
            x, y, bw, bh = d['bbox_xywh']
            dx = max(x - cx, 0, cx - (x + bw))
            dy = max(y - cy, 0, cy - (y + bh))
            dd = (dx * dx + dy * dy) ** 0.5
            if dd < best_d:
                best_d, best = dd, d['id']
        return best

    # 导线经端子圆关联到的器件
    net_terminal_device = {}  # net_idx -> (terminal_xy, dev_id)
    for (cx, cy, r) in terminals:
        # 圆环 50px 内的所有导线端点（两侧的导线都要关联）
        near_nets = set()
        for i, n in enumerate(wire_nets):
            for (ey, ex) in n['endpoints']:
                if ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5 < 50:
                    near_nets.add(i)
                    break
        if not near_nets:
            continue
        dev = nearest_device(cx, cy)
        if dev is None:
            continue
        for ni in near_nets:
            if ni not in net_terminal_device:
                net_terminal_device[ni] = ((cx, cy), dev)
    say('nets linked via terminals=%d' % len(net_terminal_device))

    # 缓存中间结果（供关联逻辑迭代）
    import pickle
    with open(cache_path, 'wb') as f:
        pickle.dump({'wire_nets': wire_nets, 'devices': devices,
                     'terminals': terminals, 'W': W, 'H': H,
                     'net_terminal_device': net_terminal_device}, f)

    # 6. From-To：导线网络关联器件框（按像素 proximity，不只看端点）
    # 先识别器件框自身的边框网络（像素大比例贴合某器件框边框 -> 标记排除）
    def net_border_overlap(n, d, tol=8):
        x, y, bw, bh = d['bbox_xywh']
        px = n['pixels']
        near = np.sum((np.abs(px[:, 1] - x) < tol) | (np.abs(px[:, 1] - (x + bw)) < tol) |
                      (np.abs(px[:, 0] - y) < tol) | (np.abs(px[:, 0] - (y + bh)) < tol))
        return near / max(len(px), 1)

    net_kind = {}  # idx -> ('border', dev_id) | ('wire',)
    for i, n in enumerate(wire_nets):
        best, best_r = None, 0
        for d in devices:
            r = net_border_overlap(n, d)
            if r > best_r:
                best_r, best = r, d['id']
        net_kind[i] = ('border', best) if best_r > 0.4 else ('wire', None)

    def net_touches_device(n, d, tol=18):
        """导线网络是否接到器件框边上。
        边框自身网络已由 net_border_overlap 提前排除，这里只需 proximity。"""
        x, y, bw, bh = d['bbox_xywh']
        px = n['pixels']
        in_ring = ((px[:, 1] > x - tol) & (px[:, 1] < x + bw + tol) &
                   (px[:, 0] > y - tol) & (px[:, 0] < y + bh + tol))
        return bool(np.any(in_ring))

    # 框边界端口：端点距图像边缘 < 25px
    rows = []
    n_ports = n_dangling = n_border = 0
    for i, n in enumerate(wire_nets):
        kind, info = net_kind[i]
        if kind == 'border':
            n_border += 1
            continue
        devs = [d['id'] for d in devices
                if len(n['pixels']) > 200 and net_touches_device(n, d)]
        ends = ['器件#%d' % d for d in devs]
        # 经端子圆环的关联（导线在圆环处终止，圆环指向器件）——主要关联方式
        if i in net_terminal_device:
            (cx, cy), tdev = net_terminal_device[i]
            tag = '器件#%d(经端子)' % tdev
            if tag not in ends and '器件#%d' % tdev not in ends:
                ends.append(tag)
        for ep in n['endpoints']:
            ey, ex = ep
            if ex < 25 or ey < 25 or ex > W - 25 or ey > H - 25:
                if '框边界端口' not in ends:
                    ends.append('框边界端口')
                    n_ports += 1
        if not ends:
            ends = ['悬空']
            n_dangling += 1
        tlist = [t for t, _, _ in attached.get(i, []) if t.strip()]
        px = n['pixels']
        rows.append({
            'id': i,
            'label': '; '.join(tlist) if tlist else '未标注',
            'endpoints': ends,
            'n_endpoints': n['n_endpoints'],
            'length_px': len(px),
        })
    with open(os.path.join(OUT, 'from_to.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, 'devices.json'), 'w', encoding='utf-8') as f:
        json.dump(devices, f, ensure_ascii=False, indent=1)
    say('from_to rows=%d devices=%d ports=%d dangling=%d border_nets_excluded=%d'
        % (len(rows), len(devices), n_ports, n_dangling, n_border))

    # 7. 叠加 QA 图：导线颜色 + 器件框编号 + 框边界
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    vis = np.stack([gray] * 3, axis=-1).copy()
    colors = (cm.get_cmap('hsv')(np.linspace(0, 1, max(1, min(len(wire_nets), 400))))[:, :3] * 255).astype(np.uint8)
    for i, n in enumerate(wire_nets[:400]):
        px = n['pixels']
        vis[px[:, 0], px[:, 1]] = colors[i % len(colors)]
    for d in devices:
        x, y, bw, bh = d['bbox_xywh']
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 4)
        cv2.putText(vis, str(d['id']), (x + 6, y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.rectangle(vis, (2, 2), (W - 3, H - 3), (255, 0, 0), 6)
    cv2.putText(vis, 'FRAME #52', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (255, 0, 0), 3)
    scale = min(1.0, 2800.0 / max(H, W))
    if scale < 1.0:
        vis = cv2.resize(vis, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(OUT, 'overlay.png'), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    say('overlay -> overlay.png')

    with open(os.path.join(OUT, 'run.log'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log) + '\n')
    say('DONE')


if __name__ == '__main__':
    main()
