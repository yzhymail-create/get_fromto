#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用单框纯视觉管线：任意分图框 -> From-To 表 + overlay。

流程（只看渲染像素，绝不解析 DXF 矢量图元做追踪）：
  渲染框区域 -> 二值化/细化 -> 导线网络提取
  -> 器件框检测（含标题栏过滤） -> RapidOCR 分块识别
  -> 管脚锚定 -> From-To 三段链 -> Excel + overlay

From-To 三段链（P0-1 统一正式入口）：
  a. build_fromto.build_rows（net-hub 法）
  b. build_label_rows（按线号组装；仅 pilot_112 生效，其余 fallback 到 a）
  c. add_jump_targets（跨图跳转目标补录）

用法：python3 run_frame.py <frame_id> [--no-render]
  --no-render：复用已有 render.png（跳过渲染；OCR/管脚等仍用缓存则秒级）
约束：skill local-visual-tracer（开工前已读）；Windows 硬约束。
"""
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from render import render_dxf
from trace import (binarize_otsu, skeleton_of, extract_nets, filter_nets,
                   neighbor_degree)
from ocr import recognize_tiled

PIN_RE = re.compile(r'^[A-Z]{2,}\d+-\d+[A-Z]?$')
PIN_ANCHOR_MAX_PX = 80  # #52 实测管脚-端点距离 15~45px


def imread_gray(path):
    """读灰度图（PIL，支持中文路径；cv2.imread 在 Windows 上不支持中文路径）。

    返回 uint8 numpy 数组。PIL 的 convert('L') 直接得灰度，
    避免 RGB/BGR 通道反转问题。
    """
    from PIL import Image
    return np.array(Image.open(path).convert('L'))


def norm(s):
    """标准化文本：去空白转大写，并统一横杠变体。

    跨平台 OCR 常把 '-' 误读为 '–'（en-dash U+2013）/ '—'（em-dash U+2014），
    导致 PIN_RE 匹配失败（2026-10-18 Windows 实测：管脚 27→1）。统一后再匹配。
    """
    s = re.sub(r'\s+', '', s or '').upper()
    return re.sub(r'[–—−‐]', '-', s)


def find_rectangles(fg, min_area=1500, max_area_ratio=0.4):
    H, W = fg.shape
    img = fg.astype(np.uint8) * 255
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > H * W * max_area_ratio:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype(float)
            ok = True
            for i in range(4):
                v1 = pts[(i - 1) % 4] - pts[i]
                v2 = pts[(i + 1) % 4] - pts[i]
                cosang = abs(np.dot(v1, v2) /
                             (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
                if cosang > 0.25:
                    ok = False
                    break
            if ok:
                x, y, w, h = cv2.boundingRect(approx)
                rects.append((x, y, w, h))
    return rects


def detect_devices(fg, skel, degree, log):
    H, W = fg.shape
    rects = find_rectangles(fg)
    log.append('rectangle candidates=%d' % len(rects))
    title_top = int(H * 0.80)
    header_bottom = int(H * 0.10)
    bottom_cells = [r for r in rects if r[1] > title_top]

    def similar(a, b):
        return abs(a[2] - b[2]) < 40 and abs(a[3] - b[3]) < 40

    grid_flags = set()
    for i, a in enumerate(bottom_cells):
        if sum(1 for b in bottom_cells if similar(a, b)) >= 4:
            grid_flags.add(i)
    log.append('bottom grid cells excluded=%d' % len(grid_flags))

    devices = []
    for (x, y, bw, bh) in rects:
        if y > title_top:
            continue
        if y + bh < header_bottom and bw * bh < 60000:
            continue
        # uint8 代替 bool：OpenCV 4.14 不支持 bool 数组（Windows 实测）。
        # 先画外圈 255，再挖内圈 0，等价于 bool 版的 ring & ~inner。
        ring = np.zeros(skel.shape, dtype=np.uint8)
        cv2.rectangle(ring, (x - 6, y - 6), (x + bw + 6, y + bh + 6),
                      255, 12)
        cv2.rectangle(ring, (x + 4, y + 4), (x + bw - 4, y + bh - 4),
                      0, -1)
        if not np.any((ring > 0) & (skel > 0)):
            continue
        junc = int(np.sum((degree >= 3) & (ring > 0)))
        devices.append({'bbox_xywh': [int(x), int(y), int(bw), int(bh)],
                        'border_junctions': junc, 'texts': []})
    devices.sort(key=lambda d: (d['bbox_xywh'][1] // 200,
                                d['bbox_xywh'][0]))
    for i, d in enumerate(devices):
        d['id'] = i
    log.append('device boxes kept=%d' % len(devices))
    return devices


def associate_pins_v2(gray, texts, wire_nets, devices, log):
    """两跳匹配（2026-10-18 框#112 根因修复）：

    根因：导线在端子圆环处闭环相切，骨架上没有度数为1的端点；
    旧逻辑"管脚文本->最近端点<80px"依赖骨架毛刺运气（BD184-2 中、
    BD184-1 不中）。按画法约定（skill §5）：端子圆环必在管脚文本
    25px 内，且圆环骨架属于导线 net——直接找文本盘内骨架像素
    最多的 wire net，不依赖端点距离。
    """
    from pin_ring_match import border_net_ids, norm as _norm, cx as _cx, cy as _cy
    pin_texts = [t for t in texts if PIN_RE.match(_norm(t['text']))]
    border_ids = border_net_ids(wire_nets, devices, gray.shape)
    if log is not None:
        log.append('border nets excluded=%d' % len(border_ids))
    R = 25
    pin_match, unmatched = {}, []
    for t in pin_texts:
        name = t['text'].strip()
        a = t['aabb']
        px, py = _cx(a), _cy(a)
        best_ni, best_c = None, 0
        for ni, n in enumerate(wire_nets):
            if ni in border_ids:
                continue
            p = n['pixels']
            if (p[:, 1].max() < px - R or p[:, 1].min() > px + R or
                    p[:, 0].max() < py - R or p[:, 0].min() > py + R):
                continue
            c = int(np.sum((p[:, 1] - px) ** 2 + (p[:, 0] - py) ** 2 <= R * R))
            if c > best_c:
                best_c, best_ni = c, ni
        if best_ni is not None and best_c >= 3:
            pin_match[name] = (best_ni, float(px), float(py), 0.0)
        else:
            unmatched.append(name)
    if log is not None:
        log.append('pins matched=%d/%d (ring-disk v2)' %
                   (len(pin_match), len(pin_texts)))
        if unmatched:
            log.append('WARN pins unmatched: %s' % unmatched[:12])
    net_eps = {}
    for ni, n in enumerate(wire_nets):
        net_eps[ni] = [(int(x), int(y)) for (y, x) in n['endpoints']]
    return {'pin_match': pin_match, 'net_eps': net_eps}


def associate_pins(texts, wire_nets, log):
    """管脚号文本 -> 最近导线端点。返回 pin_assoc dict。

    （#52 沉淀逻辑：管脚标注文本锚定最近导线端点，实测距离 15~45px）
    2026-10-18 框#112 废弃：改用 associate_pins_v2（圆环盘匹配）。
    保留作 fallback。
    """
    pin_match = {}
    for t in texts:
        if not PIN_RE.match(norm(t['text'])):
            continue
        a = t['aabb']
        px, py = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        best, bd = None, PIN_ANCHOR_MAX_PX
        for ni, n in enumerate(wire_nets):
            for (ey, ex) in n['endpoints']:
                d = ((ex - px) ** 2 + (ey - py) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, (ni, ex, ey)
        if best is not None:
            ni, ex, ey = best
            pin_match[t['text'].strip()] = (ni, ex, ey, round(bd, 1))
    net_eps = {}
    for ni, n in enumerate(wire_nets):
        net_eps[ni] = [(int(ex), int(ey)) for (ey, ex) in n['endpoints']]
    from collections import Counter
    cnt = Counter(ni for ni, _, _, _ in pin_match.values())
    multi = {k: v for k, v in cnt.items() if v > 2}
    log.append('pins matched=%d (nets: %d)' % (len(pin_match), len(cnt)))
    if multi:
        log.append('WARN nets with >2 pins: %s' % multi)
    unmatched = [t['text'] for t in texts
                 if PIN_RE.match(norm(t['text']))
                 and t['text'].strip() not in pin_match]
    if unmatched:
        log.append('WARN pins unmatched(>%dpx): %s'
                   % (PIN_ANCHOR_MAX_PX, unmatched[:10]))
    return {'pin_match': pin_match, 'net_eps': net_eps}


def make_overlay(gray, wire_nets, devices, frame_id, out_path, log):
    import matplotlib
    matplotlib.use('Agg')
    try:
        _cmap = matplotlib.colormaps['hsv']
    except AttributeError:
        from matplotlib import cm as _cm
        _cmap = _cm.get_cmap('hsv')
    H, W = gray.shape
    vis = np.stack([gray] * 3, axis=-1).copy()
    colors = (_cmap(
        np.linspace(0, 1, max(1, min(len(wire_nets), 400)))
    )[:, :3] * 255).astype(np.uint8)
    for i, n in enumerate(wire_nets[:400]):
        px = n['pixels']
        vis[px[:, 0], px[:, 1]] = colors[i % len(colors)]
    for d in devices:
        x, y, bw, bh = d['bbox_xywh']
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 4)
        cv2.putText(vis, str(d['id']), (x + 6, y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.rectangle(vis, (2, 2), (W - 3, H - 3), (255, 0, 0), 6)
    cv2.putText(vis, 'FRAME #%d' % frame_id, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 0, 0), 3)
    scale = min(1.0, 2800.0 / max(H, W))
    if scale < 1.0:
        vis = cv2.resize(vis, (int(W * scale), int(H * scale)),
                         interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    log.append('overlay -> %s' % os.path.basename(out_path))


def main():
    no_render = '--no-render' in sys.argv
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    frame_id = int(argv[0]) if argv else 112
    t_all = time.time()
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)

    BASE = os.path.dirname(HERE)
    frames = json.load(open(os.path.join(
        BASE, 'frames', 'subframes.json'), encoding='utf-8'))['frames']
    f = next(x for x in frames if x['id'] == frame_id)
    x1, y1, x2, y2 = f['units']
    say('frame #%d units bbox=%s' % (frame_id, f['units']))

    OUT = os.path.join(BASE, 'frames', 'pilot_%d' % frame_id)
    os.makedirs(OUT, exist_ok=True)
    v2dir = os.path.join(BASE, 'incoming')
    DXF = os.path.join(v2dir, [x for x in os.listdir(v2dir)
                               if x.endswith('.dxf')][0])

    # 1. 渲染（--no-render 时复用已有 render.png）
    t0 = time.time()
    png = os.path.join(OUT, 'render.png')
    if no_render and os.path.exists(png):
        gray0 = imread_gray(png)
        h, w = gray0.shape
        del gray0
        say('render: skipped (--no-render), reuse %dx%d' % (w, h))
    else:
        w, h = render_dxf(DXF, png, bbox=(x1, y1, x2, y2), px_per_unit=8.0)
        say('render: %dx%d px (%.1fs)' % (w, h, time.time() - t0))
    gray = imread_gray(png)
    H, W = gray.shape

    # 2-3. 二值化 + 细化 + 导线提取
    fg = binarize_otsu(gray)
    say('binarize fg_ratio=%.4f' % fg.mean())
    skel = skeleton_of(fg)
    say('skeleton px=%d' % skel.sum())
    nets, wire_width, dots = extract_nets(fg, skel)
    say('raw nets=%d wire_width=%.1fpx dots=%d'
        % (len(nets), wire_width, len(dots)))
    wire_nets, small_nets = filter_nets(nets, min_pixels=50)
    say('wire nets=%d small=%d' % (len(wire_nets), len(small_nets)))
    degree = neighbor_degree(skel)

    # 4. 器件框
    devices = detect_devices(fg, skel, degree, log)
    with open(os.path.join(OUT, 'devices.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(devices, fh, ensure_ascii=False, indent=1)

    # 5. OCR（分块端到端识别，结果缓存）
    # 注意：空/损坏缓存视为未命中，重跑 OCR（2026-10-18 教训：
    # pilot_52 的 ocr_cache.pkl 为空，旧逻辑直接信任导致 ocr_texts.json 被清空）
    ocr_cache = os.path.join(OUT, 'ocr_cache.pkl')
    texts_path = os.path.join(OUT, 'ocr_texts.json')
    items = None
    if os.path.exists(ocr_cache):
        try:
            with open(ocr_cache, 'rb') as fh:
                items = pickle.load(fh)
        except Exception as e:
            say('ocr cache corrupt (%s), re-running' % e)
            items = None
    if items:
        say('ocr loaded from cache=%d' % len(items))
    else:
        if os.path.exists(ocr_cache):
            say('ocr cache empty/invalid, running OCR')
        t0 = time.time()
        items = recognize_tiled(gray)
        say('ocr tiled: %d texts (%.1fs)' % (len(items), time.time() - t0))
        with open(ocr_cache, 'wb') as fh:
            pickle.dump(items, fh)
    texts = [{'aabb': [float(v) for v in aabb], 'text': t, 'conf': c}
             for aabb, t, c in items]
    # 保留已有的人工补丁（manual_patch），不被缓存覆盖掉
    # （2026-10-18 教训：框#112 有 3 条人工补线号，只存在 ocr_texts.json）
    if os.path.exists(texts_path):
        try:
            old_texts = json.load(open(texts_path, encoding='utf-8'))
            patched = [t for t in old_texts if t.get('manual_patch')]
            if patched:
                have = set((tuple(t['aabb']), t['text']) for t in texts)
                for t in patched:
                    if (tuple(t['aabb']), t['text']) not in have:
                        texts.append(t)
                say('kept %d manual_patch texts' % len(patched))
        except Exception as e:
            say('WARN failed to keep manual_patch: %s' % e)
    with open(texts_path, 'w', encoding='utf-8') as fh:
        json.dump(texts, fh, ensure_ascii=False, indent=1)

    # 6. 管脚锚定（v2：圆环盘匹配；旧端点法已废弃，见函数注释）
    pa = associate_pins_v2(gray, texts, wire_nets, devices, log)
    with open(os.path.join(OUT, 'pin_assoc.pkl'), 'wb') as fh:
        pickle.dump(pa, fh)

    # 7. From-To：三段链（P0-1 统一正式入口，2026-10-18）
    #   a. build_fromto.build_rows（net-hub 法；#52 已验证路径，保持不变）
    #   b. build_label_rows（按线号组装；仅 pilot_112 生效，其余返回空走 fallback）
    #   c. add_jump_targets（跨图跳转目标补录，直接改 xlsx）
    import build_fromto
    rows, notes_log, wire_labels = build_fromto.build_rows(OUT)
    say('from_to rows=%d wire_labels=%d' % (len(rows), len(wire_labels)))
    for n in notes_log[:10]:
        say('note: ' + n)
    xlsx = os.path.join(OUT, 'FromTo_机器版.xlsx')
    build_fromto.write_excel(rows, xlsx, frame_id=frame_id)
    say('wrote %s (build_fromto)' % xlsx)
    assembly = 'build_fromto'

    # b. 按线号组装（框#112 专属；返回空/异常则保留 a 的结果）
    try:
        import build_label_rows as blr
        lrows, _, _ = blr.build_label_rows(OUT)
        if lrows:
            build_fromto.write_excel(lrows, xlsx, frame_id=frame_id)
            rows = lrows
            assembly = 'label-rows'
            say('label-rows assembly used: %d rows (overwrote xlsx)' %
                len(lrows))
        else:
            say('label-rows skipped: empty (fallback to build_fromto)')
    except Exception as e:
        say('label-rows skipped: %s' % e)

    # c. 跨图跳转目标补录
    try:
        import add_jump_targets as ajt
        nj, nn = ajt.apply_jump_targets(OUT)
        say('jump targets: %d patched, %d no-symbol' % (nj, nn))
        jump_info = '%d patched, %d no-symbol' % (nj, nn)
    except Exception as e:
        jump_info = 'skipped: %s' % e
        say('jump targets %s' % jump_info)

    # 8. overlay
    make_overlay(gray, wire_nets, devices, frame_id,
                 os.path.join(OUT, 'overlay.png'), log)

    # 9. 报告
    n_full = sum(1 for r in rows if '出框' not in r['to_dev'])
    n_bnd = len(rows) - n_full
    n_unrec = sum(1 for r in rows if r['wire'] == '未识别')
    rep = [
        '# 框#%d 机器版验证报告（纯视觉，%s）' % (frame_id, time.strftime('%Y-%m-%d')),
        '',
        '- 渲染：%dx%d px；导线网络 %d；器件框 %d；OCR 文本 %d' %
        (w, h, len(wire_nets), len(devices), len(texts)),
        '- From-To 行数：%d（完整 %d / 边界 %d / 线号未识别 %d）' %
        (len(rows), n_full, n_bnd, n_unrec),
        '- 组装方法：%s；跨图跳转补录：%s' % (assembly, jump_info),
        '- 管脚锚定：%d 个管脚号文本匹配到导线端点' % len(pa['pin_match']),
        '',
        '## 运行日志',
        '',
    ] + log
    with open(os.path.join(OUT, '机器版验证报告.md'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(rep) + '\n')

    with open(os.path.join(OUT, 'run.log'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(log) + '\n')
    say('DONE total %.1fs' % (time.time() - t_all))


if __name__ == '__main__':
    main()
