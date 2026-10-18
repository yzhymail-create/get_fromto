#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两跳管脚匹配：管脚文本 -> Hough 端子圆环 -> 骨架像素重叠 -> 导线 net。

根因（2026-10-18 框#112）：导线在端子圆环处闭环相切，骨架上没有度数
为1的端点；旧逻辑"管脚文本->最近端点<80px"依赖骨架毛刺运气，
BD184-2 中、BD184-1 不中。改按画法约定做两跳，不依赖端点距离。
"""
import os
import re
import numpy as np
import cv2

PIN_RE = re.compile(r'^[A-Z]{2,}\d+-\d+[A-Z]?$')


def norm(s):
    return re.sub(r'\s+', '', s or '').upper()


def cx(a):
    return (a[0] + a[2]) / 2


def cy(a):
    return (a[1] + a[3]) / 2


def detect_rings(gray, pin_texts, log=None):
    """每个管脚文本窗口内 Hough 找圆。返回 {pin_text: (rcx, rcy, r)}。"""
    H, W = gray.shape
    rings = {}
    for t in pin_texts:
        name = t['text'].strip()
        a = t['aabb']
        px, py = cx(a), cy(a)
        x0, x1 = max(0, int(px - 70)), min(W, int(px + 70))
        y0, y1 = max(0, int(py - 70)), min(H, int(py + 70))
        win = gray[y0:y1, x0:x1]
        blur = cv2.GaussianBlur(win, (5, 5), 1.0)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=80, param2=14,
                                   minRadius=5, maxRadius=16)
        best = None
        if circles is not None:
            for (ccx, ccy, r) in circles[0]:
                gx, gy = ccx + x0, ccy + y0
                # 排除字母字腔（B/D/8/0 等）：圆心必须在管脚文本框外面
                if (a[0] - 3 <= gx <= a[2] + 3 and
                        a[1] - 3 <= gy <= a[3] + 3):
                    continue
                d = ((gx - px) ** 2 + (gy - py) ** 2) ** 0.5
                if d < 60 and (best is None or d < best[3]):
                    best = (gx, gy, float(r), d)
        if best is not None:
            rings[name] = best[:3]
    if log is not None:
        log.append('rings detected=%d/%d' % (len(rings), len(pin_texts)))
    return rings


def border_net_ids(wire_nets, devices, skel_shape):
    """器件边框 net：像素大部分落在器件边框带上。返回 set(net_id)。"""
    H, W = skel_shape
    out = set()
    for ni, n in enumerate(wire_nets):
        px = n['pixels']
        if len(px) == 0:
            continue
        hit = 0
        for d in devices:
            x, y, bw, bh = [int(v) for v in d['bbox_xywh']]
            # 边框带：框外扩 8px 的环
            on_border = (
                (np.abs(px[:, 0] - y) <= 8) |
                (np.abs(px[:, 0] - (y + bh)) <= 8) |
                (np.abs(px[:, 1] - x) <= 8) |
                (np.abs(px[:, 1] - (x + bw)) <= 8)
            )
            inside = ((px[:, 0] >= y - 8) & (px[:, 0] <= y + bh + 8) &
                      (px[:, 1] >= x - 8) & (px[:, 1] <= x + bw + 8))
            hit += int(np.sum(on_border & inside))
        if hit > 0.5 * len(px):
            out.add(ni)
    return out


def associate_pins_v2(gray, texts, wire_nets, devices, log):
    """返回 pin_match {pin_text: (net_id, rcx, rcy, dist)}，dist 为圆环中心
    到 net 像素的最小距离（恒小，应 <15px；仅作记录）。"""
    pin_texts = [t for t in texts if PIN_RE.match(norm(t['text']))]
    rings = detect_rings(gray, pin_texts, log)
    border_ids = border_net_ids(wire_nets, devices, gray.shape)
    if log is not None:
        log.append('border nets excluded=%d' % len(border_ids))

    # net 像素 bbox 预过滤
    net_bboxes = []
    for n in wire_nets:
        px = n['pixels']
        if len(px):
            net_bboxes.append((px[:, 0].min(), px[:, 1].min(),
                               px[:, 0].max(), px[:, 1].max()))
        else:
            net_bboxes.append((0, 0, -1, -1))

    pin_match = {}
    unmatched = []
    for t in pin_texts:
        name = t['text'].strip()
        ring = rings.get(name)
        if ring is None:
            unmatched.append(name + '(no ring)')
            continue
        rcx, rcy, rr = ring
        rad = rr + 6
        x0, x1 = rcx - rad, rcx + rad
        y0, y1 = rcy - rad, rcy + rad
        best_ni, best_d = None, 1e9
        for ni, n in enumerate(wire_nets):
            if ni in border_ids:
                continue
            by0, bx0, by1, bx1 = net_bboxes[ni]
            if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
                continue
            px = n['pixels']
            sel = px[(px[:, 0] >= y0) & (px[:, 0] <= y1) &
                     (px[:, 1] >= x0) & (px[:, 1] <= x1)]
            if len(sel) == 0:
                continue
            d = float(np.min(np.hypot(sel[:, 1] - rcx, sel[:, 0] - rcy)))
            if d < best_d:
                best_d, best_ni = d, ni
        if best_ni is not None:
            pin_match[name] = (best_ni, rcx, rcy, round(best_d, 1))
        else:
            unmatched.append(name)
    if log is not None:
        log.append('pins matched=%d/%d' % (len(pin_match), len(pin_texts)))
        if unmatched:
            log.append('WARN pins unmatched: %s' % unmatched[:12])
    net_eps = {}
    for ni, n in enumerate(wire_nets):
        net_eps[ni] = [(int(x), int(y)) for (y, x) in n['endpoints']]
    return {'pin_match': pin_match, 'net_eps': net_eps}
