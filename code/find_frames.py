# -*- coding: utf-8 -*-
"""Phase 1: 总览图多尺度矩形检测 + 包含关系树（纯视觉：只看渲染像素）。"""
import cv2, json, math
import numpy as np

def detect_rectangles(img_path, min_area_px=150, max_fill_ratio=0.98):
    """返回矩形列表，每项含 bbox_px、面积、hierarchy 信息。"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    H, W = img.shape
    # 黑线白底 -> 反转为白线黑底
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = 255 - bw
    # 轻微闭运算弥合细线 1-2px 断裂
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hier = hierarchy[0]  # [next, prev, child, parent]
    rects = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        pts = approx.reshape(4, 2).astype(float)
        # 四角接近直角
        ok = True
        for k in range(4):
            p0, p1, p2 = pts[k], pts[(k + 1) % 4], pts[(k + 2) % 4]
            v1, v2 = p0 - p1, p2 - p1
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                ok = False; break
            if abs(np.dot(v1, v2) / (n1 * n2)) > 0.18:  # cos>0.18 视为非直角
                ok = False; break
        if not ok:
            continue
        # axis-aligned：边与水平/竖直夹角 < 4°
        for k in range(4):
            p0, p1 = pts[k], pts[(k + 1) % 4]
            ang = abs(math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))) % 90
            ang = min(ang, 90 - ang)
            if ang > 4.0:
                ok = False; break
        if not ok:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w < 8 or h < 8:
            continue
        # 过滤实心大色块（填充比过高）：框线应是细线，轮廓面积应远大于bbox面积
        # 用 bbox 面积 vs 轮廓面积比判断（细线框：contourArea≈bbox面积；实心块也≈bbox面积，无法区分，暂保留）
        rects.append({
            'idx': i, 'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
            'area': float(area),
            'parent': int(hier[i][3]), 'child': int(hier[i][2]),
        })
    return rects, (W, H)


def build_tree(rects):
    """按 bbox 包含关系建树（不依赖 cv2 hierarchy，更鲁棒）。返回每框的 children 列表。"""
    n = len(rects)
    children = [[] for _ in range(n)]
    for i in range(n):
        a = rects[i]
        ax1, ay1, ax2, ay2 = a['x'], a['y'], a['x'] + a['w'], a['y'] + a['h']
        best_parent, best_area = -1, float('inf')
        for j in range(n):
            if i == j:
                continue
            b = rects[j]
            bx1, by1, bx2, by2 = b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']
            # a 被 b 包含（留 3px 容差），且 b 明显更大
            if (bx1 - 3 <= ax1 and by1 - 3 <= ay1 and ax2 <= bx2 + 3 and ay2 <= by2 + 3
                    and b['area'] > a['area'] * 1.5):
                if b['area'] < best_area:
                    best_area, best_parent = b['area'], j
        if best_parent >= 0:
            children[best_parent].append(i)
    # 直接父节点
    parent_of = [-1] * n
    for j, ch in enumerate(children):
        for i in ch:
            # 只保留最小包含者为直接父
            if parent_of[i] == -1:
                parent_of[i] = j
    return children, parent_of


if __name__ == '__main__':
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/hatch/workspace/visual_tracer/qa/frames/overview.png'
    out = sys.argv[2] if len(sys.argv) > 2 else \
        '/home/hatch/workspace/visual_tracer/qa/frames/rectangles.json'
    rects, (W, H) = detect_rectangles(img)
    children, parent_of = build_tree(rects)
    for i, r in enumerate(rects):
        r['n_children'] = len(children[i])
        r['tree_parent'] = parent_of[i]
    json.dump({'W': W, 'H': H, 'rects': rects}, open(out, 'w'))
    print('rects:', len(rects), 'image %dx%d' % (W, H))
    # 面积 Top20
    for r in sorted(rects, key=lambda r: -r['area'])[:20]:
        print('  area=%8.0f bbox=%dx%d@(%d,%d) children=%d parent=%d' % (
            r['area'], r['w'], r['h'], r['x'], r['y'], r['n_children'], r['tree_parent']))
