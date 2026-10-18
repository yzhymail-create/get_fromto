# -*- coding: utf-8 -*-
"""纯视觉 OCR：只接收 numpy 图像，不碰矢量图元。

识别引擎：rapidocr_onnxruntime（纯 pip 包，有 Windows wheel，
无系统依赖）。Windows 下 pip install -r requirements.txt 即可用。

Windows 硬约束：不调用 shell、不依赖 apt/系统二进制、
路径一律 os.path。
"""
import os

import numpy as np
import cv2

_engine = None


def get_engine():
    """RapidOCR 单例（延迟加载，模型首次加载较慢）。"""
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def find_text_regions(foreground, min_h=8, max_h=120, min_w=8):
    """找文字块。foreground: bool 数组(True=前景)。

    返回 [(x, y, w, h), ...] 文字块 bbox 列表（像素坐标）。
    原理：文字笔画细小，用闭运算把字符连成块，再按尺寸过滤。
    """
    fg = foreground.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8)
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (min_h <= h <= max_h and w >= min_w):
            continue
        if w / max(h, 1) > 60:  # 太长的大概率是导线
            continue
        fill = area / max(w * h, 1)
        if fill < 0.05 or fill > 0.95:
            continue
        regions.append((int(x), int(y), int(w), int(h)))
    return regions


def recognize_image(gray):
    """整图识别。gray: 灰度 numpy 数组。

    返回 [(bbox4, text, conf), ...]，bbox4 为四点坐标。
    用 RapidOCR 自带的检测+识别（纯像素）。
    """
    engine = get_engine()
    if gray.ndim == 2:
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        img = gray
    result, _ = engine(img)
    out = []
    if result:
        for box, text, conf in result:
            out.append((box, text.strip(), float(conf)))
    return out


def recognize_region(gray, bbox, pad=8):
    """识别单个文字块 crop。bbox=(x,y,w,h)。返回 (text, conf)。"""
    x, y, w, h = [int(v) for v in bbox]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = x + w + pad, y + h + pad
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return '', 0.0
    res = recognize_image(crop)
    if not res:
        return '', 0.0
    # 取最长的识别结果（通常为整块文字）
    best = max(res, key=lambda r: len(r[1]))
    return best[1], best[2]


def _as_aabb(bbox):
    """归一化文字框为 (x1, y1, x2, y2) 浮点四元组。

    接受：
      - (x1, y1, x2, y2)
      - 四点坐标 [[x,y],[x,y],[x,y],[x,y]]（RapidOCR 原生格式）
    注意：(x, y, w, h) 格式**不再**被隐式接受，调用方必须先转换。
    这是 CODEX 评审指出的坐标格式不一致 bug 的修复。
    """
    import numpy as _np
    b = _np.asarray(bbox, dtype=float)
    if b.shape == (4, 2):
        (x1, y1), (x2, y2) = b.min(axis=0), b.max(axis=0)
        return (float(x1), float(y1), float(x2), float(y2))
    x1, y1, x2, y2 = (float(v) for v in b.flat[:4])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def xywh_to_aabb(x, y, w, h):
    """(x, y, w, h) -> (x1, y1, x2, y2)。find_text_regions 的输出用此转换。"""
    return (float(x), float(y), float(x + w), float(y + h))


def recognize_text(gray, tiled_threshold=3000):
    """统一文字识别接口（CODEX 评审建议，主程序只调这一个）。

    gray: 灰度 numpy 数组（纯像素）。
    返回 [{"bbox": (x1,y1,x2,y2), "text": str, "confidence": float}, ...]，
    bbox 为整图像素坐标。识别不到文字返回 []（与"引擎不可用"区分：
    引擎不可用时本函数抛 ImportError）。
    """
    H, W = gray.shape[:2]
    if max(H, W) > tiled_threshold:
        raw = recognize_tiled(gray)
    else:
        raw = [(box, t, c) for box, t, c in recognize_image(gray)]
        # recognize_image 返回四点 box，转 aabb
        raw = [(_as_aabb(b), t, c) for b, t, c in raw]
    return [{"bbox": _as_aabb(b), "text": t, "confidence": float(c)}
            for b, t, c in raw]


def _boxes_iou(a, b):
    """轴对齐 bbox 的 IoU。a,b 为 (x1,y1,x2,y2)。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def _nms(items, iou_thr=0.4):
    """items: [(aabb, text, conf), ...]，按 conf 降序去重。"""
    items = sorted(items, key=lambda r: r[2], reverse=True)
    kept = []
    for aabb, text, conf in items:
        if all(_boxes_iou(aabb, k[0]) < iou_thr for k in kept):
            kept.append((aabb, text, conf))
    return kept


def recognize_tiled(gray, tile_h=1500, tile_w=1700, overlap=250):
    """分块端到端识别（大图标准做法）。返回 [(aabb, text, conf), ...]。

    aabb 为整图像素坐标 (x1,y1,x2,y2)。重叠区用 NMS 去重。
    纯像素，不碰矢量图元。
    """
    H, W = gray.shape[:2]
    items = []
    ys = list(range(0, H, tile_h - overlap))
    xs = list(range(0, W, tile_w - overlap))
    for yt in ys:
        for xt in xs:
            y1 = min(yt + tile_h, H)
            x1 = min(xt + tile_w, W)
            tile = gray[yt:y1, xt:x1]
            for box, text, conf in recognize_image(tile):
                xs4 = [p[0] for p in box]
                ys4 = [p[1] for p in box]
                aabb = (xt + min(xs4), yt + min(ys4),
                        xt + max(xs4), yt + max(ys4))
                items.append((aabb, text, conf))
    return _nms(items)


def recognize_text_blocks(gray, regions, pad=8):
    """批量识别文字块（纯识别，不重复做检测）。返回 [(bbox, text, conf), ...]。

    认不出记 ''（诚实纪律：不编造）。
    """
    engine = get_engine()
    crops, idx = [], []
    for i, bbox in enumerate(regions):
        x, y, w, h = [int(v) for v in bbox]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        crop = gray[y0:y + h + pad, x0:x + w + pad]
        if crop.size == 0:
            continue
        # 小字放大到可识别高度
        scale = max(1.0, 32.0 / max(h, 1))
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        crops.append(crop)
        idx.append(i)
    out = [(bbox, '', 0.0) for bbox in regions]
    if not crops:
        return out
    results, _ = engine.text_rec(crops)
    for j, (text, conf) in enumerate(results):
        out[idx[j]] = (regions[idx[j]], text.strip(), float(conf))
    return out


def attach_text_to_nets(text_results, nets, max_dist_px=60):
    """文字按距离绑到最近导线网络。

    text_results: 统一格式 [{"bbox": (x1,y1,x2,y2), "text":..., "confidence":...}]
    （recognize_text 的输出）。兼容旧式 [(aabb4, text, conf)] 元组，
    其中 aabb4 为 (x1,y1,x2,y2)。(x,y,w,h) 格式不再隐式接受。
    nets: trace.py 的 nets 列表
    返回 {net_index: [(text, conf, dist), ...]}
    """
    from collections import defaultdict
    attached = defaultdict(list)
    net_bboxes = []
    for n in nets:
        px = n['pixels']
        net_bboxes.append(
            (px[:, 1].min(), px[:, 0].min(), px[:, 1].max(), px[:, 0].max()))
    items = []
    for r in text_results:
        if isinstance(r, dict):
            bbox, text, conf = r['bbox'], r['text'], r.get('confidence', 0)
        else:
            bbox, text, conf = r
        x1, y1, x2, y2 = _as_aabb(bbox)
        items.append(((x1, y1, x2, y2), text, conf))
    for (x1, y1, x2, y2), text, conf in items:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best, best_d = -1, max_dist_px
        for i, (nx1, ny1, nx2, ny2) in enumerate(net_bboxes):
            dx = max(nx1 - cx, 0, cx - nx2)
            dy = max(ny1 - cy, 0, cy - ny2)
            d = (dx * dx + dy * dy) ** 0.5
            if d < best_d:
                best_d, best = d, i
        if best >= 0:
            attached[best].append((text, conf, round(best_d, 1)))
    return attached
