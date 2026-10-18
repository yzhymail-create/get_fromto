# -*- coding: utf-8 -*-
"""纯视觉导线追踪：只接收 numpy 图像，绝不接触 DXF 数据。

管线：
  灰度图 -> Otsu 二值化 -> Zhang-Suen 细化 -> 骨架图分析 ->
  提取导线段(arc) -> 节点圆点判定 -> 合并成电气网络(net)

关键概念：
- 端点：骨架中 8 邻域=1 的像素
- 节点：8 邻域>=3 的像素；有实心圆点=拼接连接，无圆点=跨线不连
"""
import numpy as np
import cv2
from skimage.morphology import skeletonize


def binarize_otsu(gray):
    """Otsu 二值化，返回前景=True 的 bool 数组。"""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 前景应为暗像素（黑线白底）；若前景占比>50% 说明反了
    fg = bw < 128
    if fg.mean() > 0.5:
        fg = ~fg
    return fg


def skeleton_of(foreground):
    """Zhang-Suen 细化，返回骨架 bool 数组。"""
    return skeletonize(foreground)


def neighbor_degree(skel):
    """每个骨架像素的 8 邻域度数（不含自身）。"""
    k = np.ones((3, 3), dtype=np.uint8)
    k[1, 1] = 0
    cnt = cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    return cnt * skel


def estimate_wire_width(foreground, skeleton):
    """用距离变换估计导线宽度（像素）。"""
    dist = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 3)
    vals = dist[skeleton]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 3.0
    # 导线骨架处的距离值中位数*2 = 线宽；用较低分位避开圆点干扰
    return float(np.percentile(vals, 30)) * 2.0


def has_dot(foreground, dist, jr, jc, wire_width):
    """节点处是否有实心圆点（拼接点）。

    圆点半径明显大于线宽的一半；在节点邻域内找距离变换局部最大值。
    """
    r = int(max(wire_width * 2.5, 6))
    r0, r1 = max(0, jr - r), jr + r + 1
    c0, c1 = max(0, jc - r), jc + r + 1
    patch = dist[r0:r1, c0:c1]
    if patch.size == 0:
        return False
    # 圆点处的局部厚度应显著大于线宽
    return bool(patch.max() > wire_width * 0.9)


def extract_nets(foreground, skeleton):
    """从骨架提取电气网络。

    返回 nets: list of dict(yx_coords, endpoints, junctions, has_dot_junctions)
    """
    degree = neighbor_degree(skeleton)
    junction_mask = (degree >= 3) & skeleton
    # 去掉节点像素，剩下的连通域就是导线段(arc)
    skel_no_j = skeleton & ~junction_mask
    n_arc, arc_labels = cv2.connectedComponents(skel_no_j.astype(np.uint8), connectivity=8)

    dist = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 3)
    wire_width = estimate_wire_width(foreground, skeleton)

    # 每个 arc 的像素集合
    arc_pixels = {}
    ys, xs = np.nonzero(arc_labels)
    for y, x, lb in zip(ys, xs, arc_labels[ys, xs]):
        arc_pixels.setdefault(int(lb), []).append((y, x))

    # union-find 合并 arc
    parent = {i: i for i in arc_pixels}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 节点像素坐标
    jy, jx = np.nonzero(junction_mask)
    # 每个节点相邻的 arc id
    from collections import defaultdict
    junc_arcs = defaultdict(set)
    H, W = skeleton.shape
    for y, x in zip(jy, jx):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    lb = arc_labels[ny, nx]
                    if lb > 0:
                        junc_arcs[(y, x)].add(int(lb))

    dot_junctions = []
    for (y, x), arcs in junc_arcs.items():
        if len(arcs) < 2:
            continue
        if has_dot(foreground, dist, y, x, wire_width):
            dot_junctions.append((y, x))
            arcs = list(arcs)
            for a in arcs[1:]:
                union(arcs[0], a)
        # 无圆点=跨线：不合并，各 arc 保持独立

    # 组装 nets
    nets = defaultdict(list)
    for aid, px in arc_pixels.items():
        nets[find(aid)].extend(px)

    result = []
    full_deg = neighbor_degree(skeleton)  # 只算一次，勿放循环内
    for px in nets.values():
        arr = np.array(px)
        # 端点：在原骨架中度数为1的像素（属于本 net）
        eps = [(int(y), int(x)) for y, x in px if full_deg[y, x] == 1]
        result.append({
            'pixels': arr,
            'endpoints': eps,
            'n_endpoints': len(eps),
        })
    return result, wire_width, dot_junctions


def filter_nets(nets, min_pixels=30):
    """过滤太小的连通域（文字、噪点）。返回 (wire_nets, small_nets)。"""
    wire, small = [], []
    for n in nets:
        (wire if len(n['pixels']) >= min_pixels else small).append(n)
    return wire, small


def overlay_nets(gray, nets, out_path, max_nets=400):
    """叠加图：原图灰色 + 每个网络不同颜色 + 端点编号。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    H, W = gray.shape
    vis = np.stack([gray] * 3, axis=-1).copy()
    colors = (cm.get_cmap('hsv')(np.linspace(0, 1, max(1, min(len(nets), max_nets))))[:, :3] * 255).astype(np.uint8)
    for i, n in enumerate(nets[:max_nets]):
        c = colors[i % len(colors)]
        px = n['pixels']
        vis[px[:, 0], px[:, 1]] = c
        for j, (y, x) in enumerate(n['endpoints'][:6]):
            cv2.circle(vis, (x, y), 8, (255, 255, 255), 2)
            cv2.circle(vis, (x, y), 8, (0, 0, 0), 1)
    # 存图（限制最大边 3000px 便于查看）
    scale = min(1.0, 3000.0 / max(H, W))
    if scale < 1.0:
        vis = cv2.resize(vis, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return out_path
