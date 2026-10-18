#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
矢量校验工具: 为框#52 生成独立 From-To 表, 与视觉管线结果交叉校对。

定位: 纯校验工具, 用户明确授权的例外。生产管线(skill local-visual-tracer)
保持纯视觉、输入格式无关; 本脚本只利用"这次输入恰好是 DXF"做交叉验证,
代码单独存放于 vector_check/, 不许混入视觉管线。

方法(矢量):
  1. LINE(导线图层 '1'/'201'/'0')按端点重合连成网络(union-find, 容差 END_TOL)
  2. CIRCLE 找端子圆环(半径 RING_RMIN~RING_RMAX), 关联到网络端点
  3. 闭合矩形找器件框; 有端子圆环邻近的矩形才算器件(过滤标题栏格子)
  4. TEXT 按正则+邻近关联: 管脚号(BD176-21)→圆环; 线号(IPB021_01)→导线;
     器件名/管脚功能→器件框内文字
  5. 每条"两端都有圆环"的网络出一行 From-To; 单端出框的记"框边界端口"

Windows 硬约束: 只用 ezdxf/openpyxl(pip, 均有 Windows wheel); 路径 pathlib。
"""
import pickle
import re
import math
from pathlib import Path
from collections import defaultdict

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
PKL = HERE / "frame52_entities.pkl"
XLSX = HERE.parent / "qa" / "frames" / "pilot_52" / "FromTo_矢量版.xlsx"

# ---------------- 可调参数(按框#52 普查结果设定) ----------------
WIRE_LAYERS = {"1", "201", "0"}  # 本框实际只有 '0' 有导线; '2FRAME'/'3FRAME' 是框线
TEXT_SKIP_LAYERS = {"1FILAMENT"}  # 标题栏文字层, 排除
END_TOL = 2.0          # 导线端点重合容差(图纸单位)
RING_RMIN, RING_RMAX = 0.5, 0.8   # 端子圆环: CIRCLE r=0.62(37个, 每个另有一 r=1.25 ARC 同心装饰); r=0.31 的 9 个是实心接点
RING_WIRE_TOL = 5.0    # 圆环中心与导线端点关联距离
RING_PIN_TOL = 30.0    # 管脚号文字与圆环关联距离(实测 1.6~9.6)
WIRE_LABEL_TOL = 25.0  # 线号文字与导线关联距离
RING_DEV_TOL = 100.0   # 圆环与器件框关联距离
DEV_RING_TOL = 100.0   # 器件框判定: 边框附近有圆环的最大距离(本框改用 pline 直接检测, 此值备用)
PIN_FUNC_DY = 6.0      # 管脚功能行带宽: pin 文字与功能文字同行(|dy|<=6), 行内按 x 最近配对
RECT_TOL = 1.0         # 矩形检测端点量化容差
DEV_MIN_AREA = 500.0   # 器件框最小面积(排除 ~6 平方单位的小三角符号)

RE_PIN = re.compile(r"^[A-Z0-9]{2,}-\d{1,3}$")       # BD176-21 / BD219-2
RE_WIRE = re.compile(r"^[A-Z]{2,}\d+_\d+")            # IPB021_01...
RE_DEVCODE = re.compile(r"^BD\d+$")                   # BD176

# ---------------- union-find ----------------
class UF:
    def __init__(self):
        self.p = {}
    def find(self, a):
        p = self.p
        if a not in p:
            p[a] = a
            return a
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

def pt_key(x, y, tol):
    return (round(x / tol), round(y / tol))

def dist_pt_seg(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

def main():
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    bb = data["bbox"]
    lines = [l for l in data["lines"] if l[0] in WIRE_LAYERS]
    circles = [c for c in data["circles"] if RING_RMIN <= c[3] <= RING_RMAX]
    plines = data.get("plines", [])
    arcs = data.get("arcs", [])
    texts = [t for t in data["texts"] if t[0] not in TEXT_SKIP_LAYERS]
    print(f"wire LINEs: {len(lines)}, terminal rings: {len(circles)}, "
          f"plines: {len(plines)}, TEXTs: {len(texts)} (excl. {TEXT_SKIP_LAYERS})",
          flush=True)

    # ---- 0. 器件框: 闭合 4 点 LWPOLYLINE, 面积过滤 ----
    devices = []
    for (layer, closed, pts) in plines:
        if not closed or len(pts) != 4:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        rx, ry, rw, rh = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
        if rw * rh < DEV_MIN_AREA:
            continue
        # 排除分图框边框本身
        if abs(rx - bb[0]) < 5 and abs(ry - bb[1]) < 5 and \
           abs(rx + rw - bb[2]) < 5 and abs(ry + rh - bb[3]) < 5:
            continue
        devices.append({"layer": layer, "x": rx, "y": ry, "w": rw, "h": rh})
    print(f"device boxes: {len(devices)}", flush=True)
    for d in sorted(devices, key=lambda d: -d["w"] * d["h"]):
        print(f"  x={d['x']:.1f} y={d['y']:.1f} w={d['w']:.1f} h={d['h']:.1f}")

    # ---- 1. 导线网络(含非圆环装饰的 ARC 段) ----
    # 圆环装饰 ARC(圆心与端子圆环重合)不算导线
    ring_centers = [(c[1], c[2]) for c in circles]
    def is_ring_deco(a):
        return any(math.hypot(a[1] - cx, a[2] - cy) < 1.0 for cx, cy in ring_centers)
    wire_arcs = [a for a in arcs if a[0] in WIRE_LAYERS and not is_ring_deco(a)]
    print(f"wire ARCs (non decoration): {len(wire_arcs)}", flush=True)
    uf = UF()
    for _, x1, y1, x2, y2 in lines:
        uf.union(pt_key(x1, y1, END_TOL), pt_key(x2, y2, END_TOL))
    for a in wire_arcs:
        _, _, _, _, x1, y1, x2, y2 = a
        uf.union(pt_key(x1, y1, END_TOL), pt_key(x2, y2, END_TOL))
    nets = defaultdict(list)  # root -> segments
    for seg in lines:
        _, x1, y1, x2, y2 = seg
        nets[uf.find(pt_key(x1, y1, END_TOL))].append(seg)
    for a in wire_arcs:
        _, _, _, _, x1, y1, x2, y2 = a
        nets[uf.find(pt_key(x1, y1, END_TOL))].append(("ARC", x1, y1, x2, y2))
    print(f"wire networks: {len(nets)}", flush=True)

    # 网络端点(degree==1 的量化端点)与线段表
    net_endpoints = {}
    net_segs = {}
    for root, segs in nets.items():
        deg = defaultdict(int)
        for _, x1, y1, x2, y2 in segs:
            deg[pt_key(x1, y1, END_TOL)] += 1
            deg[pt_key(x2, y2, END_TOL)] += 1
        ends = [k for k, d in deg.items() if d == 1]
        net_endpoints[root] = [(k[0] * END_TOL, k[1] * END_TOL) for k in ends]
        net_segs[root] = segs

    # ---- 2. 圆环 -> 网络端点 ----
    ring_net = {}   # ring_idx -> (net_root, ex, ey)
    for i, (_, cx, cy, r) in enumerate(circles):
        best = None
        for root, ends in net_endpoints.items():
            for ex, ey in ends:
                d = math.hypot(cx - ex, cy - ey)
                if d <= r + RING_WIRE_TOL and (best is None or d < best[0]):
                    best = (d, root, ex, ey)
        if best:
            ring_net[i] = (best[1], best[2], best[3])
    print(f"rings attached to wire ends: {len(ring_net)}/{len(circles)}", flush=True)

    # ---- 3. 器件框已在第 0 步由 pline 直接检测; 保留边框距离函数 ----
    def near_border(rx, ry, rw, rh, cx, cy, tol):
        # 点到矩形边框的最短距离
        dx = max(rx - cx, 0, cx - (rx + rw))
        dy = max(ry - cy, 0, cy - (ry + rh))
        if dx == 0 and dy == 0:
            # 在框内: 到边的距离
            return min(cx - rx, rx + rw - cx, cy - ry, ry + rh - cy)
        return math.hypot(dx, dy)

    for d in sorted(devices, key=lambda d: -d["w"] * d["h"]):
        print(f"  x={d['x']:.1f} y={d['y']:.1f} w={d['w']:.1f} h={d['h']:.1f}")

    # ---- 4. TEXT 关联 ----
    # 4a. 管脚号 -> 圆环(记录管脚文字坐标, 供功能关联用 pin 文字 y)
    ring_pin = {}
    ring_pin_xy = {}
    for li, (layer, ix, iy, s, h) in enumerate(texts):
        s = s.strip()
        if not RE_PIN.match(s):
            continue
        best = None
        for i, (_, cx, cy, r) in enumerate(circles):
            d = math.hypot(ix - cx, iy - cy)
            if d <= RING_PIN_TOL and (best is None or d < best[0]):
                best = (d, i)
        if best:
            ring_pin[best[1]] = s
            ring_pin_xy[best[1]] = (ix, iy)
    print(f"pin labels attached to rings: {len(ring_pin)}", flush=True)

    # 4b. 线号 -> 网络
    net_wire = {}
    for li, (layer, ix, iy, s, h) in enumerate(texts):
        s = s.strip()
        m = RE_WIRE.match(s)
        if not m:
            continue
        label = m.group(0)
        best = None
        for root, segs in net_segs.items():
            for _, x1, y1, x2, y2 in segs:
                d = dist_pt_seg(ix, iy, x1, y1, x2, y2)
                if d <= WIRE_LABEL_TOL and (best is None or d < best[0]):
                    best = (d, root)
                if best and best[0] < 5:
                    break
            if best and best[0] < 5:
                break
        if best:
            # 同一网络取最近的线号
            if best[1] not in net_wire or best[0] < net_wire[best[1]][0]:
                net_wire[best[1]] = (best[0], s)
    net_wire = {k: v[1] for k, v in net_wire.items()}
    print(f"wire labels attached to networks: {len(net_wire)}", flush=True)

    # 4c. 器件名/代号: 框附近(40 单位, 按文字外框估计)文字, 字高优先
    #     管脚功能: 框内文字(排除名/代号/管脚号/线号)
    def text_span(ix, iy, s, h):
        return (ix, iy, ix + len(s) * h * 0.55, iy + h)

    def rect_dist(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
        dx = max(ax0 - bx1, bx0 - ax1, 0)
        dy = max(ay0 - by1, by0 - ay1, 0)
        return math.hypot(dx, dy)

    for d in devices:
        bx0, by0, bx1, by1 = d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]
        near = []
        for (_, ix, iy, s, h) in texts:
            s = s.strip()
            if not s:
                continue
            tx0, ty0, tx1, ty1 = text_span(ix, iy, s, h)
            dd = rect_dist(bx0, by0, bx1, by1, tx0, ty0, tx1, ty1)
            if dd <= 40.0:
                near.append((dd, h, s))
        codes = [(dd, s) for dd, h, s in near if RE_DEVCODE.match(s)]
        d["code"] = min(codes)[1] if codes else ""
        names = [(dd, -len(s), s) for dd, h, s in near
                 if h >= 4.0 and not RE_DEVCODE.match(s)]
        d["name"] = min(names)[2] if names else ""
        d["_code_cands"] = codes  # (dist, text) 留给后一步配对
        inside = [(ix, iy, s.strip(), h) for (_, ix, iy, s, h) in texts
                  if bx0 < ix < bx1 and by0 < iy < by1 and s.strip()]
        d["texts"] = inside
        # 管脚功能候选: 排除名/代号/管脚号/线号
        d["funcs"] = [(ix, iy, s) for (ix, iy, s, h) in inside
                      if s != d["name"] and s != d["code"]
                      and not RE_PIN.match(s) and not RE_WIRE.match(s)]

    # ---- 4d. 代号-名称配对: 代号文字先找最近的大字高名称, 再把代号赋给该名称所属的框
    # (代号常夹在两个框之间, 直接按"离框最近"会配错, 如 BD221 离 FL 框更近)
    name_spans = []  # (text, x0,y0,x1,y1)
    for (_, ix, iy, s, h) in texts:
        s = s.strip()
        if h >= 4.0 and s and not RE_DEVCODE.match(s):
            name_spans.append((s,) + text_span(ix, iy, s, h))
    code_owner = {}  # code text -> name text
    for (_, ix, iy, s, h) in texts:
        s = s.strip()
        if not RE_DEVCODE.match(s):
            continue
        tx0, ty0, tx1, ty1 = text_span(ix, iy, s, h)
        best = None
        for ns in name_spans:
            dd = rect_dist(tx0, ty0, tx1, ty1, ns[1], ns[2], ns[3], ns[4])
            if dd <= 60.0 and (best is None or dd < best[0]):
                best = (dd, ns[0])
        if best:
            code_owner[s] = best[1]
    for d in devices:
        if d["name"] and d["name"] in code_owner.values():
            # 该名称配到的代号(离该框最近的优先)
            cands = [c for c, nm in code_owner.items() if nm == d["name"]]
            if cands:
                bx0, by0 = d["x"], d["y"]
                d["code"] = min(cands, key=lambda c: min(
                    math.hypot(ix - bx0, iy - by0)
                    for (_, ix, iy, s, h) in texts if s.strip() == c))
    print("device names/codes:", flush=True)
    for d in devices:
        print(f"  {d['name']} ({d['code']})", flush=True)

    # ---- 5. 组装 From-To ----
    def device_of_ring(cx, cy):
        best = None
        for di, d in enumerate(devices):
            dist = near_border(d["x"], d["y"], d["w"], d["h"], cx, cy, 1e9)
            if dist <= RING_DEV_TOL and (best is None or dist < best[0]):
                best = (dist, di)
        return best[1] if best else None

    def pin_func(di, cx, cy):
        # 行带 + 2D: 与圆环同行(|dy|<=PIN_FUNC_DY)的功能文字中取 2D 最近的
        # (圆环位于 pin 标注与功能文字之间, 2D 能区分同行不同列; 行带排除跨行干扰如 Ag 屏蔽标记)
        d = devices[di]
        best = None
        for ix, iy, s in d["funcs"]:
            if abs(iy - cy) > PIN_FUNC_DY:
                continue
            dd = math.hypot(ix - cx, iy - cy)
            if best is None or dd < best[0]:
                best = (dd, s)
        return best[1] if best else ""

    def dev_label(di):
        d = devices[di]
        if d["name"] and d["code"]:
            return f'{d["name"]} ({d["code"]})'
        return d["name"] or d["code"] or f"器件#{di}"

    # 每个网络的圆环按端点分组: 同一网络上>=2 个圆环 -> 两两配对出 From-To
    net_rings = defaultdict(list)
    for i, (root, ex, ey) in ring_net.items():
        net_rings[root].append((i, ex, ey))

    def code_mismatch_note(pin, di):
        # 管脚代号前缀(如 BD228)与器件代号(如 BD221)不一致 -> 备注待确认
        if di is None or pin == "未识别":
            return ""
        m = re.match(r"^([A-Z0-9]+)-\d+$", pin)
        code = devices[di]["code"]
        if m and code and m.group(1) != code:
            return f"管脚代号{pin}与器件代号{code}不一致, 待人工确认"
        return ""

    rows = []
    notes = []
    for root, rl in net_rings.items():
        if len(rl) < 2:
            continue
        # 简单处理: 取最远的两个圆环为一对; 多圆环网络逐对记录并备注
        wire = net_wire.get(root, "未识别")
        pairs = []
        if len(rl) == 2:
            pairs = [(rl[0], rl[1], "")]
        else:
            # 星形/拼接: 以第一个为基准逐个配对, 备注拼接点
            for j in range(1, len(rl)):
                pairs.append((rl[0], rl[j], "多圆环网络(疑似拼接点)"))
        for (i1, _, _), (i2, _, _), note in pairs:
            _, cx1, cy1, _ = circles[i1]
            _, cx2, cy2, _ = circles[i2]
            d1, d2 = device_of_ring(cx1, cy1), device_of_ring(cx2, cy2)
            p1, p2 = ring_pin.get(i1, "未识别"), ring_pin.get(i2, "未识别")
            note_full = "；".join(x for x in
                                 [note, code_mismatch_note(p1, d1),
                                  code_mismatch_note(p2, d2)] if x)
            rows.append({
                "from_dev": dev_label(d1) if d1 is not None else "框外/未关联",
                "from_pin": p1,
                "from_func": pin_func(d1, cx1, cy1) if d1 is not None else "",
                "to_dev": dev_label(d2) if d2 is not None else "框外/未关联",
                "to_pin": p2,
                "to_func": pin_func(d2, cx2, cy2) if d2 is not None else "",
                "wire": wire,
                "note": note_full,
            })
    # 单圆环网络: 框边界端口
    n_port = 0
    for root, rl in net_rings.items():
        if len(rl) == 1:
            n_port += 1
            i1, _, _ = rl[0]
            _, cx1, cy1, _ = circles[i1]
            d1 = device_of_ring(cx1, cy1)
            p1 = ring_pin.get(i1, "未识别")
            note_full = "；".join(x for x in
                                 ["框边界端口(单端)", code_mismatch_note(p1, d1)] if x)
            rows.append({
                "from_dev": dev_label(d1) if d1 is not None else "框外/未关联",
                "from_pin": p1,
                "from_func": pin_func(d1, cx1, cy1) if d1 is not None else "",
                "to_dev": "—", "to_pin": "—", "to_func": "—",
                "wire": net_wire.get(root, "未识别"),
                "note": note_full,
            })
    print(f"rows: {len(rows)} (ports: {n_port})", flush=True)

    # ---- 6. 写 Excel ----
    wb = Workbook()
    ws = wb.active
    ws.title = "From-To矢量版"
    headers = ["序号", "起始用电器", "起始管脚", "管脚功能",
               "目标用电器", "目标管脚", "管脚功能", "导线线号", "备注"]
    hdr_fill = PatternFill("solid", fgColor="4472C4")
    hdr_font = Font(color="FFFFFF", bold=True)
    for c, htxt in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=htxt)
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for idx, row in enumerate(rows, 1):
        r = idx + 1  # Excel 行号(第1行为表头)
        vals = [idx, row["from_dev"], row["from_pin"], row["from_func"],
                row["to_dev"], row["to_pin"], row["to_func"], row["wire"], row["note"]]
        for c, v in enumerate(vals, 1):
            ws.cell(row=r, column=c, value=v)
    widths = [8, 34, 14, 26, 34, 14, 26, 28, 24]
    for c, wdt in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(c)].width = wdt
    ws.freeze_panes = "A2"

    # 说明页
    ws2 = wb.create_sheet("说明")
    ws2["A1"] = "矢量校验版 From-To（框#52）"
    ws2["A2"] = ("本表由 DXF 矢量图元直接生成，仅用于与视觉管线结果交叉校对；"
                 "生产管线保持纯视觉，本表代码独立存放于 vector_check/，不许混入。")
    ws2["A3"] = ("矢量版不是完美真值：端点重合容差(2.0)是启发式，制图缝隙会导致网络断裂；"
                 "TEXT 邻近关联是启发式，密集区可能误配；假设一根导线两端各一个圆环，多圆环按星形配对；"
                 "器件框检测依赖闭合矩形 LWPOLYLINE 画法；跨框导线只记单端端口。两版差异项需人工仲裁。")
    ws2["A4"] = ("已知差异示例：实心圆点矢量只找到 9 个(r=0.31)，视觉版报 67 个；"
                 "FR 传感器管脚标注为 BD228-x 而器件为 BD221，已在备注标出；"
                 "中英双语功能标注按图面如实记录。")
    ws2["A5"] = f"导线网络 {len(nets)} 个；端子圆环 {len(circles)} 个(已关联 {len(ring_net)})；器件框 {len(devices)} 个；管脚标注 {len(ring_pin)} 个；线号 {len(net_wire)} 个。"
    ws2.column_dimensions["A"].width = 140

    XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(XLSX)
    print(f"saved {XLSX}", flush=True)

    # ---- 7. 真值命中检查 ----
    print("\n=== 真值命中检查 ===")
    def hit(pin_a, wire_sub, pin_b):
        for row in rows:
            if (row["from_pin"] == pin_a and wire_sub in row["wire"] and row["to_pin"] == pin_b) or \
               (row["to_pin"] == pin_a and wire_sub in row["wire"] and row["from_pin"] == pin_b):
                return row
        return None
    for pa, w, pb in [("BD176-21", "IPB021_01", "BD219-2"),
                      ("BD176-7", "IPB07_01", "BD219-1")]:
        r = hit(pa, w, pb)
        print(f"{pa} --{w}-- {pb}: {'HIT ' + str({k: r[k] for k in ('from_dev','from_func','to_dev','to_func','wire')}) if r else 'MISS'}")

if __name__ == "__main__":
    main()
