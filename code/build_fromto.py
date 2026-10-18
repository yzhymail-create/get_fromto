# -*- coding: utf-8 -*-
"""框#52 机器版 From-To 表生成（纯视觉，net-hub 关联）。

关联链（skill 纪律第5条的等效实现）：
  管脚号文本 --(最近导线端点)--> net（trace 连通域）
  双管脚 net / 经导线线号配对的单管脚 net 对 --> 行
  配不成对的单管脚 net --> 边界端口行（目标记"—（出框）"）

根因修复（2026-10-18）：
  1. IPB03_01 漏检：OCR 把下划线吞了（IPB03_01→IPB0301），WIRE_RE 太严。
     放宽为允许分隔符缺失：[A-Z]{2,}\\d{2,3}[_\\s]?\\d{2}
  2. IPB04_01/IPB38_02 漏成行：旧逻辑用管脚端点（器件处）去套线号标签
     的 x 距离（<400px），长导线标签在中部必然失败。
     改为 net-hub：标签落在两管脚端点连成的走廊内即归属该 net。
  3. 边界端口：单管脚 net 配不成对 -> 边界行，不丢行（skill 第4条）。

诚实纪律：无文字依据记"待确认"/"未识别"，不编造。
Windows 约束：只用 os.path，无 shell 调用；依赖见 requirements.txt。
"""
import os
import json
import pickle
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
P52 = os.path.join(HERE, 'qa', 'frames', 'pilot_52')

PIN_RE = re.compile(r'^[A-Z]{2,}\d+-\d+[A-Z]?$')
DEV_RE = re.compile(r'^BD\d+$')
# 导线线号：字母+2~3位数字+可选分隔符（OCR 常吞下划线）+2位数字
WIRE_RE = re.compile(r'[A-Z]{2,}\d{2,3}[_\s]?\d{2}')
# 展示用归一化：IPB0301 -> IPB03_01（本图线号约定：2~3位+2位）
WIRE_NORM_RE = re.compile(r'([A-Z]{2,}\d{2,3})(\d{2})(?![\d_])')

COLS = ['序号', '起始用电器', '起始管脚', '管脚功能',
        '目标用电器', '目标管脚', '管脚功能', '导线线号', '备注']


def cx(a):
    return (a[0] + a[2]) / 2


def cy(a):
    return (a[1] + a[3]) / 2


def norm(s):
    return re.sub(r'\s+', '', s or '').upper()


def wire_display(txt):
    """IPB0301Y2.50... -> IPB03_01Y2.50...（只补分隔符，不改内容）。"""
    return WIRE_NORM_RE.sub(r'\1_\2', txt)


def load(fdir=P52):
    texts = json.load(open(os.path.join(fdir, 'ocr_texts.json'),
                           encoding='utf-8'))
    devs = json.load(open(os.path.join(fdir, 'devices.json'),
                         encoding='utf-8'))
    pa = pickle.load(open(os.path.join(fdir, 'pin_assoc.pkl'), 'rb'))
    return texts, devs, pa


def device_info(texts, devs):
    """返回 dev_id 到 {name, code, bbox} 的映射。"""
    out = {}
    for d in devs:
        x, y, w, h = d['bbox_xywh']
        cands = []
        for t in texts:
            a = t['aabb']
            tx, ty = cx(a), cy(a)
            if x - 60 < tx < x + w + 60 and y - 320 < ty < y + 40:
                cands.append((t['text'].strip(), t['conf']))
        code, name = '', ''
        for txt, _ in cands:
            if DEV_RE.match(norm(txt)):
                code = txt.strip()
        long_cands = sorted(
            [t for t, _ in cands
             if len(t) > 6 and not DEV_RE.match(norm(t))],
            key=len, reverse=True)
        if long_cands:
            name = long_cands[0]
        out[d['id']] = {'name': name, 'code': code,
                        'bbox': d['bbox_xywh']}
    return out


def pin_function_nearest(texts, dev_bbox, pin_y, y_tol=50):
    """器件框内与管脚同高度的最近功能文字（排除管脚号/器件代码/线号）。

    2026-10-18 改用最近文本优先：条带 OCR 在相邻两行文字间距小时会
    串行（如 BD176-25 串到 "RLSpeedSehsorPower"）。条带只做 fallback。
    """
    x, y, w, h = dev_bbox
    best, bd = '', 1e9
    for t in texts:
        a = t['aabb']
        tx, ty = cx(a), cy(a)
        if not (x < tx < x + w and y < ty < y + h):
            continue
        txt = t['text'].strip()
        tn = norm(txt)
        if PIN_RE.match(tn) or DEV_RE.match(tn):
            continue
        if WIRE_RE.search(txt) and len(txt) > 12:
            continue
        if len(txt) < 3:
            continue
        d = abs(ty - pin_y)
        if d < bd and d <= y_tol:
            bd, best = d, txt
    return best


def _is_garbage(txt):
    """纯数字/符号串视为垃圾识别（如 '000000000'）。"""
    t = norm(txt).replace('□', '')
    return len(t) == 0 or not re.search(r'[A-Z\u4e00-\u9fff]{2,}', t)


def pin_function_side_topk(texts, dev_bbox, pin_ex, pin_ey, y_tol=45, k=3):
    """返回 [(text, cy), ...] 按同侧+y接近排序的前 k 个。"""
    x, y, w, h = dev_bbox
    box_cx = x + w / 2
    pin_left = pin_ex < box_cx
    scored = []
    for t in texts:
        a = t['aabb']
        tx, ty = cx(a), cy(a)
        if not (x < tx < x + w and y < ty < y + h):
            continue
        txt = t['text'].strip()
        tn = norm(txt)
        if PIN_RE.match(tn) or DEV_RE.match(tn):
            continue
        if WIRE_RE.search(txt) and len(txt) > 12:
            continue
        if len(txt) < 3:
            continue
        d = abs(ty - pin_ey)
        if d > y_tol:
            continue
        same_side = (tx < box_cx) == pin_left
        scored.append((d + (0 if same_side else 25), txt, ty))
    scored.sort()
    return [(txt, ty) for _, txt, ty in scored[:k]]


def pin_function_side(texts, dev_bbox, pin_ex, pin_ey, y_tol=45):
    """(兼容) 返回最优 (text, cy)。"""
    r = pin_function_side_topk(texts, dev_bbox, pin_ex, pin_ey, y_tol, k=1)
    return r[0] if r else ('', 0)


def pin_function_band(gray, dev_bbox, pin_y, y_half=32):
    """器件框内按管脚高度裁水平条带直接识别（避免检测框切字）。

    返回 (text, cy)，cy 为选中文本框的中心 y；找不到返回 ('', 0)。
    """
    import sys
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from ocr import recognize_image
    x, y, w, h = [int(v) for v in dev_bbox]
    y0 = max(y, int(pin_y) - y_half)
    y1 = min(y + h, int(pin_y) + y_half)
    band = gray[y0:y1, x:x + w]
    if band.size == 0:
        return '', 0
    res = recognize_image(band)
    cands = []
    for box, txt, conf in res:
        t = txt.strip()
        tn = norm(t)
        if PIN_RE.match(tn) or DEV_RE.match(tn) or len(t) < 3:
            continue
        ys = [p[1] for p in box]
        cands.append((len(t), t, (min(ys) + max(ys)) / 2 + y0, conf))
    if not cands:
        return [], ('', 0)
    cands.sort(reverse=True)
    top = [(c[1], c[2]) for c in cands[:2]]
    return top, (cands[0][1], cands[0][2])


def _skip_func_text(txt):
    """过滤管脚号/器件代码/线号/过短文本。"""
    t = txt.strip()
    tn = norm(t)
    if PIN_RE.match(tn) or DEV_RE.match(tn):
        return True
    if WIRE_RE.search(t) and len(t) > 12:
        return True
    return len(t) < 3


def pin_function_final(texts, gray, dev_bbox, pin_ex, pin_ey):
    """功能文字最终规则（2026-10-18）：

    CAD 把功能文字放在管脚旁边的同一列，因此：
    1. 条带重识别（ey±40，修补检测碎片），取管脚同侧、水平离管脚
       最近的文本（y 已被条带约束，x 区分左右列，解决串行）。
    2. 条带无结果时，回退到检测文本：y 加权（行结构优先）+ 同侧优先。
    """
    import sys
    import math
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from ocr import recognize_image
    x, y, w, h = dev_bbox
    box_cx = x + w / 2
    pin_left = pin_ex < box_cx

    def ok_txt(t):
        return not _is_garbage(t) and not _skip_func_text(t)

    # 1. 条带重识别
    y_half = 40
    bx, by, bw, bh = [int(v) for v in dev_bbox]
    y0 = max(by, int(pin_ey) - y_half)
    y1 = min(by + bh, int(pin_ey) + y_half)
    band = gray[y0:y1, bx:bx + bw]
    best, bs = '', None
    if band.size > 0:
        for box, txt, conf in recognize_image(band):
            t = txt.strip()
            if not ok_txt(t):
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            tx = bx + (min(xs) + max(xs)) / 2
            ty = (min(ys) + max(ys)) / 2 + y0
            if not (x < tx < x + w and y < ty < y + h):
                continue
            if abs(ty - pin_ey) > y_half:
                continue
            side_ok = (tx < box_cx) == pin_left
            score = (0 if side_ok else 10000, abs(tx - pin_ex))
            if bs is None or score < bs:
                bs, best = score, t
    if best:
        return best

    # 2. 回退：检测文本
    cands = []
    for t in texts:
        a = t['aabb']
        tx, ty = cx(a), cy(a)
        if not (x < tx < x + w and y < ty < y + h):
            continue
        ttxt = t['text'].strip()
        if not ok_txt(ttxt):
            continue
        side_ok = (tx < box_cx) == pin_left
        score = (0 if side_ok else 10000,
                 abs(ty - pin_ey) * 4 + abs(tx - pin_ex) / 4)
        cands.append((score, ttxt))
    if not cands:
        return ''
    cands.sort()
    return cands[0][1]


def build_rows(fdir=P52):
    import numpy as np
    from PIL import Image
    texts, devs, pa = load(fdir)
    gray = np.array(Image.open(os.path.join(fdir, 'render.png')).convert('L'))
    pm = pa['pin_match']   # pin_text -> (net, ex, ey, dist)
    ne = pa['net_eps']     # net -> [(x, y), ...] 端点
    devs_info = device_info(texts, devs)

    code2id = {}
    for did, info in devs_info.items():
        if info['code']:
            code2id[norm(info['code'])] = did

    pin_pos = {}
    for t in texts:
        if PIN_RE.match(norm(t['text'])):
            a = t['aabb']
            pin_pos[t['text'].strip()] = (cx(a), cy(a))

    def nearest_device(px, py, max_dist=500):
        """管脚号前缀查不到器件时（如 BD228-*/BD221），按管脚文本位置
        找最近的器件框。返回 dev_id 或 None。"""
        best, bd = None, max_dist
        for did, info in devs_info.items():
            x, y, w, h = info['bbox']
            dx = max(x - px, 0, px - (x + w))
            dy = max(y - py, 0, py - (y + h))
            d = (dx * dx + dy * dy) ** 0.5
            if d < bd:
                bd, best = d, did
        return best

    def end_info(pin_text, ex=None, ey=None):
        """管脚 -> {device, pin, func}。ex/ey 为管脚锚定的导线端点
        （无则回退管脚号文本位置）。"""
        m = re.match(r'^([A-Z]{2}\d+)-', norm(pin_text))
        dev_id = code2id.get(m.group(1)) if m else None
        px, py = pin_pos.get(pin_text, (0, 0))
        if ex is None:
            ex, ey = px, py
        note_dev = ''
        if dev_id is None:
            # 图面管脚代号与器件代号不一致（如 BD228-*/BD221）：按位置找框
            dev_id = nearest_device(px, py)
            note_dev = '管脚代号与器件代号不一致，按位置归属' if dev_id else ''
        if dev_id is None:
            return {'device': '待确认', 'pin': pin_text, 'func': '未识别',
                    'code': '', 'note_dev': ''}
        info = devs_info[dev_id]
        device = ('%s(%s)' % (info['name'], info['code'])
                  if info['name'] else info['code'])
        func = pin_function_final(texts, gray, info['bbox'], ex, ey)
        return {'device': device, 'pin': pin_text,
                'func': func if func else '未识别',
                'code': norm(info['code']), 'note_dev': note_dev}

    # net -> pins / far 端点
    net_pins = defaultdict(list)   # ni -> [(pin_text, ex, ey)]
    for pin_text, (ni, ex, ey, d) in pm.items():
        net_pins[ni].append((pin_text, ex, ey))

    def far_eps(ni, pin_eps):
        out = []
        for (x, y) in ne.get(ni, []):
            if all(abs(x - px) > 3 or abs(y - py) > 3 for px, py in pin_eps):
                out.append((x, y))
        return out

    # 导线线号文本
    wire_labels = []
    for t in texts:
        txt = t['text'].strip()
        if WIRE_RE.search(txt) and len(txt) > 12:
            wire_labels.append({'aabb': t['aabb'], 'text': txt,
                                'conf': t['conf']})

    def label_in_corridor(lab, x1, y1, x2, y2, x_margin=200, y_tol=40):
        lx1, ly1, lx2, ly2 = lab['aabb']
        lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        return (min(x1, x2) - x_margin <= lcx <= max(x1, x2) + x_margin
                and abs(lcy - (y1 + y2) / 2) <= y_tol)

    def label_near_point(lab, px, py, dx_tol=250, dy_tol=40):
        lx1, ly1, lx2, ly2 = lab['aabb']
        lcy = (ly1 + ly2) / 2
        if abs(py - lcy) > dy_tol:
            return None
        if px < lx1 and lx1 - px <= dx_tol:
            return ('L', lx1 - px)
        if px > lx2 and px - lx2 <= dx_tol:
            return ('R', px - lx2)
        if lx1 <= px <= lx2:
            return ('M', 0)
        return None

    rows = []
    used_label_ids = set()
    notes_log = []

    def order_ends(e1, e2):
        """方向约定：非 BD176 在前（与矢量版一致）；BD176-仅一端时该端为 from。"""
        c1 = e1.get('code', '')
        c2 = e2.get('code', '')
        if c1 == 'BD176' and c2 != 'BD176':
            return e2, e1
        return e1, e2

    def make_complete(e1, e2, wire_txt, note=''):
        a, b = order_ends(e1, e2)
        notes = []
        if note:
            notes.append(note)
        if a.get('note_dev'):
            notes.append('起始' + a['note_dev'])
        if b.get('note_dev'):
            notes.append('目标' + b['note_dev'])
        if a['func'] == '未识别' or b['func'] == '未识别':
            notes.append('待确认：管脚功能未识别')
        return {'from_dev': a['device'], 'from_pin': a['pin'],
                'from_func': a['func'],
                'to_dev': b['device'], 'to_pin': b['pin'],
                'to_func': b['func'],
                'wire': wire_txt, 'note': '；'.join(notes)}

    def make_boundary(e_pin, wire_txt):
        notes = ['框边界端口']
        if e_pin.get('note_dev'):
            notes.append(e_pin['note_dev'])
        return {'from_dev': e_pin['device'], 'from_pin': e_pin['pin'],
                'from_func': e_pin['func'],
                'to_dev': '—（出框）', 'to_pin': '—', 'to_func': '—',
                'wire': wire_txt, 'note': '；'.join(notes)}

    # ---- 1. 双管脚 net：直接成行 ----
    two_pin_nets = {ni: ps for ni, ps in net_pins.items() if len(ps) == 2}
    for ni, ps in sorted(two_pin_nets.items()):
        (p1, x1, y1), (p2, x2, y2) = ps
        best, best_d = None, 1e9
        for li, lab in enumerate(wire_labels):
            if li in used_label_ids:
                continue
            if label_in_corridor(lab, x1, y1, x2, y2):
                lcx = cx(lab['aabb'])
                d = abs(lcx - (x1 + x2) / 2)
                if d < best_d:
                    best_d, best = d, li
        wire_txt = '未识别'
        if best is not None:
            used_label_ids.add(best)
            wire_txt = wire_display(wire_labels[best]['text'])
        else:
            notes_log.append('net %d 双管脚但无线号标签' % ni)
        rows.append(make_complete(end_info(p1, x1, y1), end_info(p2, x2, y2), wire_txt))

    # ---- 2. 单管脚 net：每个 net 按管脚端点认领最匹配的线号（互斥） ----
    # 管脚端点的 y 最可靠（导线水平出端子环）；far 端点是 trace 断点，
    # 位置不稳定，不参与认领。
    single = {ni: ps[0] for ni, ps in net_pins.items() if len(ps) == 1}
    net_claim = {}  # ni -> (li, side, score)
    for ni, (ptext, ex, ey) in single.items():
        best = None
        for li, lab in enumerate(wire_labels):
            if li in used_label_ids:
                continue
            hit = label_near_point(lab, ex, ey, dx_tol=300, dy_tol=40)
            if hit:
                side, dist = hit
                score = abs(ey - cy(lab['aabb'])) * 10 + dist
                if best is None or score < best[1]:
                    best = (li, score, side)
        if best:
            net_claim[ni] = (best[0], best[2], best[1])

    label_nets = defaultdict(list)
    for ni, (li, side, score) in net_claim.items():
        label_nets[li].append((ni, side, score))

    paired_nets = set()
    for li, lst in sorted(label_nets.items()):
        left = sorted([ni for ni, s, _ in lst if s in ('L', 'M')],
                      key=lambda n: next(sc for nn, ss, sc in lst if nn == n))
        right = sorted([ni for ni, s, _ in lst if s in ('R', 'M')],
                       key=lambda n: next(sc for nn, ss, sc in lst if nn == n))
        pair = None
        for ln in left:
            for rn in right:
                if ln != rn:
                    pair = (ln, rn)
                    break
            if pair:
                break
        lab = wire_labels[li]
        wire_txt = wire_display(lab['text'])
        if pair:
            used_label_ids.add(li)
            paired_nets.update(pair)
            ln, rn = pair
            e1 = end_info(single[ln][0], single[ln][1], single[ln][2])
            e2 = end_info(single[rn][0], single[rn][1], single[rn][2])
            rows.append(make_complete(e1, e2, wire_txt))
        else:
            # 只有一侧（或 M）：边界端口（标签认出来了）
            used_label_ids.add(li)
            ni = (left + right)[0]
            paired_nets.add(ni)
            rows.append(make_boundary(end_info(single[ni][0], single[ni][1], single[ni][2]), wire_txt))

    # ---- 3. 落单的单管脚 net：先沿导线方向找标签（碎片回退），找不到才记未识别 ----
    # 背景（框#112）：导线碎成 stub，标签在下游 fragment 上，label_near_point
    # 的 300px 不够。导线水平走线，标签 y 与管脚 y 差 <40px，沿 x 找最近。
    for ni in sorted(single):
        if ni in paired_nets:
            continue
        ptext, ex, ey = single[ni]
        best_li, best_dx = None, 1e9
        for li, lab in enumerate(wire_labels):
            if li in used_label_ids:
                continue
            lcx, lcy = cx(lab['aabb']), cy(lab['aabb'])
            if abs(lcy - ey) > 40:
                continue
            adx = abs(lcx - ex)
            if adx < best_dx and adx < 2500:
                best_dx, best_li = adx, li
        if best_li is not None:
            used_label_ids.add(best_li)
            wire_txt = wire_display(wire_labels[best_li]['text'])
            rows.append(make_boundary(
                end_info(ptext, ex, ey), wire_txt))
        else:
            rows.append(make_boundary(
                end_info(ptext, ex, ey), '未识别'))

    return rows, notes_log, wire_labels


def write_excel(rows, path, frame_id=52):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    wb = Workbook()
    ws = wb.active
    ws.title = 'From-To机器版'
    hdr_fill = PatternFill('solid', fgColor='4472C4')
    hdr_font = Font(color='FFFFFF', bold=True)
    for j, c in enumerate(COLS, 1):
        cell = ws.cell(row=1, column=j, value=c)
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal='center',
                                   vertical='center', wrap_text=True)
    for i, r in enumerate(rows, 2):
        ws.cell(row=i, column=1, value=i - 1)
        ws.cell(row=i, column=2, value=r['from_dev'])
        ws.cell(row=i, column=3, value=r['from_pin'])
        ws.cell(row=i, column=4, value=r['from_func'])
        ws.cell(row=i, column=5, value=r['to_dev'])
        ws.cell(row=i, column=6, value=r['to_pin'])
        ws.cell(row=i, column=7, value=r['to_func'])
        ws.cell(row=i, column=8, value=r['wire'])
        ws.cell(row=i, column=9, value=r['note'])
    widths = [6, 34, 12, 24, 34, 12, 24, 34, 22]
    for j, w in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = w
    ws.freeze_panes = 'A2'
    # 说明页
    ws2 = wb.create_sheet('说明')
    ws2['A1'] = ('框#%d 机器版 From-To（纯视觉管线输出，2026-10-18）。\n'
                 '方法：管脚号文本锚定最近导线端点 -> net-hub 关联 -> '
                 '导线线号配对；配不成对的单管脚 net 记框边界端口。\n'
                 '诚实纪律："未识别"/"待确认"为机器未能识别，如实标注，未编造。'
                 ) % frame_id
    ws2.column_dimensions['A'].width = 120
    wb.save(path)


if __name__ == '__main__':
    import sys
    fdir = sys.argv[1] if len(sys.argv) > 1 and \
        not sys.argv[1].startswith('--') else P52
    rows, notes_log, wire_labels = build_rows(fdir)
    print('rows:', len(rows))
    for r in rows:
        print('%s %s [%s] -> %s %s [%s] | %s | %s' % (
            r['from_dev'][:24], r['from_pin'], r['from_func'][:20],
            r['to_dev'][:24], r['to_pin'], r['to_func'][:20],
            r['wire'][:30], r['note'][:24]))
    print('--- notes ---')
    for n in notes_log:
        print(n)
    print('--- wire labels matched by WIRE_RE:', len(wire_labels))
    if '--write' in sys.argv:
        out = os.path.join(fdir, 'FromTo_机器版.xlsx')
        fid = 52
        m = re.search(r'pilot_(\d+)', fdir)
        if m:
            fid = int(m.group(1))
        write_excel(rows, out, frame_id=fid)
        print('wrote', out)
