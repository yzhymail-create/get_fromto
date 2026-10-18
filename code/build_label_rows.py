# -*- coding: utf-8 -*-
"""按线号成行（纯视觉，框#112）。

口径（用户 2026-10-18 明确，skill 第4条）：
  一行 = 一个线号 = 一根导线 = 一个 From-To 关系。
  同一线号出现多次合并为一行；INLINE/拼接点两侧线号不同 = 两行。

只复用 pilot_112/ 缓存（render.png / ocr_texts.json / pin_assoc.pkl /
devices.json），不重渲染、不重跑 OCR、不重做骨架。

处理：
  - 下划线被吞（RF0802->RF08_02）、O/0误识（BFO1 01R->BF01_01）
  - 碎片按"严格前缀"合并；短尾标签按位置就近归入
  - 粘连色码（DKCANL_04R->DKCANL_04）
  - 同一管脚被多线号认领 -> 标签最近者得，其余为拼接点右段

诚实纪律：归属不确定的记"待确认"，不编造。
Windows 约束：os.path，无 shell 调用。
"""
import os
import re
import json
import pickle
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PIN_RE = re.compile(r'^[A-Z]{2,}\d+-\d+[A-Z]?$')


def cx(a):
    return (a[0] + a[2]) / 2


def cy(a):
    return (a[1] + a[3]) / 2


def norm(s):
    s = re.sub(r'\s+', '', s or '').upper()
    return re.sub(r'[–—−‐]', '-', s)  # 统一横杠变体（跨平台 OCR 兼容，见 run_frame.norm）


def des_from_text(txt):
    """从整段 OCR 文本提取线号；返回 (des, is_short)。"""
    t = txt.strip()
    if not t:
        return None, False
    toks = t.split()
    tok0 = toks[0]
    # 字母间下划线是 OCR 误读：DK_CAN_H -> DKCANH
    tok0 = re.sub(r'(?<=[A-Z])_(?=[A-Z])', '', tok0)
    # BFO1 01R 形：O->0，且线号跨两个 token
    m = re.match(r'^([A-Z]{2})O(\d+)$', tok0)
    if m and len(toks) > 1:
        m2 = re.match(r'^(\d+)[A-Z]?$', toks[1])
        if m2:
            return '%s0%s_%s' % (m.group(1), m.group(2), m2.group(1)), False
    # BFO1 -> BF01（单个 token 内）
    tok0 = re.sub(r'^([A-Z]{2})O(\d)', r'\g<1>0\g<2>', tok0)
    # 主模式：字母数字(-段)* (_段)+
    m = re.match(r'^([A-Z0-9]+(?:-[A-Z0-9]+)*(?:_[A-Z0-9]+)+)', tok0)
    if m:
        des = m.group(1)
        des = re.sub(r'(\d)[A-Z]$', r'\1', des)  # 去尾部色码
        return des, False
    # 无下划线形：RF0802Y... -> RF08_02
    m = re.match(r'^([A-Z]{2,4})(\d{4})(?=[A-Z])', tok0)
    if m:
        d = m.group(2)
        return '%s_%s' % (m.group(1) + d[:2], d[2:]), False
    # 短号形：FRZD39（短尾标签）
    m = re.match(r'^([A-Z]+\d+)$', tok0)
    if m:
        return m.group(1), True
    return None, False


def extract_designations(texts):
    """返回 {des: [{'aabb':..., 'text':..., 'manual':bool}]}。"""
    raw = defaultdict(list)   # des -> occs（含碎片/短号）
    shorts = set()
    for t in texts:
        txt = t['text'].strip()
        if not txt or len(txt) < 3:
            continue
        tn = norm(txt)
        if PIN_RE.match(tn):
            continue
        if re.fullmatch(r'(BD|IP)\d{2,4}', tn):
            continue  # 纯器件代号
        des, is_short = des_from_text(txt)
        if not des:
            continue
        # 排除端口/信号标签：无下划线的纯字母数字（如 CWC05/WB003/ONUSB）
        # 短号保留用于归入
        if '_' not in des and not is_short:
            continue
        # 真线号特征：含 _数字；短号(is_short)保留用于归入
        if not re.search(r'_\w*\d', des) and not is_short:
            continue
        if des.endswith('_CWC') or des == 'KL30':
            continue
        raw[des].append({'aabb': t['aabb'], 'text': txt,
                         'manual': bool(t.get('manual_patch'))})
        if is_short:
            shorts.add(des)

    full = {}
    frag = {}
    for d, occs in raw.items():
        # 碎片：是其他 des 的严格前缀，或以下划线/短横结尾
        is_frag = d.endswith('_') or d.endswith('-')
        full[d] = occs
    # 第二遍：严格前缀 -> 碎片
    keys = list(full.keys())
    for d in keys:
        if d not in full:
            continue
        for e in keys:
            if e != d and e in full and e.startswith(d) and len(e) > len(d):
                # d 是 e 的前缀 -> d 为碎片（短号也按此处理，多候选时按位置）
                frag[d] = full.pop(d)
                break
    # 碎片归入：单候选直接并入；多候选按出现位置就近
    for fd, occs in frag.items():
        cands = [d for d in full if d.startswith(fd)]
        if len(cands) == 1:
            full[cands[0]].extend(occs)
        elif len(cands) > 1:
            for o in occs:
                ox, oy = cx(o['aabb']), cy(o['aabb'])
                best, bd = None, 1e18
                for d in cands:
                    for fo in full[d]:
                        dx = cx(fo['aabb']) - ox
                        dy = cy(fo['aabb']) - oy
                        dd = dx * dx + dy * dy
                        if dd < bd:
                            bd, best = dd, d
                full[best].append(o)
        # 无候选则丢弃
    # 短号若未被合并（无完整候选），也丢弃，不单独成行
    for d in [x for x in list(full.keys()) if re.match(r'^[A-Z]+\d+$', x)]:
        if not any(e != d and e.startswith(d) for e in full):
            # 检查它是否作为碎片被处理过；若 raw 中它是短号且无完整形，丢弃
            pass
    # 短号未合并则丢弃，不单独成行
    for d in list(shorts):
        if d in full:
            del full[d]
    return full


# 框#112 专属人工规则（看图确认），仅 pilot_112 目录生效。
# （ChatGPT P1-4：人工规则必须标注适用范围，不许 silently 作用于其他框。
# build_label_rows() 开头有目录门控，非 pilot_112 直接返回空走 fallback。）
# 框#112 拼接点（看图确认）：name, 左段线号, 右段线号
SPLICES = {
    'RF23_05': ('IPS67', None),
    'RF23_01': (None, 'IPS67'),
    'RF23_02': (None, 'IPS67/J1'),  # 两处走线，备注说明
    'RF23_03': ('J1', None),
    'RF23_04': ('J1', None),
    'RF23_06': ('J1', None),
    'FRZD39_02': ('INLINE 35', None),
    'FRZD39_01': (None, 'INLINE 35'),
    'FRZD40_02': ('INLINE 36', None),
    'FRZD40_01': (None, 'INLINE 36'),
    'FRZD37_03': ('INLINE 35', None),
    'FRZD37_01': (None, 'INLINE 35'),
    'FRZD36_03': ('INLINE 36', None),
    'FRZD36_01': (None, 'INLINE 36'),
}

# 同一线号出现在两根物理导线上（看图确认）：按 y 簇拆分，各计一行
# des -> [(簇y, from_kind, from_val, to_kind, to_val, 备注)]
# kind: 'pin'（用 y 带找管脚）/ 'splice' / 'port'
SPLIT_WIRES = {
    'RF23_02': [
        (746, 'splice', 'IPS67', 'port', 'RF23 0NUSB', 'IPS67上分支'),
        (1735, 'splice', 'J1', 'port', 'RF23 0NUSB', 'J1右侧段'),
    ],
    'RF23_04': [
        (2003, 'pin', None, 'port', 'WB003', ''),
        (1652, 'port', 'WB003', 'splice', 'J1', ''),
    ],
    'RF23_06': [
        (2380, 'pin', None, 'port', 'WB035', ''),
        (1690, 'port', 'WB035', 'splice', 'J1', ''),
    ],
}


def build_label_rows(fdir):
    """按线号组装 From-To 行。返回 (rows, des_map, des_pin)。

    框#112 专属：SPLICES / SPLIT_WIRES 为看图确认的人工规则，
    仅当 fdir 为 pilot_112 时生效；其他框返回 ([], {}, {})，
    由调用方 fallback 到 build_fromto.build_rows。
    """
    import os as _os
    if 'pilot_112' not in _os.path.basename(_os.path.normpath(fdir)):
        return [], {}, {}
    import sys
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import build_fromto as bf

    texts, devs, pa = bf.load(fdir)
    pm = pa['pin_match']
    devs_info = bf.device_info(texts, devs)
    # 补非 BD 器件代号（如 IP03）
    _code_re = re.compile(r'^[A-Z]{2,}\d+$')
    for did, info in devs_info.items():
        if not info['code']:
            x, y, w, h = info['bbox']
            for t in texts:
                a = t['aabb']
                tx, ty = cx(a), cy(a)
                if x - 60 < tx < x + w + 60 and y - 320 < ty < y + 40:
                    if _code_re.match(norm(t['text'])):
                        info['code'] = t['text'].strip()
                        break
    # 器件名修正：优先取框内文本（device_info 取最长易串到别家）
    for did, info in devs_info.items():
        x, y, w, h = info['bbox']
        inside = []
        for t in texts:
            a = t['aabb']
            tx, ty = cx(a), cy(a)
            if x < tx < x + w and y < ty < y + h:
                txt = t['text'].strip()
                tn = norm(txt)
                if PIN_RE.match(tn) or re.fullmatch(r'(BD|IP)\d{2,4}', tn):
                    continue
                # 管脚功能文字短（<15），器件名长；只取长的
                if len(txt) >= 15:
                    inside.append((len(txt), txt))
        if inside:
            inside.sort(reverse=True)
            # 选最长且不含明显别家关键词的；否则取最长
            pick = inside[0][1]
            info['name'] = pick
    code2id = {}
    for did, info in devs_info.items():
        if info['code']:
            code2id[norm(info['code'])] = did

    def fix_pin_ocr(txt):
        # Q/D 误识：BQ19-1->BD19-1，RRQ06-1->RRD06-1，RLQ06-1->RLD06-1
        t = txt.strip()
        t = re.sub(r'^BQ(\d)', r'BD\g<1>', t)
        t = re.sub(r'^RRQ(\d)', r'RRD\g<1>', t)
        t = re.sub(r'^RLQ(\d)', r'RLD\g<1>', t)
        return t
    pin_xy = {}
    pin_raw = {}
    for t in texts:
        if PIN_RE.match(norm(t['text'])):
            a = t['aabb']
            fixed = fix_pin_ocr(t['text'])
            pin_xy[fixed] = (cx(a), cy(a))
            pin_raw[fixed] = t['text'].strip()
    # pm 的键也做同样纠错
    pin_anchor = {}
    for p, v in pm.items():
        pin_anchor[fix_pin_ocr(p)] = (v[1], v[2])
    # pin_xy 若 pm 缺失则回退
    for p in list(pin_xy.keys()):
        if p not in pin_anchor:
            pin_anchor[p] = pin_xy[p]

    des_map = extract_designations(texts)

    # 1. 每个线号找 y 带内最近管脚（管脚在标签左侧）
    des_pin = {}
    for des, occs in des_map.items():
        lys = sorted(cy(o['aabb']) for o in occs)
        ly = lys[len(lys) // 2]
        min_lx = min(cx(o['aabb']) for o in occs)
        best, bd = None, 1e18
        for p, (px, py) in pin_xy.items():
            if abs(py - ly) > 40:
                continue
            if px > min_lx + 150:
                continue
            d2 = (min_lx - px) ** 2 + (py - ly) ** 2
            if d2 < bd:
                bd, best = d2, p
        des_pin[des] = best

    # 2. 同一管脚被多线号认领 -> 标签最近者得
    pin_des = defaultdict(list)
    for des, p in des_pin.items():
        if p:
            pin_des[p].append(des)
    for p, ds in pin_des.items():
        if len(ds) > 1:
            px, py = pin_xy[p]
            ds_sorted = sorted(
                ds, key=lambda d: min(abs(cx(o['aabb']) - px)
                                     for o in des_map[d]))
            for d in ds_sorted[1:]:
                des_pin[d] = None

    def end_of_pin(pin_text):
        ex, ey = pin_anchor.get(pin_text, pin_xy.get(pin_text, (0, 0)))
        m = re.match(r'^([A-Z]+\d+)-', norm(pin_text))
        dev_id = code2id.get(m.group(1)) if m else None
        px, py = pin_xy.get(pin_text, (0, 0))
        note_dev = ''
        if dev_id is None:
            best, bd = None, 500
            for did, info in devs_info.items():
                x, y, w, h = info['bbox']
                dx = max(x - px, 0, px - (x + w))
                dy = max(y - py, 0, py - (y + h))
                dd = (dx * dx + dy * dy) ** 0.5
                if dd < bd:
                    bd, best = dd, did
            dev_id = best
            note_dev = '管脚代号与器件代号不一致，按位置归属' if best else ''
        if dev_id is None:
            return {'device': '待确认', 'pin': pin_text, 'func': '未识别',
                    'note_dev': ''}
        info = devs_info[dev_id]
        device = ('%s(%s)' % (info['name'], info['code'])
                  if info['name'] else info['code'])
        # 快速版：只用检测文本找功能（跳过条带重识别，提速）
        func = bf.pin_function_side(
            texts, info['bbox'], ex, ey, y_tol=45)[0]
        return {'device': device, 'pin': pin_text,
                'func': func if func else '未识别', 'note_dev': note_dev}

    def wire_full_text(des):
        occs = des_map[des]
        best = max(occs, key=lambda o: len(o['text']))
        return best['text'].strip()

    def make_end(kind, val, pin_hint_y=None):
        # kind: pin / splice / port / out
        if kind == 'pin':
            # 按 y 带找管脚
            best, bd = None, 1e18
            for p, (px, py) in pin_xy.items():
                if abs(py - pin_hint_y) > 40:
                    continue
                d2 = (py - pin_hint_y) ** 2
                if d2 < bd:
                    bd, best = d2, p
            if best:
                return end_of_pin(best), []
            return {'device': '待确认', 'pin': '—', 'func': '未识别'}, \
                ['待确认：未找到管脚']
        if kind == 'splice':
            return {'device': '拼接点', 'pin': val, 'func': '—'}, []
        if kind == 'port':
            return {'device': val + '端口', 'pin': '—', 'func': '—'}, []
        return {'device': '—（出框）', 'pin': '—', 'func': '—'}, []

    def wire_text_for(des, cy_target=None):
        occs = des_map[des]
        if cy_target is not None:
            occs = sorted(occs, key=lambda o: abs(cy(o['aabb']) - cy_target))
        best = max(occs[:3], key=lambda o: len(o['text']))
        return best['text'].strip()

    rows = []
    for des in sorted(des_map):
        # 拆分线号：同一线号多根物理导线，各计一行
        if des in SPLIT_WIRES:
            for (cy_t, fk, fv, tk, tv, sp_note) in SPLIT_WIRES[des]:
                fro, n1 = make_end(fk, fv, cy_t)
                to, n2 = make_end(tk, tv, cy_t)
                notes = n1 + n2
                if sp_note:
                    notes.append(sp_note)
                if isinstance(fro, dict) and fro.get('note_dev'):
                    notes.append(fro['note_dev'])
                # end_of_pin 返回 dict；make_end 的 pin 分支返回 (dict, notes)
                # 统一处理：fro 已是 dict
                rows.append({
                    'from_dev': fro['device'], 'from_pin': fro['pin'],
                    'from_func': fro['func'],
                    'to_dev': to['device'], 'to_pin': to['pin'],
                    'to_func': to['func'],
                    'wire': wire_text_for(des, cy_t),
                    'note': '；'.join(notes), 'des': des})
            continue
        pin = des_pin.get(des)
        sp_left, sp_right = SPLICES.get(des, (None, None))
        wire_txt = wire_full_text(des)
        notes = []
        if pin:
            e = end_of_pin(pin)
            if e.get('note_dev'):
                notes.append(e['note_dev'])
            fro = e
            if sp_left:
                to = {'device': '拼接点', 'pin': sp_left, 'func': '—'}
                notes.append('经拼接点' + sp_left)
            else:
                to = {'device': '—（出框）', 'pin': '—', 'func': '—'}
        elif sp_right:
            fro = {'device': '拼接点', 'pin': sp_right, 'func': '—'}
            to = {'device': '—（出框）', 'pin': '—', 'func': '—'}
            notes.append('拼接点右侧段')
        else:
            notes.append('待确认：未找到管脚归属')
            fro = {'device': '待确认', 'pin': '—', 'func': '未识别'}
            to = {'device': '—（出框）', 'pin': '—', 'func': '—'}
        if fro['func'] == '未识别' and fro['device'] not in ('待确认', '拼接点'):
            notes.append('待确认：管脚功能未识别')
        rows.append({'from_dev': fro['device'], 'from_pin': fro['pin'],
                     'from_func': fro['func'],
                     'to_dev': to['device'], 'to_pin': to['pin'],
                     'to_func': to['func'],
                     'wire': wire_txt, 'note': '；'.join(notes),
                     'des': des})
    # 按线号排序，拆分行紧随其线号
    rows.sort(key=lambda r: r['des'])
    return rows, des_map, des_pin


if __name__ == '__main__':
    import sys
    import time
    t0 = time.time()
    fdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, 'qa', 'frames', 'pilot_112')
    rows, des_map, des_pin = build_label_rows(fdir)
    print('designations=%d rows=%d (%.1fs)' % (
        len(des_map), len(rows), time.time() - t0))
    for i, r in enumerate(rows, 1):
        print('%2d %-10s %-30s %-10s -> %-12s %s' % (
            i, r['des'], r['from_dev'][:30], r['from_pin'],
            r['to_dev'][:12], r['note'][:50]))
    print('无管脚归属:',
          [d for d, p in des_pin.items() if not p])
