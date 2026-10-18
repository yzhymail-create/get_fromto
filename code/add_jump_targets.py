#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨图跳转符号目标补录（纯视觉）。

导线末端的三角箭头是超链接跳转指示符号，其旁文字（如 XF11 IBS POWER）
是跳转目标，填入 From-To 表的目标用电器列。

方法（框#112 实证）：
  跳转文字位于导线标签同一 y 带、标签右侧、导线上方 8~70px、
  距标签右边缘 <800px。排除线号规格文字、管脚号、器件名。
  歧义时用"线号头前缀"加分（如 FRZD35_01 -> FRZD35 PEPS-）。

用法：
  python3 add_jump_targets.py <pilot_dir>   # 更新该目录下 FromTo_机器版.xlsx

Windows 硬约束：pathlib、无 shell 调用。
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from build_label_rows import extract_designations, des_from_text, norm  # noqa

WIRE_SPEC = re.compile(r'\d\.\d{2}|FLH|FLRY')
PIN_RE = re.compile(r'^(BD|IP|RLQ|RRQ|BQ|RRD|RLD)\d+(-\d+)?$')
DEVNAME_RE = re.compile(r'Antenna|Module|Sensor|Moudle|Door|Perform', re.I)
# 手工核对过的显示修正（OCR 丢空格）
MANUAL_PRETTY = {
    'XF11 IBSPOWER': 'XF11 IBS POWER',
    'ONUSBGND1': 'ONUSB GND 1',
    'ONUSBGND2': 'ONUSB GND 2',
    'FUSBGND': 'FUSB GND',
}


def _cx(a):
    return (a[0] + a[2]) / 2


def _cy(a):
    return (a[1] + a[3]) / 2


def _prefix_of(des):
    m = re.match(r'^([A-Z]{2,}\d+)', re.sub(r'[^A-Z0-9]', '', des or ''))
    return m.group(1) if m else ''


def _pretty(s):
    if s in MANUAL_PRETTY:
        return MANUAL_PRETTY[s]
    m = re.match(r'^([A-Z]{2,}\d+)([A-Z]{3,}\d*)$', s)
    if m and len(s) >= 9:
        return m.group(1) + ' ' + m.group(2)
    return s


def find_jump_targets(pilot_dir):
    """返回 {行号: (线号key, 跳转文字或None)}。"""
    import openpyxl
    oc = json.load(open(os.path.join(pilot_dir, 'ocr_texts.json'),
                         encoding='utf-8'))
    des_map = extract_designations(oc)
    wb = openpyxl.load_workbook(
        os.path.join(pilot_dir, 'FromTo_机器版.xlsx'))
    ws = wb[wb.sheetnames[0]]
    out = {}
    for r in ws.iter_rows(min_row=2):
        if not any(c.value for c in r):
            continue
        row = [str(c.value) if c.value is not None else '' for c in r[:9]]
        if '出框' not in row[4]:
            continue
        idx, wire = row[0], row[7]
        des, _ = des_from_text(wire)
        key = des if des in des_map else next(
            (k for k in des_map if norm(k) == norm(des or '')), None)
        if not key:
            out[idx] = (None, None)
            continue
        lab = max(des_map[key], key=lambda o: len(o['text']))
        a = lab['aabb']
        lcy, lx2 = _cy(a), a[2]
        pre = _prefix_of(key)
        scored = []
        for t in oc:
            ta = t['aabb']
            txt = t['text'].strip()
            dyc = _cy(ta) - lcy
            dx = _cx(ta) - lx2
            if not (-70 <= dyc <= 15):
                continue
            if not (0 < dx < 800 and len(txt) > 1):
                continue
            if WIRE_SPEC.search(txt):
                continue
            if PIN_RE.match(norm(txt)):
                continue
            if DEVNAME_RE.search(txt):
                continue
            if txt == lab['text'].strip():
                continue
            bonus = -100 if norm(txt).startswith(pre) else 0
            scored.append((abs(dyc) + bonus, txt))
        scored.sort()
        out[idx] = (key, _pretty(scored[0][1]) if scored else None)
    return out


def apply_jump_targets(pilot_dir):
    """把跳转目标写回 Excel，返回 (补上数, 无符号数)。"""
    import openpyxl
    targets = find_jump_targets(pilot_dir)
    xlsx = os.path.join(pilot_dir, 'FromTo_机器版.xlsx')
    wb = openpyxl.load_workbook(xlsx)
    ws = wb[wb.sheetnames[0]]
    n_jump, n_no = 0, 0
    for r in ws.iter_rows(min_row=2):
        if not any(c.value for c in r):
            continue
        idx = str(r[0].value)
        if idx not in targets:
            continue
        key, jump = targets[idx]
        if jump:
            r[4].value = jump
            r[5].value = '—'
            r[6].value = '—'
            old = str(r[8].value or '')
            if '跨图跳转' not in old:
                r[8].value = (old + '；' if old else '') + '跨图跳转'
            n_jump += 1
        else:
            old = str(r[8].value or '')
            if '无跳转符号' not in old:
                r[8].value = (old + '；' if old else '') + '无跳转符号'
            n_no += 1
    wb.save(xlsx)
    return n_jump, n_no


if __name__ == '__main__':
    d = sys.argv[1]
    j, n = apply_jump_targets(d)
    print('跨图跳转: %d, 无跳转符号: %d' % (j, n))
