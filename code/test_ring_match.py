#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试两跳匹配在框#112 上的效果（不写回，只打印）。"""
import os
import sys
import json
import time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/hatch/workspace/.pylibs')
from trace import binarize_otsu, skeleton_of, extract_nets, filter_nets
from pin_ring_match import associate_pins_v2

OUT = os.path.join(HERE, 'qa', 'frames', 'pilot_112')
t0 = time.time()
gray = cv2.imread(os.path.join(OUT, 'render.png'), cv2.IMREAD_GRAYSCALE)
fg = binarize_otsu(gray)
skel = skeleton_of(fg)
nets, ww, dots = extract_nets(fg, skel)
wire_nets, small = filter_nets(nets, min_pixels=50)
print('nets=%d wire=%d (%.1fs)' % (len(nets), len(wire_nets), time.time() - t0))

texts = json.load(open(os.path.join(OUT, 'ocr_texts.json'), encoding='utf-8'))
devs = json.load(open(os.path.join(OUT, 'devices.json'), encoding='utf-8'))
log = []
pa = associate_pins_v2(gray, texts, wire_nets, devs, log)
for s in log:
    print(s)
pm = pa['pin_match']
# 同一 net 被多个管脚认领？
from collections import Counter
c = Counter(v[0] for v in pm.values())
print('nets with >1 pin:', {k: v for k, v in c.items() if v > 1})
# 同一器件的两个管脚认领同一 net（可疑行预警）
import re
def dev_of(pin):
    m = re.match(r'^([A-Z]{2}\d+)-', pin)
    return m.group(1) if m else '?'
net_pins = {}
for pin, (ni, _, _, _) in pm.items():
    net_pins.setdefault(ni, []).append(pin)
for ni, pins in net_pins.items():
    devs_hit = set(dev_of(p) for p in pins)
    if len(pins) > 1 and len(devs_hit) == 1:
        print('SUSPECT same-device net%d:' % ni, pins)
