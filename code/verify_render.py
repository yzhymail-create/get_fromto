#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨平台渲染一致性验证（2026-10-18）。

用法：python verify_render.py <frame_id> [--render]
  --render: 在本机重新渲染（需要 DXF）；不加则用 frames/pilot_<id>/render.png

输出：OCR 文本数、管脚数、管脚列表。与 Linux 参考值对比：
  框112: OCR 247 texts, pins 27
  框52:  OCR 162 texts, pins 23

判定：管脚数一致且管脚列表一致 → 通过；否则把本输出贴给 MUSE。
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Linux 参考值（MUSE 实测）
REFERENCE = {
    112: {'ocr_texts': 247, 'pins': 27},
    52: {'ocr_texts': 162, 'pins': 23},
}


def norm(s):
    s = re.sub(r'\s+', '', s or '').upper()
    return re.sub(r'[–—−‐]', '-', s)


PIN_RE = re.compile(r'^[A-Z]{2,}\d+-\d+[A-Z]?$')


def main():
    fid = int(sys.argv[1]) if len(sys.argv) > 1 else 112
    do_render = '--render' in sys.argv
    pilot = os.path.join(HERE, 'frames', 'pilot_%d' % fid)
    png = os.path.join(pilot, 'render.png')

    if do_render:
        from render import render_dxf
        frames = json.load(open(os.path.join(
            HERE, 'frames', 'subframes.json'), encoding='utf-8'))['frames']
        f = next(x for x in frames if x['id'] == fid)
        dxf = sys.argv[sys.argv.index('--dxf') + 1] if '--dxf' in sys.argv else None
        if not dxf:
            print('需要 --dxf <路径>')
            sys.exit(1)
        w, h = render_dxf(dxf, png, bbox=tuple(f['units']), px_per_unit=8.0)
        print('rendered: %dx%d' % (w, h))

    # 读图（PIL，中文路径安全）
    from PIL import Image
    import numpy as np
    gray = np.array(Image.open(png).convert('L'))
    print('image: %dx%d' % (gray.shape[1], gray.shape[0]))

    # OCR
    from ocr import recognize_tiled
    import time
    t0 = time.time()
    items = recognize_tiled(gray)
    print('ocr: %d texts (%.1fs)' % (len(items), time.time() - t0))

    # 管脚
    pins = sorted(set(t.strip() for _, t, _ in items if PIN_RE.match(norm(t))))
    print('pins: %d' % len(pins))
    for p in pins:
        print('  ' + p)

    # 对比
    ref = REFERENCE.get(fid)
    if ref:
        ok_ocr = abs(len(items) - ref['ocr_texts']) <= 5
        ok_pin = len(pins) == ref['pins']
        print('---')
        print('OCR texts: %d (ref %d) %s' % (
            len(items), ref['ocr_texts'], 'OK' if ok_ocr else 'DIFF'))
        print('pins: %d (ref %d) %s' % (
            len(pins), ref['pins'], 'OK' if ok_pin else 'DIFF'))
        if not (ok_ocr and ok_pin):
            print('不一致！把以上输出贴给 MUSE 排查。')


if __name__ == '__main__':
    main()
