#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯视觉导线追踪器：只看渲染后的像素，绝不解析 DXF 图元做追踪。

管线：
  DXF --渲染--> PNG --二值化--> 骨架 --连通域--> 导线网络 --OCR--> 文字绑定
       (render.py)   (trace.py)              (ocr.py)

用法：
  python3 tracer.py <dxf> --bbox x1,y1,x2,y2 --dpi-px 8 --out dir

注意：
- render.py 是管线中唯一接触 DXF 的模块（仅做光栅化、开关图层、裁剪）
- trace.py / ocr.py 只接收 numpy 图像
- 需要 PYTHONPATH 包含 ezdxf、cv2、skimage
"""
import argparse, glob, json, os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import render_dxf
from trace import (binarize_otsu, skeleton_of, extract_nets, filter_nets,
                   overlay_nets)
from ocr import find_text_regions, attach_text_to_nets, recognize_text, xywh_to_aabb

# OCR 引擎：统一走 ocr.recognize_text（RapidOCR，见 requirements.txt）。
# 引擎不可用时 recognize_text 抛 ImportError；识别到 0 个文字时返回 []。
# 两者明确区分（CODEX 评审 P0 修复；旧的 tesseract/paddleocr 分支已删除，
# ocr.py 中也不存在旧代码引用的 ocr_image）。


def build_from_to(nets, text_attached):
    """组装 From-To 列表。

    nets: trace.py 的 wire nets
    text_attached: {net_idx: [(text, conf, dist)]}
    无文字的记"未标注"，绝不编造。
    """
    rows = []
    for i, n in enumerate(nets):
        texts = [t for t, _, _ in text_attached.get(i, []) if t.strip()]
        label = '; '.join(texts) if texts else '未标注'
        px = n['pixels']
        rows.append({
            'id': i,
            'label': label,
            'n_endpoints': n['n_endpoints'],
            'endpoints_yx': n['endpoints'][:8],
            'bbox_yx': [int(px[:, 0].min()), int(px[:, 0].max()),
                        int(px[:, 1].min()), int(px[:, 1].max())],
            'length_px': len(px),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description='纯视觉导线追踪')
    ap.add_argument('dxf', help='DXF 文件路径')
    ap.add_argument('--bbox', required=True, help='x1,y1,x2,y2 图纸单位')
    ap.add_argument('--dpi-px', type=float, default=8.0, dest='px',
                    help='每图纸单位像素数')
    ap.add_argument('--out', default='out', help='输出目录')
    ap.add_argument('--no-ocr', action='store_true', help='跳过 OCR')
    args = ap.parse_args()

    bbox = tuple(float(v) for v in args.bbox.split(','))
    os.makedirs(args.out, exist_ok=True)
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)

    # 1. 渲染（唯一接触 DXF 的步骤）
    t0 = time.time()
    png = os.path.join(args.out, 'render.png')
    w, h = render_dxf(args.dxf, png, bbox=bbox, px_per_unit=args.px)
    say('render: %dx%d px (%.1fs)' % (w, h, time.time() - t0))

    # 2-3. 二值化 + 细化 + 导线提取（纯视觉）
    gray = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    t0 = time.time()
    fg = binarize_otsu(gray)
    say('binarize: fg_ratio=%.4f (%.1fs)' % (fg.mean(), time.time() - t0))
    t0 = time.time()
    skel = skeleton_of(fg)
    say('skeleton: %d px (%.1fs)' % (skel.sum(), time.time() - t0))
    t0 = time.time()
    nets, wire_width, dots = extract_nets(fg, skel)
    say('nets: raw=%d wire_width=%.1fpx dot_junctions=%d (%.1fs)'
        % (len(nets), wire_width, len(dots), time.time() - t0))
    wire_nets, small_nets = filter_nets(nets, min_pixels=50)
    say('wire nets: %d (small/filtered: %d)' % (len(wire_nets), len(small_nets)))

    # 4. OCR（可选；统一接口 recognize_text）
    attached = {}
    if not args.no_ocr:
        t0 = time.time()
        try:
            texts = recognize_text(gray)
        except ImportError as e:
            say('ocr: 引擎不可用（%s），仅做文字块检测' % e)
            texts = []
            regions = find_text_regions(fg)
            say('ocr: 检测到 %d 个文字块（未识别：引擎缺失）' % len(regions))
        else:
            say('ocr(recognize_text): %d texts (%.1fs)' % (len(texts), time.time() - t0))
            if texts:
                attached = attach_text_to_nets(texts, wire_nets)
            else:
                say('ocr: 引擎正常，但本图未识别到文字')
    n_labeled = sum(1 for i in range(len(wire_nets)) if attached.get(i))
    say('nets with text: %d / %d' % (n_labeled, len(wire_nets)))

    # 5. From-To
    rows = build_from_to(wire_nets, attached)
    with open(os.path.join(args.out, 'from_to.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    say('from_to: %d rows -> from_to.json' % len(rows))

    # 6. 叠加图
    overlay_nets(gray, wire_nets, os.path.join(args.out, 'overlay.png'))
    say('overlay -> overlay.png')

    with open(os.path.join(args.out, 'run.log'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log) + '\n')
    say('DONE -> ' + args.out)


if __name__ == '__main__':
    main()
