#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PDF 渲染模块：PDF 页面 -> PNG（纯像素，后续只走视觉管线）。

与 render.py 的 render_dxf 对应：输入格式无关，输出都是 PNG 像素。
只做渲染（合法预处理），绝不提取 PDF 内嵌文字/矢量做追踪。

依赖：pymupdf（pip 有 Windows wheel，符合 Windows 交付约束）。
"""


def render_pdf(pdf_path, out_png, page=0, dpi=150, bbox=None):
    """渲染 PDF 指定页面为 PNG。

    pdf_path : PDF 文件路径
    out_png  : 输出 PNG 路径
    page     : 页码（0 起）
    dpi      : 渲染分辨率。150 dpi 在本测试（1571x674 pts）得 3273x1404 px，
               与 DXF 管线的 8 px/单位量级相当
    bbox     : (x1, y1, x2, y2) 页面点坐标裁剪；None = 整页
    返回 (width_px, height_px)
    """
    import pymupdf
    doc = pymupdf.open(pdf_path)
    pg = doc[page]
    if bbox is not None:
        pg.set_cropbox(pymupdf.Rect(*bbox))
    pix = pg.get_pixmap(dpi=dpi)
    pix.save(out_png)
    doc.close()
    return pix.width, pix.height


if __name__ == '__main__':
    import sys
    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else 'render.png'
    w, h = render_pdf(pdf, out)
    print('rendered %dx%d -> %s' % (w, h, out))
