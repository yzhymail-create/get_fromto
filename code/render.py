# -*- coding: utf-8 -*-
"""DXF -> 光栅渲染（纯视觉管线的前端）。

本模块是管线中唯一允许接触 DXF 图元的部分：
- 只做渲染（开/关图层、按 bbox 裁剪、光栅化）
- 追踪模块 (trace.py) 只接收 numpy 图像，绝不接触 DXF 数据
"""
import ezdxf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.config import Configuration, ColorPolicy


def _entity_in_bbox(e, x1, y1, x2, y2, margin):
    """快速图元-bbox 相交测试（仅用于渲染加速，不属于追踪）。"""
    try:
        t = e.dxftype()
        pts = []
        if t == 'LINE':
            pts = [e.dxf.start, e.dxf.end]
        elif t == 'LWPOLYLINE':
            pts = [(p[0], p[1]) for p in e.get_points('xy')]
        elif t in ('CIRCLE', 'ARC'):
            c, r = e.dxf.center, e.dxf.radius
            return not (c[0] + r < x1 - margin or c[0] - r > x2 + margin or
                        c[1] + r < y1 - margin or c[1] - r > y2 + margin)
        elif t in ('TEXT', 'MTEXT', 'INSERT', 'POINT'):
            p = e.dxf.insert
            pts = [p]
        elif t == 'POLYLINE':
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
        else:
            return True  # 未知类型不过滤
        for p in pts:
            if x1 - margin <= p[0] <= x2 + margin and y1 - margin <= p[1] <= y2 + margin:
                return True
        # 线段可能横跨 bbox：检查包络盒相交
        if pts:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            return not (max(xs) < x1 - margin or min(xs) > x2 + margin or
                        max(ys) < y1 - margin or min(ys) > y2 + margin)
        return False
    except Exception:
        return True


def render_dxf(dxf_path, out_png, bbox=None, px_per_unit=4.0,
               off_layers=('00_BACKGROUND',), bg_color='#ffffff',
               prefilter=True, margin=500.0):
    """渲染 DXF 指定区域为 PNG。

    dxf_path : DXF 文件路径
    out_png  : 输出 PNG 路径
    bbox     : (x1, y1, x2, y2) 图纸单位；None = 全图
    px_per_unit : 每图纸单位多少像素（决定有效分辨率）
    off_layers : 渲染时关闭的图层（预处理，不属于追踪）
    返回 (width_px, height_px)
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # 跨平台渲染一致性（2026-10-18）：DXF 文字样式多为 SHX 描边字体
    #（TXT.shx/romans.shx），ezdxf 在 Linux/Windows 上解析结果不同——
    # Windows 回退到描边渲染（空心细字），OCR 直接崩（管脚 27→1）。
    # 强制所有样式指向 DejaVu Sans TTF（matplotlib 自带，两边同一文件），
    # 绕过 ezdxf 的平台相关字体解析。rcParams 备用（部分路径仍走 matplotlib）。
    import matplotlib as _mpl
    _mpl.rcParams['font.family'] = 'DejaVu Sans'
    _mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
    try:
        import os as _os
        from matplotlib.font_manager import findfont, FontProperties
        _dejavu = _os.path.basename(
            findfont(FontProperties(family='DejaVu Sans')))
        for _st in doc.styles:
            try:
                _st.dxf.font = _dejavu  # 文件名如 DejaVuSans.ttf，不用全路径
            except Exception:
                pass
    except Exception:
        pass

    # 关闭背景装饰图层（合法的渲染预处理）
    for lname in off_layers:
        try:
            doc.layers.get(lname).off()
        except Exception:
            pass

    if bbox is None:
        from ezdxf import bbox as _bbox
        ext = _bbox.extents(msp)
        bbox = (ext.extmin.x, ext.extmin.y, ext.extmax.x, ext.extmax.y)
    x1, y1, x2, y2 = bbox
    w_units, h_units = x2 - x1, y2 - y1
    w_px = int(w_units * px_per_unit)
    h_px = int(h_units * px_per_unit)
    print('render bbox units: %.0f x %.0f -> %d x %d px' % (w_units, h_units, w_px, h_px), flush=True)

    fig = plt.figure(figsize=(w_px / 100.0, h_px / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    if prefilter and bbox is not None:
        # 空间预过滤：只把 bbox 附近图元送入渲染后端（加速，不属于追踪）
        x1, y1, x2, y2 = bbox
        import ezdxf as _ez
        tmp = _ez.new()
        tmp_msp = tmp.modelspace()
        
        # 字体修复也要应用到 tmp 文档
        import matplotlib as _mpl
        _mpl.rcParams['font.family'] = 'DejaVu Sans'
        _mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
        try:
            import os as _os
            from matplotlib.font_manager import findfont, FontProperties
            _dejavu = _os.path.basename(
                findfont(FontProperties(family='DejaVu Sans')))
            for _st in tmp.styles:
                try:
                    _st.dxf.font = _dejavu
                except Exception:
                    pass
        except Exception:
            pass
        
        n_in, n_all = 0, 0
        for e in msp:
            n_all += 1
            try:
                if e.dxf.layer in off_layers:
                    continue
            except Exception:
                pass
            if _entity_in_bbox(e, x1, y1, x2, y2, margin):
                try:
                    tmp_msp.add_foreign_entity(e)
                    n_in += 1
                except Exception:
                    pass
        print('prefilter: %d / %d entities kept' % (n_in, n_all), flush=True)
        draw_msp, draw_doc = tmp_msp, tmp
    else:
        draw_msp, draw_doc = msp, doc
    ctx = RenderContext(draw_doc)
    # 纯黑白渲染：便于后续二值化
    cfg = Configuration(color_policy=ColorPolicy.BLACK)
    backend = MatplotlibBackend(ax)
    # finalize=False：ezdxf 的 finalize 会 autoscale 并重置 figure 尺寸，
    # 破坏我们设定的 bbox 与分辨率；父类 finalize 本就是空操作，可安全关闭
    Frontend(ctx, backend, config=cfg).draw_layout(draw_msp, finalize=False)
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    fig.savefig(out_png, dpi=100, facecolor=bg_color)
    plt.close(fig)
    return w_px, h_px


if __name__ == '__main__':
    import sys, glob
    dxf = sys.argv[1] if len(sys.argv) > 1 else glob.glob(
        '/home/hatch/workspace/visual_tracer/incoming/v2/*.dxf')[0]
    out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/overview.png'
    # 全图低分辨率概览
    render_dxf(dxf, out, bbox=None, px_per_unit=0.15)
    print('saved', out)
