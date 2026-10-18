# 纯视觉导线追踪器（visual_tracer）

对 DXF 电路图做**纯视觉**导线追踪：只看渲染后的像素，绝不解析 DXF 图元做追踪。
（用户明确要求；之前解析图元的方案已被否决并关停。）

## 管线

```
DXF --render.py--> PNG --trace.py--> 导线网络 --ocr.py--> 文字绑定 --> From-To
     (唯一接触DXF)   (纯像素)          (纯像素)
```

| 模块 | 输入 | 说明 |
|------|------|------|
| `render.py` | DXF 路径 | ezdxf matplotlib 后端光栅化；可开关图层、按 bbox 裁剪、空间预过滤加速。**这是唯一允许接触 DXF 的模块** |
| `trace.py` | numpy 灰度图 | Otsu 二值化 → Zhang-Suen 细化 → 骨架连通域 → 节点圆点判定 → 电气网络 |
| `ocr.py` | numpy 灰度图 | rapidocr_onnxruntime 中文识别：分块端到端识别（`recognize_tiled`）/ 文字块检测（`find_text_regions`）/ 批量纯识别（`recognize_text_blocks`） |
| `build_fromto.py` | 试点目录 | From-To 表生成：管脚标注锚定导线端点 → 导线线号文本归组 → 器件/管脚功能关联 → Excel |
| `tracer.py` | — | 编排器：`python3 tracer.py <dxf> --bbox x1,y1,x2,y2 --dpi-px 8 --out dir` |
| `pilot.py` | — | 试点脚本（渲染+追踪+叠加图+统计） |
| `gridscan.py` | — | 渲染区域选择：在网格上统计图元密度，选最密格（纯属"看哪里"，不属于追踪） |

## Windows 安装运行（交付）

本管线最终交付在 **Windows** 上运行，Linux 仅做开发验证。

1. 安装 Python 3.10~3.12 64 位（[python.org](https://www.python.org/downloads/)，
   安装时勾选 "Add python.exe to PATH"）
2. 打开 CMD，安装依赖（全部为含 Windows wheel 的纯 pip 包，无系统依赖）：
   ```
   pip install -r requirements.txt
   ```
3. 运行（示例：框#52 出表）：
   ```
   python build_fromto.py
   ```
   输出 `qa/frames/pilot_52/FromTo_机器版.xlsx`。

说明：代码中路径一律用 `os.path`，无 shell 特有调用，可直接在 Windows 下运行。

## 核心算法（trace.py）

1. **二值化**：Otsu 自动阈值，黑线白底
2. **细化**：Zhang-Suen 把导线变成单像素骨架
3. **导线段提取**：去掉节点像素（8邻域≥3）后连通域标记，每段为一条 arc
4. **节点判定**：在原二值图上用距离变换检测节点处有无**实心圆点**
   - 有圆点 → 拼接连接，合并相邻 arc
   - 无圆点 → 跨线不连，各 arc 独立
5. **网络组装**：union-find 合并，输出每个网络的像素、端点（8邻域=1）

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--dpi-px` | 8.0 | 每图纸单位像素数。导线宽约 0.24 单位→约 1.9px；建议≥8 |
| `min_pixels` | 50 | 小于此像素数的连通域视为文字/噪点过滤 |
| 圆点判定 | — | 节点邻域距离变换最大值 > 线宽×0.9 判为实心圆点 |
| 文字绑定距离 | 60px | 文字块中心到导线 bbox 距离阈值 |

## 已知局限（诚实记录）

1. **空心圆环连接点不合并**：本图用空心圆环表示接线点，追踪器只合并实心圆点处。
   对导线级 From-To 这是可接受的（每根导线保持独立网络），但总线-导线归属需后处理。
2. **文字/符号干扰**：大号文字、箭头、符号轮廓会形成闭环网络（0端点），靠尺寸过滤，
   残留需结合 OCR 文字块掩膜剔除（待 OCR 引擎可用后实现）。
3. **线宽太细**：8px/单位下导线仅约 1.9px，骨架易断裂；更高分辨率更稳但更慢。
4. **OCR 引擎**：已解决。`rapidocr_onnxruntime`（纯 pip，有 Windows wheel），
   框#52 上 162 个文本检出，20 抽样内容准确率 90%（2 个检测框偏小）。
   已知：①识别常丢空格（"FL Speed Sensor Power"→"FLSpeedSensorPower"），比对时归一化；
   ②下划线偶被认成空格（"IPB07_01"→"IPB07 01"），线号匹配已宽容处理；
   ③端子圆环符号为复合形（小圆+外弧，非标准圆），Hough/轮廓检测均不可靠，
   关联改用"管脚标注锚定导线端点"等效实现（见 `build_fromto.py` 说明）。
5. **跨块连续**：试点为单区域；全图需分块+重叠+跨块端点拼接，未实现。
6. **环境**：虚拟机会重置 `/usr/local`，pip 包改装到 `~/workspace/.pylibs`，
   使用时需 `export PYTHONPATH=/home/hatch/workspace/.pylibs`。
7. **ezdxf 后端坑**：`Frontend.draw_layout(finalize=True)` 会 autoscale 并把 figure 重置为 4.8 英寸，
   必须用 `finalize=False` 并手动重设 xlim/ylim（父类 finalize 是空操作，可安全关闭）。

## 运行

```bash
export PYTHONPATH=/home/hatch/workspace/.pylibs
python3 tracer.py /path/to.dxf --bbox 109000,-200,109400,200 --dpi-px 8 --out qa/run1
# 输出：render.png / overlay.png / from_to.json / run.log
```
