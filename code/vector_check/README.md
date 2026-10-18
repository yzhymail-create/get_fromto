# vector_check — 矢量校验工具（框#52）

## 定位（必读）

**纯校验工具，用户明确授权的例外。** 生产管线（skill `local-visual-tracer`）
保持纯视觉、输入格式无关；本目录只利用"这次输入恰好是 DXF"做交叉验证，
**代码不许混入视觉管线**，也不许被生产管线调用。

## 输入

- DXF：`~/workspace/visual_tracer/incoming/v2/` 下 165MB 文件
- 框#52 bbox：`~/workspace/visual_tracer/qa/frames/subframes.json`（id=52）

## 文件

| 文件 | 说明 |
|---|---|
| `explore_frame52.py` | 阶段A：框内图元普查，落盘 `frame52_entities.pkl`（LINE/CIRCLE/ARC/LWPOLYLINE/TEXT 的几何+文字，纯 tuple） |
| `vector_check.py` | 阶段B：由 pickle 生成 From-To（不再读 DXF，秒级迭代） |
| `frame52_entities.pkl` | 框#52 图元缓存 |
| `requirements-vector.txt` | pip 依赖（Windows wheel） |

## 方法（矢量）

1. 导线网络：图层 `0` 的 LINE 按端点重合连成网络（union-find，容差 `END_TOL=2.0` 图纸单位）；非圆环装饰的 ARC 也纳入
2. 端子圆环：`CIRCLE r∈[0.5,0.8]`（本框 37 个，每个另有一 r=1.25 同心 ARC 装饰）；r=0.31 的 9 个是实心接点，不算端子
3. 器件框：闭合 4 点 LWPOLYLINE + 面积>500（本框 6 个，与视觉版 6/6 一致）
4. TEXT（排除标题栏层 `1FILAMENT`）：
   - 管脚号 `BD176-21` → 30 单位内最近圆环
   - 线号 `IPB021_01...` → 25 单位内最近导线段
   - 器件名/代号：框附近 40 单位内字高≥4.0 的文字为名；代号先与最近的名称配对再归属框（代号常夹在两框之间，直接按离框最近会配错）
   - 管脚功能：与圆环同行（|dy|≤6）的框内文字中取 2D 最近（圆环位于 pin 标注与功能文字之间；行带排除 Ag 屏蔽标记等跨行干扰）
5. 每个"两端都有圆环"的网络出一行 From-To；单圆环网络记"框边界端口"；
   管脚代号前缀与器件代号不一致（如 BD228 vs BD221）记备注待人工确认

## 输出

`~/workspace/visual_tracer/qa/frames/pilot_52/FromTo_矢量版.xlsx`
列：序号/起始用电器/起始管脚/管脚功能/目标用电器/目标管脚/管脚功能/导线线号/备注

## 已知弱点（矢量版不是完美真值）

1. 端点重合容差是启发式：制图缝隙 >2 单位会导致网络断裂（本框 37/37 圆环全部挂上，未见断裂）
2. TEXT 邻近关联是启发式：密集区可能误配；中英双语功能标注（如 `左后轮速传感器信号`/`RL Speed Sensor Signal`）按图面如实记录，不强行统一
3. 假设"一根导线两端各一个圆环"；多圆环网络按星形配对并备注
4. 器件框检测依赖"闭合矩形 LWPOLYLINE"画法，换框/换图需验证
5. 实心圆点：矢量只找到 9 个（r=0.31），视觉版报 67 个——**两版差异项，需人工仲裁**（视觉可能多计或矢量漏检）
6. 跨框导线只记单端端口，跨框匹配是 Phase 3 的事
7. 仅针对框#52 调参，未验证其他框

## 运行

```bat
:: Windows
pip install -r requirements-vector.txt
python explore_frame52.py   :: 约4分钟（读165MB DXF，可选；pkl 已备好可跳过）
python vector_check.py      :: 秒级，输出 FromTo_矢量版.xlsx
```
