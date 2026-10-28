# 巴恩斯迷宫与 DLC

在视频导入页选择「巴恩斯迷宫 Barnes Maze」，可通过快速分析或项目工作台进入。
本模块使用完整分辨率和源视频帧，不启用快速评分的时空降采样。
当前实现口径为 `barnes-1`；运行快照保存解析后的参数和模型版本。

## 经典追踪的正式实现

经典分割和几何估点统一位于 `core/classical_mouse.py` 的 `ClassicalMouseDetector`。
正式巴恩斯分析、单帧预览、关键点验证和对照视频均调用该实现；实验视频目录里的旧脚本不再作为项目方法入口。
计分规则版本仍为 `barnes-1`，新版经典追踪方法 ID 为 `local_otsu_v1`，两者分别记录。

处理顺序为：时序背景差分 → 按鼠体明暗方向保留前景 → 3×3开运算清理 → 候选区域局部Otsu抑制较浅阴影 → 3×3闭运算。
由此得到的同帧分割证据用于估计鼻尖、尾根；在它的副本上做身体腐蚀，得到身体范围和质心。
默认仅身体使用3×3椭圆核腐蚀1次，经典质量阈值与DLC筛选阈值均为0.4。

**身体范围与头尾证据用途不同。** 身体可以收紧，鼻尖和尾根不会因这一步腐蚀而丢掉输入依据；鼻尖仍落在额外腐蚀前的可见轮廓上。
灰色轮廓为估点证据，青色轮廓为身体范围，N/B/T分别表示鼻尖、身体中心、尾根。
若身体被腐蚀到不满足面积条件，则明确返回缺失，不退回旧轮廓伪装检出。
尾部弯曲、遮挡或缺乏足够形状依据时，鼻尖和尾根仍会缺失；不会借用DLC、上一帧或插值补点。

局部Otsu仅按亮度区分前景，不能保证区分灰度相近的阴影。几何质量分数不是正确概率；测试通过与检出覆盖率都不等于已取得定位准确率。

界面中的“仅身体腐蚀核/次数”只控制身体范围，“清理核”控制前景开闭运算。
旧的腐蚀/膨胀参数仅在 `legacy` 方法中使用。已保存的旧运行快照未记录方法ID时按旧形态学流程复现；新分析明确保存方法ID、版本、参数和头尾证据来源。

## 操作

1. 在分析配置页确认画面方向、视频时间窗和帧率；仅在已知实际帧率时覆盖。
2. 沿圆盘边界标至少 5 点，或拖出椭圆。填写真实圆盘直径、全部洞数和洞 ROI 半径。
3. 逐个点击洞中心，核对洞号，指定当前试次的目标洞；多个圆盘分别配置。
4. 如有边缘遮挡，可画检测遮罩；遮罩只包含本只小鼠，但可以延伸至圆盘边缘外。
5. 选择训练、反转或探测，填写天次、试次和小鼠绑定。逐视频检查标定和关键时刻的鼻尖预览。
6. 保存配置后可刷新恢复，或返回分析配置页批量开始。

设备模板只保存圆盘、洞位和物理尺寸。应用时保留各视频的小鼠、目标洞、时间窗、人工事件和试次。
新视频的目标洞需要单独确认。椭圆仿射可以修正近似椭圆投影，不等同于完整相机标定；严重斜视及不可见洞口仍需人工复核。

## DLC 开关

- **关闭**：使用上述统一经典方法；鼻尖为尾根对侧的保守候选。没有足够尾部依据时只输出身体，不用移动方向猜鼻尖。
- **开启**：复用现有独立 DLC worker，选择模型档案与不可变版本，或使用 SuperAnimal-TopViewMouse 预训练模型。DLC 与经典背景参数分开。
- 开启但模型/worker不可用会阻止启动。推理错误停止分析；低置信度、画外和缺失关键点保持缺失，不会静默换回经典视觉。
- 预训练模型并非对任意拍摄条件都准确。需用与训练/调参视频独立的画面验证鼻尖、身体和尾根。

## 计分口径

自动事件为「鼻尖进入洞 ROI 并达到确认时长」，记录进入起点及确认帧时间。
它是二维可见鼻尖的操作定义，不能确认隐蔽的头部下探动作。
观察到撤出达到重置时长后，再进入同一洞会再次计次；丢失鼻尖不会让重复计数重新开始。
**不使用 Y 迷宫的连续同臂合并。**

| 指标 | 定义 |
|---|---|
| primary latency / errors | 分析起点至首次目标洞访问；之前的所有非目标洞访问次数 |
| primary path length | 首次目标洞事件起点前观察到的身体路径 |
| total latency / errors | 至人工确认入箱；此前的全部非目标洞访问次数 |
| target sector | 以目标洞方向为中线、左右各 45° 的身体占用；比例以有效身体观察时间为分母 |
| nose / body observed percent | 分析窗内有效关键点观察时间占比，不能作为定位准确率 |

目标洞附近消失仅产生入箱待复核候选，重新出现会撤回候选。确认入箱时间必须手动输入源视频的绝对秒数。
未找到目标洞或未确认入箱时，相应潜伏期为空并标记删失；不把超时记作成功，也不把缺失记作零潜伏期。
探测试次无需入箱。人工洞序列开启后替代自动事件；勾选但空列表明确表示人工零事件。
复核与原始自动分析建议分别运行，两次输出及快照均保留。

轨迹不跨缺失帧连接，跳点被排除。路径长度只表示观察到的路径，缺失会低估真实路程。
搜索模式仅给出基于圆周邻接比例的探索性候选，不是经独立验证的分类器。

结果包包含轨迹、事件、逐洞、汇总、分箱 CSV，轨迹图、热图、报告、运行快照与可选标注视频。
汇总携带试次、小鼠/组别、后端与模型版本，事件注明自动/人工来源。
原始视频不会被修改。

方法终点参考 [Pitts, 2018, Bio-protocol e2744](https://bio-protocol.org/pdf/Bio-protocol2744.pdf)，区分首次定位和最终入箱。
报告实验时应另写明自动鼻尖 ROI、确认/重置阈值及人工复核办法。

## 用旧旷场视频验证定位

下面在项目根目录运行，使用能够运行 MouseScope 的 Python。
每个裁剪框只包含一只鼠；示例框需要按实际视频修改。

```powershell
python tools/validate_landmarks.py prepare --video "D:/实验/旷场.mpg" --output "D:/验证/oft" --samples 24 --roi Left:0:0:384:288 --roi Right:384:0:384:288
```

先打开输出目录的 `标注.html`，在未查看预测的条件下逐帧标注鼻尖、双耳、身体中心和尾根，看不到的点标不可见。
可以保存阶段结果并导入继续。保存得到 `labels.json`。
身体中心是操作性关键点，不同算法的轮廓质心与训练集定义可能不同，应在验证前统一人工标注定义。
零样本推理前进行的人工标注也不能代表对巴恩斯事件的验证。

```powershell
python tools/validate_landmarks.py predict --manifest "D:/验证/oft/manifest.json" --backend classical --output "D:/验证/oft/classical.json"
python tools/validate_landmarks.py predict --manifest "D:/验证/oft/manifest.json" --backend deeplabcut --confidence 0.4 --output "D:/验证/oft/dlc.json"
python tools/validate_landmarks.py evaluate --labels "D:/验证/oft/labels.json" --predictions "D:/验证/oft/classical.json" --tolerance-px 5 --output "D:/验证/oft/经典评估"
python tools/validate_landmarks.py evaluate --labels "D:/验证/oft/labels.json" --predictions "D:/验证/oft/dlc.json" --tolerance-px 5 --output "D:/验证/oft/DLC评估"
```

用 `predict --run-snapshot 路径/run_snapshot.json` 可复用某次运行解析的 DLC 模型引用和关键点 schema。
经典预测也从同一个运行快照读取实际分割方法和参数；批量快照用 `--video-index` 选择对应视频。
未提供快照时使用项目当前工厂默认配置。`--body-erosion-kernel` 和 `--body-erosion-iter` 可显式改变身体范围；它们不修改头尾证据。
请给出预先确定的容差。评估按关键点输出人工可见点数、检出覆盖率、容差内命中率、匹配点中位误差和第90百分位误差。
容差内命中率以所有人工可见点为分母，预测缺失算未命中；误差只在配对点上计算，需和覆盖率一起看。
标注页的“human”标签只是来源记录，不能替代独立、盲法和标注者一致性要求。

## 从项目方法直接生成对照视频

```powershell
python tools/review_mouse_tracking.py --manifest "D:/验证/oft/manifest.json" --segment 60:70 --segment 300:310 --segment 540:550 --confidence 0.4 --output "D:/验证/正式方法复核"
```

这是正式入口，不在视频脚本里重写分割。它输出 `review.mp4`、逐帧坐标 `points.csv`、完整配置 `review_config.json`、方法版本与覆盖记录 `summary.json`。
没有未标点的原图面板；每个连续片段使用固定窗口与倍率，不对坐标平滑或补点。
若已有逐帧DLC预测，可加 `--compare-predictions 路径/raw_predictions.jsonl --prediction-metadata 路径/summary.json`，只比较共同的N/B/T三点。
保存的DLC预测必须与源视频、源帧和ROI名称匹配，缺少预测输入会报错，不能当作模型漏检。
`--viewports` 可复用已保存的窗口JSON；所有对照列共用相同的固定映射。
两边的青/灰轮廓都来自统一经典分割，DLC列只替换标点，不表示DLC输出了分割掩膜。

## 开发验收

```powershell
python -m pytest tests/test_classical_mouse.py tests/test_classical_review.py tests/test_barnes_maze.py tests/test_barnes_runtime.py tests/test_landmark_validation.py -q
python tools/e2e_barnes.py --url http://127.0.0.1:8093 --output "D:/验证/browser"
```

浏览器脚本需要 Playwright 和本机 Edge。对独立 `MOUSESCOPE_DATA_DIR` 测试服务运行，脚本使用生成视频，不读取真实实验。
它覆盖标定、预览、保存/恢复、DLC 开关、几何模板、逐视频目标洞、批处理、标注视频和 ZIP 下载。
