# 巴恩斯迷宫与 DLC

在视频导入页选择「巴恩斯迷宫 Barnes Maze」，可通过快速分析或项目工作台进入。
本模块使用完整分辨率和源视频帧，不启用快速评分的时空降采样。
当前实现口径为 `barnes-1`；运行快照保存解析后的参数和模型版本。

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

- **关闭**：背景差分、形态学和轮廓身体/细尾分离；鼻尖为尾根对侧的保守候选。没有足够尾部依据时只输出身体，不用移动方向猜鼻尖。
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
python tools/validate_landmarks.py predict --manifest "D:/验证/oft/manifest.json" --backend deeplabcut --confidence 0.8 --output "D:/验证/oft/dlc.json"
python tools/validate_landmarks.py evaluate --labels "D:/验证/oft/labels.json" --predictions "D:/验证/oft/classical.json" --tolerance-px 5 --output "D:/验证/oft/经典评估"
python tools/validate_landmarks.py evaluate --labels "D:/验证/oft/labels.json" --predictions "D:/验证/oft/dlc.json" --tolerance-px 5 --output "D:/验证/oft/DLC评估"
```

用 `predict --run-snapshot 路径/run_snapshot.json` 可复用某次运行解析的 DLC 模型引用和关键点 schema。
请给出预先确定的容差。评估按关键点输出人工可见点数、检出覆盖率、容差内命中率、匹配点中位误差和第90百分位误差。
容差内命中率以所有人工可见点为分母，预测缺失算未命中；误差只在配对点上计算，需和覆盖率一起看。
标注页的“human”标签只是来源记录，不能替代独立、盲法和标注者一致性要求。

## 开发验收

```powershell
python -m pytest tests/test_barnes_maze.py tests/test_barnes_runtime.py tests/test_landmark_validation.py -q
python tools/e2e_barnes.py --url http://127.0.0.1:8093 --output "D:/验证/browser"
```

浏览器脚本需要 Playwright 和本机 Edge。对独立 `MOUSESCOPE_DATA_DIR` 测试服务运行，脚本使用生成视频，不读取真实实验。
它覆盖标定、预览、保存/恢复、DLC 开关、几何模板、逐视频目标洞、批处理、标注视频和 ZIP 下载。
