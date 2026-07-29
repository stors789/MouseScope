# DeepLabCut 姿态模型档案（第一版）

OFT 与 NOR 的第三阶段配置页现在把“追踪后端”和“背景建模方法”作为两个独立配置：

- `tracking_backend`: `classical`、`deeplabcut`、`auto`；NOR 额外支持 `hybrid`
- `bg_method`: 灰度阈值、静态/时序背景等经典视觉背景模型

选择 DeepLabCut 不会修改 `bg_method`。如果 DLC 依赖、模型或工作进程不可用，预览会返回明确的回退状态，经典视觉预览和分析仍可继续。

## 档案与不可变版本

姿态模型档案保存在应用数据目录的 `pose_model_profiles.json`，不是 ROI 模板的一部分。每个档案包含：

- 适用实验、拍摄环境元数据；
- `nose`、双耳、`body_center`、`tail_base` 等关键点 schema；
- 基础模型与权重引用；
- SHA-256 校验值及验证状态；
- 人工训练标注和固定验证标注；
- 不可变版本号、`version_id` 和 `parent_version`；
- 训练适配器状态。

创建新版本时，MouseScope 会复制父版本的全部历史标注和固定验证集，再加入本次纠错帧。旧版本不会被覆盖。第一版只登记 `awaiting_external_training`，不会将“已保存训练请求”冒充“训练成功”。自定义版本只有同时登记权重引用和 SHA-256 后才能标记为 `ready`。

## 配置页流程

1. 在姿态识别区域选择 `SuperAnimal-TopViewMouse（零样本）` 或已保存的模型版本。
2. 点击“零样本/模型预览”。
3. 外部 DLC 预览工作进程可用时，关键点会叠加在右侧结果图；可拖动点位纠正。
4. 先“另存为模型档案”，然后保存当前纠错帧（训练集或固定验证集）。
5. 点击“基于所选版本生成新版本”。新版本进入等待外部训练状态。

## 可选 DLC 工作进程

为避免 DeepLabCut、PyTorch 和模型权重进入 PyInstaller 主包，使用可选的隔离工作进程。仓库已提供真实的 SuperAnimal-TopViewMouse worker。首次安装：

```text
python tools/setup_dlc_worker.py
```

它会创建 `.venv-dlc`；重启后 MouseScope 自动发现该环境。也可使用已有环境：

```text
MOUSESCOPE_DLC_PYTHON=/path/to/dlc/environment/python
```

仍可通过 `MOUSESCOPE_DLC_WORKER` 指向自定义兼容 worker。仓库 worker 支持：

```text
python tools/mousescope_dlc_worker.py status
python tools/mousescope_dlc_worker.py preview /temporary/request.json
python tools/mousescope_dlc_worker.py serve
```

`serve` 使用逐行 JSON 协议并让官方 detector/pose runner 常驻内存。请求包含 `image_path`、`model_reference`、`keypoint_schema` 和 `max_individuals`；响应支持多动物：

```json
{
  "success": true,
  "individuals": [{
    "individual_id": "animal0",
    "keypoints": {
      "nose": {"x": 120.5, "y": 83.2, "likelihood": 0.97},
      "body_center": {"x": 94.0, "y": 108.1, "likelihood": 0.99}
    }
  }]
}
```

官方 TopViewMouse 的 `mouse_center` 映射为 MouseScope 的 `body_center`。其余规范映射是 `nose`、`left_ear`、`right_ear` 和 `tail_base`。

正式 OFT/NOR 分析也使用同一工作进程。每个采样帧只执行一次 DLC 推理，再根据 `body_center` 所在多边形把多个个体分配给各 ROI：

- OFT：DLC `body_center` 直接进入坐标变换、路程、速度和区域统计。
- NOR：DLC `body_center` 用于轨迹；`nose` 与双耳中点用于鼻尖距离、头向夹角和探索评分。
- NOR `hybrid`：经典视觉负责全程轨迹；经典轮廓进入“探索半径 + DLC
  触发余量”后，按小鼠外接框扩边裁切，只对小图运行 DLC。裁切关键点会还原到
  全帧坐标，DLC 只负责近物体姿态和探索判定。
- 某个 ROI 的身体中心、鼻尖或耳部低于置信度阈值时，仅该 ROI/帧回退经典视觉。
- 工作进程连续失败 3 次后，本视频剩余帧停止重复启动 DLC，统一回退经典视觉。

轨迹 CSV 每帧记录 `tracking_source`、`tracking_fallback_reason` 和身体中心置信度；汇总 CSV 同时记录请求后端、启动时解析出的后端、实际观察到的后端、DLC 帧数和回退帧数。因此不会把经典视觉结果误标为 DLC 结果。

### 性能与缓存

正式分析会按 `pose_batch_size` 进行批量推理。评分阶段的姿态结果保存在内存中，
标注视频直接复用，不会再次请求 DLC。

NOR `hybrid` 还会把近物体裁切结果持久化到应用数据目录相邻的
`dlc_pose_cache`。缓存身份包含视频内容采样指纹、模型/权重版本、关键点 schema、
ROI、采样、经典检测、触发距离和裁切扩边配置；帧号、ROI 与实际裁切框也必须一致。
因此重复分析相同输入时可跳过已有 DLC 推理，相关配置变化时会自动使用新的缓存
命名空间。运行时失败不写入持久缓存，以免临时故障污染后续分析。

官方当前推荐顶视小鼠使用 `superanimal_topviewmouse`、`hrnet_w32` 和 `fasterrcnn_resnet50_fpn_v2`；SuperAnimal 输出应由 sidecar 映射为 MouseScope 的规范名称 `nose`、`left_ear`、`right_ear`、`body_center`、`tail_base`。参考 [DeepLabCut Model Zoo](https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html) 与 [SuperAnimal 推理文档](https://deeplabcut.github.io/DeepLabCut/dev/latest-release/reference/deeplabcut/modelzoo/video_inference/)。
