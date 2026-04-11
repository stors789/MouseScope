# MouseScope

<p align="center">
  <img src="static/img/logo.png" alt="MouseScope logo" width="112"/>
</p>

<p align="center">
  <strong>面向真实实验流程的小鼠行为学视频分析桌面平台</strong><br/>
  <strong>Desktop platform for practical mouse behavioral video analysis in real lab workflows.</strong>
</p>

<p align="center">
  支持悬尾实验 TST、旷场实验 OFT、新奇物体识别 NOR、高架十字迷宫 EPM 与 Y 迷宫 Y-Maze<br/>
  Supports Tail Suspension Test (TST), Open Field Test (OFT), Novel Object Recognition (NOR), Elevated Plus Maze (EPM), and Y-Maze.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Version-v2.0.0-c96442" alt="Version"/>
  <img src="https://img.shields.io/badge/Platform-Windows%2010%2F11-7a5136" alt="Platform"/>
  <img src="https://img.shields.io/badge/UI-Desktop%20Workbench-f59e0b" alt="UI"/>
  <img src="https://img.shields.io/badge/Modules-5%20Assays-16a34a" alt="Modules"/>
  <img src="https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87%20%7C%20English-0f766e" alt="Language"/>
</p>

<p align="center">
  <a href="https://github.com/fujiacheng3/MouseScope/releases">
    <img src="https://img.shields.io/badge/%E4%B8%8B%E8%BD%BD%E6%89%93%E5%8C%85%E7%89%88-Download%20Package-c96442?style=for-the-badge" alt="Download Package"/>
  </a>
  <a href="#workflow-demo">
    <img src="https://img.shields.io/badge/%E5%B7%A5%E4%BD%9C%E6%B5%81%E6%BC%94%E7%A4%BA-Workflow%20Demo-2563eb?style=for-the-badge" alt="Workflow Demo"/>
  </a>
  <a href="docs/releases/v2.0.0.md">
    <img src="https://img.shields.io/badge/Release%20Notes-v2.0.0-7a5136?style=for-the-badge" alt="Release Notes"/>
  </a>
  <a href="docs/system_design.md">
    <img src="https://img.shields.io/badge/%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1-System%20Design-0f766e?style=for-the-badge" alt="System Design"/>
  </a>
</p>

<p align="center">
  <img src="static/img/ui_home.png" alt="MouseScope v2.0.0 home page" width="960"/>
  <br/>
  <sub>MouseScope v2.0.0 当前首页界面 · Current home interface</sub>
</p>

## MouseScope 是什么 | Overview

MouseScope 是一套面向实验室日常工作的桌面分析平台，用来把小鼠行为学视频从上传、标定、ROI 或迷宫区域设置、参数调整、批量检查、正式分析到结果复核串成一条统一流程。它尽量让不同实验模块共用一致的操作逻辑，方便直接交给实验人员使用。  

MouseScope is a desktop analysis platform built for everyday lab use. It turns mouse behavioral video processing into one unified workflow, from upload and calibration to ROI or maze setup, parameter tuning, batch review, final analysis, and result checking. The goal is to keep the workflow consistent enough to hand over directly to lab users.

## 平台特点 | Highlights

- 一个桌面工作台支持 `TST / OFT / NOR / EPM / Y-Maze` 五类实验。  
  One desktop workbench supports five common assays: `TST / OFT / NOR / EPM / Y-Maze`.
- 支持单视频分析，也支持按视频独立配置的批量工作流。  
  Supports both single-video analysis and per-video configuration in batch sessions.
- ROI、迷宫区域、目标物、时间窗和检测遮掩框均可视化设置。  
  ROI, maze zones, objects, time windows, and detection masks can all be configured visually.
- 输出核心行为学指标、结果视频、热图和轨迹图，便于复核与汇报。  
  Exports core behavioral metrics, annotated videos, heatmaps, and trackmaps for review and reporting.
- 提供中英双语界面、手册与 TXT 报告字段。  
  Provides bilingual Chinese/English UI text, manuals, and TXT report labels.

## 为什么用 MouseScope | Why MouseScope

MouseScope 更适合真实实验室场景，而不是只适合作者本人使用的脚本流程。它的重点不只是“能算出结果”，而是尽量减少手工整理、脚本切换和重复配置带来的时间成本，同时让结果更方便交接、复核和汇报。  

MouseScope is meant for real lab workflows rather than author-only scripts. The goal is not only to produce results, but also to reduce repetitive setup, manual file handling, and script switching, while making outputs easier to review, share, and hand over to other lab members.

<a id="workflow-demo"></a>

## 统一工作流演示 | Workflow Demo

<p align="center">
  <img src="docs/assets/workflow_v2.gif" alt="MouseScope workflow demo" width="960"/>
  <br/>
  <sub>统一流程：选择实验 → 上传视频 → 配置 ROI / 迷宫 → 预览与调参 → 分析与导出</sub><br/>
  <sub>Unified workflow: choose assay → upload video → configure ROI / maze → preview and tune → analyze and export</sub>
</p>

## 支持实验模块 | Supported Assays

| 实验模块 Assay | 典型用途 Typical Use | 核心设置 Core Setup | 主要指标 Main Metrics |
|---|---|---|---|
| `TST` 悬尾实验 / Tail Suspension Test | 不动与挣扎状态分析 / immobility and struggling analysis | 身长标定、ROI、时间窗 / body-length calibration, ROI, time windows | 不动时间、挣扎时间、不动比例、首次不动潜伏期 / immobility time, struggling time, immobility percent, latency |
| `OFT` 旷场实验 / Open Field Test | 自主活动与中心区偏好 / locomotion and center preference | 4 角场地、可选遮掩框、时间窗 / 4-corner arena, optional mask, time windows | 总路程、速度、中心区时间与比例、壁区时间与比例 / distance, speed, center metrics, wall metrics |
| `NOR` 新奇物体识别 / Novel Object Recognition | 新奇与熟悉物体探索差异 / novel vs familiar object exploration | 场地 4 角、2 个物体中心、可选遮掩框 / arena corners, 2 object centers, optional mask | 新奇探索时间、熟悉探索时间、辨别指数、偏好指数 / novel exploration, familiar exploration, discrimination index, preference index |
| `EPM` 高架十字迷宫 / Elevated Plus Maze | 开放臂与封闭臂探索 / open-arm vs closed-arm exploration | 中心区 + 2 开放臂 + 2 封闭臂手动画区 / manual center + 2 open arms + 2 closed arms | 开放臂时间、开放臂进入次数、开放臂进入占比、首次进入开放臂潜伏期 / open-arm time, open entries, open-entry ratio, first open-entry latency |
| `Y-Maze` Y 迷宫 / Y-Maze | 入臂序列与自发交替 / arm sequence and spontaneous alternation | 三角形中心区 + 3 条臂手动画区 / triangular center + 3 manual arm zones | 各臂停留、总入臂数、交替次数、交替率 / arm occupancy, total entries, alternation count, alternation rate |

## 结果示例 | Featured Results

### 1. 悬尾实验 TST | Tail Suspension Test

TST 模块用于记录每个 ROI 的不动与挣扎状态，并输出便于回看的结果视频与统计结果。  

The TST module records immobility and struggling states for each ROI and exports both review-friendly videos and summary metrics.

<p align="center">
  <img src="static/img/demo_result.gif" alt="TST annotated result" width="960"/>
  <br/>
  <sub>TST 结果演示 · TST result preview</sub>
</p>

<table>
  <tr>
    <td align="center">
      <img src="static/img/validation_correlation_33_mice.png" alt="TST validation correlation" width="420"/>
      <br/>
      <sub>软件评分与人工评分相关性 · Software vs manual correlation</sub>
    </td>
    <td align="center">
      <img src="static/img/validation_ethogram_33_mice.png" alt="TST validation ethogram" width="470"/>
      <br/>
      <sub>逐时程对照示例 · Ethogram-style comparison</sub>
    </td>
  </tr>
</table>

### 2. 旷场实验 OFT | Open Field Test

OFT 模块关注自主活动、中心区偏好与空间占据情况，适合快速获得常用的运动学和空间统计指标。  

The OFT module focuses on locomotion, center preference, and spatial occupancy, making it useful for routine kinematic and arena-based summaries.

<p align="center">
  <img src="docs/assets/oft_demo.gif" alt="Open Field demo" width="960"/>
  <br/>
  <sub>OFT 演示 · OFT demo</sub>
</p>

<table>
  <tr>
    <td align="center">
      <img src="docs/assets/oft_heatmaps.png" alt="Open field heatmaps" width="430"/>
      <br/>
      <sub>旷场热图 · Open-field heatmaps</sub>
    </td>
    <td align="center">
      <img src="docs/assets/oft_trackmaps.png" alt="Open field trackmaps" width="430"/>
      <br/>
      <sub>旷场轨迹图 · Open-field trackmaps</sub>
    </td>
  </tr>
</table>

### 3. 新奇物体识别 NOR | Novel Object Recognition

NOR 模块用于量化小鼠对熟悉物体和新奇物体的探索差异，支持探索时间、偏好与辨别指标输出。  

The NOR module quantifies exploration differences between familiar and novel objects, including exploration time, preference, and discrimination metrics.

<table>
  <tr>
    <td align="center">
      <img src="docs/assets/nor_heatmaps.png" alt="NOR heatmaps" width="430"/>
      <br/>
      <sub>NOR 热图 · NOR heatmaps</sub>
    </td>
    <td align="center">
      <img src="docs/assets/nor_trackmaps.png" alt="NOR trackmaps" width="430"/>
      <br/>
      <sub>NOR 轨迹图 · NOR trackmaps</sub>
    </td>
  </tr>
</table>

### 4. 高架十字迷宫 EPM | Elevated Plus Maze

EPM 模块用于量化开放臂、封闭臂与中心区的停留和进入情况，适合观察开放空间探索偏好与相关焦虑样行为指标。  

The EPM module quantifies occupancy and entries across open arms, closed arms, and the center, making it useful for open-space exploration and related anxiety-like measures.

<p align="center">
  <img src="docs/assets/epm_trackmap.png" alt="EPM trackmap" width="700"/>
  <br/>
  <sub>高架十字迷宫 EPM 轨迹图 · EPM trackmap</sub>
</p>

### 5. Y 迷宫 Y-Maze

Y 迷宫模块用于量化各臂停留、入臂序列和自发交替情况，适合快速查看探索模式与交替率。  

The Y-Maze module quantifies arm occupancy, entry sequence, and spontaneous alternation, making it useful for quick inspection of exploration patterns and alternation rate.

<p align="center">
  <img src="docs/assets/ymaze_trackmap.png" alt="Y-Maze trackmap" width="700"/>
  <br/>
  <sub>Y 迷宫 Y-Maze 轨迹图 · Y-Maze trackmap</sub>
</p>

## 可以得到什么结果 | Outputs

MouseScope 的输出重点不是文件名本身，而是对实验更有意义的行为学指标与复核材料。  

MouseScope is designed to output meaningful behavioral metrics and review materials, rather than just a collection of raw files.

- 核心行为学指标：如不动时间、中心区时间、开放臂进入比例、交替率、辨别指数等。  
  Core behavioral metrics: such as immobility time, center time, open-entry ratio, alternation rate, and discrimination index.
- 逐帧与分箱统计：便于查看实验过程中行为随时间的变化。  
  Framewise and binned summaries: useful for tracking how behavior changes over time.
- 标注结果视频：适合回看 ROI、区域切换和事件判定。  
  Annotated result videos: useful for checking ROI logic, zone switching, and event calls.
- 空间可视化结果：包括热图和轨迹图，便于展示空间占据与探索路径。  
  Spatial visualizations: including heatmaps and trackmaps for occupancy and exploration patterns.

## 统一使用路径 | Unified Workflow

1. 选择实验模块并上传一个或多个视频。  
   Choose an assay and upload one or more videos.
2. 根据实验类型完成场地、ROI、迷宫分区或目标物设置。  
   Configure the arena, ROI, maze zones, or objects depending on the assay.
3. 设置时间窗、检测遮掩框与关键参数，并用预览检查是否稳定。  
   Set time windows, detection masks, and key parameters, then validate them with preview.
4. 多视频时先进入批量工作台，为每个视频单独保存配置。  
   For multi-video sessions, enter the batch workspace and save settings separately for each video.
5. 统一启动分析。  
   Start analysis after all required setups are complete.
6. 查看结果视频、统计指标和空间可视化结果。  
   Review result videos, behavioral metrics, and spatial visualizations.

## 快速开始 | Quick Start

### 方式一：直接下载打包版本 | Option 1: Download the packaged build

1. 打开 [Releases](https://github.com/fujiacheng3/MouseScope/releases)。  
   Open the [Releases](https://github.com/fujiacheng3/MouseScope/releases) page.
2. 下载最新压缩包并解压。  
   Download the latest archive and extract it.
3. 运行 `MouseScope.exe`。  
   Run `MouseScope.exe`.
4. 软件会在本机浏览器中打开 `http://localhost:8080`。  
   The app will open `http://localhost:8080` in your local browser.

大多数实验室用户建议直接使用打包版本。  
For most lab users, the packaged desktop build is the recommended option.

### 方式二：从源码运行 | Option 2: Run from source

```bash
git clone https://github.com/fujiacheng3/MouseScope.git
cd MouseScope
pip install -r requirements.txt
python app.py
```

## 适用场景 | Typical Use Cases

- 日常神经科学与行为学实验视频后处理。  
  Routine post-processing for neuroscience and behavioral videos.
- 多视频、多 arena 或多 ROI 的批量分析。  
  Batch analysis across multiple videos, arenas, or ROIs.
- 需要统一配置、减少人工整理错误的实验室工作流。  
  Lab workflows that need more standardized setup and fewer manual mistakes.
- 需要快速生成组会汇报或论文前检查用结果图的场景。  
  Scenarios that require quick figures for lab meetings or pre-publication checks.
- 希望把分析流程交接给非程序背景同学的场景。  
  Cases where the workflow should be handed over to non-programmer lab members.

## 局限与说明 | Limitations and Notes

- MouseScope 是研究辅助工具，正式统计分析前仍建议抽查原始视频与结果视频。  
  MouseScope is a research-assistance tool, and raw videos should still be reviewed before formal statistical interpretation.
- 跟踪效果会受到光照、背景、阴影、反光、视角稳定性和视频质量影响。  
  Tracking quality depends on lighting, background, shadows, glare, camera stability, and video quality.
- 当前 OFT / NOR 仍以经典计算机视觉方法为主，并非完整的姿态估计级别行为解析。  
  Current OFT / NOR analysis is still mainly based on classical computer vision rather than full pose-estimation-grade behavior parsing.
- NOR 的探索判定包含头部方向启发式逻辑，不能完全替代人工专业判断。  
  NOR exploration scoring includes heuristic head-direction logic and should not fully replace expert review.

## 仓库结构 | Repository Structure

```text
MouseScope/
|- app.py
|- core/
|  |- vision_tracker.py
|  |- open_field.py
|  |- novel_object.py
|  |- elevated_plus.py
|  |- y_maze.py
|  |- maze_zone_utils.py
|  \- spatial_render.py
|- services/
|  |- experiment_registry.py
|  |- analysis_builders.py
|  \- session_store.py
|- templates/
|- static/
|- tools/
|  |- smoke_routes_v2.py
|  \- smoke_short_analyses_v2.py
|- docs/
|  |- releases/
|  \- system_design.md
\- MouseScope.spec
```

## 引用 | Citation

如果 MouseScope 对你的工作有帮助，欢迎按平台版本引用。  
If MouseScope helps your work, please cite the platform version below.

```bibtex
@software{fu2026mousescope,
  author  = {Fu, Jiacheng},
  title   = {MouseScope: Mouse Behavioral Video Analysis Platform},
  year    = {2026},
  url     = {https://github.com/fujiacheng3/MouseScope},
  version = {2.0.0}
}
```

Plain text:

> Fu, J. (2026). *MouseScope: Mouse behavioral video analysis platform* (Version 2.0.0) [Software]. GitHub. https://github.com/fujiacheng3/MouseScope

## 许可与联系 | License & Contact

- License: `GPL-3.0-or-later`
- Author: Jiacheng Fu
- Email: 2583348593@qq.com
- GitHub: https://github.com/fujiacheng3/MouseScope

<p align="center">
  为真实实验流程而设计，而不是为了展示脚本本身。<br/>
  Built for real lab workflows rather than for showcasing scripts alone.
</p>
