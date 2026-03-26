# MouseScope

<p align="center">
  <img src="static/img/logo.png" alt="MouseScope logo" width="112"/>
</p>

<p align="center">
  <strong>面向小鼠行为学视频分析的桌面软件平台</strong><br/>
  当前已支持 <strong>悬尾实验（TST）</strong>、<strong>旷场实验（OFT）</strong> 与 <strong>新奇物体识别（NOR）</strong>
</p>

<p align="center">
  A desktop platform for practical mouse behavioral video analysis, currently supporting
  <strong>TST</strong>, <strong>OFT</strong>, and <strong>NOR</strong>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Version-1.2-2563eb" alt="Version"/>
  <img src="https://img.shields.io/badge/Platform-Windows%2010%2F11-1d4ed8" alt="Platform"/>
  <img src="https://img.shields.io/badge/Python-3.9%2B-0f766e" alt="Python"/>
  <img src="https://img.shields.io/badge/UI-Flask%20Desktop-0369a1" alt="UI"/>
  <img src="https://img.shields.io/badge/Experiments-TST%20%7C%20OFT%20%7C%20NOR-f59e0b" alt="Experiments"/>
  <img src="https://img.shields.io/badge/Language-中文%20%7C%20English-7c3aed" alt="Language"/>
</p>

> 小鼠行为视频分析不应该被繁琐脚本和手工统计拖慢。MouseScope 希望把常见行为学实验逐步整合进一套真正能落地的软件工作流。

---

## 项目简介 | Overview

**MouseScope** 是一个面向实验室日常使用场景的小鼠行为学视频分析平台。  
它采用本地浏览器界面（local web UI）+ Python / OpenCV 分析后端的方式，帮助用户完成从视频导入、参数标定、批量分析到结果导出的完整流程，并以模块化方式持续扩展不同实验范式。

当前版本已经集成以下实验模块：

| 实验模块 Module | 主要用途 | 典型输出 |
|---|---|---|
| **悬尾实验 TST** | 多 ROI 小鼠不动 / 挣扎自动分析 | 标注视频、逐帧日志 CSV、不动性汇总 CSV、TXT 报告 |
| **旷场实验 OFT** | 多 arena 轨迹追踪、中心区 / 墙区统计、热图与轨迹图输出 | 标注视频、轨迹 CSV、summary CSV、bins CSV、热图、轨迹图、TXT 报告 |
| **新奇物体识别 NOR** | 多 ROI 场地 + 双物体探索偏好分析 | 标注视频、轨迹 CSV、summary CSV、bins CSV、热图、轨迹图、TXT 报告 |

---

## 核心亮点 | Highlights

- **一个软件支持多实验**：统一工作流下完成 TST、OFT 与 NOR 分析。
- **适合拼接视频 / 多笼位视频**：支持多 ROI / 多 arena 并行处理。
- **每个 ROI 可独立命名与设时窗**：更贴近真实实验里的多组、多笼位批处理。
- **OFT / NOR 支持场地四角标定**：便于透视校正与厘米坐标统计。
- **NOR 支持双物体标定与探索偏好指标**：输出 `DI`、偏好百分比、bouts、latency 等结果。
- **边缘畸变场景可单独绘制检测遮罩**：更适合广角镜头、黑边或暗角较明显的视频。
- **支持中英界面切换**：软件前端支持 Chinese / English。
- **可打包为 EXE**：适合直接分发给实验室成员使用。

---

## 展示与效果 | Demo

> 下面按实际使用顺序展示软件界面与典型输出：先看统一入口，再看 TST、OFT 与 NOR 的分析效果。

### 1. 软件主界面

<p align="center">
  <img src="docs/assets/cover.png" alt="MouseScope main interface" width="920"/>
  <br/>
  <sub>统一入口支持 TST / OFT / NOR，适合实验室日常分析流程</sub>
</p>

### 2. 悬尾实验（TST）

<p align="center">
  <img src="static/img/demo_result.gif" alt="TST result visualization" width="920"/>
  <br/>
  <sub>自动识别不动状态，并在视频中实时输出可视化分析结果</sub>
</p>

<table>
  <tr>
    <td align="center">
      <img src="static/img/validation_correlation_33_mice.png" alt="TST validation correlation" width="420"/>
      <br/>
      <sub>软件分析与人工评分一致性</sub>
    </td>
    <td align="center">
      <img src="static/img/validation_ethogram_33_mice.png" alt="TST validation ethogram" width="470"/>
      <br/>
      <sub>33 只小鼠逐段行为对照示意</sub>
    </td>
  </tr>
</table>

### 3. 旷场实验（OFT）

<p align="center">
  <img src="docs/assets/oft_demo.gif" alt="Open Field demo GIF" width="920"/>
  <br/>
  <sub>多 arena 旷场分析示例，支持轨迹统计、中心区分析与透视校正</sub>
</p>

<table>
  <tr>
    <td align="center">
      <img src="docs/assets/oft_heatmaps.png" alt="OFT heatmaps" width="430"/>
      <br/>
      <sub>旷场热图（heatmaps）</sub>
    </td>
    <td align="center">
      <img src="docs/assets/oft_trackmaps.png" alt="OFT track maps" width="430"/>
      <br/>
      <sub>透视校正后的轨迹图（track maps）</sub>
    </td>
  </tr>
</table>

### 4. 新奇物体识别（NOR）

<p align="center">
  <img src="docs/assets/nor_preview.png" alt="NOR annotated preview" width="920"/>
  <br/>
  <sub>多 ROI / 多 arena 新奇物体识别分析预览，叠加场地、物体、交互区和轨迹结果</sub>
</p>

<table>
  <tr>
    <td align="center">
      <img src="docs/assets/nor_heatmaps.png" alt="NOR heatmaps" width="430"/>
      <br/>
      <sub>NOR 热图（heatmaps）</sub>
    </td>
    <td align="center">
      <img src="docs/assets/nor_trackmaps.png" alt="NOR trajectory maps" width="430"/>
      <br/>
      <sub>NOR 轨迹图（track maps）</sub>
    </td>
  </tr>
</table>

---

## 快速开始 | Quick Start

### 方式一：直接运行打包版 EXE

1. 前往 [Releases](https://github.com/fujiacheng3/MouseScope/releases) 下载最新版本。
2. 解压压缩包。
3. 运行 `MouseScope.exe`。
4. 软件会自动在浏览器中打开 `http://localhost:8080`。

### 方式二：从源码运行

```bash
git clone https://github.com/fujiacheng3/MouseScope.git
cd MouseScope
pip install -r requirements.txt
python app.py
```

随后在浏览器中打开 `http://localhost:8080`。

---

## 典型工作流 | Workflow

### TST 工作流

1. 上传一个或多个悬尾实验视频。
2. 如有需要，先做体长标定。
3. 绘制一个或多个 ROI。
4. 为每个 ROI 命名，并设置统计时间窗。
5. 调整检测参数并开始分析。
6. 导出标注视频、逐帧日志、不动性统计表和 TXT 报告。

### OFT 工作流

1. 上传一个或多个旷场实验视频。
2. 选择实验类型为 `Open Field Test`。
3. 为每个 arena 点击四个角点，并输入场地边长。
4. 为每个 ROI / arena 命名，并按需设置精确统计时间。
5. 调整阈值、形态学参数、中心区比例、墙区边界、分箱时长等参数。
6. 导出轨迹 CSV、summary CSV、分箱统计、热图、轨迹图和 TXT 报告。

### NOR 工作流

1. 上传一个或多个 NOR 视频。
2. 选择实验类型为 `Novel Object Recognition`。
3. 为每个 ROI 依次标定 4 个 arena 角点和 2 个物体中心点。
4. 如有需要，单独绘制检测遮罩以适应广角畸变、黑边或暗角。
5. 设置阶段、物体角色、统计时间窗，并调整检测与统计参数。
6. 导出轨迹 CSV、summary CSV、分箱统计、热图、轨迹图和 TXT 报告。

---

## 输出文件 | Output Files

### TST 输出

| 文件名模式 | 含义 |
|---|---|
| `*_analyzed.mp4` | 标注后的视频结果 |
| `*_logs.csv` | 逐帧日志 |
| `*_immobility.csv` | 各 ROI 不动性统计汇总 |
| `*_report.txt` | 文本报告 |

### OFT 输出

| 文件名模式 | 含义 |
|---|---|
| `*_oft_analyzed.mp4` | 旷场标注视频 |
| `*_oft_tracks.csv` | 逐帧轨迹数据 |
| `*_oft_summary.csv` | 各 ROI 汇总指标 |
| `*_oft_bins.csv` | 按时间分箱的统计结果 |
| `*_oft_report.txt` | 文本报告 |
| `*_oft_heatmaps.png` | 热图结果 |
| `*_oft_trackmaps.png` | 轨迹图结果 |

### NOR 输出

| 文件名模式 | 含义 |
|---|---|
| `*_nor_analyzed.mp4` | NOR 标注视频 |
| `*_nor_track.csv` | 逐帧轨迹与探索状态 |
| `*_nor_summary.csv` | 各 ROI 汇总指标 |
| `*_nor_bins.csv` | 按时间分箱的统计结果 |
| `*_nor_report.txt` | 文本报告 |
| `*_nor_heatmaps.png` | 热图结果 |
| `*_nor_trackmaps.png` | 轨迹图结果 |

---

## 适用场景 | Suitable Use Cases

- 实验室内部日常视频分析
- 多笼位 / 多 arena 拼接视频处理
- TST 不动性自动统计
- OFT 路程、速度、中心区时间、墙区时间等指标统计
- NOR 新奇偏好、探索时间、bouts、latency 等指标统计
- 需要给同事直接发 EXE 使用的 Windows 环境

---

## 技术栈 | Tech Stack

| 层级 | 技术 |
|---|---|
| 后端 Backend | Python, Flask |
| 视觉分析 CV Pipeline | OpenCV、背景分离、轮廓检测、传统运动分析 |
| 数据处理 Data | NumPy, Pandas |
| 前端 Frontend | HTML, CSS, Vanilla JavaScript |
| 打包 Packaging | PyInstaller |

---

## 项目结构 | Repository Structure

```text
MouseScope/
|- app.py                      # Flask app and workflow orchestration
|- core/
|  |- vision_tracker.py        # TST 分析主流程
|  |- open_field.py            # OFT 轨迹分析、统计与绘图
|  |- novel_object.py          # NOR 轨迹分析、探索统计与绘图
|  \- performance_utils.py
|- templates/                  # 页面模板
|- static/                     # 静态资源
|- docs/assets/                # README 展示素材
|- data/results/               # 本地分析输出示例
\- MouseScope.spec             # PyInstaller 打包配置
```

---

## 运行环境 | Requirements

- Windows 10 / 11 x64
- Python 3.9 至 3.11（源码运行时）
- 建议 8 GB 及以上内存
- 建议使用多核 CPU 以提高批处理速度

---

## 使用说明与注意事项 | Notes

- 本软件主要作为 **科研辅助工具（research tool）** 使用。
- 分析质量会受到视频清晰度、光照、背景均匀性、相机角度和实验布置影响。
- 对于正式实验数据，建议在发表或关键结论使用前进行人工复核。
- OFT / NOR 当前为经典视觉方案，不包含姿态估计级别的细粒度行为识别。
- NOR 当前探索判定是基于“接近物体 + 低速条件”的传统启发式，而不是鼻尖级追踪。

---

## 引用 | Citation

如果 MouseScope 对你的研究有帮助，欢迎按以下格式引用：

```bibtex
@software{fu2026mousescope,
  author  = {Fu, Jiacheng},
  title   = {MouseScope: Automated Mouse Behavior Analysis Platform},
  year    = {2026},
  url     = {https://github.com/fujiacheng3/MouseScope},
  version = {1.2}
}
```

Plain text:

> Fu, J. (2026). *MouseScope: Automated mouse behavior analysis platform* (Version 1.2) [Software]. GitHub. https://github.com/fujiacheng3/MouseScope

---

## 许可证 | License

仅限学术与非商业用途。  
详见 [LICENSE](LICENSE)。

---

## 联系方式 | Contact

- 作者 Author: Jiacheng Fu
- 邮箱 Email: 2583348593@qq.com
- GitHub: https://github.com/fujiacheng3/MouseScope

---

<p align="center">
  Built for practical behavioral video analysis in real lab workflows.
</p>
