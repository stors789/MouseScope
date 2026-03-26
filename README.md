# MouseScope

<p align="center">
  <img src="docs/assets/cover.png" alt="MouseScope cover" width="100%"/>
</p>

<p align="center">
  <img src="static/img/logo.png" alt="MouseScope Logo" width="120"/>
</p>

<p align="center">
  <strong>Desktop software for automated mouse behavior analysis</strong><br/>
  Supports <strong>Tail Suspension Test (TST)</strong> and <strong>Open Field Test (OFT)</strong>
</p>

<p align="center">
  面向小鼠行为学视频分析的桌面软件，当前支持 <strong>悬尾实验（TST）</strong> 与 <strong>旷场实验（OFT）</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey.svg" alt="Platform"/>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/interface-Flask%20Desktop-2563eb.svg" alt="Interface"/>
  <img src="https://img.shields.io/badge/experiments-TST%20%7C%20OFT-0f766e.svg" alt="Experiments"/>
  <img src="https://img.shields.io/github/stars/fujiacheng3/MouseScope?style=social" alt="GitHub stars"/>
</p>

---

## Overview

**MouseScope** is a Windows-first desktop application for automated mouse behavior analysis from video.
It provides a browser-based local UI, classical computer-vision pipelines, batch processing, and exportable reports for common behavioral assays.

**MouseScope** 是一个面向 Windows 的小鼠行为学视频分析桌面软件，采用本地 Web 界面，结合传统计算机视觉流程，支持批处理、结果导出与实验工作流管理。

### Current experiment modules

| Module | What it does | Key outputs |
|---|---|---|
| **Tail Suspension Test (TST)** | Multi-ROI hanging-mouse analysis with immobility / struggle scoring | Annotated video, frame log CSV, immobility summary CSV, TXT report |
| **Open Field Test (OFT)** | Multi-arena tracking with corner calibration, perspective normalization, center / wall statistics | Annotated video, track CSV, summary CSV, bins CSV, TXT report, heatmaps, track maps |

---

## Highlights

- **One desktop app, multiple assays**: TST and OFT now live in the same workflow.
- **No coding required**: upload files or paste local Windows paths directly in the UI.
- **Multi-ROI / multi-arena ready**: suitable for stitched recordings containing multiple cages or arenas.
- **Per-ROI naming and time windows**: each ROI can be named and assigned its own statistics window.
- **Perspective-normalized OFT maps**: OFT track maps and heatmaps are rectified to standard square coordinates.
- **Batch-friendly**: supports multi-video analysis with shared calibration where appropriate.
- **Bilingual UI**: Chinese / English interface switching.
- **Windows packaging**: can be distributed as a standalone EXE via PyInstaller.

---

## Demo

### Open Field workflow preview

<p align="center">
  <img src="docs/assets/oft_demo.gif" alt="Open Field demo GIF" width="900"/>
  <br/>
  <em>Short GIF excerpt from an analyzed OFT video</em>
</p>

### Open Field result maps

<table>
  <tr>
    <td align="center">
      <img src="docs/assets/oft_heatmaps.png" alt="OFT heatmaps" width="430"/>
      <br/>
      <sub>Multi-arena OFT heatmaps</sub>
    </td>
    <td align="center">
      <img src="docs/assets/oft_trackmaps.png" alt="OFT track maps" width="430"/>
      <br/>
      <sub>Perspective-normalized OFT track maps</sub>
    </td>
  </tr>
</table>

### Tail Suspension Test demos

<table>
  <tr>
    <td align="center">
      <img src="static/img/demo_tutorial.gif" alt="TST workflow tutorial" width="430"/>
      <br/>
      <sub>TST workflow: upload, calibrate, analyze, export</sub>
    </td>
    <td align="center">
      <img src="static/img/demo_result.gif" alt="TST result visualization" width="430"/>
      <br/>
      <sub>TST analysis visualization with automated immobility scoring</sub>
    </td>
  </tr>
</table>

### TST desktop UI

<p align="center">
  <img src="static/img/ui_home.png" alt="MouseScope desktop UI" width="900"/>
</p>

---

## TST validation snapshots

<table>
  <tr>
    <td align="center">
      <img src="static/img/validation_correlation_33_mice.png" alt="TST validation correlation" width="420"/>
      <br/>
      <sub>Correlation between software and manual scoring</sub>
    </td>
    <td align="center">
      <img src="static/img/validation_ethogram_33_mice.png" alt="TST validation ethogram" width="470"/>
      <br/>
      <sub>Ethogram-style comparison across 33 mice</sub>
    </td>
  </tr>
</table>

---

## Quick Start

### Option 1: Run the packaged EXE

1. Download the latest release from [Releases](https://github.com/fujiacheng3/MouseScope/releases).
2. Extract the archive.
3. Run `MouseScope.exe`.
4. The app opens in your browser at `http://localhost:8080`.

### Option 2: Run from source

```bash
git clone https://github.com/fujiacheng3/MouseScope.git
cd MouseScope
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:8080`.

---

## Typical workflow

### TST

1. Upload one or more TST videos.
2. Perform body-length calibration if needed.
3. Draw one or more ROIs.
4. Name each ROI and optionally set ROI-level time windows.
5. Tune detection parameters and start analysis.
6. Export annotated video, frame logs, immobility summary, and TXT report.

### OFT

1. Upload one or more OFT videos.
2. Switch the experiment type to `Open Field Test`.
3. For each arena, click the four corners and enter the arena side length.
4. Name each ROI / arena and optionally assign precise statistics windows.
5. Tune shared tracking parameters with multi-arena preview.
6. Export trajectory CSV, summary CSV, bin statistics, heatmaps, track maps, and report files.

---

## Output files

### TST outputs

| File | Description |
|---|---|
| `*_analyzed.mp4` | Annotated analysis video |
| `*_logs.csv` | Frame-by-frame logs |
| `*_immobility.csv` | ROI summary statistics |
| `*_report.txt` | Text report |

### OFT outputs

| File | Description |
|---|---|
| `*_oft_analyzed.mp4` | Annotated OFT video |
| `*_oft_tracks.csv` | Frame-by-frame trajectory records |
| `*_oft_summary.csv` | Per-ROI summary metrics |
| `*_oft_bins.csv` | Time-binned statistics |
| `*_oft_report.txt` | Text report |
| `*_oft_heatmaps.png` | Heatmap output |
| `*_oft_trackmaps.png` | Perspective-normalized track maps |

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| CV pipeline | OpenCV, background subtraction, contour detection, optical-flow-assisted TST analysis |
| Data | NumPy, Pandas |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Packaging | PyInstaller |

---

## Repository structure

```text
MouseScope/
|- app.py                      # Flask app and workflow orchestration
|- core/
|  |- vision_tracker.py        # TST CV pipeline
|  |- open_field.py            # OFT tracking / statistics / map rendering
|  \- performance_utils.py
|- templates/                  # UI pages
|- static/                     # Static assets
|- data/results/               # Example local outputs
\- MouseScope.spec             # PyInstaller build spec
```

---

## System requirements

- Windows 10 / 11 x64
- Python 3.9 to 3.11 for source mode
- 8 GB RAM recommended
- Multi-core CPU recommended for faster batch analysis

---

## Disclaimer

This software is intended as a **research tool**.
Behavioral analysis quality can be affected by video quality, lighting, camera angle, and experimental setup.
Please manually review results before using them in publications or critical decisions.

本软件仅作为**科研辅助工具**使用。
分析结果会受到视频质量、光照、机位和实验条件影响，正式使用前请务必进行人工复核。

---

## Citation

If MouseScope helps your research, please cite:

```bibtex
@software{fu2026mousescope,
  author  = {Fu, Jiacheng},
  title   = {MouseScope: Automated Mouse Behavior Analysis Software for Tail Suspension Test and Open Field Test},
  year    = {2026},
  url     = {https://github.com/fujiacheng3/MouseScope},
  version = {1.0}
}
```

Plain text:

> Fu, J. (2026). *MouseScope: Automated mouse behavior analysis software for tail suspension test and open field test* (Version 1.0) [Software]. GitHub. https://github.com/fujiacheng3/MouseScope

---

## License

Academic and non-commercial use only.
See [LICENSE](LICENSE) for details.

---

## Contact

- **Author**: Jiacheng Fu
- **Email**: 2583348593@qq.com
- **GitHub**: https://github.com/fujiacheng3/MouseScope

---

<p align="center">
  Built for practical behavioral video analysis in real lab workflows.
</p>
