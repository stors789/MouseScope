# MouseScope

<p align="center">
  <img src="static/img/logo.png" alt="MouseScope logo" width="112"/>
</p>

<p align="center">
  <strong>面向真实实验工作流的小鼠行为学视频分析桌面平台</strong><br/>
  <strong>A modular desktop platform for practical mouse behavioral video analysis in real lab workflows.</strong>
</p>

<p align="center">
  Current release supports <strong>TST</strong>, <strong>Open Field</strong>, <strong>NOR</strong>, <strong>Elevated Plus Maze</strong>, and <strong>Y-Maze</strong> in one unified workflow.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Version-v2.0.0-c96442" alt="Version"/>
  <img src="https://img.shields.io/badge/Platform-Windows%2010%2F11-7a5136" alt="Platform"/>
  <img src="https://img.shields.io/badge/Python-3.9%2B-2563eb" alt="Python"/>
  <img src="https://img.shields.io/badge/UI-Flask%20Desktop-f59e0b" alt="UI"/>
  <img src="https://img.shields.io/badge/Experiments-5%20Modules-16a34a" alt="Experiments"/>
  <img src="https://img.shields.io/badge/Language-中文%20%7C%20English-0f766e" alt="Language"/>
</p>

<p align="center">
  <a href="https://github.com/fujiacheng3/MouseScope/releases">Download</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#supported-assays">Supported Assays</a> ·
  <a href="docs/system_design.md">System Design</a>
</p>

<p align="center">
  <img src="static/img/ui_home.png" alt="MouseScope v2.0.0 home page" width="920"/>
  <br/>
  <sub>MouseScope v2.0.0 current home interface</sub>
</p>

## What MouseScope Is

MouseScope is a desktop application for mouse behavioral video analysis. It combines a local browser UI with a Python / OpenCV backend so lab users can finish upload, calibration, ROI or zone setup, parameter tuning, batch review, analysis, and export inside one consistent workflow.

v2.0.0 is a platform release rather than a small feature patch. It adds `EPM` and `Y-Maze`, rebuilds batch processing into a real workspace flow, unifies ROI / mask / timing logic across modules, and leaves behind a system design document for future expansion.

## Why MouseScope

- One software entry for multiple assays instead of separate scripts and ad hoc notebooks.
- Built around real lab usage: single-video editing, batch workspace, per-video drafts, and result export.
- Supports ROI, zone, detection mask, time-window, and tuning workflows that can be handed to non-programmer colleagues.
- Exports annotated videos, CSV tables, TXT reports, trajectory maps, heatmaps, and ZIP-ready result folders.

## v2.0.0 Highlights

- Added `Elevated Plus Maze (EPM)` and `Y-Maze` modules.
- Reworked multi-video sessions into `Upload -> Batch Workspace -> Per-video Setup -> Unified Start`.
- Unified ROI and detection-mask editing logic across OFT / NOR / EPM / Y-Maze.
- Added standard-template heatmaps and trackmaps for maze assays instead of raw skewed ROI-shaped maps.
- Kept bilingual UI and UTF-8 / Chinese-path-safe output handling.
- Added route and short-analysis smoke checks for safer future refactors.

## Workflow Preview

<p align="center">
  <img src="static/img/demo_tutorial.gif" alt="MouseScope workflow preview" width="920"/>
  <br/>
  <sub>Typical workflow: choose assay, calibrate, tune, analyze, export</sub>
</p>

## Supported Assays

| Assay | Typical use | Core setup | Main outputs |
|---|---|---|---|
| `TST` | Tail suspension immobility analysis | Body-length calibration, ROI, time windows | Annotated video, logs CSV, immobility CSV, TXT report |
| `Open Field` | Trajectory, center / wall statistics | 4-corner arena ROI, optional detection mask | Annotated video, track CSV, summary CSV, bins CSV, heatmaps, trackmaps, TXT report |
| `NOR` | Object exploration and preference | 4-corner arena + 2 object centers + optional detection mask | Annotated video, track CSV, summary CSV, bins CSV, heatmaps, trackmaps, TXT report |
| `EPM` | Open/closed arm exploration | Manually drawn center + open arms + closed arms + optional detection mask | Annotated video, summary CSV, bins CSV, heatmaps, trackmaps, TXT report |
| `Y-Maze` | Arm entries and spontaneous alternation | 3-point center triangle + 3 arm zones + optional detection mask | Annotated video, summary CSV, bins CSV, heatmaps, trackmaps, TXT report |

## Featured Results

### Tail Suspension Test

<p align="center">
  <img src="static/img/demo_result.gif" alt="TST annotated result" width="920"/>
  <br/>
  <sub>Annotated TST result preview</sub>
</p>

<table>
  <tr>
    <td align="center">
      <img src="static/img/validation_correlation_33_mice.png" alt="TST validation correlation" width="420"/>
      <br/>
      <sub>Software vs. manual scoring correlation</sub>
    </td>
    <td align="center">
      <img src="static/img/validation_ethogram_33_mice.png" alt="TST validation ethogram" width="470"/>
      <br/>
      <sub>Ethogram-style comparison across 33 mice</sub>
    </td>
  </tr>
</table>

### Open Field

<table>
  <tr>
    <td align="center">
      <img src="docs/assets/oft_heatmaps.png" alt="Open field heatmaps" width="430"/>
      <br/>
      <sub>Open-field heatmaps</sub>
    </td>
    <td align="center">
      <img src="docs/assets/oft_trackmaps.png" alt="Open field trackmaps" width="430"/>
      <br/>
      <sub>Open-field trackmaps</sub>
    </td>
  </tr>
</table>

### Novel Object Recognition

<table>
  <tr>
    <td align="center">
      <img src="docs/assets/nor_preview.png" alt="NOR preview" width="430"/>
      <br/>
      <sub>NOR preview with arena, objects, and interaction regions</sub>
    </td>
    <td align="center">
      <img src="docs/assets/nor_heatmaps.png" alt="NOR heatmaps" width="430"/>
      <br/>
      <sub>NOR heatmaps and exploration-space visualization</sub>
    </td>
  </tr>
</table>

## Unified Workflow

1. Upload one or multiple videos and choose an assay.
2. Configure ROI / zones / objects / masks and time windows.
3. Tune detection and statistics parameters with live preview.
4. Run analysis, then export videos, tables, reports, and spatial figures.

For multi-video sessions in v2.0.0:

1. Upload videos.
2. Enter the batch workspace.
3. Open each video card for its own setup page.
4. Return to the workspace and start the batch only after every card is ready.

## Output Files

MouseScope outputs vary by assay, but the v2.0.0 platform consistently supports:

- Annotated result videos
- Per-frame or per-ROI CSV tables
- Summary CSV tables
- TXT reports with bilingual metric labels
- Heatmaps and trajectory maps for spatial assays
- Batch-friendly result folders that can be zipped and shared

## Quick Start

### Windows package

1. Download the latest build from [Releases](https://github.com/fujiacheng3/MouseScope/releases).
2. Unzip the package.
3. Run `MouseScope.exe`.
4. The app will open `http://localhost:8080` in your browser.

### Run from source

```bash
git clone https://github.com/fujiacheng3/MouseScope.git
cd MouseScope
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:8080`.

## Repository Structure

```text
MouseScope/
|- app.py
|- core/
|  |- vision_tracker.py
|  |- open_field.py
|  |- novel_object.py
|  |- elevated_plus.py
|  |- y_maze.py
|  \- maze_zone_utils.py
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
|  |- system_design.md
|  \- releases/
\- MouseScope.spec
```

## Notes and Limitations

- MouseScope is a research tool and should still be checked against raw video in critical analyses.
- Performance depends on lighting, contrast, background cleanliness, camera angle, and video quality.
- OFT / NOR currently use classic computer-vision tracking rather than pose-estimation-grade behavior parsing.
- NOR exploration is currently based on distance, speed, and optional head-direction heuristics rather than full nose-point tracking.

## Citation

If MouseScope helps your work, please cite the platform version:

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

## License

This repository is distributed under `GPL-3.0-or-later`. See [LICENSE](LICENSE).

## Contact

- Author: Jiacheng Fu
- Email: 2583348593@qq.com
- GitHub: https://github.com/fujiacheng3/MouseScope

<p align="center">
  Built for practical behavioral analysis in real lab workflows.
</p>
