from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from services.video_export import find_ffmpeg_exe

TYPE_LABELS = {
    "old_right_new_wrong_new_false_struggle": "新增误判挣扎",
    "old_right_new_wrong_new_missed_struggle": "新增漏判挣扎",
    "old_wrong_new_right_rescued_struggle": "救回漏判挣扎",
    "old_wrong_new_right_removed_false_struggle": "纠正旧误判挣扎",
}

TYPE_LIMITS = {
    "old_right_new_wrong_new_false_struggle": 5,
    "old_right_new_wrong_new_missed_struggle": 3,
    "old_wrong_new_right_rescued_struggle": 4,
}

VIDEO_NAMES = {
    "11-02-42-882": "bandicam 2024-11-29 11-02-42-882.mp4",
    "11-14-37-601": "bandicam 2024-11-29 11-14-37-601.mp4",
    "11-28-29-994": "bandicam 2024-11-29 11-28-29-994.mp4",
    "11-40-48-715": "bandicam 2024-11-29 11-40-48-715.mp4",
    "11-56-10-908": "bandicam 2024-11-29 11-56-10-908.mp4",
    "12-08-06-116": "bandicam 2024-11-29 12-08-06-116.mp4",
    "12-21-27-977": "bandicam 2024-11-29 12-21-27-977.mp4",
    "12-34-22-351": "bandicam 2024-11-29 12-34-22-351.mp4",
    "12-44-36-199": "bandicam 2024-11-29 12-44-36-199.mp4",
}


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) or default)
    except Exception:
        return default


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key) or default))
    except Exception:
        return default


def load_segments(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    selected: list[dict[str, Any]] = []
    for segment_type, limit in TYPE_LIMITS.items():
        group = [row for row in rows if row.get("segment_type") == segment_type]
        group.sort(key=lambda row: _float(row, "duration_s"), reverse=True)
        selected.extend(group[:limit])

    out: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        video_ts = str(row.get("video_ts") or "")
        start_s = _float(row, "start_s")
        end_s = _float(row, "end_s")
        out.append(
            {
                "id": idx,
                "segment_type": row.get("segment_type"),
                "type_label": TYPE_LABELS.get(str(row.get("segment_type") or ""), row.get("segment_type")),
                "mouse_id": _int(row, "mouse_id"),
                "video_ts": video_ts,
                "video": VIDEO_NAMES.get(video_ts, row.get("video_name") or ""),
                "clip": "",
                "roi": f"ROI {_int(row, 'roi_id')}",
                "roi_id": _int(row, "roi_id"),
                "start": round(start_s, 3),
                "end": round(end_s, 3),
                "duration": round(max(0.0, end_s - start_s), 3),
                "hidden_human": "struggle" if str(row.get("human_struggle")).lower() == "true" else "non_struggle",
                "hidden_old": "struggle" if str(row.get("old_struggle")).lower() == "true" else "non_struggle",
                "hidden_new": "struggle" if str(row.get("new_struggle")).lower() == "true" else "non_struggle",
                "new_long_gap_count": _int(row, "new_long_gap_count"),
                "new_mean_flow_p90": round(_float(row, "new_mean_flow_p90"), 3),
                "new_mean_speed_std": round(_float(row, "new_mean_speed_std"), 3),
                "new_mean_entropy": round(_float(row, "new_mean_entropy"), 4),
                "notes": "",
            }
        )
    return out


def clip_file_name(segment: dict[str, Any]) -> str:
    return (
        f"clip_{int(segment['id']):02d}_"
        f"m{int(segment['mouse_id']):02d}_"
        f"roi{int(segment['roi_id'])}_"
        f"{str(segment['video_ts']).replace('-', '')}_"
        f"{float(segment['start']):07.3f}_{float(segment['end']):07.3f}.mp4"
    ).replace(".", "p", 2)


def make_clips(
    segments: list[dict[str, Any]],
    *,
    source_dir: Path,
    clip_dir: Path,
    html_dir: Path,
    overwrite: bool = False,
) -> None:
    ffmpeg_exe = find_ffmpeg_exe()
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg executable not found; cannot create real video clips.")
    clip_dir.mkdir(parents=True, exist_ok=True)
    for segment in segments:
        source = source_dir / str(segment["video"])
        if not source.exists():
            raise FileNotFoundError(f"Source video not found: {source}")
        clip_path = clip_dir / clip_file_name(segment)
        duration = max(0.04, float(segment["end"]) - float(segment["start"]))
        pre_seek = max(0.0, float(segment["start"]) - 2.0)
        local_seek = max(0.0, float(segment["start"]) - pre_seek)
        if overwrite or not clip_path.exists() or clip_path.stat().st_size <= 0:
            cmd = [
                ffmpeg_exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{pre_seek:.6f}",
                "-i",
                str(source),
                "-ss",
                f"{local_seek:.6f}",
                "-t",
                f"{duration:.6f}",
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(clip_path),
            ]
            subprocess.run(cmd, check=True)
        segment["clip"] = clip_path.relative_to(html_dir).as_posix()


def build_html(segments: list[dict[str, Any]]) -> str:
    payload = json.dumps(segments, ensure_ascii=False, indent=2)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>TST 短片段仲裁</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --border: #d1d5db;
      --red: #dc2626;
      --blue: #2563eb;
      --green: #16a34a;
      --amber: #d97706;
      --gray: #4b5563;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    main {{
      width: min(1040px, calc(100vw - 32px));
      margin: 18px auto;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 24px;
    }}
    .topbar, .meta, .actions, .utility {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .topbar {{
      justify-content: space-between;
      margin-bottom: 12px;
    }}
    .pill {{
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 14px;
      color: var(--muted);
      background: #f9fafb;
    }}
    .focus {{
      margin: 10px 0;
      padding: 10px 12px;
      border-left: 5px solid var(--red);
      background: #fff7ed;
      font-size: 20px;
      font-weight: 700;
    }}
    video {{
      display: block;
      width: 100%;
      max-height: 68vh;
      background: #000;
      border-radius: 6px;
      margin: 10px 0 14px;
    }}
    button {{
      border: 0;
      border-radius: 7px;
      padding: 12px 18px;
      font-size: 17px;
      font-weight: 700;
      cursor: pointer;
      color: white;
    }}
    button:active {{ transform: translateY(1px); }}
    .struggle {{ background: var(--red); }}
    .immobile {{ background: var(--blue); }}
    .swing {{ background: var(--amber); }}
    .uncertain {{ background: var(--gray); }}
    .secondary {{ background: var(--green); }}
    .ghost {{
      background: #e5e7eb;
      color: #111827;
      border: 1px solid var(--border);
    }}
    textarea {{
      width: 100%;
      min-height: 54px;
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px;
      font-size: 15px;
      margin: 6px 0 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 7px 6px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{ color: var(--muted); font-weight: 700; }}
    .hidden-labels {{
      display: none;
      background: #f9fafb;
      border: 1px dashed var(--border);
      border-radius: 6px;
      padding: 8px;
      margin: 8px 0;
      color: var(--muted);
    }}
    .done {{
      text-align: center;
      padding: 24px;
      color: var(--green);
      font-weight: 700;
      display: none;
    }}
  </style>
</head>
<body>
<main>
  <div class="topbar">
    <h1>TST 典型短片段仲裁</h1>
    <div class="pill" id="progress">0 / {len(segments)}</div>
  </div>
  <div class="meta">
    <span class="pill">只含 {len(segments)} 个片段</span>
    <span class="pill">直接播放片段本身，无前后延长</span>
    <span class="pill">ROI 从左到右为 ROI 1-4</span>
  </div>
  <div class="focus" id="focusText"></div>
  <div class="meta" id="segmentMeta"></div>
  <div class="hidden-labels" id="hiddenLabels"></div>
  <video id="video" controls muted autoplay playsinline loop></video>
  <textarea id="note" placeholder="可选备注：例如边界不确定、像摆动、像低强度挣扎、看不清等"></textarea>
  <div class="actions">
    <button class="struggle" onclick="recordChoice('struggle')">挣扎</button>
    <button class="immobile" onclick="recordChoice('immobile')">不动/静止</button>
    <button class="swing" onclick="recordChoice('swing_non_struggle')">摆动非挣扎</button>
    <button class="uncertain" onclick="recordChoice('uncertain')">不确定/边界</button>
  </div>
  <div class="utility" style="margin-top: 12px;">
    <button class="ghost" onclick="previousSegment()">上一个</button>
    <button class="ghost" onclick="replaySegment()">重播</button>
    <button class="ghost" onclick="toggleLabels()">显示/隐藏旧新标签</button>
    <button class="secondary" onclick="exportCSV()">导出 CSV</button>
  </div>
  <div class="done" id="doneText">已完成全部片段，请导出 CSV。</div>
  <table>
    <thead>
      <tr><th>#</th><th>类型</th><th>鼠</th><th>视频</th><th>ROI</th><th>时间</th><th>你的判断</th></tr>
    </thead>
    <tbody id="reviewTable"></tbody>
  </table>
</main>
<script>
const segments = {payload};
let index = 0;
let results = JSON.parse(localStorage.getItem("tst_short_arbitration_results") || "[]");
let showLabels = false;
const video = document.getElementById("video");
const focusText = document.getElementById("focusText");
const segmentMeta = document.getElementById("segmentMeta");
const progress = document.getElementById("progress");
const note = document.getElementById("note");
const hiddenLabels = document.getElementById("hiddenLabels");
const doneText = document.getElementById("doneText");
const reviewTable = document.getElementById("reviewTable");

function existingResult(id) {{
  return results.find(r => Number(r.id) === Number(id));
}}

function saveResults() {{
  localStorage.setItem("tst_short_arbitration_results", JSON.stringify(results));
}}

function loadSegment() {{
  doneText.style.display = index >= segments.length ? "block" : "none";
  if (index >= segments.length) {{
    video.removeAttribute("src");
    video.load();
    focusText.textContent = "全部片段已完成";
    segmentMeta.innerHTML = "";
    progress.textContent = `${{segments.length}} / ${{segments.length}}`;
    renderTable();
    return;
  }}
  const seg = segments[index];
  const old = existingResult(seg.id);
  note.value = old ? (old.note || "") : "";
  progress.textContent = `${{index + 1}} / ${{segments.length}}`;
  focusText.textContent = `请只关注：鼠 ${{seg.mouse_id}}，${{seg.roi}}`;
  segmentMeta.innerHTML = `
    <span class="pill">${{seg.type_label}}</span>
    <span class="pill">${{seg.video}}</span>
    <span class="pill">${{seg.start.toFixed(2)}}-${{seg.end.toFixed(2)}}s，共 ${{seg.duration.toFixed(2)}}s</span>
    <span class="pill">长间隔 ${{seg.new_long_gap_count}}</span>
  `;
  hiddenLabels.innerHTML = `
    人工: <b>${{seg.hidden_human}}</b>　
    旧算法: <b>${{seg.hidden_old}}</b>　
    新算法: <b>${{seg.hidden_new}}</b>　
    flow: ${{seg.new_mean_flow_p90}}　
    speed std: ${{seg.new_mean_speed_std}}　
    entropy: ${{seg.new_mean_entropy}}
  `;
  hiddenLabels.style.display = showLabels ? "block" : "none";
  video.src = seg.clip || seg.video;
  video.load();
  video.play().catch(() => {{}});
  renderTable();
}}

video.addEventListener("timeupdate", () => {{
  const seg = segments[index];
  if (!seg) return;
  if (video.duration && video.currentTime >= video.duration - 0.03) {{
    video.currentTime = 0;
    video.play().catch(() => {{}});
  }}
}});

function recordChoice(choice) {{
  const seg = segments[index];
  if (!seg) return;
  const row = {{
    id: seg.id,
    segment_type: seg.segment_type,
    type_label: seg.type_label,
    mouse_id: seg.mouse_id,
    video_ts: seg.video_ts,
    video: seg.video,
    clip: seg.clip,
    roi_id: seg.roi_id,
    roi: seg.roi,
    start_s: seg.start,
    end_s: seg.end,
    duration_s: seg.duration,
    user_choice: choice,
    note: note.value.replace(/[\\r\\n]+/g, " ").trim(),
    hidden_human: seg.hidden_human,
    hidden_old: seg.hidden_old,
    hidden_new: seg.hidden_new,
    new_long_gap_count: seg.new_long_gap_count,
    new_mean_flow_p90: seg.new_mean_flow_p90,
    new_mean_speed_std: seg.new_mean_speed_std,
    new_mean_entropy: seg.new_mean_entropy
  }};
  results = results.filter(r => Number(r.id) !== Number(seg.id));
  results.push(row);
  results.sort((a, b) => Number(a.id) - Number(b.id));
  saveResults();
  index += 1;
  loadSegment();
}}

function previousSegment() {{
  if (index > 0) {{
    index -= 1;
    loadSegment();
  }}
}}

function replaySegment() {{
  const seg = segments[index];
  if (!seg) return;
  video.currentTime = 0;
  video.play().catch(() => {{}});
}}

function toggleLabels() {{
  showLabels = !showLabels;
  hiddenLabels.style.display = showLabels ? "block" : "none";
}}

function renderTable() {{
  reviewTable.innerHTML = segments.map(seg => {{
    const result = existingResult(seg.id);
    return `<tr>
      <td>${{seg.id}}</td>
      <td>${{seg.type_label}}</td>
      <td>${{seg.mouse_id}}</td>
      <td>${{seg.video_ts}}</td>
      <td>${{seg.roi}}</td>
      <td>${{seg.start.toFixed(2)}}-${{seg.end.toFixed(2)}}s</td>
      <td>${{result ? result.user_choice : ""}}</td>
    </tr>`;
  }}).join("");
}}

function exportCSV() {{
  const header = [
    "id","segment_type","type_label","mouse_id","video_ts","video","clip","roi_id","roi",
    "start_s","end_s","duration_s","user_choice","note",
    "hidden_human","hidden_old","hidden_new",
    "new_long_gap_count","new_mean_flow_p90","new_mean_speed_std","new_mean_entropy"
  ];
  const escapeCell = value => `"${{String(value ?? "").replace(/"/g, '""')}}"`;
  const rows = [header.join(",")];
  results.forEach(row => rows.push(header.map(k => escapeCell(row[k])).join(",")));
  const blob = new Blob(["\\uFEFF" + rows.join("\\n")], {{ type: "text/csv;charset=utf-8" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "tst_short_arbitration_results.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}

window.addEventListener("keydown", event => {{
  if (event.target && event.target.tagName === "TEXTAREA") return;
  if (event.key === "1") recordChoice("struggle");
  if (event.key === "2") recordChoice("immobile");
  if (event.key === "3") recordChoice("swing_non_struggle");
  if (event.key === "4") recordChoice("uncertain");
  if (event.key === "ArrowLeft") previousSegment();
  if (event.key === " ") {{
    event.preventDefault();
    if (video.paused) video.play(); else video.pause();
  }}
}});

loadSegment();
</script>
</body>
</html>
"""


def write_manifest(segments: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        fieldnames = list(segments[0].keys()) if segments else []
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(segments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a short TST arbitration HTML from priority segments.")
    parser.add_argument(
        "--segments-csv",
        default=str(ROOT / "data" / "results" / "tst_timebase_regression_current" / "arbitration_priority_segments.csv"),
    )
    parser.add_argument("--out-html", default=r"E:\20241129\tst_timebase_short_arbitration.html")
    parser.add_argument("--source-dir", default=r"E:\20241129")
    parser.add_argument("--clip-dir", default=r"E:\20241129\tst_timebase_short_arbitration_clips")
    parser.add_argument("--no-clips", action="store_true", help="Do not cut MP4 clips; fall back to source videos.")
    parser.add_argument("--overwrite-clips", action="store_true")
    parser.add_argument(
        "--manifest-csv",
        default=str(ROOT / "data" / "results" / "tst_timebase_regression_current" / "short_arbitration_manifest.csv"),
    )
    args = parser.parse_args()

    segments = load_segments(Path(args.segments_csv))
    out_html = Path(args.out_html)
    if not args.no_clips:
        make_clips(
            segments,
            source_dir=Path(args.source_dir),
            clip_dir=Path(args.clip_dir),
            html_dir=out_html.parent,
            overwrite=bool(args.overwrite_clips),
        )
    html_text = build_html(segments)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_text, encoding="utf-8")
    write_manifest(segments, Path(args.manifest_csv))
    print(f"Wrote {out_html}")
    print(f"Wrote {args.manifest_csv}")
    print(f"Segments: {len(segments)}")


if __name__ == "__main__":
    main()
