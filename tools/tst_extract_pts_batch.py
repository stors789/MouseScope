from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from tst_pts_timebase import load_or_extract_pts, summarize_pts


TS_RE = re.compile(r"(\d{2}-\d{2}-\d{2}-\d{3})")


def latest_session_state(results_root: Path) -> Path:
    candidates = sorted(
        (results_root / "_work").glob("*/session_state.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No session_state.json under {results_root / '_work'}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and cache video PTS timebase for TST videos.")
    parser.add_argument("--results-root", default=r"E:\MouseScope_gui\data\results")
    parser.add_argument("--out-dir", default=r"E:\MouseScope_gui\data\results\tst_pts_timebase")
    parser.add_argument("--session-state", default=None)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session_state = Path(args.session_state) if args.session_state else latest_session_state(results_root)
    state = json.loads(session_state.read_text(encoding="utf-8"))
    video_paths = state.get("video_paths") or []

    rows = []
    merged = []
    for order, video_path in enumerate(video_paths, start=1):
        match = TS_RE.search(str(video_path))
        video_ts = match.group(1) if match else ""
        df = load_or_extract_pts(video_path, out_dir)
        item = {"video_order": order, "video_ts": video_ts, "video_path": str(video_path)}
        item.update(summarize_pts(df))
        rows.append(item)
        tmp = df.copy()
        tmp["video_order"] = order
        tmp["video_ts"] = video_ts
        merged.append(tmp)
        print(
            f"{video_ts}: frames={item['frames']} median_dt={item['dt_median']:.6f} "
            f"max_dt={item['dt_max']:.6f} long_gaps={item['long_gap_gt_50ms']}",
            flush=True,
        )

    pd.DataFrame(rows).to_csv(out_dir / "pts_timebase_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(merged, ignore_index=True).to_csv(out_dir / "pts_timebase_all_frames.csv", index=False, encoding="utf-8-sig")
    print(f"wrote {out_dir / 'pts_timebase_summary.csv'}")


if __name__ == "__main__":
    main()
