from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_y_maze_annotation_uses_the_summary_analysis_pass() -> None:
    source = (ROOT / "services" / "y_maze_session.py").read_text(encoding="utf-8")
    analyze_body = source.split("def _analyze_single_video", 1)[1].split(
        "def _draw_manual_overlays", 1
    )[0]

    assert analyze_body.count("tracker.update(") == 1
    assert "annotated_writer.write(result_frame)" in analyze_body
    assert analyze_body.index("tracker.update(") < analyze_body.index(
        "annotated_writer.write(result_frame)"
    )
    assert "def _render_annotated_video" not in source
