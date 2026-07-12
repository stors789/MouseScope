from __future__ import annotations

import re
import time
from pathlib import Path


def _safe_component(value: object, fallback: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_-]+", "_", str(value or "").strip()).strip("_-")
    return (text or fallback)[:48]


def build_run_output_dir(
    output_root: str | Path,
    *,
    session_id: str,
    experiment_type: str,
    timestamp: str | None = None,
    attempt: int = 1,
) -> str:
    """Return a deterministic, session-unique result directory below a root.

    The timestamp keeps folders readable; the session component supplies the
    uniqueness guarantee even when two analyses start in the same second.
    """

    root = Path(output_root)
    stamp = _safe_component(timestamp or time.strftime("%Y%m%d_%H%M%S"), "run")
    experiment = _safe_component(experiment_type, "analysis")
    session = _safe_component(session_id, "session")[:12]
    attempt_number = max(1, int(attempt or 1))
    attempt_suffix = f"_retry{attempt_number}" if attempt_number > 1 else ""
    return str(root / f"{stamp}_{experiment}_{session}{attempt_suffix}")


def prepare_session_run_output(
    sess: dict,
    *,
    session_id: str,
    default_output_root: str | Path,
    resolve_output_dir,
    force_new: bool = False,
) -> str:
    """Convert a quick-analysis output root into one unique run directory.

    Project sessions reserve their run directory when the project run is
    created. Quick-analysis sessions keep the user's selected folder as
    ``output_root_dir`` and expose the concrete run folder as ``output_dir``.
    """

    existing_run_dir = str(sess.get("run_output_dir") or "").strip()
    if existing_run_dir and not force_new:
        sess["output_dir"] = existing_run_dir
        return existing_run_dir

    previous_attempt = int(sess.get("run_attempt") or (1 if existing_run_dir else 0))
    run_attempt = previous_attempt + 1 if force_new or not existing_run_dir else previous_attempt

    if isinstance(sess.get("project_context"), dict):
        run_dir = "" if force_new else str(sess.get("output_dir") or "").strip()
        if not run_dir:
            project_context = sess.get("project_context") or {}
            project_root = (
                sess.get("output_root_dir")
                or project_context.get("output_root_dir")
                or (Path(existing_run_dir).parent if existing_run_dir else default_output_root)
            )
            run_dir = build_run_output_dir(
                resolve_output_dir(project_root),
                session_id=session_id,
                experiment_type=sess.get("experiment_type") or "analysis",
                attempt=run_attempt,
            )
        sess.setdefault("output_root_dir", str(Path(run_dir).parent))
    else:
        configured_root = sess.get("output_root_dir") or sess.get("output_dir") or default_output_root
        resolved_root = resolve_output_dir(configured_root)
        sess["output_root_dir"] = resolved_root
        run_dir = build_run_output_dir(
            resolved_root,
            session_id=session_id,
            experiment_type=sess.get("experiment_type") or "analysis",
            attempt=run_attempt,
        )

    sess["run_attempt"] = run_attempt
    sess["run_output_dir"] = run_dir
    sess["output_dir"] = run_dir
    return run_dir
