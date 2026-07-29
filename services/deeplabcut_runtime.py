from __future__ import annotations

from collections import deque
import atexit
import importlib.util
import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from typing import Any

import cv2


class DeepLabCutRuntimeError(RuntimeError):
    pass


class DeepLabCutRuntime:
    """Optional DLC process boundary that keeps PyTorch out of PyInstaller."""

    def __init__(
        self,
        command: str | None = None,
        *,
        timeout_seconds: float = 120.0,
        status_timeout_seconds: float = 15.0,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.status_timeout_seconds = float(status_timeout_seconds)
        self._explicit_command = command is not None
        configured = command if command is not None else os.environ.get("MOUSESCOPE_DLC_WORKER")
        self.command = str(configured or "").strip()
        self._argv, self._command_origin = self._resolve_command()
        self._status_cache: dict[str, Any] | None = None
        self._process: subprocess.Popen[str] | None = None
        self._response_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_lines: deque[str] = deque(maxlen=30)
        self._process_lock = threading.RLock()
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self.effective_batch_size: int | None = None

    @staticmethod
    def _main_module_installed() -> bool:
        try:
            return importlib.util.find_spec("deeplabcut") is not None
        except (ImportError, ValueError):
            return False

    @staticmethod
    def _worker_script() -> Path:
        return Path(__file__).resolve().parents[1] / "tools" / "mousescope_dlc_worker.py"

    @staticmethod
    def _python_candidates() -> list[tuple[Path, str]]:
        root = Path(__file__).resolve().parents[1]
        application_root = (
            Path(sys.executable).resolve().parent
            if getattr(sys, "frozen", False)
            else root
        )
        home = Path.home()
        configured_python = str(os.environ.get("MOUSESCOPE_DLC_PYTHON") or "").strip()
        candidates: list[tuple[Path, str]] = []
        if configured_python:
            candidates.append((Path(configured_python).expanduser(), "MOUSESCOPE_DLC_PYTHON"))
        candidates.extend(
            [
                (application_root / ".venv-dlc" / "bin" / "python", "application_dlc_venv"),
                (application_root / ".venv-dlc" / "Scripts" / "python.exe", "application_dlc_venv"),
                (application_root / "venv-dlc" / "bin" / "python", "application_dlc_venv"),
                (application_root / "venv-dlc" / "Scripts" / "python.exe", "application_dlc_venv"),
                (root / ".venv-dlc" / "bin" / "python", "project_dlc_venv"),
                (root / ".venv-dlc" / "Scripts" / "python.exe", "project_dlc_venv"),
                (root / "venv-dlc" / "bin" / "python", "project_dlc_venv"),
                (root / "venv-dlc" / "Scripts" / "python.exe", "project_dlc_venv"),
                (
                    home / "Documents" / "MouseScope" / ".venv-dlc" / "bin" / "python",
                    "documents_mousescope_dlc_venv",
                ),
                (
                    home / "Documents" / "MouseScope" / ".venv-dlc" / "Scripts" / "python.exe",
                    "documents_mousescope_dlc_venv",
                ),
                (
                    home / "Library" / "Application Support" / "MouseScope" / ".venv-dlc" / "bin" / "python",
                    "application_support_dlc_venv",
                ),
            ]
        )
        return candidates

    def _resolve_command(self) -> tuple[list[str], str | None]:
        if self.command:
            try:
                return shlex.split(
                    self.command,
                    posix=os.name != "nt",
                ), "MOUSESCOPE_DLC_WORKER"
            except ValueError:
                return [], None
        # command="" is an intentional opt-out used by diagnostics/tests.
        if self._explicit_command:
            return [], None
        worker = self._worker_script()
        if not worker.is_file():
            return [], None
        for python, origin in self._python_candidates():
            if python.is_file():
                argv = [str(python), str(worker)]
                self.command = shlex.join(argv)
                return argv, origin
        if self._main_module_installed() and not getattr(sys, "frozen", False):
            argv = [sys.executable, str(worker)]
            self.command = shlex.join(argv)
            return argv, "current_python"
        return [], None

    @staticmethod
    def _command_exists(argv: list[str]) -> bool:
        if not argv:
            return False
        executable = argv[0]
        if os.path.isabs(executable) or os.sep in executable:
            return Path(executable).is_file()
        return shutil.which(executable) is not None

    @staticmethod
    def _last_json_object(text: str) -> dict[str, Any] | None:
        for line in reversed((text or "").splitlines()):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def status(self) -> dict[str, Any]:
        if self._status_cache is not None:
            return dict(self._status_cache)
        module_installed = self._main_module_installed()
        if not self._command_exists(self._argv):
            code = "worker_not_configured" if module_installed else "dependency_missing"
            message = (
                "检测到 DeepLabCut，但未找到隔离 worker；将继续使用经典视觉。"
                if module_installed
                else "未找到 DeepLabCut worker；可运行 python tools/setup_dlc_worker.py 安装。"
            )
            self._status_cache = {
                "available": False,
                "preview_available": False,
                "analysis_available": False,
                "code": code,
                "message": message,
                "module_installed_in_app": module_installed,
                "worker_origin": None,
            }
            return dict(self._status_cache)
        try:
            completed = subprocess.run(
                [*self._argv, "status"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.status_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._status_cache = {
                "available": False,
                "preview_available": False,
                "analysis_available": False,
                "code": "worker_status_failed",
                "message": f"DeepLabCut worker 状态检查失败：{exc}",
                "module_installed_in_app": module_installed,
                "worker_origin": self._command_origin,
            }
            return dict(self._status_cache)
        payload = self._last_json_object(completed.stdout)
        ready = bool(
            completed.returncode == 0
            and payload
            and payload.get("available")
        )
        if ready:
            operations = payload.get("supported_operations") or []
            ready = "preview" in operations
        if not ready:
            detail = (
                (payload or {}).get("message")
                or completed.stderr.strip()[-1000:]
                or "worker status contract is unavailable"
            )
            self._status_cache = {
                "available": False,
                "preview_available": False,
                "analysis_available": False,
                "code": "worker_dependency_missing",
                "message": f"DeepLabCut worker 尚不可用：{detail}",
                "module_installed_in_app": module_installed,
                "worker_origin": self._command_origin,
            }
            return dict(self._status_cache)
        persistent = bool(payload.get("persistent")) and "serve" in (
            payload.get("supported_operations") or []
        )
        self._status_cache = {
            **payload,
            "available": True,
            "preview_available": True,
            "analysis_available": True,
            "persistent": persistent,
            "code": "worker_ready",
            "message": (
                "DeepLabCut worker 已就绪；正式分析将复用常驻模型。"
                if persistent
                else "DeepLabCut worker 已就绪。"
            ),
            "module_installed_in_app": module_installed,
            "worker_origin": self._command_origin,
        }
        return dict(self._status_cache)

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            payload = self._last_json_object(line)
            if payload is not None:
                self._response_queue.put(payload)
            elif line.strip():
                self._stderr_lines.append(f"unexpected stdout: {line.strip()}")
        self._response_queue.put(
            {
                "success": False,
                "error": "DeepLabCut persistent worker exited.",
                "_worker_eof": True,
            }
        )

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            if line.strip():
                self._stderr_lines.append(line.strip())

    def _ensure_persistent_process(self) -> subprocess.Popen[str]:
        process = self._process
        if process is not None and process.poll() is None:
            return process
        while True:
            try:
                self._response_queue.get_nowait()
            except queue.Empty:
                break
        try:
            process = subprocess.Popen(
                [*self._argv, "serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise DeepLabCutRuntimeError(f"DLC 常驻 worker 启动失败：{exc}") from exc
        self._process = process
        self._reader_thread = threading.Thread(
            target=self._read_stdout,
            args=(process,),
            daemon=True,
            name="mousescope-dlc-stdout",
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(process,),
            daemon=True,
            name="mousescope-dlc-stderr",
        )
        self._reader_thread.start()
        self._stderr_thread.start()
        return process

    def _persistent_request(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        request_payload = {**request_payload, "request_id": request_id}
        with self._process_lock:
            process = self._ensure_persistent_process()
            if process.stdin is None:
                raise DeepLabCutRuntimeError("DLC 常驻 worker 缺少输入管道。")
            try:
                process.stdin.write(
                    json.dumps(request_payload, ensure_ascii=False, allow_nan=False) + "\n"
                )
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self.close()
                raise DeepLabCutRuntimeError(f"DLC 常驻 worker 通信失败：{exc}") from exc
            try:
                payload = self._response_queue.get(timeout=self.timeout_seconds)
            except queue.Empty as exc:
                detail = " | ".join(self._stderr_lines)[-1000:]
                self.close()
                raise DeepLabCutRuntimeError(
                    f"DLC 常驻 worker 推理超时。{(' ' + detail) if detail else ''}"
                ) from exc
            if payload.get("request_id") not in (None, request_id):
                self.close()
                raise DeepLabCutRuntimeError("DLC 常驻 worker 响应顺序无效。")
            if not payload.get("success", True):
                detail = str(payload.get("error") or "unknown worker error")
                stderr = " | ".join(self._stderr_lines)[-600:]
                raise DeepLabCutRuntimeError(
                    f"DLC worker 返回错误：{detail}{('；' + stderr) if stderr else ''}"
                )
            return payload

    def _one_shot_request(
        self,
        request_payload: dict[str, Any],
        request_path: Path,
    ) -> dict[str, Any]:
        request_path.write_text(
            json.dumps(request_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [*self._argv, "preview", str(request_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DeepLabCutRuntimeError(f"DLC worker 启动失败：{exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[-1000:]
            raise DeepLabCutRuntimeError(
                f"DLC worker 返回错误：{detail or completed.returncode}"
            )
        payload = self._last_json_object(completed.stdout)
        if payload is None:
            raise DeepLabCutRuntimeError("DLC worker 没有返回有效 JSON。")
        return payload

    def preview_frame(
        self,
        frame,
        *,
        model_reference: str,
        keypoint_schema: list[dict[str, Any]],
        max_individuals: int = 1,
    ) -> dict[str, Any]:
        status = self.status()
        if not status["available"]:
            raise DeepLabCutRuntimeError(status["message"])
        with tempfile.TemporaryDirectory(prefix="mousescope-dlc-preview-") as temp_dir:
            image_path = Path(temp_dir) / "frame.png"
            request_path = Path(temp_dir) / "request.json"
            if not cv2.imwrite(str(image_path), frame):
                raise DeepLabCutRuntimeError("无法准备 DLC 预览帧。")
            request_payload = {
                "operation": "preview",
                "image_path": str(image_path),
                "model_reference": model_reference,
                "keypoint_schema": keypoint_schema,
                "max_individuals": max(1, int(max_individuals)),
            }
            if status.get("persistent"):
                payload = self._persistent_request(request_payload)
            else:
                payload = self._one_shot_request(request_payload, request_path)
        keypoints = payload.get("keypoints")
        individuals = payload.get("individuals")
        if not isinstance(keypoints, dict) and not isinstance(individuals, list):
            raise DeepLabCutRuntimeError("DLC worker 结果缺少 keypoints/individuals。")
        return payload

    def infer_frame(
        self,
        frame,
        *,
        model_reference: str,
        keypoint_schema: list[dict[str, Any]],
        max_individuals: int = 1,
    ) -> dict[str, Any]:
        return self.infer_frames(
            [frame],
            model_reference=model_reference,
            keypoint_schema=keypoint_schema,
            max_individuals=max_individuals,
            batch_size=1,
        )[0]

    def infer_frames(
        self,
        frames,
        *,
        model_reference: str,
        keypoint_schema: list[dict[str, Any]],
        max_individuals: int = 1,
        batch_size: int = 4,
    ) -> list[dict[str, Any]]:
        frame_list = list(frames or [])
        if not frame_list:
            return []
        status = self.status()
        if not status["available"]:
            raise DeepLabCutRuntimeError(status["message"])
        requested_batch_size = max(1, min(16, int(batch_size or 1)))
        with tempfile.TemporaryDirectory(prefix="mousescope-dlc-batch-") as temp_dir:
            image_paths: list[str] = []
            for index, frame in enumerate(frame_list):
                image_path = Path(temp_dir) / f"frame-{index:04d}.png"
                if not cv2.imwrite(str(image_path), frame):
                    raise DeepLabCutRuntimeError(
                        f"Unable to prepare DLC batch frame {index}."
                    )
                image_paths.append(str(image_path))
            payload = self._persistent_request(
                {
                    "operation": "infer_batch",
                    "image_paths": image_paths,
                    "model_reference": model_reference,
                    "keypoint_schema": keypoint_schema,
                    "max_individuals": max(1, int(max_individuals)),
                    "batch_size": requested_batch_size,
                }
            )
        items = payload.get("items")
        if not isinstance(items, list) or len(items) != len(frame_list):
            raise DeepLabCutRuntimeError(
                "DLC worker batch result count does not match the input frame count."
            )
        try:
            self.effective_batch_size = max(
                1,
                int(payload.get("effective_batch_size") or requested_batch_size),
            )
        except (TypeError, ValueError):
            self.effective_batch_size = requested_batch_size
        return items

    def close(self) -> None:
        with self._process_lock:
            process, self._process = self._process, None
            if process is None:
                return
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except OSError:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


_SHARED_RUNTIME: DeepLabCutRuntime | None = None
_SHARED_RUNTIME_LOCK = threading.Lock()


def shared_deeplabcut_runtime() -> DeepLabCutRuntime:
    global _SHARED_RUNTIME
    with _SHARED_RUNTIME_LOCK:
        if _SHARED_RUNTIME is None:
            _SHARED_RUNTIME = DeepLabCutRuntime()
        return _SHARED_RUNTIME


def close_shared_deeplabcut_runtime() -> None:
    global _SHARED_RUNTIME
    with _SHARED_RUNTIME_LOCK:
        runtime, _SHARED_RUNTIME = _SHARED_RUNTIME, None
    if runtime is not None:
        runtime.close()


atexit.register(close_shared_deeplabcut_runtime)


def resolve_tracking_backend(
    requested: Any,
    *,
    runtime_status: dict[str, Any] | None = None,
    model_ready: bool = False,
) -> dict[str, Any]:
    value = str(requested or "classical").strip().lower()
    if value not in {"classical", "deeplabcut", "auto"}:
        value = "classical"
    status = runtime_status or DeepLabCutRuntime().status()
    dlc_ready = bool(status.get("available")) and bool(model_ready)
    if value == "classical":
        return {"requested": value, "effective": "classical", "fallback_reason": None}
    if dlc_ready:
        return {"requested": value, "effective": "deeplabcut", "fallback_reason": None}
    return {
        "requested": value,
        "effective": "classical",
        "fallback_reason": str(status.get("code") or "pose_model_unavailable"),
    }
