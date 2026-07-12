from __future__ import annotations

import re
from typing import Any


ERROR_MESSAGES: dict[str, tuple[str, str]] = {
    "operation_failed": ("操作失败，请检查输入后重试。", "Operation failed. Check the inputs and try again."),
    "session_not_found": ("会话不存在或已过期。", "The analysis session does not exist or has expired."),
    "project_not_found": ("项目不存在。", "The project does not exist."),
    "group_not_found": ("组别不存在。", "The group does not exist."),
    "subject_not_found": ("小鼠不存在。", "The animal does not exist."),
    "video_not_found": ("视频不存在。", "The video does not exist."),
    "assignment_not_found": ("视频实验记录不存在。", "The video experiment record does not exist."),
    "project_run_not_found": ("项目分析记录不存在。", "The project analysis record does not exist."),
    "task_not_found": ("任务记录不存在。", "The task record does not exist."),
    "task_running": ("任务仍在运行，请先停止分析。", "The task is still running. Stop it before continuing."),
    "analysis_already_running": ("分析任务已在运行，请勿重复启动。", "The analysis is already running."),
    "missing_saved_config": ("当前没有可用于启动分析的已保存配置。", "No saved configuration is available to start this analysis."),
    "no_assignments": ("请至少选择一个视频实验记录。", "Select at least one video experiment record."),
    "mixed_experiments": ("一次项目分析只能包含同一种实验类型。", "One project run can contain only one experiment type."),
    "project_subject_required": (
        "请先在项目中添加小鼠，并为每个完整 ROI 选择对应小鼠。",
        "Add animals to the project and assign one to every complete ROI.",
    ),
    "video_file_missing": ("视频文件不存在。", "The video file does not exist."),
    "no_video_files": ("没有找到可导入的视频文件。", "No supported video files were found."),
    "result_dir_missing": ("结果目录不存在或已移动。", "The result folder does not exist or has been moved."),
    "no_results": ("没有可用的结果文件。", "No result files are available."),
    "snapshot_missing": ("该任务没有可用的运行快照，无法按原配置重跑。", "This task has no usable run snapshot and cannot be replayed exactly."),
    "snapshot_read_failed": ("无法读取运行快照。", "The run snapshot could not be read."),
    "snapshot_write_failed": ("无法写入运行快照，分析未启动。", "The run snapshot could not be written, so analysis was not started."),
    "replay_source_running": ("原任务仍在运行，完成或停止后才能按原配置重跑。", "The source task is still running. Wait for it to finish or stop it before replaying."),
    "replay_project_missing": ("原项目不存在，无法按原项目配置重跑。", "The original project no longer exists, so this project run cannot be replayed."),
    "replay_project_unavailable": ("项目工作台不可用，无法重建该项目任务。", "Project Workspace is unavailable, so this project run cannot be rebuilt."),
    "three_chamber_hidden": ("三箱社交模块未开放。", "The three-chamber social module is not enabled."),
    "invalid_task_limit": ("任务数量参数无效。", "The task limit is invalid."),
    "invalid_video_index": ("视频索引无效。", "The video index is invalid."),
    "invalid_draft": ("草稿数据格式无效。", "The saved configuration format is invalid."),
}


_EXACT_CODE_BY_ZH = {
    "项目不存在": "project_not_found",
    "组别不存在": "group_not_found",
    "小鼠不存在": "subject_not_found",
    "视频不存在": "video_not_found",
    "视频实验记录不存在": "assignment_not_found",
    "项目分析记录不存在": "project_run_not_found",
    "任务记录不存在": "task_not_found",
    "历史记录不存在": "task_not_found",
    "结果目录不存在或已移动": "result_dir_missing",
    "没有结果文件": "no_results",
    "没有可下载的结果文件": "no_results",
    "没有找到可导入的视频文件": "no_video_files",
    "请至少选择一个视频实验记录": "no_assignments",
    "一次项目分析只能包含同一种实验类型": "mixed_experiments",
    "会话不存在": "session_not_found",
    "分析任务已在运行，请勿重复启动。": "analysis_already_running",
    "当前没有可用于启动分析的已保存配置": "missing_saved_config",
}


def _contains_chinese(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value or ""))


def _infer_code(message: str) -> str | None:
    normalized = str(message or "").strip().rstrip("。")
    if normalized in _EXACT_CODE_BY_ZH:
        return _EXACT_CODE_BY_ZH[normalized]
    if normalized.startswith("视频文件不存在"):
        return "video_file_missing"
    if "未绑定项目小鼠" in normalized or "为每个 ROI 选择" in normalized:
        return "project_subject_required"
    if normalized.startswith("无法读取运行快照"):
        return "snapshot_read_failed"
    if normalized.startswith("无法写入运行快照"):
        return "snapshot_write_failed"
    if normalized.startswith("三箱社交模块未开放"):
        return "three_chamber_hidden"
    return None


def build_api_error(
    message_zh: str | None = None,
    *,
    message_en: str | None = None,
    code: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Return a stable bilingual API error payload.

    Chinese remains the default UI language. English clients can consistently
    select ``error_en`` without having to translate arbitrary backend text.
    """

    resolved_code = str(code or "").strip() or _infer_code(str(message_zh or "")) or "operation_failed"
    catalog_zh, catalog_en = ERROR_MESSAGES.get(resolved_code, ERROR_MESSAGES["operation_failed"])
    zh = str(message_zh or catalog_zh)
    if message_en:
        en = str(message_en)
    elif not _contains_chinese(zh):
        en = zh
    else:
        en = catalog_en
    payload: dict[str, Any] = {
        "success": False,
        "error": zh,
        "error_en": en,
        "error_code": resolved_code,
    }
    payload.update(extra)
    return payload
