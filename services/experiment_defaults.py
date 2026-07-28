from __future__ import annotations

import copy
import math
from typing import Any


# These are the application's factory defaults.  They intentionally contain
# only analysis/configuration fields (never ROI geometry or user data).
BUILTIN_EXPERIMENT_DEFAULTS: dict[str, dict[str, Any]] = {
    "tst": {
        "bg_method": "threshold", "bg_type": "light", "gray_threshold": 127,
        "erosion_kernel": 3, "erosion_iter": 2, "dilation_kernel": 5,
        "dilation_iter": 3, "min_area": 100, "max_area": 50000,
        "auto_behavior_threshold": False, "behavior_threshold_sensitivity": 0.0,
        # The optimized/classic choice is one canonical switch. The inverse
        # legacy flag and runtime enable flag are derived by the builder so the
        # global page cannot save contradictory combinations.
        "use_tst_lis_primary": False, "pixel_to_mm": 1.0, "generate_video": True,
    },
    "forced_swim": {
        "bg_method": "threshold", "bg_type": "light", "gray_threshold": 127,
        "erosion_kernel": 3, "erosion_iter": 1, "dilation_kernel": 5,
        "dilation_iter": 2, "min_area": 80, "max_area": 50000,
        "cylinder_diameter_cm": 10.0, "auto_threshold": True,
        "immobility_threshold": None, "immobility_high_threshold": None,
        "threshold_sensitivity": 0.0, "smoothing_window_s": 0.5,
        "min_immobility_duration_s": 1.0, "min_active_duration_s": 0.3,
        "gap_merge_s": 0.5, "bin_seconds": 60.0, "generate_video": True,
    },
    "water_maze": {
        "trial_type": "acquisition_hidden", "day_label": "", "trial_no": 1,
        "pool_diameter_cm": 120.0, "platform_radius_cm": 5.0,
        "target_quadrant": "", "start_position": "N", "quadrant_rotation_deg": 0.0,
        "correction_mode": "ellipse_affine", "distortion_k1": 0.0,
        "distortion_k2": 0.0, "thigmotaxis_band_cm": 10.0,
        "platform_dwell_s": 1.0, "track_smoothing_max_jump_cm": 25.0,
        "max_gap_s": 0.5, "movement_threshold_cm_s": 2.0,
        "whishaw_corridor_width_cm": None, "exclude_platform_from_detection": False,
        "bin_seconds": 10.0, "bg_method": "threshold", "bg_type": "light",
        "gray_threshold": 127, "erosion_kernel": 3, "erosion_iter": 1,
        "dilation_kernel": 5, "dilation_iter": 2, "min_area": 80,
        "max_area": 50000, "generate_video": True,
    },
    "open_field": {
        "tracking_backend": "classical", "pose_confidence_threshold": 0.8,
        "bg_method": "threshold", "bg_type": "light", "gray_threshold": 127,
        "erosion_kernel": 3, "erosion_iter": 1, "dilation_kernel": 5,
        "dilation_iter": 2, "min_area": 120, "max_area": 50000,
        "arena_side_cm": 50.0, "center_ratio_x": 0.5, "center_ratio_y": 0.5,
        "wall_margin_cm": 8.0, "movement_threshold_cm_s": 2.0,
        "bin_seconds": 300.0, "generate_video": True,
        "temporal_bg_samples": 90, "temporal_diff_threshold": 25,
    },
    "novel_object": {
        "tracking_backend": "classical", "pose_confidence_threshold": 0.8,
        "phase": "recognition", "bg_method": "threshold", "bg_type": "light",
        "gray_threshold": 127, "body_gray_threshold": 95, "erosion_kernel": 3,
        "erosion_iter": 1, "dilation_kernel": 5, "dilation_iter": 2,
        "min_area": 120, "max_area": 50000, "arena_width_cm": 50.0,
        "arena_height_cm": 50.0, "center_ratio_x": 0.5, "center_ratio_y": 0.5,
        "interaction_radius_cm": 4.0, "exploration_max_speed_cm_s": 12.0,
        "use_head_direction": True, "head_angle_threshold_deg": 50.0,
        "head_confidence_threshold": 0.35, "movement_threshold_cm_s": 2.0,
        "min_valid_exploration_s": 20.0, "bin_seconds": 300.0,
        "generate_video": True,
        "temporal_bg_samples": 90, "temporal_diff_threshold": 25,
    },
    "elevated_plus": {
        "bg_method": "threshold", "bg_type": "light", "gray_threshold": 127,
        "erosion_kernel": 3, "erosion_iter": 1, "dilation_kernel": 5,
        "dilation_iter": 2, "min_area": 120, "max_area": 50000,
        "center_size_cm": 5.0, "open_arm_length_cm": 30.0,
        "closed_arm_length_cm": 30.0, "open_axis": "horizontal",
        "movement_threshold_cm_s": 2.0, "entry_depth_ratio": 0.2,
        "bin_seconds": 60.0, "generate_video": True,
    },
    "y_maze": {
        "bg_method": "threshold", "bg_type": "light", "gray_threshold": 127,
        "erosion_kernel": 3, "erosion_iter": 1, "dilation_kernel": 5,
        "dilation_iter": 2, "min_area": 120, "max_area": 50000,
        "arm_length_cm": 30.0, "arm_width_cm": 10.0,
        "movement_threshold_cm_s": 2.0, "entry_depth_ratio": 0.2,
        "tracking_max_jump_cm": 12.0, "occlusion_hold_s": 0.4,
        "zone_confirm_s": 0.12, "center_reset_s": 0.2,
        "require_center_passage": True,
        "temporal_bg_samples": 90, "temporal_diff_threshold": 25,
        "bin_seconds": 60.0, "generate_video": True,
    },
    "light_dark_box": {
        "bg_method": "threshold", "bg_type": "light", "gray_threshold": 127,
        "erosion_kernel": 3, "erosion_iter": 1, "dilation_kernel": 5,
        "dilation_iter": 2, "min_area": 120, "max_area": 50000,
        "arena_width_cm": 40.0, "arena_height_cm": 40.0,
        "light_box_length_cm": 40.0, "light_box_width_cm": 20.0,
        "dark_box_length_cm": 40.0, "dark_box_width_cm": 20.0,
        "shared_edge_length_cm": 40.0, "dark_box_visible": True,
        "door_open_delay_s": 3.0, "transition_min_dwell_s": 0.12,
        "doorway_reset_s": 0.2, "movement_threshold_cm_s": 2.0,
        "bin_seconds": 60.0, "generate_video": True,
    },
    "three_chamber": {
        "analysis_mode": "full_protocol", "selected_phase": "sociability",
        "phase": "sociability", "protocol_segments": [
            {"id": "habituation_center", "phase": "habituation_center", "start_s": 0, "end_s": 300, "analyze": True},
            {"id": "gap_1", "phase": "ignore", "start_s": 300, "end_s": 420, "analyze": False},
            {"id": "habituation_all", "phase": "habituation_all", "start_s": 420, "end_s": 1020, "analyze": True},
            {"id": "gap_2", "phase": "ignore", "start_s": 1020, "end_s": 1140, "analyze": False},
            {"id": "sociability", "phase": "sociability", "start_s": 1140, "end_s": 1740, "analyze": True},
            {"id": "gap_3", "phase": "ignore", "start_s": 1740, "end_s": 1860, "analyze": False},
            {"id": "social_novelty", "phase": "social_novelty", "start_s": 1860, "end_s": 2460, "analyze": True},
        ],
        "role_schedule": {
            "habituation_center": {"left": "", "right": ""},
            "habituation_all": {"left": "empty_object", "right": "empty_object"},
            "sociability": {"left": "social_stranger_1", "right": "empty_object"},
            "social_novelty": {"left": "familiar_stranger_1", "right": "novel_stranger_2"},
        },
        "bg_method": "threshold", "bg_type": "light", "gray_threshold": 127,
        "erosion_kernel": 3, "erosion_iter": 1, "dilation_kernel": 5,
        "dilation_iter": 2, "min_area": 120, "max_area": 50000,
        "arena_width_cm": 60.0, "arena_height_cm": 40.0,
        "interaction_radius_cm": 2.0, "cup_exclusion_radius_cm": 1.5,
        "use_head_direction": False, "head_angle_threshold_deg": 60.0,
        "head_confidence_threshold": 0.35, "transition_min_dwell_s": 0.12,
        "center_reset_s": 0.2, "movement_threshold_cm_s": 2.0,
        "bin_seconds": 60.0, "generate_video": True,
    },
}


_FIELD_LABELS: dict[str, tuple[str, str]] = {
    "bg_method": ("背景建模方法", "Background method"), "bg_type": ("背景类型", "Background type"),
    "gray_threshold": ("灰度阈值", "Gray threshold"), "body_gray_threshold": ("身体灰度阈值", "Body gray threshold"),
    "erosion_kernel": ("腐蚀核大小", "Erosion kernel"), "erosion_iter": ("腐蚀次数", "Erosion iterations"),
    "dilation_kernel": ("膨胀核大小", "Dilation kernel"), "dilation_iter": ("膨胀次数", "Dilation iterations"),
    "min_area": ("最小检测面积（px²）", "Minimum area (px²)"), "max_area": ("最大检测面积（px²）", "Maximum area (px²)"),
    "generate_video": ("生成标注视频", "Generate annotated video"), "arena_side_cm": ("场地边长（cm）", "Arena side (cm)"),
    "arena_width_cm": ("场地宽度（cm）", "Arena width (cm)"), "arena_height_cm": ("场地高度（cm）", "Arena height (cm)"),
    "center_ratio_x": ("中心区横向比例", "Center width ratio"), "center_ratio_y": ("中心区纵向比例", "Center height ratio"),
    "wall_margin_cm": ("贴壁区宽度（cm）", "Wall zone width (cm)"), "movement_threshold_cm_s": ("运动阈值（cm/s）", "Movement threshold (cm/s)"),
    "bin_seconds": ("时间分箱（s）", "Time-bin size (s)"), "phase": ("默认测试阶段", "Default phase"),
    "interaction_radius_cm": ("探索半径（cm）", "Interaction radius (cm)"), "exploration_max_speed_cm_s": ("探索最大速度（cm/s）", "Exploration max speed (cm/s)"),
    "use_head_direction": ("启用头部朝向", "Use head direction"), "head_angle_threshold_deg": ("头向夹角阈值（°）", "Head-angle threshold (°)"),
    "head_confidence_threshold": ("头向置信度阈值", "Head confidence threshold"), "min_valid_exploration_s": ("最短有效探索（s）", "Minimum valid exploration (s)"),
    "center_size_cm": ("中心区尺寸（cm）", "Center size (cm)"), "open_arm_length_cm": ("开放臂长度（cm）", "Open-arm length (cm)"),
    "closed_arm_length_cm": ("封闭臂长度（cm）", "Closed-arm length (cm)"), "open_axis": ("开放臂方向", "Open-arm axis"),
    "arm_length_cm": ("臂长（cm）", "Arm length (cm)"), "arm_width_cm": ("臂宽（cm）", "Arm width (cm)"),
    "entry_depth_ratio": ("进入深度比例", "Entry-depth ratio"), "cylinder_diameter_cm": ("圆筒直径（cm）", "Cylinder diameter (cm)"),
    "tracking_max_jump_cm": ("最大追踪跳变（cm）", "Maximum tracking jump (cm)"),
    "occlusion_hold_s": ("短时遮挡保持（s）", "Short occlusion hold (s)"),
    "zone_confirm_s": ("区域确认时长（s）", "Zone confirmation time (s)"),
    "require_center_passage": ("入臂切换必须经过中心", "Require center passage between arms"),
    "temporal_bg_samples": ("时序背景抽样帧数", "Temporal background samples"),
    "temporal_diff_threshold": ("时序背景差分阈值", "Temporal difference threshold"),
    "auto_threshold": ("自动不动阈值", "Auto immobility threshold"), "immobility_threshold": ("不动阈值", "Immobility threshold"),
    "immobility_high_threshold": ("高不动阈值", "High immobility threshold"), "threshold_sensitivity": ("阈值灵敏度", "Threshold sensitivity"),
    "smoothing_window_s": ("平滑窗口（s）", "Smoothing window (s)"), "min_immobility_duration_s": ("最短不动时长（s）", "Minimum immobility (s)"),
    "min_active_duration_s": ("最短活动时长（s）", "Minimum activity (s)"), "gap_merge_s": ("间隙合并（s）", "Gap merge (s)"),
    "pixel_to_mm": ("像素到毫米比例", "Pixels-to-mm ratio"), "auto_behavior_threshold": ("自动行为阈值", "Auto behavior threshold"),
    "behavior_threshold_sensitivity": ("行为阈值灵敏度", "Behavior threshold sensitivity"), "use_tst_lis_primary": ("使用微小挣扎优化算法", "Use subtle-struggle algorithm"),
    "use_tst_legacy_primary": ("使用经典行为算法", "Use classic behavior algorithm"), "trial_type": ("试验类型", "Trial type"),
    "day_label": ("试验日标签", "Day label"), "trial_no": ("试验序号", "Trial number"), "pool_diameter_cm": ("水池直径（cm）", "Pool diameter (cm)"),
    "platform_radius_cm": ("平台半径（cm）", "Platform radius (cm)"), "target_quadrant": ("目标象限", "Target quadrant"),
    "start_position": ("起始位置", "Start position"), "quadrant_rotation_deg": ("象限旋转（°）", "Quadrant rotation (°)"),
    "correction_mode": ("透视/畸变校正", "Geometry correction"), "distortion_k1": ("畸变系数 K1", "Distortion K1"),
    "distortion_k2": ("畸变系数 K2", "Distortion K2"), "thigmotaxis_band_cm": ("贴边环宽（cm）", "Thigmotaxis band (cm)"),
    "platform_dwell_s": ("平台停留判定（s）", "Platform dwell (s)"), "track_smoothing_max_jump_cm": ("最大轨迹跳变（cm）", "Max trajectory jump (cm)"),
    "max_gap_s": ("最大补点间隔（s）", "Maximum gap (s)"), "whishaw_corridor_width_cm": ("Whishaw 走廊宽度（cm）", "Whishaw corridor width (cm)"),
    "exclude_platform_from_detection": ("检测时排除平台", "Exclude platform from detection"), "light_box_length_cm": ("明室长度（cm）", "Light-box length (cm)"),
    "light_box_width_cm": ("明室宽度（cm）", "Light-box width (cm)"), "dark_box_length_cm": ("暗室长度（cm）", "Dark-box length (cm)"),
    "dark_box_width_cm": ("暗室宽度（cm）", "Dark-box width (cm)"), "shared_edge_length_cm": ("共用边长度（cm）", "Shared edge (cm)"),
    "dark_box_visible": ("暗室可见", "Dark box visible"), "door_open_delay_s": ("开门延迟（s）", "Door-open delay (s)"),
    "transition_min_dwell_s": ("最短转区停留（s）", "Minimum transition dwell (s)"), "doorway_reset_s": ("门区重置（s）", "Doorway reset (s)"),
    "analysis_mode": ("分析模式", "Analysis mode"), "selected_phase": ("选择阶段", "Selected phase"),
    "protocol_segments": ("完整流程时间轴（JSON）", "Full-protocol timeline (JSON)"), "role_schedule": ("对象角色安排（JSON）", "Object-role schedule (JSON)"),
    "cup_exclusion_radius_cm": ("杯体排除半径（cm）", "Cup exclusion radius (cm)"), "center_reset_s": ("中心区重置（s）", "Center reset (s)"),
}

_SELECT_OPTIONS: dict[str, list[tuple[str, str, str]]] = {
    "tracking_backend": [
        ("classical", "经典视觉", "Classical vision"),
        ("deeplabcut", "DeepLabCut", "DeepLabCut"),
        ("auto", "自动回退", "Automatic fallback"),
    ],
    "bg_method": [("threshold", "阈值", "Threshold"), ("static", "静态背景", "Static background"), ("mog2", "MOG2", "MOG2"), ("knn", "KNN", "KNN")],
    "bg_type": [("light", "浅色背景/深色鼠", "Light background / dark mouse"), ("dark", "深色背景/浅色鼠", "Dark background / light mouse")],
    "phase": [("recognition", "识别期", "Recognition"), ("training", "训练期", "Training")],
    "open_axis": [("horizontal", "水平", "Horizontal"), ("vertical", "垂直", "Vertical")],
    "trial_type": [("acquisition_hidden", "隐藏平台训练", "Hidden-platform acquisition"), ("probe", "探测试验", "Probe"), ("visible_platform", "可见平台", "Visible platform")],
    "start_position": [(item, item, item) for item in ("N", "E", "S", "W")],
    "correction_mode": [("ellipse_affine", "椭圆仿射", "Ellipse affine"), ("none", "不校正", "None")],
    "analysis_mode": [("full_protocol", "完整流程", "Full protocol"), ("single_phase", "单阶段", "Single phase")],
    "selected_phase": [(item, item, item.replace("_", " ").title()) for item in ("habituation_center", "habituation_all", "sociability", "social_novelty")],
}

# Keep the schema text in real UTF-8.  An early generated revision of this
# file contained mojibake labels; updating the mapping here also makes old
# saved defaults render correctly without changing their keys.
_FIELD_LABELS.update({
    "tracking_backend": ("追踪后端", "Tracking backend"),
    "pose_confidence_threshold": ("姿态关键点置信度阈值", "Pose keypoint confidence threshold"),
    "bg_method": ("背景建模方法", "Background method"), "bg_type": ("背景类型", "Background type"),
    "gray_threshold": ("灰度阈值", "Gray threshold"), "body_gray_threshold": ("身体灰度阈值", "Body gray threshold"),
    "erosion_kernel": ("腐蚀核大小", "Erosion kernel"), "erosion_iter": ("腐蚀次数", "Erosion iterations"),
    "dilation_kernel": ("膨胀核大小", "Dilation kernel"), "dilation_iter": ("膨胀次数", "Dilation iterations"),
    "min_area": ("最小检测面积（px²）", "Minimum area (px²)"), "max_area": ("最大检测面积（px²）", "Maximum area (px²)"),
    "generate_video": ("生成标注视频", "Generate annotated video"), "arena_side_cm": ("场地边长（cm）", "Arena side (cm)"),
    "arena_width_cm": ("场地宽度（cm）", "Arena width (cm)"), "arena_height_cm": ("场地高度（cm）", "Arena height (cm)"),
    "center_ratio_x": ("中心区横向比例", "Center width ratio"), "center_ratio_y": ("中心区纵向比例", "Center height ratio"),
    "wall_margin_cm": ("贴壁区宽度（cm）", "Wall zone width (cm)"), "movement_threshold_cm_s": ("运动阈值（cm/s）", "Movement threshold (cm/s)"),
    "bin_seconds": ("时间分箱（s）", "Time-bin size (s)"), "phase": ("默认测试阶段", "Default phase"),
    "interaction_radius_cm": ("探索半径（cm）", "Interaction radius (cm)"), "exploration_max_speed_cm_s": ("探索最大速度（cm/s）", "Exploration max speed (cm/s)"),
    "use_head_direction": ("启用头部朝向", "Use head direction"), "head_angle_threshold_deg": ("头向夹角阈值（°）", "Head-angle threshold (°)"),
    "head_confidence_threshold": ("头向置信度阈值", "Head confidence threshold"), "min_valid_exploration_s": ("最短有效探索（s）", "Minimum valid exploration (s)"),
    "center_size_cm": ("中心区尺寸（cm）", "Center size (cm)"), "open_arm_length_cm": ("开放臂长度（cm）", "Open-arm length (cm)"),
    "closed_arm_length_cm": ("封闭臂长度（cm）", "Closed-arm length (cm)"), "open_axis": ("开放臂方向", "Open-arm axis"),
    "arm_length_cm": ("臂长（cm）", "Arm length (cm)"), "arm_width_cm": ("臂宽（cm）", "Arm width (cm)"),
    "entry_depth_ratio": ("进入深度比例", "Entry-depth ratio"), "cylinder_diameter_cm": ("圆筒直径（cm）", "Cylinder diameter (cm)"),
    "auto_threshold": ("自动不动阈值", "Auto immobility threshold"), "immobility_threshold": ("不动阈值", "Immobility threshold"),
    "immobility_high_threshold": ("高不动阈值", "High immobility threshold"), "threshold_sensitivity": ("阈值灵敏度", "Threshold sensitivity"),
    "smoothing_window_s": ("平滑窗口（s）", "Smoothing window (s)"), "min_immobility_duration_s": ("最短不动时长（s）", "Minimum immobility (s)"),
    "min_active_duration_s": ("最短活动时长（s）", "Minimum activity (s)"), "gap_merge_s": ("间隙合并（s）", "Gap merge (s)"),
    "pixel_to_mm": ("像素到毫米比例", "Pixels-to-mm ratio"), "auto_behavior_threshold": ("自动行为阈值", "Auto behavior threshold"),
    "behavior_threshold_sensitivity": ("行为阈值灵敏度", "Behavior threshold sensitivity"), "use_tst_lis_primary": ("使用微小挣扎优化算法", "Use subtle-struggle algorithm"),
    "use_tst_legacy_primary": ("使用经典行为算法", "Use classic behavior algorithm"), "enable_tst_lis": ("启用微小挣扎分析", "Enable subtle-struggle analysis"),
    "trial_type": ("试验类型", "Trial type"), "day_label": ("试验日标签", "Day label"), "trial_no": ("试验序号", "Trial number"),
    "pool_diameter_cm": ("水池直径（cm）", "Pool diameter (cm)"), "platform_radius_cm": ("平台半径（cm）", "Platform radius (cm)"),
    "target_quadrant": ("目标象限", "Target quadrant"), "start_position": ("起始位置", "Start position"),
    "quadrant_rotation_deg": ("象限旋转（°）", "Quadrant rotation (°)"), "correction_mode": ("透视/畸变校正", "Geometry correction"),
    "distortion_k1": ("畸变系数 K1", "Distortion K1"), "distortion_k2": ("畸变系数 K2", "Distortion K2"),
    "thigmotaxis_band_cm": ("贴边环宽（cm）", "Thigmotaxis band (cm)"), "platform_dwell_s": ("平台停留判定（s）", "Platform dwell (s)"),
    "track_smoothing_max_jump_cm": ("最大轨迹跳变（cm）", "Max trajectory jump (cm)"), "max_gap_s": ("最大补点间隔（s）", "Maximum gap (s)"),
    "whishaw_corridor_width_cm": ("Whishaw 走廊宽度（cm）", "Whishaw corridor width (cm)"), "exclude_platform_from_detection": ("检测时排除平台", "Exclude platform from detection"),
    "light_box_length_cm": ("明室长度（cm）", "Light-box length (cm)"), "light_box_width_cm": ("明室宽度（cm）", "Light-box width (cm)"),
    "dark_box_length_cm": ("暗室长度（cm）", "Dark-box length (cm)"), "dark_box_width_cm": ("暗室宽度（cm）", "Dark-box width (cm)"),
    "shared_edge_length_cm": ("共用边长度（cm）", "Shared edge (cm)"), "dark_box_visible": ("暗室可见", "Dark box visible"),
    "door_open_delay_s": ("开门延迟（s）", "Door-open delay (s)"), "transition_min_dwell_s": ("最短转区停留（s）", "Minimum transition dwell (s)"),
    "doorway_reset_s": ("门区重置（s）", "Doorway reset (s)"), "analysis_mode": ("分析模式", "Analysis mode"),
    "selected_phase": ("选择阶段", "Selected phase"), "protocol_segments": ("完整流程时间轴（JSON）", "Full-protocol timeline (JSON)"),
    "role_schedule": ("对象角色安排（JSON）", "Object-role schedule (JSON)"), "cup_exclusion_radius_cm": ("杯体排除半径（cm）", "Cup exclusion radius (cm)"),
    "center_reset_s": ("中心区重置（s）", "Center reset (s)"),
    "tracking_max_jump_cm": ("最大追踪跳变（cm）", "Maximum tracking jump (cm)"),
    "occlusion_hold_s": ("短时遮挡保持（s）", "Short occlusion hold (s)"),
    "zone_confirm_s": ("区域确认时长（s）", "Zone confirmation time (s)"),
    "require_center_passage": ("入臂切换必须经过中心", "Require center passage between arms"),
    "temporal_bg_samples": ("时序背景抽样帧数", "Temporal background samples"),
    "temporal_diff_threshold": ("时序背景差分阈值", "Temporal difference threshold"),
})

_SELECT_OPTIONS.update({
    "bg_method": [("threshold", "阈值", "Threshold"), ("static", "静态背景", "Static background"), ("mog2", "MOG2", "MOG2"), ("knn", "KNN", "KNN")],
    "bg_type": [("light", "浅色背景/深色鼠", "Light background / dark mouse"), ("dark", "深色背景/浅色鼠", "Dark background / light mouse")],
    "phase": [("recognition", "识别期", "Recognition"), ("training", "训练期", "Training")],
    "open_axis": [("horizontal", "水平", "Horizontal"), ("vertical", "垂直", "Vertical")],
    "trial_type": [("acquisition_hidden", "隐藏平台训练", "Hidden-platform acquisition"), ("probe", "探测试验", "Probe"), ("visible_platform", "可见平台", "Visible platform")],
    "correction_mode": [("ellipse_affine", "椭圆仿射", "Ellipse affine"), ("none", "不校正", "None")],
    "analysis_mode": [("full_protocol", "完整流程", "Full protocol"), ("single_phase", "单阶段", "Single phase")],
})

_EXPERIMENT_SELECT_OPTIONS: dict[tuple[str, str], list[tuple[str, str, str]]] = {
    ("y_maze", "bg_method"): [
        ("threshold", "阈值", "Threshold"),
        ("static", "静态背景", "Static background"),
        ("mog2", "MOG2", "MOG2"),
        ("knn", "KNN", "KNN"),
        ("temporal", "时序背景", "Temporal background"),
    ],
    ("open_field", "bg_method"): [
        ("threshold", "阈值", "Threshold"),
        ("static", "静态背景", "Static background"),
        ("mog2", "MOG2", "MOG2"),
        ("knn", "KNN", "KNN"),
        ("temporal", "时序背景", "Temporal background"),
    ],
    ("novel_object", "bg_method"): [
        ("threshold", "阈值", "Threshold"),
        ("static", "静态背景", "Static background"),
        ("mog2", "MOG2", "MOG2"),
        ("knn", "KNN", "KNN"),
        ("temporal", "时序背景", "Temporal background"),
    ],
    ("three_chamber", "phase"): [
        ("habituation_center", "中央室适应", "Center habituation"),
        ("habituation_all", "全场适应", "Full-apparatus habituation"),
        ("sociability", "社交偏好", "Sociability"),
        ("social_novelty", "社会新颖性", "Social novelty"),
    ],
}

_POSITIVE_CM_FIELDS = {
    "arena_side_cm", "arena_width_cm", "arena_height_cm", "wall_margin_cm", "interaction_radius_cm",
    "center_size_cm", "open_arm_length_cm", "closed_arm_length_cm", "arm_length_cm", "arm_width_cm",
    "cylinder_diameter_cm", "pool_diameter_cm", "platform_radius_cm", "thigmotaxis_band_cm",
    "track_smoothing_max_jump_cm", "whishaw_corridor_width_cm", "light_box_length_cm",
    "light_box_width_cm", "dark_box_length_cm", "dark_box_width_cm", "shared_edge_length_cm",
    "cup_exclusion_radius_cm",
    "tracking_max_jump_cm",
}
_NONNEGATIVE_SECONDS_FIELDS = {
    "smoothing_window_s", "min_immobility_duration_s", "min_active_duration_s", "gap_merge_s",
    "min_valid_exploration_s", "platform_dwell_s", "max_gap_s", "door_open_delay_s",
    "transition_min_dwell_s", "doorway_reset_s", "center_reset_s",
    "occlusion_hold_s", "zone_confirm_s",
}
_FIELD_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "gray_threshold": {"min": 0, "max": 255, "step": 1}, "body_gray_threshold": {"min": 0, "max": 255, "step": 1},
    "erosion_kernel": {"min": 1, "max": 99, "step": 1}, "dilation_kernel": {"min": 1, "max": 99, "step": 1},
    "erosion_iter": {"min": 0, "max": 50, "step": 1}, "dilation_iter": {"min": 0, "max": 50, "step": 1},
    "min_area": {"min": 1, "max": 1_000_000_000, "step": 1}, "max_area": {"min": 1, "max": 1_000_000_000, "step": 1},
    "temporal_bg_samples": {"min": 5, "max": 500, "step": 1},
    "temporal_diff_threshold": {"min": 1, "max": 255, "step": 1},
    "center_ratio_x": {"min": 0.01, "max": 1.0, "step": 0.01}, "center_ratio_y": {"min": 0.01, "max": 1.0, "step": 0.01},
    "pose_confidence_threshold": {"min": 0.0, "max": 1.0, "step": 0.01},
    "entry_depth_ratio": {"min": 0.0, "max": 1.0, "step": 0.01}, "head_confidence_threshold": {"min": 0.0, "max": 1.0, "step": 0.01},
    "head_angle_threshold_deg": {"min": 0.0, "max": 180.0, "step": 1},
    "behavior_threshold_sensitivity": {"min": -3.0, "max": 3.0, "step": 0.1}, "threshold_sensitivity": {"min": -3.0, "max": 3.0, "step": 0.1},
    "immobility_threshold": {"min": 0.000001, "max": 1_000_000.0, "step": 0.001}, "immobility_high_threshold": {"min": 0.000001, "max": 1_000_000.0, "step": 0.001},
    "pixel_to_mm": {"min": 0.000001, "max": 1_000_000.0, "step": 0.001},
    "movement_threshold_cm_s": {"min": 0.0, "max": 10_000.0, "step": 0.1}, "exploration_max_speed_cm_s": {"min": 0.0, "max": 10_000.0, "step": 0.1},
    "bin_seconds": {"min": 0.001, "max": 86_400.0, "step": 0.1}, "trial_no": {"min": 1, "max": 100_000, "step": 1},
    "quadrant_rotation_deg": {"min": -360.0, "max": 360.0, "step": 1},
    "distortion_k1": {"min": -10.0, "max": 10.0, "step": 0.001}, "distortion_k2": {"min": -10.0, "max": 10.0, "step": 0.001},
}
for _key in _POSITIVE_CM_FIELDS:
    _FIELD_CONSTRAINTS.setdefault(_key, {"min": 0.001, "max": 10_000.0, "step": 0.1})
for _key in _NONNEGATIVE_SECONDS_FIELDS:
    _FIELD_CONSTRAINTS.setdefault(_key, {"min": 0.0, "max": 86_400.0, "step": 0.01})


def _select_options(experiment: str, key: str) -> list[tuple[str, str, str]]:
    return _EXPERIMENT_SELECT_OPTIONS.get((experiment, key), _SELECT_OPTIONS.get(key, []))


def _field_group(key: str) -> str:
    if key in {"tracking_backend", "pose_confidence_threshold", "bg_method", "bg_type", "gray_threshold", "body_gray_threshold", "erosion_kernel", "erosion_iter", "dilation_kernel", "dilation_iter", "min_area", "max_area", "temporal_bg_samples", "temporal_diff_threshold", "exclude_platform_from_detection"}:
        return "detection"
    if key in {"arena_side_cm", "arena_width_cm", "arena_height_cm", "center_size_cm", "open_arm_length_cm", "closed_arm_length_cm", "arm_length_cm", "arm_width_cm", "cylinder_diameter_cm", "pool_diameter_cm", "platform_radius_cm", "light_box_length_cm", "light_box_width_cm", "dark_box_length_cm", "dark_box_width_cm", "shared_edge_length_cm", "open_axis"}:
        return "geometry"
    if key in {"protocol_segments", "role_schedule", "analysis_mode", "selected_phase", "phase", "trial_type", "day_label", "trial_no", "target_quadrant", "start_position", "quadrant_rotation_deg"}:
        return "protocol"
    return "analysis"


def experiment_defaults_schema() -> dict[str, Any]:
    modules: dict[str, list[dict[str, Any]]] = {}
    for experiment, defaults in BUILTIN_EXPERIMENT_DEFAULTS.items():
        fields: list[dict[str, Any]] = []
        for key, value in defaults.items():
            label_zh, label_en = _FIELD_LABELS.get(key, (key, key.replace("_", " ").title()))
            options = _select_options(experiment, key)
            if options:
                field_type = "select"
            elif isinstance(value, bool):
                field_type = "bool"
            elif isinstance(value, (dict, list)):
                field_type = "json"
            elif isinstance(value, int) and not isinstance(value, bool):
                field_type = "int"
            elif isinstance(value, float) or value is None:
                field_type = "number"
            else:
                field_type = "text"
            field = {
                "key": key, "type": field_type, "group": _field_group(key),
                "label": {"zh": label_zh, "en": label_en},
                "nullable": value is None or value == "",
                "options": [{"value": option_value, "zh": zh, "en": en} for option_value, zh, en in options],
            }
            field.update(copy.deepcopy(_FIELD_CONSTRAINTS.get(key, {})))
            fields.append(field)
        modules[experiment] = fields
    return {"modules": modules}


def _normalise_scalar(value: Any, factory_value: Any) -> Any:
    if isinstance(factory_value, bool):
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(value)
    if isinstance(factory_value, int) and not isinstance(factory_value, bool):
        try:
            parsed = int(round(float(value)))
        except (TypeError, ValueError):
            return factory_value
        return parsed if math.isfinite(float(parsed)) else factory_value
    if isinstance(factory_value, float) or factory_value is None:
        if value in (None, ""):
            return None if factory_value is None else factory_value
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return factory_value
        return parsed if math.isfinite(parsed) else factory_value
    if isinstance(factory_value, (dict, list)):
        return copy.deepcopy(value) if isinstance(value, type(factory_value)) else copy.deepcopy(factory_value)
    return str(value) if value is not None else factory_value


def normalize_experiment_defaults(payload: Any) -> dict[str, dict[str, Any]]:
    raw = payload if isinstance(payload, dict) else {}
    normalized: dict[str, dict[str, Any]] = {}
    for experiment, factory in BUILTIN_EXPERIMENT_DEFAULTS.items():
        incoming = raw.get(experiment) if isinstance(raw.get(experiment), dict) else {}
        normalized[experiment] = {
            key: _normalise_scalar(incoming.get(key, default), default)
            for key, default in factory.items()
        }
    return normalized


class ExperimentDefaultsValidationError(ValueError):
    def __init__(self, errors: list[dict[str, str]]) -> None:
        self.errors = errors
        first = errors[0] if errors else {}
        super().__init__(first.get("message") or "实验默认配置无效")

    @property
    def message_en(self) -> str:
        first = self.errors[0] if self.errors else {}
        return first.get("message_en") or "Invalid experiment default configuration."


def _validation_item(experiment: str, key: str, zh: str, en: str) -> dict[str, str]:
    return {"experiment": experiment, "field": key, "message": zh, "message_en": en}


def _validate_protocol_segments(experiment: str, value: Any) -> list[dict[str, str]]:
    key = "protocol_segments"
    if not isinstance(value, list) or not value:
        return [_validation_item(experiment, key, "完整流程时间轴必须是非空数组", "The full-protocol timeline must be a non-empty array.")]
    errors: list[dict[str, str]] = []
    previous_end: float | None = None
    for index, segment in enumerate(value, start=1):
        if not isinstance(segment, dict):
            errors.append(_validation_item(experiment, key, f"时间段 {index} 必须是对象", f"Protocol segment {index} must be an object."))
            continue
        if not str(segment.get("id") or "").strip() or not str(segment.get("phase") or "").strip():
            errors.append(_validation_item(experiment, key, f"时间段 {index} 缺少 id 或 phase", f"Protocol segment {index} requires id and phase."))
            continue
        try:
            start = float(segment.get("start_s"))
            end = float(segment.get("end_s"))
        except (TypeError, ValueError):
            errors.append(_validation_item(experiment, key, f"时间段 {index} 的起止时间必须是数字", f"Protocol segment {index} start/end must be numeric."))
            continue
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            errors.append(_validation_item(experiment, key, f"时间段 {index} 必须满足 0 ≤ 开始时间 < 结束时间", f"Protocol segment {index} must satisfy 0 ≤ start < end."))
            continue
        if previous_end is not None and start < previous_end:
            errors.append(_validation_item(experiment, key, f"时间段 {index} 与前一个时间段重叠", f"Protocol segment {index} overlaps the previous segment."))
        previous_end = end
    return errors


def _validate_role_schedule(experiment: str, value: Any) -> list[dict[str, str]]:
    key = "role_schedule"
    if not isinstance(value, dict) or not value:
        return [_validation_item(experiment, key, "对象角色安排必须是非空对象", "The object-role schedule must be a non-empty object.")]
    for phase, roles in value.items():
        if not str(phase or "").strip() or not isinstance(roles, dict):
            return [_validation_item(experiment, key, "对象角色安排中的阶段格式无效", "A role-schedule phase has an invalid format.")]
        if "left" not in roles or "right" not in roles:
            return [_validation_item(experiment, key, f"阶段 {phase} 必须同时包含 left 和 right", f"Role phase {phase} must contain both left and right.")]
        if not isinstance(roles.get("left"), str) or not isinstance(roles.get("right"), str):
            return [_validation_item(experiment, key, f"阶段 {phase} 的角色必须是文本", f"Roles for phase {phase} must be text values.")]
    return []


def validate_experiment_defaults(payload: Any) -> list[dict[str, str]]:
    raw = payload if isinstance(payload, dict) else {}
    errors: list[dict[str, str]] = []
    for experiment, factory in BUILTIN_EXPERIMENT_DEFAULTS.items():
        incoming = raw.get(experiment)
        if incoming is None:
            continue
        if not isinstance(incoming, dict):
            errors.append(_validation_item(experiment, "", "实验模块配置必须是对象", "Experiment module defaults must be an object."))
            continue

        for key, factory_value in factory.items():
            if key not in incoming:
                continue
            value = incoming.get(key)
            label_zh, label_en = _FIELD_LABELS.get(key, (key, key.replace("_", " ").title()))
            if value in (None, ""):
                if factory_value not in (None, ""):
                    errors.append(_validation_item(experiment, key, f"{label_zh}不能为空", f"{label_en} cannot be empty."))
                continue

            options = _select_options(experiment, key)
            if options:
                allowed = {str(item[0]) for item in options}
                if str(value) not in allowed:
                    errors.append(_validation_item(experiment, key, f"{label_zh}不是支持的选项", f"{label_en} is not a supported option."))
                continue

            if isinstance(factory_value, bool):
                valid_bool = isinstance(value, bool) or value in (0, 1)
                if isinstance(value, str):
                    valid_bool = value.strip().lower() in {"0", "1", "true", "false", "yes", "no", "on", "off"}
                if not valid_bool:
                    errors.append(_validation_item(experiment, key, f"{label_zh}必须是开关值", f"{label_en} must be a boolean value."))
                continue

            if isinstance(factory_value, (int, float)) or factory_value is None:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    errors.append(_validation_item(experiment, key, f"{label_zh}必须是数字", f"{label_en} must be numeric."))
                    continue
                if not math.isfinite(number):
                    errors.append(_validation_item(experiment, key, f"{label_zh}必须是有限数字", f"{label_en} must be finite."))
                    continue
                if isinstance(factory_value, int) and not isinstance(factory_value, bool) and not number.is_integer():
                    errors.append(_validation_item(experiment, key, f"{label_zh}必须是整数", f"{label_en} must be an integer."))
                    continue
                constraint = _FIELD_CONSTRAINTS.get(key, {})
                minimum = constraint.get("min")
                maximum = constraint.get("max")
                if minimum is not None and number < float(minimum):
                    errors.append(_validation_item(experiment, key, f"{label_zh}不能小于 {minimum}", f"{label_en} cannot be less than {minimum}."))
                elif maximum is not None and number > float(maximum):
                    errors.append(_validation_item(experiment, key, f"{label_zh}不能大于 {maximum}", f"{label_en} cannot exceed {maximum}."))
                continue

            if isinstance(factory_value, list):
                if key == "protocol_segments":
                    errors.extend(_validate_protocol_segments(experiment, value))
                elif not isinstance(value, list):
                    errors.append(_validation_item(experiment, key, f"{label_zh}必须是数组", f"{label_en} must be an array."))
                continue
            if isinstance(factory_value, dict):
                if key == "role_schedule":
                    errors.extend(_validate_role_schedule(experiment, value))
                elif not isinstance(value, dict):
                    errors.append(_validation_item(experiment, key, f"{label_zh}必须是对象", f"{label_en} must be an object."))

        merged = copy.deepcopy(factory)
        merged.update({key: value for key, value in incoming.items() if key in factory})
        try:
            min_area = float(merged.get("min_area"))
            max_area = float(merged.get("max_area"))
            if math.isfinite(min_area) and math.isfinite(max_area) and min_area > max_area:
                errors.append(_validation_item(experiment, "min_area", "最小检测面积不能大于最大检测面积", "Minimum detection area cannot exceed maximum detection area."))
        except (TypeError, ValueError):
            pass
        try:
            if experiment == "open_field" and float(merged["wall_margin_cm"]) * 2 >= float(merged["arena_side_cm"]):
                errors.append(_validation_item(experiment, "wall_margin_cm", "贴壁区宽度必须小于场地边长的一半", "Wall-zone width must be less than half the arena side."))
            if experiment == "water_maze":
                pool_radius = float(merged["pool_diameter_cm"]) / 2.0
                if float(merged["platform_radius_cm"]) >= pool_radius:
                    errors.append(_validation_item(experiment, "platform_radius_cm", "平台半径必须小于水池半径", "Platform radius must be smaller than the pool radius."))
                if float(merged["thigmotaxis_band_cm"]) >= pool_radius:
                    errors.append(_validation_item(experiment, "thigmotaxis_band_cm", "贴边环宽度必须小于水池半径", "Thigmotaxis-band width must be smaller than the pool radius."))
            if experiment == "light_dark_box":
                shared = float(merged["shared_edge_length_cm"])
                light_max = max(float(merged["light_box_length_cm"]), float(merged["light_box_width_cm"]))
                dark_max = max(float(merged["dark_box_length_cm"]), float(merged["dark_box_width_cm"]))
                if shared > min(light_max, dark_max):
                    errors.append(_validation_item(experiment, "shared_edge_length_cm", "共用边长度不能超过明室或暗室的对应最大边长", "Shared-edge length cannot exceed the available edge of either box."))
        except (KeyError, TypeError, ValueError):
            pass
        if experiment == "forced_swim":
            low = merged.get("immobility_threshold")
            high = merged.get("immobility_high_threshold")
            try:
                if low not in (None, "") and high not in (None, "") and float(high) <= float(low):
                    errors.append(_validation_item(experiment, "immobility_high_threshold", "高不动阈值必须大于不动阈值", "High immobility threshold must exceed the low threshold."))
            except (TypeError, ValueError):
                pass
    return errors


def ensure_valid_experiment_defaults(payload: Any) -> None:
    errors = validate_experiment_defaults(payload)
    if errors:
        raise ExperimentDefaultsValidationError(errors)


def safe_experiment_defaults(payload: Any) -> dict[str, dict[str, Any]]:
    """Normalize persisted data and replace unsafe legacy values with defaults."""

    normalized = normalize_experiment_defaults(payload)
    for _attempt in range(3):
        errors = validate_experiment_defaults(normalized)
        if not errors:
            break
        for error in errors:
            experiment = error.get("experiment") or ""
            key = error.get("field") or ""
            factory = BUILTIN_EXPERIMENT_DEFAULTS.get(experiment)
            if not factory:
                continue
            if not key:
                normalized[experiment] = copy.deepcopy(factory)
                continue
            reset_keys = {key}
            if key == "min_area":
                reset_keys.update({"min_area", "max_area"})
            if key == "immobility_high_threshold":
                reset_keys.update({"immobility_threshold", "immobility_high_threshold"})
            if key == "wall_margin_cm":
                reset_keys.update({"wall_margin_cm", "arena_side_cm"})
            if key in {"platform_radius_cm", "thigmotaxis_band_cm"}:
                reset_keys.update({"pool_diameter_cm", "platform_radius_cm", "thigmotaxis_band_cm"})
            if key == "shared_edge_length_cm":
                reset_keys.update({
                    "shared_edge_length_cm", "light_box_length_cm", "light_box_width_cm",
                    "dark_box_length_cm", "dark_box_width_cm",
                })
            for reset_key in reset_keys:
                if reset_key in factory:
                    normalized[experiment][reset_key] = copy.deepcopy(factory[reset_key])
    return normalized


def get_experiment_defaults(payload: Any, experiment_type: str | None) -> dict[str, Any]:
    normalized = normalize_experiment_defaults(payload)
    key = str(experiment_type or "").strip().lower()
    return copy.deepcopy(normalized.get(key, {}))


def merge_experiment_defaults(
    experiment_type: str | None,
    data: dict[str, Any] | None,
    defaults_payload: Any,
) -> dict[str, Any]:
    """Overlay user/video input on the current global defaults.

    The same global baseline is added to each per-video payload, so a video
    can override only the fields it needs without accidentally falling back to
    a different hard-coded value.
    """
    defaults = get_experiment_defaults(defaults_payload, experiment_type)
    result = copy.deepcopy(defaults)
    incoming = copy.deepcopy(data) if isinstance(data, dict) else {}
    raw_per_video = incoming.pop("per_video_configs", None)
    result.update(incoming)
    if isinstance(raw_per_video, list):
        merged_configs: list[dict[str, Any]] = []
        for raw_config in raw_per_video:
            config = copy.deepcopy(defaults)
            if isinstance(raw_config, dict):
                config.update(copy.deepcopy(raw_config))
            merged_configs.append(config)
        result["per_video_configs"] = merged_configs
    return result
