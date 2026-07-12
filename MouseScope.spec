# -*- mode: python ; coding: utf-8 -*-
"""
MouseScope Desktop — PyInstaller 打包配置
用法:  pyinstaller MouseScope.spec
"""

import os
import sys
import shutil
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# 项目根目录
PROJECT_DIR = os.path.dirname(os.path.abspath(SPEC))

try:
    IMAGEIO_FFMPEG_DATAS = collect_data_files('imageio_ffmpeg')
except Exception:
    IMAGEIO_FFMPEG_DATAS = []

def _collect_optional_binary(env_name, executable_names):
    candidates = []
    env_path = os.environ.get(env_name)
    if env_path:
        candidates.append(env_path)
    for executable_name in executable_names:
        found = shutil.which(executable_name)
        if found:
            candidates.append(found)
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return [(candidate, '.')]
    return []

FFPROBE_BINARIES = _collect_optional_binary('MOUSESCOPE_FFPROBE_EXE', ['ffprobe.exe', 'ffprobe'])

a = Analysis(
    [os.path.join(PROJECT_DIR, 'app.py')],
    pathex=[PROJECT_DIR, os.path.join(PROJECT_DIR, 'core')],
    binaries=FFPROBE_BINARIES,
    datas=[
        # 模板和静态资源
        (os.path.join(PROJECT_DIR, 'templates'), 'templates'),
        (os.path.join(PROJECT_DIR, 'static'), 'static'),
        # core 模块
        (os.path.join(PROJECT_DIR, 'core'), 'core'),
        # 许可证
        (os.path.join(PROJECT_DIR, 'LICENSE'), '.'),
    ] + IMAGEIO_FFMPEG_DATAS,
    hiddenimports=[
        'flask',
        'jinja2',
        'markupsafe',
        'cv2',
        'numpy',
        'pandas',
        'PIL',
        'encodings.idna',
        'stringprep',
        'forced_swim',
        'open_field',
        'novel_object',
        'elevated_plus',
        'y_maze',
        'light_dark_box',
        'three_chamber',
        'water_maze',
        'maze_zone_utils',
        'spatial_render',
        'file_utils',
        'vision_tracker',
        'tst_lis',
        'tst_legacy',
        'performance_utils',
        'core',
        'core.forced_swim',
        'core.open_field',
        'core.novel_object',
        'core.elevated_plus',
        'core.y_maze',
        'core.light_dark_box',
        'core.three_chamber',
        'core.water_maze',
        'core.maze_zone_utils',
        'core.spatial_render',
        'core.file_utils',
        'core.vision_tracker',
        'core.tst_lis',
        'core.tst_legacy',
        'core.performance_utils',
        'services',
        'services.analysis_builders',
        'services.analysis_history',
        'services.analysis_runtime',
        'services.analysis_sessions',
        'services.atomic_json',
        'services.api_errors',
        'services.debug_preview',
        'services.experiment_defaults',
        'services.experiment_defaults_store',
        'services.elevated_plus_session',
        'services.experiment_registry',
        'services.forced_swim_session',
        'services.light_dark_session',
        'services.novel_object_session',
        'services.output_runs',
        'services.open_field_session',
        'services.payload_geometry',
        'services.project_store',
        'services.resource_policy',
        'services.reproducibility',
        'services.session_store',
        'services.tst_session',
        'services.video_export',
        'services.video_frame_reader',
        'services.video_metadata',
        'services.video_orientation',
        'services.video_timebase',
        'services.three_chamber_session',
        'services.water_maze_session',
        'services.y_maze_session',
        'imageio_ffmpeg',
        'imageio_ffmpeg._utils',
        'routes',
        'routes.api',
        'routes.pages',
        'routes.projects',
        'routes.project_api',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不需要的大模块，减小体积
        'tkinter',
        'matplotlib',
        'scipy',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MouseScope',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,       # 保留控制台窗口，方便看日志；发布时可改 False
    # icon='static/img/icon.ico',  # 如果有图标，取消注释
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MouseScope',
)
