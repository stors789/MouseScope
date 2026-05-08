# -*- mode: python ; coding: utf-8 -*-
"""
MouseScope Desktop — PyInstaller 打包配置
用法:  pyinstaller MouseScope.spec
"""

import os
import sys

block_cipher = None

# 项目根目录
PROJECT_DIR = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    [os.path.join(PROJECT_DIR, 'app.py')],
    pathex=[PROJECT_DIR, os.path.join(PROJECT_DIR, 'core')],
    binaries=[],
    datas=[
        # 模板和静态资源
        (os.path.join(PROJECT_DIR, 'templates'), 'templates'),
        (os.path.join(PROJECT_DIR, 'static'), 'static'),
        # core 模块
        (os.path.join(PROJECT_DIR, 'core'), 'core'),
        # 许可证
        (os.path.join(PROJECT_DIR, 'LICENSE'), '.'),
    ],
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
        'maze_zone_utils',
        'spatial_render',
        'file_utils',
        'vision_tracker',
        'performance_utils',
        'core',
        'core.forced_swim',
        'core.open_field',
        'core.novel_object',
        'core.elevated_plus',
        'core.y_maze',
        'core.maze_zone_utils',
        'core.spatial_render',
        'core.file_utils',
        'core.vision_tracker',
        'core.performance_utils',
        'services',
        'services.analysis_builders',
        'services.analysis_history',
        'services.analysis_runtime',
        'services.analysis_sessions',
        'services.debug_preview',
        'services.elevated_plus_session',
        'services.experiment_registry',
        'services.forced_swim_session',
        'services.novel_object_session',
        'services.open_field_session',
        'services.payload_geometry',
        'services.session_store',
        'services.tst_session',
        'services.y_maze_session',
        'routes',
        'routes.api',
        'routes.pages',
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
