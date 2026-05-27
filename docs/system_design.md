# MouseScope 系统信息架构文档

> 当前基线：v3.1.0 时间轴、TST LIS 与项目工作区版本
> 文档状态：按 2026-05-18 代码实现更新
> 适用范围：项目真实架构、主要流程、运行时数据、打包和验证路径

## 1. 文档定位

本文是 MouseScope 当前代码架构的中文主文档，用来回答“系统由哪些部分组成、数据怎样流动、各实验模块怎样接入、哪些目录承担什么职责”。

需要特别区分：

- `docs/system_design.md`：当前系统信息架构的主文档，应跟随真实代码更新。
- `docs/plans/*.md`：历史规划和阶段计划，可作为背景材料，但不是当前实现的唯一依据。
- `DESIGN.md`：外部下载的英文视觉/界面规则参考，不是 MouseScope 系统架构文档。
- `BUILD_GUIDE.md`：打包和发布说明。
- `README.md`：面向使用者和开发者的入口说明。

## 2. 产品形态

MouseScope 是一个本地运行的鼠标行为视频分析平台。当前实现是 Flask Web 应用，但使用方式接近桌面软件：用户在本机启动 `app.py`，浏览器访问 `http://127.0.0.1:8080` 完成上传、校准、分析、监控和结果导出。

系统当前支持 6 类实验：

- TST：Tail Suspension Test，悬尾实验。
- FST：Forced Swim Test，强迫游泳。
- OFT：Open Field Test，开放场。
- NOR：Novel Object Recognition，新物体识别。
- EPM：Elevated Plus Maze，高架十字迷宫。
- Y-Maze：Y 迷宫。

当前有两条主要使用路径：

- 快速分析：适合直接上传一个或多个视频，完成校准后立即分析。
- 项目工作区：适合按项目、分组、受试个体、视频、实验类型组织长期数据，并保留运行记录。

系统不依赖云端账号、远程服务或多用户权限体系。`data/` 下的上传、结果、项目数据库和历史记录是本机用户数据。

## 3. 总体架构

整体调用链可以概括为：

```text
Browser UI
  -> Flask routes
  -> services 配置、存储、运行时
  -> experiment session
  -> core 视觉算法和指标计算
  -> data/results、history、project store
```

更具体地说：

- 页面和静态资源由 `templates/`、`static/` 提供。
- HTTP 入口由 `routes/` 下的页面路由和 API 路由拆分维护。
- 上传、草稿、批量状态、任务历史、项目数据等由 `services/` 层负责。
- 每类实验有对应的 session 类，负责读取视频、执行逐帧分析、写出 CSV/JSON/视频结果。
- 底层视觉算法、轨迹、区域判定、TST 特征和绘制逻辑位于 `core/`。
- 打包版本通过 `MouseScope.spec` 收集 Python 模块、模板、静态资源和必要二进制依赖。

## 4. 目录职责

### 4.1 根目录入口

- `app.py`：应用入口。负责解析开发态/打包态路径、初始化 Flask、配置上传目录和结果目录、注册路由、初始化任务历史和项目存储、启动本地浏览器。
- `MouseScope.spec`：PyInstaller 打包配置。收集 `templates/`、`static/`、`core/`、`imageio_ffmpeg` 数据，以及可选 `ffprobe` 二进制。
- `requirements.txt`：运行依赖，当前核心依赖包括 Flask、OpenCV、NumPy、Pandas、imageio-ffmpeg。
- `BUILD_GUIDE.md`：构建、打包和 ffmpeg/ffprobe 相关说明。
- `README.md`：项目介绍、运行方式、功能概览。

### 4.2 路由层

- `routes/pages.py`：页面路由，包括首页、上传页、任务页、手动说明、许可证、上传处理、批量页、校准页和监控页。
- `routes/api.py`：快速分析相关 API，包括任务列表、历史记录、首帧读取、批量状态、草稿保存、调试预览、启动分析、状态查询、停止任务、下载结果、打开结果目录等。
- `routes/projects.py`：项目工作区页面路由。
- `routes/project_api.py`：项目 API，包括项目、分组、受试个体、视频、实验分配、运行记录和项目模式启动分析。

路由层只承担请求解析、参数校验、响应组织和页面跳转。实验类型定义、配置归一化、任务运行、视频处理和存储细节都下沉到 `services/`。

### 4.3 服务层

- `services/experiment_registry.py`：实验注册表。定义实验类型、显示名、默认配置、校准字段、批量配置字段和校验逻辑。
- `services/analysis_builders.py`：把前端草稿、批量配置和项目上下文归一化为 session 可执行配置，核心结构是 `per_video_configs`。
- `services/analysis_runtime.py`：分析运行时基类和通用处理器。负责批量循环、进度更新、采样/降采样、时间基准辅助、结果登记和“0 帧完成”防护。
- `services/analysis_sessions.py`：实验 session 的兼容导出层，保留旧引用入口。
- `services/*_session.py`：各实验的实际分析 session，包括 `tst_session.py`、`forced_swim_session.py`、`open_field_session.py`、`novel_object_session.py`、`elevated_plus_session.py`、`y_maze_session.py`、`light_dark_session.py`、`three_chamber_session.py`。
- `services/debug_preview.py`：各实验调试预览入口。
- `services/payload_geometry.py`：ROI、zone、mask 等几何 payload 归一化工具。
- `services/session_store.py`：运行中任务存储。内存保存 session，同时把关键状态快照写入 `data/results/_work/<session_id>/session_state.json`，便于页面和 API 恢复。
- `services/analysis_history.py`：SQLite 分析历史，位于 `data/history/analysis_history.sqlite`。
- `services/project_store.py`：项目索引和项目数据库。索引位于 `data/projects/project_index.sqlite`，每个项目有自己的 `project.sqlite`。
- `services/video_metadata.py`：视频元数据探测，整合 OpenCV、ffprobe/ffmpeg 和 FPS 覆盖逻辑。
- `services/video_timebase.py`：视频时间轴服务。优先使用 PTS 时间戳，无法取得时回退为基于 FPS 的均匀时间轴。
- `services/video_frame_reader.py`：预览帧读取。按可用性使用 PyAV、FFmpeg、OpenCV 等 fallback。
- `services/video_export.py`：结果视频导出。优先使用 FFmpeg H.264，必要时回退 OpenCV writer。

v3.2.0 的公开实验列表默认包含 TST、FST、OFT、NOR、EPM、Y-Maze 和 LDB。`three_chamber` 代码仍保留在注册表和内部实现中，但默认通过 release-hidden feature flag 隐藏；只有设置 `MOUSESCOPE_ENABLE_3CST=1` 或 `MOUSESCOPE_ENABLE_THREE_CHAMBER=1` 后才会出现在上传页、手册入口和验收 smoke 中。

### 4.4 算法层

- `core/tst_lis.py`：TST 可选新算法。基于轮廓残差、形状、外周、高频变化、周期性等特征计算不动状态。
- `core/tst_legacy.py`：TST 原本算法兼容实现。当前作为默认主路径，用于稳定输出和回归对照。
- `core/vision_tracker.py`：TST 共享视觉组件、绘制辅助和部分旧算法兼容逻辑。
- `core/forced_swim.py`、`core/open_field.py`、`core/novel_object.py`、`core/elevated_plus.py`、`core/y_maze.py`、`core/light_dark_box.py`、`core/three_chamber.py`：对应实验的区域判定、运动指标和绘制逻辑。
- `core/maze_zone_utils.py`、`core/spatial_render.py`：迷宫/空间类实验的几何和可视化辅助。
- `core/file_utils.py`、`core/performance_utils.py`：文件路径和性能优化辅助。

当前运行时没有 `core/tst_v2.py` 这个主模块。TST 的真实主路径是 `services/tst_session.py` 调用 `core/tst_lis.py`，旧路径由 `core/tst_legacy.py` 保留。

### 4.5 前端层

- `templates/base.html`：全局布局、导航和公共资源入口。
- `templates/home.html`：入口页，承载快速分析和项目工作区入口。
- `templates/upload.html`：上传和本地路径选择。
- `templates/calibrate.html`、`templates/calibrate_forced_swim.html`、`templates/calibrate_openfield.html`、`templates/calibrate_nor.html`、`templates/calibrate_maze.html`、`templates/calibrate_light_dark.html`、`templates/calibrate_three_chamber.html`：单视频校准、ROI/区域设置、调试预览和启动分析。
- `templates/batch_workspace.html`：批量配置工作台。
- `templates/monitor.html`、`templates/monitor_forced_swim.html`、`templates/monitor_openfield.html`、`templates/monitor_nor.html`、`templates/monitor_maze.html`、`templates/monitor_light_dark.html`、`templates/monitor_three_chamber.html`：分析进度和结果入口。
- `templates/tasks.html`：任务和历史记录。
- `templates/projects.html`、`templates/project_home.html`：项目工作区页面。
- `static/js/`：页面交互、国际化、项目工作区、批量配置和引导逻辑。
- `static/css/`：主题、校准工作台和引导样式。

前端当前以服务端模板加原生 JavaScript 为主，没有独立前端构建系统。

### 4.6 测试和开发工具

- `tests/`：单元和集成测试。
- `tools/smoke_routes_v2.py`：路由/API 冒烟测试。
- `tools/smoke_short_analyses_v2.py`：短视频分析冒烟测试。
- `tools/tst_*.py`：TST 研发、诊断、对比和标定相关工具。它们不属于用户主流程，但对算法回归和研究仍有价值，不应仅因主程序不直接 import 就删除。

## 5. 数据目录和持久化

当前默认数据目录为项目根下的 `data/`。开发态和打包态都通过 `app.py` 里的路径解析统一处理。

主要子目录：

- `data/uploads/`：上传视频和项目视频缓存。
- `data/uploads/_staged/<session_id>/`：本次分析启动前的安全 staging 区，避免原始上传路径变化影响后台任务。
- `data/results/`：分析输出结果。
- `data/results/_work/<session_id>/session_state.json`：任务状态快照。
- `data/history/analysis_history.sqlite`：快速分析和通用任务历史。
- `data/projects/project_index.sqlite`：项目索引。
- `data/projects/<project_id>/project.sqlite`：单个项目数据库。

注意：

- `data/uploads/`、`data/results/`、`data/history/`、`data/projects/` 都可能包含用户真实数据。
- 清理时不能把整个 `data/` 当成缓存目录删除。
- 临时审计目录、发布冒烟目录和 `__pycache__` 这类生成物可以在确认没有被引用后清理，但应逐项确认。

## 6. 快速分析流程

快速分析流程如下：

```text
/upload
  -> /do_upload
  -> 创建 session
  -> /calibrate/<session_id> 或 /batch/<session_id>
  -> 保存 draft_payload
  -> /api/debug_preview/<session_id>
  -> /api/start_analysis/<session_id>
  -> /monitor/<session_id>
  -> /api/status、/api/result_dir、下载或打开结果目录
```

单视频通常进入 `calibrate.html`。多视频进入 `batch_workspace.html`，并为每个视频维护独立配置。无论入口来自快速分析还是项目工作区，最终都会通过 `services/analysis_builders.py` 形成 session 可执行的 `per_video_configs`。

启动分析前会进行实验级校验，例如：

- TST/FST 需要有效 ROI。
- OFT 需要场地区域和中心区参数。
- NOR 需要场地 ROI、两个物体 ROI 和对象尺寸。
- EPM/Y-Maze 需要对应臂区/区域配置。
- LDB 暗箱可见时需要明室、暗室和可选门洞区；暗箱不可见时只要求明室，门洞区仍可选，用于排除门口探头和门缝误检。
- 3CST 需要左/中/右三室、两侧杯点、phase/role 配置；完整 protocol 模式还需要时间轴片段。v3.2.0 默认不公开该模块，内部测试需开启 `MOUSESCOPE_ENABLE_3CST=1` 或 `MOUSESCOPE_ENABLE_THREE_CHAMBER=1`。

## 7. 项目工作区流程

项目工作区用于长期组织实验数据：

```text
项目
  -> 分组
  -> 受试个体
  -> 视频导入
  -> 实验类型分配
  -> 项目模式启动分析
  -> 运行记录和结果同步
```

项目模式的关键特征：

- 视频可以链接原路径，也可以复制到项目上传缓存。
- 分析前需要把 ROI/配置绑定到具体受试个体。
- `project_context` 会随 session 保存，用于结果和运行记录回写。
- 项目分析完成后，会同步 run 状态、输出目录、汇总指标，并清理可安全释放的上传缓存。

项目工作区不是快速分析的替代实现，而是在相同分析 session 和算法层之上增加数据组织、分组、个体、分配和运行历史。

## 8. Session 配置模型

session 是快速分析和项目分析之间的核心桥梁。常见字段包括：

- `experiment_type`：实验类型。
- `video_paths`：待分析视频列表。
- `video_infos`：每个视频的元数据。
- `video_info`：兼容单视频页面的当前视频元数据。
- `start_time` / `end_time`：分析时间范围。
- `output_dir`：结果输出目录。
- `fps_override`：用户指定 FPS，用于处理元数据不可靠的视频。
- `is_batch`：是否批量模式。
- `draft_payload`：前端保存的校准草稿。
- `per_video_configs`：每个视频最终执行配置。
- `selected_video_idx`：当前预览/编辑的视频索引。
- `project_context`：项目模式上下文。
- `analysis`：后台线程、进度、状态、错误和结果。

当前应把 `per_video_configs` 视为分析执行的权威配置。前端可能存在 `workspace_video_configs` 等编辑态字段，但启动分析前必须归一化到 `per_video_configs`。

## 9. TST 当前实现

TST 是当前变化最多的模块，真实实现状态如下：

- 默认主算法是原本路径：`use_tst_lis_primary=False`，`use_tst_legacy_primary=True`。
- LIS / 微小挣扎优化路径只在调参页打开“使用新算法”后作为主路径。
- `services/tst_session.py` 负责 TST 视频循环、特征计算、状态平滑、CSV/JSON 写出和结果视频生成。
- `core/tst_legacy.py` 是当前默认识别逻辑，保留旧版背景差分/阈值逻辑，用于稳定输出和回归对照。
- `core/tst_lis.py` 是可选新算法，使用多类特征综合判定不动状态。
- 自动行为阈值相关逻辑仅适用于 legacy 路径，LIS 主路径不依赖该阈值作为核心判定。
- 结果视频采用最终标签回填绘制，避免在线状态抖动导致导出视频和最终 CSV 不一致。

TST 当前重点不再是单一像素变化阈值，而是围绕整鼠形态、尾部固定约束下的身体摆动、运动残差、频域/周期性和连续状态机做综合判断。

## 10. 其他实验模块

### FST

FST 使用水池 ROI、运动强度、质心变化和状态阈值输出不动/活动相关指标。结果包含逐帧状态、汇总 JSON/CSV 和标注视频。

### OFT

OFT 使用场地 ROI 和中心区定义，计算总距离、速度、中心区时间、边缘区时间、进入次数和轨迹相关指标。支持 PTS 时间轴和批量分析。

### NOR

NOR 需要场地 ROI、两个物体 ROI、对象尺寸和互动距离参数。输出对两个物体的探索时间、次数、偏好和识别指数。

### EPM

EPM 需要开放臂、闭合臂和中心区配置，输出各区域时间、进入次数、风险相关指标和轨迹标注。

### Y-Maze

Y-Maze 需要三臂区域配置，输出进入序列、自发交替、臂停留时间和运动指标。

## 11. 视频时间轴和解码策略

视频处理当前有三层关注点：

- 元数据探测：`services/video_metadata.py` 使用 OpenCV 和可用的 ffprobe/ffmpeg 获取帧数、FPS、时长、编码等信息。
- 时间轴计算：`services/video_timebase.py` 优先使用 PTS 时间戳，无法取得时回退为均匀 FPS 时间轴。
- 帧读取：预览和分析路径不同。预览可以通过 `video_frame_reader.py` 使用多种 fallback；分析主循环通常依赖 OpenCV 解码。

“0 帧完成”的本质风险是：元数据可以显示视频有帧，预览也可能正常，但分析主循环实际没有成功解码任何帧。如果没有防护，任务可能看起来完成却没有有效样本。

当前已在通用运行时增加防护：每个实验分析结束前会检查实际处理帧数/样本数。如果源视频声明有帧但分析结果为 0 帧，会抛出错误并把任务标记为失败，而不是生成一个看似正常的空结果。

这不影响正常视频分析。用户实际使用中能正常处理的视频，会显示真实 source frames 和 analyzed samples，例如 OFT 日志里的 `902 source frames (902 analyzed samples)`。

## 12. 打包和二进制依赖

打包入口为 `MouseScope.spec`。当前打包策略包括：

- 收集 `templates/` 和 `static/`。
- 收集 `core/` 算法模块。
- 收集 `imageio_ffmpeg` 相关数据。
- 从 `MOUSESCOPE_FFPROBE_EXE` 或 PATH 中可选收集 `ffprobe`。
- 补充 PyInstaller 可能静态分析不到的 hidden imports。

建议发布前检查：

- `BUILD_GUIDE.md` 中的构建步骤是否与当前环境一致。
- 打包环境里是否能找到 ffmpeg/ffprobe。
- 打包产物是否可以完成上传、校准、预览、启动分析和打开结果目录。

ffprobe 不是所有视频都绝对必需，但它能显著提升 PTS/VFR 视频的元数据和时间轴可靠性，建议随发布包一起准备。

## 13. 校验建议

代码变更后建议至少运行：

```powershell
E:\miniconda\envs\mouse\python.exe -B -m unittest discover -s tests -v
E:\miniconda\envs\mouse\python.exe -B tools\smoke_routes_v2.py
E:\miniconda\envs\mouse\python.exe -B tools\smoke_short_analyses_v2.py
```

涉及打包时还应额外做人工路径：

- 启动打包产物。
- 上传单视频并读取首帧。
- 保存校准草稿。
- 执行一次 debug preview。
- 启动至少一个短视频分析。
- 检查 `data/results/` 输出、任务历史和打开结果目录。

涉及项目工作区时还应检查：

- 创建项目、分组、受试个体。
- 导入视频。
- 分配实验类型。
- 启动项目分析。
- 完成后 run 状态和输出路径正确回写。

## 14. 文档维护规则

当以下内容变化时，应同步更新本文：

- 新增或删除实验类型。
- 修改 `routes/` API 或页面主流程。
- 修改 session 配置模型，尤其是 `draft_payload`、`per_video_configs`、`project_context`。
- 修改视频解码、时间轴、导出或打包策略。
- 修改 TST 主算法路径或 legacy/LIS 开关语义。
- 修改项目工作区数据模型。

历史规划文档可以保留，但如果与本文冲突，以当前代码和本文为准。
