# MouseScope 系统设计文档

当前基线版本：`v2.3.0`

本文档是 MouseScope 后续演进的统一设计约束。以后新增实验、重构 UI、调整批量逻辑或拆分后端时，都应优先遵守这里的结构与原则，避免再次出现“功能逐步叠加后，各模块交互和代码组织越来越不一致”的问题。

## 1. 产品目标

MouseScope 是一个面向小鼠行为学视频分析的桌面软件平台，当前支持：

- `TST` 悬尾实验
- `FST` 强迫游泳
- `OFT` 旷场实验
- `NOR` 新奇物体识别
- `EPM` 高架十字迷宫
- `Y-Maze` Y 迷宫

统一产品目标：

- 所有实验共享一致的上传、配置、批量、监控、导出主流程
- ROI / zone / mask / time window / 参数调试逻辑在不同实验之间尽量同构
- 批量模式必须是“批次工作台 -> 单视频编辑 -> 返回工作台 -> 统一启动”
- 保留中文优先体验，同时保证英文切换完整可用
- 所有会写文件的结果都要兼容中文路径与中文文件名显示

## 2. 核心设计原则

### 2.1 薄入口

`app.py` 负责：

- Flask 应用初始化
- 路由注册
- 调用服务层和分析会话层

`app.py` 不应继续承担：

- 越来越多的实验元数据分发
- 批量状态规则定义
- 草稿持久化策略
- 大量重复的配置校验逻辑

### 2.2 单一来源

以下信息必须只有一个权威来源：

- 实验列表、实验标签、手册模块、配置页模板、监控页模板：`services/experiment_registry.py`
- 会话草稿与恢复：`services/session_store.py`
- 版本号与发布说明：发布文档和公共文案
- 公共 UI 规则：本设计文档

### 2.3 批量优先一致性

批量模式不是在单视频页面上硬塞一个视频切换器，而是一个单独的工作流：

1. 上传多个视频
2. 进入批量工作台
3. 每个视频用单独入口进入配置页
4. 保存草稿并返回工作台
5. 工作台统一检查完整性
6. 从工作台统一启动分析

### 2.4 ROI 与检测区域分离

以后所有实验都遵守：

- `统计区域` 决定行为学指标归属
- `检测遮罩` 决定前景检测实际参与的图像范围
- 如果用户未单独绘制检测遮罩，则默认回退到统计区域或其并集

这条规则已经适用于 `FST / OFT / NOR / EPM / Y-Maze`。

## 3. 当前目录结构

建议长期维持以下职责边界：

### 3.1 Web 层

- `app.py`
  - Flask 入口
  - 路由注册
  - 许可拦截
  - 运行时服务装配
- `routes/pages.py`
  - 页面路由
  - 上传、批量工作台、标定页、监控页、任务中心
- `routes/api.py`
  - API 路由
  - 分析启动、状态查询、下载、任务中心数据接口

### 3.2 核心分析层

- `core/vision_tracker.py`
  - TST 分析底层
- `core/forced_swim.py`
  - FST 分析底层
- `core/open_field.py`
  - OFT 分析底层
- `core/novel_object.py`
  - NOR 分析底层
- `core/elevated_plus.py`
  - EPM 分析底层
- `core/y_maze.py`
  - Y-Maze 分析底层
- `core/maze_zone_utils.py`
  - 迷宫几何与多边形工具
- `core/file_utils.py`
  - 中文路径 / 安全文件名工具

### 3.3 服务层

- `services/experiment_registry.py`
  - 实验注册表
  - 批量完整性校验
  - 手册映射
- `services/analysis_builders.py`
  - 各实验 `start_analysis` 配置构建
  - 会话公共参数与 `per_video_configs` 的统一归一化
  - 新实验接入时的分析 payload 入口
- `services/session_store.py`
  - 会话内存态
  - 草稿磁盘快照
  - TTL 清理
- `services/analysis_history.py`
  - SQLite 分析历史索引
  - 任务中心搜索、筛选、重启后历史展示
  - 软件关闭时未完成任务标记为 `interrupted`
- `services/analysis_sessions.py`
  - TST / FST / OFT / NOR / EPM / Y-Maze 分析会话类
- `services/debug_preview.py`
  - 各实验参数调试预览
- `services/payload_geometry.py`
  - ROI / zone / mask 几何 payload 归一化

### 3.4 模板层

- `templates/upload.html`
  - 上传与入口页
- `templates/batch_workspace.html`
  - 批量工作台
- `templates/calibrate*.html`
  - 各实验配置页
- `templates/monitor*.html`
  - 各实验监控页
- `templates/manual*.html`
  - 模块手册

### 3.5 资产层

- `static/js/i18n.js`
  - 公共中英文文案
- `static/css/tech-theme.css`
  - 全局主题样式

### 3.6 质量保障

- `tools/smoke_routes_v2.py`
  - 页面渲染、批量路由、inline JS 语法的最小回归检查
- `tools/smoke_short_analyses_v2.py`
  - 使用样例视频对 6 个实验模块做短程端到端分析
  - 用于发现“页面能打开，但真实分析配置已失效”的问题

## 4. 实验注册表设计

每个实验在注册表中至少声明：

- `key`
- `label`
- `short_label`
- `manual_module`
- `calibrate_template`
- `monitor_template`
- `validator`
- `maze_module`（迷宫类实验可选）

未来新增实验时，优先按照以下顺序接入：

1. 在 `services/experiment_registry.py` 注册实验元数据
2. 准备 core 分析器
3. 新增或复用配置页模板
4. 新增或复用监控页模板
5. 接入 debug preview
6. 接入 start_analysis builder
7. 补手册和 smoke 验证

不要再采用“先在模板里做一份，再在 app.py 里零散补分支”的方式。

## 5. 会话与草稿模型

### 5.1 会话对象

会话至少包含：

- `experiment_type`
- `video_paths`
- `video_infos`
- `video_info`
- `start_seconds`
- `end_seconds`
- `output_dir`
- `fps_override`
- `is_batch`
- `draft_payload`
- `draft_updated_at`
- `selected_video_idx`
- `created_at`
- `last_accessed_at`
- `analysis`

### 5.2 草稿模型

草稿分为两层：

- `per_video_configs`
  - 用于正式分析启动
  - 只保留完成态配置
- `workspace_video_configs`
  - 用于配置页编辑中的过程态
  - 允许保留正在编辑的 ROI、多边形点、mask 编辑状态等

这两层必须同时存在，原因是：

- 工作台需要看到“哪些视频已经可启动”
- 配置页需要保存“还没画完但不能丢”的中间状态

### 5.3 持久化策略

`SessionStore` 负责：

- 内存里的运行态会话
- 将可序列化字段写入 `data/results/_work/<session_id>/session_state.json`
- 恢复草稿
- 清理过期会话

约束：

- `analysis` 实例不写入 JSON
- JSON 必须用 `utf-8`
- JSON 写出时 `ensure_ascii=False`

`AnalysisHistoryStore` 负责：

- 将已经启动的分析写入 `data/history/analysis_history.sqlite`
- 记录实验类型、视频名、输出目录、状态、进度、结果文件与错误信息
- 任务中心必须展示当次分析的结果目录；旧记录缺少输出目录时，可从已有结果文件反推父目录
- 任务完成、停止、出错时更新历史记录
- 软件重新启动时，将仍处于 `waiting` / `analyzing` 的历史记录标记为 `interrupted`

约束：

- 历史库只负责“可搜索、可追溯、可下载”的任务索引
- 不承诺断点续跑；软件关闭后后台线程已经终止，重启后只能显示中断状态
- 任务中心删除记录默认只删除历史索引和非运行内存会话，不删除结果文件
- 历史库属于用户数据，不能提交到版本库

## 6. 批量工作流设计

### 6.1 上传页

上传页负责：

- 选择实验模块
- 导入单个或多个视频
- 指定全局时间窗
- 指定输出目录
- 指定 FPS 覆盖

多视频上传后：

- 不直接进入配置页
- 必须先进入 `batch_workspace`

### 6.2 批量工作台

工作台页面必须包含：

- 批次摘要
- 每个视频一张卡片
- 完成 / 未完成状态
- 缺项清单
- 进入编辑按钮
- 复制配置快捷操作
- 重置当前视频草稿
- 从工作台统一启动分析

### 6.3 单视频编辑页

批量模式下，配置页是“单视频编辑视图”：

- 只编辑当前视频
- 顶部明确显示当前视频索引和返回工作台入口
- 主按钮变成“保存并返回批量工作台”
- 不再要求用户在配置页里反复切换视频

## 7. 配置页统一规则

所有 `calibrate*.html` 应尽量遵守同一信息架构：

### 7.1 顶部

- 实验名称
- 简短说明
- 手册入口
- 返回批量工作台按钮（批量模式）
- 步骤条

### 7.2 左主区

- 视频帧与画布
- ROI / zone / object / mask 绘制
- 当前状态条

### 7.3 右侧区

- ROI 卡片列表
- 名称编辑
- 时间窗模式
- 精确时间范围
- 遮罩编辑入口

### 7.4 第二步

- 调试预览
- 检测参数
- 统计参数
- 主操作按钮

### 7.5 监控页统一规则

所有 `monitor*.html` 尽量遵守同一信息层次：

- 顶部：实验标题、简短说明、停止 / 刷新操作
- 第一主卡：处理状态、消息、进度条
- 第二主卡：会话概览指标
- 第三主卡：ROI / arena / zone 实时结果表
- 侧栏：实时流式状态和日志
- 完成后：统一显示下载区与结果目录入口

### 7.6 时间窗规则

所有实验统一：

- `global` 模式下隐藏精确开始 / 结束输入
- `precise` 模式下再展示开始 / 结束输入

不要继续保留“明明是全局模式，但界面上还展示被禁用输入框”的旧行为。

### 7.7 Mask 规则

支持 mask 的页面必须统一：

- ROI 卡片内有 `编辑检测遮罩 / 重置默认遮罩`
- 画布有明确的 mask 编辑模式
- 未完成的 mask 不进入正式分析 payload
- 若 mask 已存在但点数少于 4，启动前必须阻止

## 8. 迷宫实验特殊规则

### 8.1 EPM

每个 ROI 必须显式手动画：

- `center`
- `open_1`
- `open_2`
- `closed_1`
- `closed_2`

不再允许用中心点自动推导所有臂。

### 8.2 Y-Maze

每个 ROI 必须显式手动画：

- `center`
- `arm_a`
- `arm_b`
- `arm_c`

### 8.3 点顺序规则

四边形区域的点击顺序不应影响结果：

- 保存前统一做多边形归一化排序
- 入臂、区域归属和统计只能依赖命名后的区域，不依赖用户点击顺序

## 9. 中文编码与路径规则

必须始终遵守：

- Python 文件使用 `utf-8`
- CSV 使用 `utf-8-sig`
- 会话快照 JSON 使用 `utf-8`
- 文件显示名和文件实际保存名分离
- 内部结果文件名优先使用安全 ASCII stem
- UI 上向用户显示原始中文文件名

所有新增代码都应继续复用：

- `core/file_utils.py`
- `_stage_local_video_if_needed`
- `_register_original_name`
- `_safe_display_stem`

## 10. 中英文切换规则

多语言设计原则：

- 通用文案优先进入 `static/js/i18n.js`
- 模块特有且高度动态的配置页文案可以保留页内 `PAGE_I18N`
- 同一条文案不要同时在模板硬编码和 i18n 文件各维护一份不同版本

新增页面时优先判断：

- 如果是全局导航、上传页、首页、公共页脚：放入 `i18n.js`
- 如果是复杂的实验配置页细节：允许页内局部 i18n，但必须中英文成对出现

## 11. 测试与回归要求

每次较大改动后，至少执行：

1. `mouse` 环境下 Python 编译检查
2. 公共页面路由渲染检查
3. 单视频 / 批量模式配置页路由检查
4. 批量工作台接口检查
5. inline JS 语法检查

建议命令：

```powershell
E:\miniconda\envs\mouse\python.exe -m py_compile app.py services\analysis_builders.py services\session_store.py services\experiment_registry.py
E:\miniconda\envs\mouse\python.exe tools\smoke_routes_v2.py
E:\miniconda\envs\mouse\python.exe tools\smoke_short_analyses_v2.py
```

如果涉及真实算法调整，还应补：

- 真实样例视频回归
- 导出 CSV 字段核对
- 热图 / 轨迹图抽查
- 配置选项覆盖关系检查，例如 `generate_video`、时间窗和 `fps_override`

## 12. 后续拆分建议

当前 `app.py` 仍然偏大，下一阶段应继续朝以下方向收敛：

- `routes/pages.py`
  - 页面路由
- `routes/api.py`
  - API 路由
- `services/analysis_builders.py`
  - 按实验构建 `vision_config`
- `services/debug_preview.py`
  - preview 分发
- `services/analysis_sessions.py`
  - 各实验 session 类

拆分原则：

- 先抽重复最重、风险最高的部分
- 每拆一层都保留现有 URL 和 payload 兼容
- 优先通过“注册表 + builder + service”减少 `if / elif` 扩散

## 13. 新增实验的标准流程

以后如果新增实验模块，必须按这个顺序做：

1. 明确实验 protocol 和 ROI / zone / mask / time-window 规则
2. 在 `core/` 实现分析器
3. 在 `services/experiment_registry.py` 注册元数据和完整性校验
4. 接入上传页实验入口
5. 实现配置页 / 监控页 / 手册
6. 接入批量工作台缺项摘要
7. 接入 debug preview 和 start_analysis
8. 补 i18n
9. 补 smoke 回归
10. 更新本设计文档

如果没更新文档，就视为设计没有真正完成。
