(function () {
    const registry = new Map();
    let active = null;
    let renderQueued = false;

    function currentLang() {
        return typeof getLang === 'function' ? getLang() : (localStorage.getItem('mousescope_lang') || 'zh');
    }

    function tr(key, fallback = '') {
        const source = (typeof I18N !== 'undefined' ? I18N : window.I18N) || {};
        const dict = source[currentLang()] || {};
        return dict[key] ?? fallback;
    }

    function localText(value, fallback = '') {
        if (value && typeof value === 'object') {
            return value[currentLang()] ?? value.zh ?? value.en ?? fallback;
        }
        return value ?? fallback;
    }

    function stepText(step, field) {
        const key = step[`${field}Key`];
        return key ? tr(key, localText(step[field], '')) : localText(step[field], '');
    }

    function visibleElement(selector) {
        const nodes = Array.from(document.querySelectorAll(selector || ''));
        return nodes.find(node => {
            const rect = node.getBoundingClientRect();
            const style = window.getComputedStyle(node);
            return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
        }) || null;
    }

    function storageKey(tour) {
        return `mousescope_tour_${tour.id}_${tour.version || 'v1'}`;
    }

    function isBlockedByModal() {
        return !!document.querySelector('.modal.show');
    }

    function createLayer() {
        const scrim = document.createElement('div');
        scrim.className = 'tour-scrim';
        const spotlight = document.createElement('div');
        spotlight.className = 'tour-spotlight';
        const popover = document.createElement('div');
        popover.className = 'tour-popover';
        document.body.append(scrim, spotlight, popover);
        return {scrim, spotlight, popover};
    }

    function removeLayer() {
        if (!active) return;
        if (active.target) active.target.classList.remove('tour-target-active');
        active.layer.scrim.remove();
        active.layer.spotlight.remove();
        active.layer.popover.remove();
        active = null;
        renderQueued = false;
    }

    function usableSteps(tour) {
        const rawSteps = typeof tour.steps === 'function' ? tour.steps() : (tour.steps || []);
        const seen = new Set();
        return rawSteps.filter(step => {
            if (!(visibleElement(step.target) || (step.showTab && document.querySelector(step.target)))) return false;
            const key = [
                step.target || '',
                step.showTab || '',
                step.titleKey || localText(step.title, ''),
                step.bodyKey || localText(step.body, ''),
            ].join('|');
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    }

    function queueRender() {
        if (!active || renderQueued) return;
        renderQueued = true;
        window.requestAnimationFrame(() => {
            renderQueued = false;
            render();
        });
    }

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    function navBottom() {
        const nav = document.querySelector('.navbar');
        if (!nav) return 0;
        return Math.max(0, Math.ceil(nav.getBoundingClientRect().bottom));
    }

    function placePopover(popover, rect) {
        const margin = 12;
        const navLimit = navBottom() + margin;
        const popWidth = Math.min(360, window.innerWidth - margin * 2);
        popover.style.width = `${popWidth}px`;
        const aboveSpace = rect.top - navLimit;
        const belowSpace = window.innerHeight - rect.bottom;
        let top = belowSpace >= 190 || belowSpace >= aboveSpace ? rect.bottom + margin : rect.top - popover.offsetHeight - margin;
        if (top < navLimit) top = navLimit;
        top = clamp(top, navLimit, Math.max(navLimit, window.innerHeight - popover.offsetHeight - margin));
        const left = clamp(rect.left, margin, window.innerWidth - popWidth - margin);
        popover.style.top = `${top}px`;
        popover.style.left = `${left}px`;
    }

    function render() {
        if (!active) return;
        const step = active.steps[active.index];
        if (step.showTab && typeof window.showTab === 'function') {
            window.showTab(step.showTab);
        }
        const target = visibleElement(step.target);
        if (!target) {
            next();
            return;
        }
        if (active.target && active.target !== target) {
            active.target.classList.remove('tour-target-active');
        }
        active.target = target;
        active.target.classList.add('tour-target-active');
        const isNavTarget = !!target.closest('.navbar');
        if (isNavTarget && window.scrollY > 0) {
            window.scrollTo({top: 0, left: window.scrollX, behavior: 'auto'});
            queueRender();
            return;
        }
        const rect = target.getBoundingClientRect();
        const visibleTop = navBottom() + 12;
        const visibleBottom = window.innerHeight - 12;
        const visibleHeight = Math.max(120, visibleBottom - visibleTop);
        const targetIsTall = rect.height >= visibleHeight;
        const needsScroll = !isNavTarget && (
            targetIsTall
                ? (rect.top > visibleTop || rect.bottom < visibleTop + Math.min(160, visibleHeight * 0.4))
                : (rect.top < visibleTop || rect.bottom > visibleBottom)
        );
        if (needsScroll) {
            target.scrollIntoView({block: targetIsTall ? 'start' : 'center', inline: 'nearest', behavior: 'auto'});
            queueRender();
            return;
        }
        const pad = Number(step.padding ?? 8);
        const top = isNavTarget ? Math.max(0, rect.top - pad) : Math.max(navBottom(), rect.top - pad);
        const left = Math.max(8, rect.left - pad);
        const width = Math.min(window.innerWidth - left - 8, rect.width + pad * 2);
        const height = Math.min(window.innerHeight - top - 8, rect.height + pad * 2);

        active.layer.scrim.style.setProperty('--tour-top', `${navBottom()}px`);
        active.layer.spotlight.classList.toggle('in-navbar', isNavTarget);
        Object.assign(active.layer.spotlight.style, {
            top: `${top}px`,
            left: `${left}px`,
            width: `${width}px`,
            height: `${height}px`,
        });

        const isLast = active.index >= active.steps.length - 1;
        active.layer.popover.innerHTML = `
            <h3>${escapeHtml(stepText(step, 'title'))}</h3>
            <p>${escapeHtml(stepText(step, 'body'))}</p>
            <div class="tour-progress">${active.index + 1} / ${active.steps.length}</div>
            <div class="tour-actions">
                <button type="button" class="btn btn-outline-secondary btn-sm" data-tour-action="skip">${escapeHtml(tr('tour.action.skip', '跳过'))}</button>
                <div class="d-flex gap-2">
                    <button type="button" class="btn btn-outline-primary btn-sm" data-tour-action="prev" ${active.index === 0 ? 'disabled' : ''}>${escapeHtml(tr('tour.action.prev', '上一步'))}</button>
                    <button type="button" class="btn btn-primary btn-sm" data-tour-action="next">${escapeHtml(tr(isLast ? 'tour.action.done' : 'tour.action.next', isLast ? '完成' : '下一步'))}</button>
                </div>
            </div>
        `;
        active.layer.popover.querySelector('[data-tour-action="skip"]').addEventListener('click', finish);
        active.layer.popover.querySelector('[data-tour-action="prev"]').addEventListener('click', prev);
        active.layer.popover.querySelector('[data-tour-action="next"]').addEventListener('click', next);
        const placementTop = clamp(top, navBottom() + 12, window.innerHeight - 80);
        const placementBottom = clamp(top + height, placementTop + 40, window.innerHeight - 12);
        placePopover(active.layer.popover, {top: placementTop, left, width, height: placementBottom - placementTop, bottom: placementBottom});
    }

    function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, ch => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[ch]));
    }

    function finish() {
        if (active) {
            localStorage.setItem(storageKey(active.tour), 'done');
        }
        removeLayer();
    }

    function next() {
        if (!active) return;
        if (active.index >= active.steps.length - 1) {
            finish();
            return;
        }
        active.index += 1;
        render();
    }

    function prev() {
        if (!active || active.index <= 0) return;
        active.index -= 1;
        render();
    }

    function start(id, options = {}) {
        const tour = registry.get(id);
        if (!tour) return false;
        if (!options.force && localStorage.getItem(storageKey(tour))) return false;
        if (isBlockedByModal()) {
            window.setTimeout(() => start(id, options), 900);
            return false;
        }
        const steps = usableSteps(tour);
        if (!steps.length) return false;
        removeLayer();
        active = {tour, steps, index: 0, layer: createLayer(), target: null};
        render();
        return true;
    }

    function register(id, config) {
        registry.set(id, {id, version: 'v1', autoStart: true, ...(config || {})});
        if ((config || {}).autoStart !== false) {
            const schedule = () => window.setTimeout(() => start(id), 450);
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', schedule, {once: true});
            } else {
                schedule();
            }
        }
    }

    function startCurrent(options = {}) {
        const id = document.body.getAttribute('data-tour-page') || document.querySelector('[data-tour-page]')?.getAttribute('data-tour-page');
        if (id && start(id, {force: true, ...options})) return true;
        const first = Array.from(registry.keys())[0];
        return first ? start(first, {force: true, ...options}) : false;
    }

    function tourStep(target, zhTitle, zhBody, enTitle, enBody, padding = 8) {
        return {
            target,
            title: {zh: zhTitle, en: enTitle},
            body: {zh: zhBody, en: enBody},
            padding,
        };
    }

    function calibrationShell() {
        return document.querySelector('[data-calibration-module]');
    }

    function calibrationModule() {
        const shell = calibrationShell();
        return shell ? shell.getAttribute('data-calibration-module') : '';
    }

    function calibrationNavStep(moduleName, stepTextZh, stepTextEn) {
        return tourStep(
            '.calibration-steps, .fst-steps, .oft-steps, .nor-steps, .maze-steps, .steps',
            `${moduleName} 流程`,
            stepTextZh,
            `${moduleName} Workflow`,
            stepTextEn
        );
    }

    function calibrationActionStep() {
        return tourStep(
            '.calibration-action-head, .fst-toolbar, .oft-toolbar, .nor-toolbar, .maze-toolbar',
            '顶部操作区',
            '这里统一放返回、刷新预览和开始分析/保存返回按钮。调参确认后从右侧主按钮提交。',
            'Action Bar',
            'Back, refresh preview, and start/save actions live here consistently. Use the primary button after checking the preview.'
        );
    }

    function projectSubjectBindingStep() {
        return tourStep(
            '[data-tour-id="project-subject-required"]',
            '项目小鼠绑定',
            '如果是从项目里进入配置，每个完整 ROI 都必须在右侧选择项目中已有小鼠。没有选择时，“进入调参”或“开始分析”会保持不可用。',
            'Project Animal Binding',
            'When setup is launched from a project, every complete ROI must be assigned to an existing project animal. Until then, tuning or analysis stays unavailable.'
        );
    }

    function calibrationTourSteps() {
        const module = calibrationModule();
        const inDebug = !!visibleElement('#debug-step, #step2');
        const inRoi = !!visibleElement('#roi-step');

        if (module === 'light_dark_box') {
            if (inDebug) {
                return [
                    calibrationNavStep('LDB', '当前是第 2 步：调检测、检查预览，并确认真正参与统计的时间规则。', 'This is step 2: tune detection, inspect preview, and confirm timing rules used for metrics.'),
                    tourStep('[data-tour-id="ldb-action-bar"]', '调参操作区', '这里可以返回标定、刷新当前帧预览，或在确认 ROI 和参数后开始分析。多视频逐条编辑时，“开始”会保存当前视频配置并返回分析配置页。', 'Tuning Actions', 'Go back to ROI drawing, refresh the current-frame preview, or start after checking ROI and parameters. During multi-video editing, Start saves this video and returns to Analysis Configuration.'),
                    tourStep('[data-tour-id="ldb-preview-frame"]', '预览帧', '拖动这里选择用于调参的帧。建议看几帧明室、暗室和门口附近的画面，尤其检查黑门缝是否被误检。', 'Preview Frame', 'Drag this to choose the frame used for tuning. Check frames in the light chamber, dark chamber, and passage area, especially whether dark door seams are falsely detected.'),
                    tourStep('[data-tour-id="ldb-preview-grid"]', '预览画面', '左侧是原始帧，右侧会叠加鼠体检测结果、明室、暗室和可选门区。理想状态是鼠体轮廓稳定，门缝或黑边不被识别成鼠体。', 'Preview Canvases', 'The left side is the original frame; the right side overlays mouse detection plus light, dark, and optional doorway zones. Aim for a stable body contour without door seams or dark edges being detected as the animal.'),
                    tourStep('[data-tour-id="ldb-detection-params"]', '检测参数', '这些参数只负责把鼠体从背景中分出来。先选对背景类型，再调灰度阈值和最小面积；如果右侧轮廓断裂、粘住箱壁或漏检，优先在这里处理。', 'Detection Parameters', 'These settings only segment the animal from the background. Choose the correct background type, then tune gray threshold and minimum area. Use this section when the contour breaks, sticks to walls, or misses the animal.'),
                    tourStep('[data-tour-id="ldb-core-stats"]', '核心统计参数', '这一栏会影响明暗箱的事件和时间统计：什么时候开始算、一次转换怎样才成立，以及 bins 文件按多长时间切片。暗箱不可见模式下，转换由“明室可见/不可见”的稳定变化推断。', 'Core Statistics', 'This section affects LDB timing and events: when counting starts, when a transition is accepted, and how long each bin is in the binned output. In invisible-dark mode, transitions are inferred from stable visible/not-visible changes in the light chamber.'),
                    tourStep('[data-tour-id="ldb-door-delay"]', '开门延迟', '如果视频开头包含放鼠、开门、手还在画面里，填这里会从统计起点跳过这段秒数。它会影响首次入明室潜伏期和开头几秒的进入/转换事件；如果视频已经从正式实验开始，可设为 0。', 'Door-Open Delay', 'Use this when the video starts with placing the mouse, opening the door, or a hand still in frame. Those seconds are skipped from the metric start, affecting first-light latency and early entry/transition events. Set 0 if the video already starts at the formal trial.'),
                    tourStep('[data-tour-id="ldb-transition-dwell"]', '转换稳定时间', '这是主要防抖参数：小鼠进入新主室后，需要连续稳定停留这么久才算转换。即使不画门区，也会用这个规则避免门口来回抖动被算成很多次。', 'Transition Dwell', 'This is the main debounce setting: after entering a new main chamber, the animal must remain there for this long before a transition is accepted. It works even when no doorway is drawn.'),
                    tourStep('[data-tour-id="ldb-doorway-reset"]', '门区重置', '门区是可选的门缝或边界忽略带。默认检测会避开门区，适合门口较暗、阈值容易识别门缝的情况；如果小鼠在门区附近停留超过这个时间，会重置候选转换。', 'Doorway Reset', 'Doorway is an optional door-seam or boundary ignore band. Default detection excludes it, which helps when a dark passage seam is falsely segmented; lingering there longer than this value resets the pending transition.'),
                    tourStep('[data-tour-id="ldb-bin-seconds"]', '分箱时长', '只影响按时间分段导出的 bins 文件，例如每 60 秒汇总一次明室时间、暗室时间和转换次数；不改变整段实验的总指标。', 'Bin Length', 'This only affects the time-binned output, for example summarizing light time, dark time, and transitions every 60 seconds. It does not change whole-session totals.'),
                    tourStep('[data-tour-id="ldb-aux-scale"]', '箱体尺寸（辅助）', '这里分别填写明箱和暗箱的真实长、宽。长宽只用于距离、速度、移动时间和规整轨迹图比例，不影响明暗室时间、进入次数、转换次数或首次入明室潜伏期。', 'Chamber Size (Auxiliary)', 'Enter the real length and width for the light and dark chambers separately. These values only affect distance, speed, movement time, and rectified trajectory-map scale; they do not affect core LDB metrics.'),
                    tourStep('[data-tour-id="ldb-shared-edge"]', '两箱重合边', '填写明箱和暗箱贴在一起的那条公共边实际重合长度。它不是门洞宽度，也不是画面 x/y 方向；系统会结合你画的两个 ROI 中最靠近的边，自动判断轨迹图应该横向拼接还是纵向拼接。', 'Shared Edge', 'Enter the real overlap length of the side where the light and dark boxes touch. This is not the doorway width or image x/y size; MouseScope combines it with the closest ROI edges to infer horizontal or vertical layout for the trajectory map.'),
                    tourStep('[data-tour-id="ldb-movement-threshold"]', '运动阈值（辅助）', '这个阈值只用于判断移动/低活动时间等辅助 QC 字段，不是明暗箱焦虑/探索核心指标。只需要在你关心运动学字段时认真校准。', 'Movement Threshold (Auxiliary)', 'This threshold only labels moving or low-activity time for auxiliary QC fields. It is not a core anxiety/exploration metric for LDB. Tune it carefully only when those motion fields matter.'),
                    tourStep('#primaryActionBtn', '开始分析', '确认预览、ROI 和这些参数后，从这里启动分析；项目或批量编辑场景会先保存配置。', 'Start Analysis', 'After checking preview, ROI, and parameters, start analysis here. Project or batch-edit contexts save the setup first.')
                ];
            }
            return [
                calibrationNavStep('LDB', '明暗箱分两步：先画每个箱体的明室、暗室和可选门缝忽略带，再进入调参页确认检测与统计规则。', 'LDB has two steps: draw light, dark, and optional door-seam ignore zones for each box, then tune detection and statistics.'),
                tourStep('[data-tour-id="ldb-core-note"]', '先理解统计边界', 'LDB 核心指标主要依赖明室、暗室和时间窗；转换防抖由稳定停留时间完成，不必须画门区。门区只在门口黑线/门缝误检时作为可选忽略带。', 'Core Metric Boundary', 'Core LDB metrics mainly depend on light/dark zones and timing windows; transition debounce is handled by stable dwell time and does not require doorway. Doorway is optional and useful as an ignore band for door-seam false positives.'),
                tourStep('[data-tour-id="ldb-dark-visibility"]', '暗箱是否可见', '如果暗箱透明、能在暗室内追踪小鼠，选“可见”并画明室和暗室；如果暗箱不透明、暗室内看不到小鼠，选“不可见”，只画明室，系统用“明室内是否看见小鼠”推断明/暗时间，并关闭轨迹、热图、速度和距离输出。', 'Dark-Chamber Visibility', 'Choose Visible when the animal can be tracked inside the dark chamber, then draw both light and dark zones. Choose Invisible for opaque dark boxes: draw only the light chamber, infer light/dark time from whether the mouse is visible in light, and disable trajectory, heatmap, speed, and distance outputs.'),
                tourStep('[data-tour-id="ldb-draw-frame"]', '选择标定帧', '拖动这里找一帧边界清楚、没有手遮挡、明暗室和门口都能看清的画面。标定帧只帮助你画 ROI，不决定最终只分析这一帧。', 'Drawing Frame', 'Drag this to a frame with clear boundaries, no hand occlusion, and visible light/dark/passage areas. This frame helps draw ROI; it does not limit analysis to this frame.'),
                tourStep('[data-tour-id="ldb-draw-canvas"]', '绘制区域', '先点“新建 ROI”，再在右侧 ROI 卡片选择要画的区域。暗箱可见时明室和暗室各至少 3 个点；暗箱不可见时只需要明室至少 3 个点。点击顺序只决定多边形边界。', 'Draw Zones', 'Click New ROI first, then choose the zone from the ROI card. In visible-dark mode, light and dark need at least three points each; in invisible-dark mode, only the light chamber needs at least three points. Click order only defines the polygon boundary.'),
                tourStep('[data-tour-id="ldb-roi-list"]', 'ROI 卡片', '每张卡片对应一个箱体/一只小鼠。在这里改 ROI 名称、选择明室/暗室/可选门区按钮、设置时间窗。默认检测使用明室+暗室，并会排除已画门区。', 'ROI Cards', 'Each card is one box or one animal. Rename ROI, choose Light/Dark/optional Doorway buttons, and set timing windows. Default detection uses light + dark and excludes doorway if drawn.'),
                projectSubjectBindingStep(),
                tourStep('[data-tour-id="ldb-roi-status"]', '完成状态', '这里会显示 ROI 数、完整 ROI 数、当前区域点数和遮罩点数。暗箱可见时需要明室和暗室都完成；暗箱不可见时只需要明室完成。', 'Completion Status', 'This shows ROI count, complete ROI count, current zone points, and mask points. Visible-dark mode needs both light and dark zones; invisible-dark mode only needs the light zone.'),
                tourStep('[data-tour-id="ldb-draw-toolbar"]', '绘制工具', '撤销会撤掉当前区域最后一个点；清空当前区域只影响正在编辑的明室/暗室/门区；完成遮罩只在编辑检测遮罩时可用。画完后点“进入调参”。', 'Drawing Tools', 'Undo removes the last point in the current zone. Clear current zone affects only the active light/dark/doorway polygon. Finish mask is available only while editing a detection mask. Use Enter Tuning after drawing.')
            ];
        }

        if (module === 'tst') {
            if (inDebug) {
                return [
                    calibrationNavStep('TST', '当前是第 3 步：参数调试。这里主要确认鼠体分割、行为判断逻辑和导出设置。', 'Step 3 checks segmentation, behavior decision logic, and output options.'),
                    calibrationActionStep(),
                    tourStep('[data-tour-id="tst-debug-preview"]', '预览结果', '左侧看原始帧，右侧看检测轮廓。目标是让绿色轮廓尽量紧密包住小鼠除尾巴外的主体躯干。', 'Preview', 'Original frame is on the left and detection result on the right. Aim for a tight green contour around the body trunk, excluding the tail.'),
                    tourStep('[data-tour-id="tst-detection-params"]', '检测参数', '背景类型决定阈值方向；灰度阈值、腐蚀/膨胀和最小面积一起决定轮廓是否紧密包住主体躯干。', 'Detection Parameters', 'Background type sets threshold direction; threshold, erosion/dilation, and minimum area determine how tightly the contour wraps the trunk.'),
                    tourStep('[data-tour-id="tst-output-params"]', '输出与行为算法', '默认使用经典行为算法；如需增强低强度挣扎识别，可启用“微小挣扎优化算法”。自动行为阈值和灵敏度仅适用于经典行为算法。', 'Output and Behavior Algorithm', 'The classic behavior algorithm is selected by default. Enable the subtle-struggle optimized algorithm for low-intensity struggle detection. Auto-threshold and sensitivity apply only to the classic behavior algorithm.'),
                    tourStep('#primaryActionBtn', '开始分析', '单视频会直接开始；多视频逐条编辑会先保存当前配置并返回分析配置页。', 'Start Analysis', 'Single-video mode starts immediately. Multi-video editing saves this setup and returns to Analysis Configuration.')
                ];
            }
            if (inRoi) {
                return [
                    calibrationNavStep('TST', '当前是第 2 步：为每只小鼠画 ROI，并设置 ROI 名称和统计时间窗。', 'Step 2 draws one ROI per animal and sets names plus timing windows.'),
                    tourStep('#roiCanvas', 'ROI 画布', '点击多个点圈住单只小鼠的完整悬挂活动范围。每个 ROI 只放一只小鼠。', 'ROI Canvas', 'Click points around one animal’s full suspension area. Keep one animal per ROI.'),
                    tourStep('#roiList', 'ROI 列表', '这里修改 ROI 名称、绑定项目小鼠，并选择跟随视频、相对检测时间或精确时间段。', 'ROI List', 'Rename ROIs, bind project animals, and choose follow-video, relative, or precise timing.'),
                    projectSubjectBindingStep(),
                    tourStep('.stage-side', '当前规则', '右侧汇总 ROI 数量、点数和标定状态。TST 需要体长标定和至少一个有效 ROI。', 'Rules', 'The side panel summarizes ROI count, current points, and calibration state. TST needs calibration and at least one valid ROI.')
                ];
            }
            return [
                calibrationNavStep('TST', 'TST 有三步：体长标定、ROI 选择、参数调试。当前是体长标定。', 'TST has three steps: body calibration, ROI selection, and parameter tuning. This is body calibration.'),
                tourStep('.calibration-instruction', '体长标定', '体长标定用于把像素运动换算成物理尺度。选择姿态平直、无遮挡的一只小鼠，在鼻尖和尾巴根部之间画线。', 'Body Calibration', 'Body calibration converts pixel movement to physical scale. Use a clear, straight animal and draw between the nose tip and tail base.'),
                tourStep('#calibCanvas', '标定画布', '拖动帧滑块找到清晰画面，在画面上点击鼻尖和尾巴根部。', 'Calibration Canvas', 'Move the frame slider to a clear frame, then click the nose tip and tail base.'),
                tourStep('.stage-side', '标定状态', '这里显示当前视频、标定目标和下一步。确认后进入 ROI 绘制。', 'Calibration Status', 'This shows the current video, calibration target, and next step. Confirm to continue to ROI drawing.')
            ];
        }

        if (module === 'forced_swim') {
            if (inDebug) {
                return [
                    calibrationNavStep('FST', '当前是第 2 步：检查圆筒 ROI、直径线和不动阈值。', 'Step 2 checks cylinder ROIs, diameter lines, and immobility thresholds.'),
                    calibrationActionStep(),
                    tourStep('.fst-preview', '检测预览', '用原始帧和检测结果对照，确认每个圆筒里的小鼠轮廓稳定。', 'Detection Preview', 'Compare original and detected frames to confirm stable contours in every cylinder.'),
                    tourStep('.param-grid', 'FST 参数', '检测参数负责分割鼠体；圆筒直径负责尺度换算；自动不动阈值/灵敏度、平滑窗口、最短不动片段和间隙合并共同决定不动时间。', 'FST Parameters', 'Detection parameters segment the animal; cylinder diameter sets scale; auto threshold, sensitivity, smoothing, minimum immobility, and gap merge define immobility time.'),
                    tourStep('#primaryActionBtn', '开始分析', '预览和阈值确认后，从这里启动分析或保存当前视频配置。', 'Start Analysis', 'After checking preview and thresholds, start analysis or save this video setup here.')
                ];
            }
            return [
                calibrationNavStep('FST', 'FST 两步：先画圆筒 ROI 和直径线，再调检测与不动阈值。', 'FST has two steps: draw cylinder ROIs and diameter lines, then tune detection and immobility thresholds.'),
                tourStep('#defaultDiameterCm', '默认圆筒内径', '圆筒内径用于把像素运动归一化到真实尺寸。每个 ROI 也可以在右侧卡片里单独改。', 'Default Cylinder Diameter', 'Cylinder diameter converts pixels to real scale. Each ROI can override it in the right-side card.'),
                tourStep('#fstCanvas', '圆筒 ROI', '先圈出每个圆筒有效区域，再在 ROI 卡片里画一条横跨内壁的直径线。', 'Cylinder ROI', 'Draw each cylinder’s valid area, then add a diameter line across the inner wall in its ROI card.'),
                tourStep('#cylinderList', '圆筒列表', '这里检查 ROI 是否完整、设置名称、直径线和统计时间窗。', 'Cylinder List', 'Check ROI completeness and set names, diameter lines, and timing windows here.'),
                projectSubjectBindingStep()
            ];
        }

        if (module === 'open_field') {
            if (inDebug) {
                return [
                    calibrationNavStep('OFT', '当前是调参页：检查 arena 检测、中心区/壁区和运动阈值。', 'This tuning step checks arena detection, center/wall zones, and movement threshold.'),
                    calibrationActionStep(),
                    tourStep('.oft-preview', '检测预览', '预览会叠加所有 arena 的边界和检测结果。确认每个 ROI 都能稳定追踪。', 'Detection Preview', 'Preview overlays arena boundaries and detection results for all ROIs. Confirm stable tracking in each ROI.'),
                    tourStep('.param-grid', 'OFT 参数', '背景/阈值/形态学参数影响鼠体检测；中心区比例、壁区边界、运动阈值和分箱时长决定 OFT 指标统计。', 'OFT Parameters', 'Background, threshold, and morphology affect detection; center ratio, wall margin, movement threshold, and bin length define OFT metrics.'),
                    tourStep('#primaryActionBtn', '开始分析', '确认场地和参数后，从这里启动分析或保存当前视频配置。', 'Start Analysis', 'After checking arena and parameters, start analysis or save this video setup here.')
                ];
            }
            return [
                calibrationNavStep('OFT', 'OFT 两步：先画每个 arena，再调检测与统计参数。', 'OFT has two steps: draw each arena, then tune detection and statistics.'),
                tourStep('#arenaSideCm', '旷场边长', '这里填写真实 arena 边长，用于把轨迹换算成厘米。', 'Arena Side Length', 'Enter the real arena side length so trajectories can be converted to centimeters.'),
                tourStep('#arenaCanvas', '四角标定', '每个 arena 点击 4 个角点。一个 arena 对应一个 ROI。', 'Corner Calibration', 'Click four corners for each arena. One arena equals one ROI.'),
                tourStep('#arenaList', 'ROI 列表', '这里命名 ROI、设置统计时间窗；当小鼠躯体可能超出场地边界时，可额外画检测遮罩。', 'ROI List', 'Name ROIs and set timing windows here. Add a detection mask when the mouse body may extend beyond the arena boundary.'),
                projectSubjectBindingStep()
            ];
        }

        if (module === 'novel_object') {
            if (inDebug) {
                return [
                    calibrationNavStep('NOR', '当前是调参页：检查 arena、两个物体点、探索半径和头向辅助。', 'This tuning step checks arena, two object points, exploration radius, and head-direction assistance.'),
                    calibrationActionStep(),
                    tourStep('.nor-preview', '检测预览', '预览会显示 arena、物体接近范围和鼠体检测结果。重点看物体附近是否误判。', 'Detection Preview', 'Preview shows arena, object interaction radius, and animal detection. Watch for false exploration near objects.'),
                    tourStep('[data-tour-id="nor-detection-params"]', '检测参数', '先让绿色整体轮廓稳定包住小鼠主体；灰度阈值、腐蚀/膨胀和最小面积会影响鼠体检测。', 'Detection Parameters', 'First make the green whole-animal contour stable around the body. Gray threshold, erosion/dilation, and minimum area affect detection.'),
                    tourStep('[data-tour-id="nor-body-threshold"]', '第二灰度阈值', '这个阈值专门辅助头朝向：它会把躯干和尾巴分开。看预览里的蓝色躯干、紫色尾巴和黄色鼻尖线，稳定后再调头向判定。', 'Second Gray Threshold', 'This threshold supports heading estimation by separating trunk and tail. Check the blue body, purple tail, and yellow nose line before tuning heading rules.'),
                    tourStep('[data-tour-id="nor-head-direction-guide"]', '头向双阈值', '角度阈值越小越严格，置信度阈值越高越保守。若黄色鼻尖线抖动，优先调第二灰度阈值；若误判探索，再收紧角度或提高置信度。', 'Heading Thresholds', 'A smaller angle threshold is stricter; a higher confidence threshold is more conservative. If the yellow nose line jitters, tune the second gray threshold first; if exploration is overcalled, tighten angle or raise confidence.'),
                    tourStep('#primaryActionBtn', '开始分析', '确认物体点位、探索规则和检测结果后，从这里启动或保存。', 'Start Analysis', 'After checking object points, exploration rules, and detection, start or save here.')
                ];
            }
            return [
                calibrationNavStep('NOR', 'NOR 两步：先选择实验阶段并标定 arena/物体，再调探索判定参数。', 'NOR has two steps: choose phase and mark arena/objects, then tune exploration rules.'),
                tourStep('#phase', '实验阶段', 'Recognition 需要区分 familiar/novel；Familiarization 用于熟悉期记录。', 'Experiment Phase', 'Recognition distinguishes familiar and novel objects; Familiarization records the familiarization phase.'),
                tourStep('#arenaCanvas', 'Arena 与物体点', '固定顺序是 4 个 arena 角点，再点 2 个物体中心。', 'Arena and Object Points', 'Use the fixed order: four arena corners, then two object centers.'),
                tourStep('#roiList', 'ROI 列表', '这里改 ROI 名称、物体角色、物体名称、时间窗和检测遮罩。', 'ROI List', 'Edit ROI names, object roles, object names, timing windows, and detection masks here.'),
                projectSubjectBindingStep()
            ];
        }

        if (module === 'epm' || module === 'ymaze') {
            const isEpm = module === 'epm';
            const name = isEpm ? 'EPM' : 'Y-Maze';
            if (inDebug) {
                return [
                    calibrationNavStep(name, '当前是调参页：检查手画区、检测遮罩和进入区域判定。', 'This tuning step checks drawn zones, detection masks, and zone-entry rules.'),
                    calibrationActionStep(),
                    tourStep('.preview-grid', '区域预览', '左侧原始帧，右侧叠加检测结果和手画区域。确认区域边界与小鼠运动范围吻合。', 'Zone Preview', 'Original frame is on the left; detection and drawn zones are overlaid on the right. Confirm boundaries match animal movement.'),
                    tourStep('.param-grid', `${name} 参数`, isEpm ? '中心区/开臂/闭臂尺寸用于尺度估计；运动阈值、入臂深度比例、分箱时长和标注视频影响 EPM 指标输出。' : '臂长/臂宽用于尺度估计；运动阈值、入臂深度比例、分箱时长和标注视频影响 Y 迷宫交替与活动指标。',
                        `${name} Parameters`, isEpm ? 'Center/open/closed arm sizes estimate scale; movement threshold, entry-depth ratio, bin length, and annotated video affect EPM outputs.' : 'Arm length/width estimate scale; movement threshold, entry-depth ratio, bin length, and annotated video affect Y-Maze alternation and activity outputs.'),
                    tourStep('#primaryActionBtn', '开始分析', '区域和参数确认后，从这里启动分析或保存当前视频配置。', 'Start Analysis', 'After zones and parameters are checked, start analysis or save this video setup here.')
                ];
            }
            return [
                calibrationNavStep(name, isEpm ? 'EPM 两步：先画中心区、开臂、闭臂，再调检测和统计参数。' : 'Y 迷宫两步：先画中心区和三条臂，再调检测和统计参数。',
                    isEpm ? 'EPM has two steps: draw center, open arms, and closed arms, then tune detection/statistics.' : 'Y-Maze has two steps: draw center and three arms, then tune detection/statistics.'),
                tourStep('.maze-panel > .maze-grid', '几何尺寸', isEpm ? '中心区边长、开臂长度和闭臂长度用于估计厘米/像素比例。' : '臂长和臂宽用于估计厘米/像素比例。',
                    'Geometry', isEpm ? 'Center size plus open/closed arm lengths estimate cm per pixel.' : 'Arm length and width estimate cm per pixel.'),
                tourStep('#mazeCanvas', '手画区域', isEpm ? '每个 ROI 需要中心区、两条开臂和两条闭臂。' : '每个 ROI 需要三角中心区和 Arm A/B/C 三条臂。',
                    'Drawn Zones', isEpm ? 'Each ROI needs center, two open arms, and two closed arms.' : 'Each ROI needs a triangular center and Arm A/B/C.'),
                tourStep('#roiList', 'ROI 列表', '这里查看区域完成度、命名 ROI、设置时间窗，也可以编辑检测遮罩。', 'ROI List', 'Review zone completeness, name ROIs, set timing windows, and edit detection masks here.'),
                projectSubjectBindingStep()
            ];
        }

        return [
            calibrationNavStep('实验配置', '当前页面用于完成 ROI、参数和分析启动。', 'This setup page configures ROIs, parameters, and analysis start.'),
            tourStep('canvas', '视频画面', '在这里选点、画 ROI 或查看预览。', 'Video Canvas', 'Use this area to pick points, draw ROIs, or inspect previews.')
        ];
    }

    function registerAutoTours() {
        const declaredPage = document.querySelector('[data-tour-page]')?.getAttribute('data-tour-page') || '';
        if (calibrationShell() && (!declaredPage || declaredPage === 'calibration')) {
            register('calibration', {
                version: 'v5',
                steps: calibrationTourSteps,
            });
            document.body.setAttribute('data-tour-page', 'calibration');
        }
        if (document.querySelector('#progressBar') && document.querySelector('.monitor-card') && (!declaredPage || declaredPage === 'monitor')) {
            register('monitor', {
                version: 'v1',
                steps: [
                    {target: '#progressBar', titleKey: 'tour.monitor.progress.title', bodyKey: 'tour.monitor.progress.body'},
                    {target: '.monitor-grid', titleKey: 'tour.monitor.cards.title', bodyKey: 'tour.monitor.cards.body'},
                    {target: '#downloadSection', titleKey: 'tour.monitor.download.title', bodyKey: 'tour.monitor.download.body'},
                ],
            });
            document.body.setAttribute('data-tour-page', 'monitor');
        }
    }

    document.addEventListener('keydown', event => {
        if (!active) return;
        if (event.key === 'Escape') finish();
        if (event.key === 'ArrowRight') next();
        if (event.key === 'ArrowLeft') prev();
    });
    document.addEventListener('mousescope:langchange', queueRender);
    window.addEventListener('resize', queueRender);
    window.addEventListener('scroll', queueRender, true);
    document.addEventListener('DOMContentLoaded', registerAutoTours);

    window.MouseScopeTour = {register, start, startCurrent, finish};
})();
