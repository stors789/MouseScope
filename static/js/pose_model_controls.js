(function () {
    'use strict';

    const state = {
        root: null,
        assay: '',
        sessionId: '',
        profiles: [],
        runtime: null,
        prediction: null,
        baseCanvasImage: null,
        draggedKey: null,
        corrections: [],
        requestedSelection: null,
    };

    function byId(id) { return document.getElementById(id); }
    function text(id, value) { const el = byId(id); if (el) el.textContent = value; }
    function selectedProfile() {
        const id = byId('poseModelProfile')?.value || '';
        return state.profiles.find(item => item.profile_id === id) || null;
    }
    function selectedVersion() {
        const profile = selectedProfile();
        const id = byId('poseModelVersion')?.value || '';
        return (profile?.versions || []).find(item => item.version_id === id) || null;
    }
    function errorText(payload, fallback) {
        return payload?.error?.message || payload?.error || payload?.message || fallback;
    }
    function currentFrameIndex() {
        return parseInt(byId('frameSliderDebug')?.value || '0', 10) || 0;
    }
    function currentVideoIndex() {
        try { return Number(currentVideoIdx || 0); } catch (_error) { return 0; }
    }
    function currentRois() {
        try { return Array.isArray(rois) ? rois : []; } catch (_error) { return []; }
    }
    function currentVideoReference() {
        try {
            return `${state.sessionId}:${currentVideoName(currentVideoIndex())}`;
        } catch (_error) {
            return `${state.sessionId}:video-${currentVideoIndex() + 1}`;
        }
    }

    function readConfig() {
        return {
            tracking_backend: byId('trackingBackend')?.value || 'classical',
            pose_model_profile_id: byId('poseModelProfile')?.value || null,
            pose_model_version_id: byId('poseModelVersion')?.value || null,
            pose_confidence_threshold: Number(byId('poseConfidenceThreshold')?.value || 0.8),
            pose_batch_size: Math.max(1, Math.min(16, Number(byId('poseBatchSize')?.value || 4))),
        };
    }

    function applyConfig(config) {
        const raw = config || {};
        const backend = byId('trackingBackend');
        const confidence = byId('poseConfidenceThreshold');
        const batchSize = byId('poseBatchSize');
        if (backend && raw.tracking_backend) backend.value = raw.tracking_backend;
        if (confidence && raw.pose_confidence_threshold != null) confidence.value = String(raw.pose_confidence_threshold);
        if (batchSize && raw.pose_batch_size != null) batchSize.value = String(raw.pose_batch_size);
        state.requestedSelection = {
            profileId: raw.pose_model_profile_id || '',
            versionId: raw.pose_model_version_id || '',
        };
        syncBackendUi();
        applyRequestedSelection();
    }

    function syncBackendUi() {
        const strictDlc = byId('trackingBackend')?.value === 'deeplabcut';
        const classicalControls = byId('classicalDetectionControls');
        if (classicalControls) classicalControls.hidden = strictDlc;
    }

    function applyRequestedSelection() {
        if (!state.requestedSelection) return;
        const profileSelect = byId('poseModelProfile');
        const profileAvailable = profileSelect
            && Array.from(profileSelect.options).some(option => option.value === state.requestedSelection.profileId);
        if (profileAvailable) {
            profileSelect.value = state.requestedSelection.profileId;
            rebuildVersionOptions();
            const versionSelect = byId('poseModelVersion');
            if (versionSelect && Array.from(versionSelect.options).some(option => option.value === state.requestedSelection.versionId)) {
                versionSelect.value = state.requestedSelection.versionId;
            }
            state.requestedSelection = null;
        }
        syncButtons();
    }

    function rebuildProfileOptions() {
        const select = byId('poseModelProfile');
        if (!select) return;
        select.innerHTML = '<option value="">SuperAnimal-TopViewMouse（零样本）</option>';
        state.profiles.forEach(profile => {
            const option = document.createElement('option');
            option.value = profile.profile_id;
            option.textContent = `${profile.name}（${(profile.versions || []).length} 个版本）`;
            select.appendChild(option);
        });
        applyRequestedSelection();
        rebuildVersionOptions();
    }

    function rebuildVersionOptions() {
        const select = byId('poseModelVersion');
        if (!select) return;
        const profile = selectedProfile();
        select.innerHTML = '';
        if (!profile) {
            select.innerHTML = '<option value="">预训练基础版本</option>';
            return;
        }
        (profile.versions || []).slice().reverse().forEach(version => {
            const option = document.createElement('option');
            option.value = version.version_id;
            const status = version.training?.status || 'unknown';
            option.textContent = `v${version.version_number} · ${status}`;
            select.appendChild(option);
        });
        syncButtons();
    }

    async function loadProfiles() {
        const response = await fetch(`/api/pose_models?assay=${encodeURIComponent(state.assay)}`);
        const payload = await response.json();
        if (!response.ok || !payload.success) throw new Error(errorText(payload, '无法读取姿态模型档案'));
        state.profiles = payload.profiles || [];
        rebuildProfileOptions();
    }

    async function loadRuntime() {
        const response = await fetch('/api/pose_models/runtime');
        const payload = await response.json();
        state.runtime = payload;
        const badge = byId('poseRuntimeBadge');
        if (badge) {
            badge.className = `badge ${payload.available ? 'text-bg-success' : 'text-bg-warning'}`;
            badge.textContent = payload.available ? 'DLC 预览/分析可用' : '将回退经典视觉';
        }
        text('poseModelStatus', payload.message || 'DLC 运行状态未知。');
    }

    function normalizePrediction(payload) {
        const raw = payload.keypoints || {};
        const points = {};
        Object.entries(raw).forEach(([name, value]) => {
            const point = Array.isArray(value) ? {x: value[0], y: value[1], likelihood: value[2]} : value;
            if (!point || !Number.isFinite(Number(point.x)) || !Number.isFinite(Number(point.y))) return;
            points[name] = {
                x: Number(point.x),
                y: Number(point.y),
                likelihood: Number(point.likelihood ?? point.confidence ?? 1),
                visible: true,
            };
        });
        return {
            keypoints: points,
            frame: Number(payload.frame || currentFrameIndex()),
            frameSize: payload.frame_size || null,
        };
    }

    function drawPrediction() {
        const canvas = byId('previewResult');
        if (!canvas || !state.prediction) return;
        const ctx = canvas.getContext('2d');
        if (state.baseCanvasImage && state.baseCanvasImage.width === canvas.width && state.baseCanvasImage.height === canvas.height) {
            ctx.putImageData(state.baseCanvasImage, 0, 0);
        }
        ctx.save();
        ctx.font = `${Math.max(12, canvas.width / 80)}px sans-serif`;
        ctx.lineWidth = Math.max(2, canvas.width / 400);
        Object.entries(state.prediction.keypoints).forEach(([name, point]) => {
            ctx.beginPath();
            ctx.fillStyle = name === 'nose' ? '#facc15' : (name === 'body_center' ? '#22c55e' : '#38bdf8');
            ctx.strokeStyle = '#111827';
            ctx.arc(point.x, point.y, Math.max(5, canvas.width / 160), 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = '#ffffff';
            ctx.strokeStyle = '#111827';
            ctx.strokeText(name, point.x + 8, point.y - 8);
            ctx.fillText(name, point.x + 8, point.y - 8);
        });
        ctx.restore();
    }

    function installCanvasEditing() {
        const canvas = byId('previewResult');
        if (!canvas || canvas.dataset.poseEditingInstalled === '1') return;
        canvas.dataset.poseEditingInstalled = '1';
        function pointFromEvent(event) {
            const rect = canvas.getBoundingClientRect();
            return {
                x: (event.clientX - rect.left) * canvas.width / Math.max(rect.width, 1),
                y: (event.clientY - rect.top) * canvas.height / Math.max(rect.height, 1),
            };
        }
        canvas.addEventListener('pointerdown', event => {
            if (!state.prediction) return;
            const here = pointFromEvent(event);
            let best = null;
            let bestDistance = Infinity;
            Object.entries(state.prediction.keypoints).forEach(([name, point]) => {
                const distance = Math.hypot(point.x - here.x, point.y - here.y);
                if (distance < bestDistance) { best = name; bestDistance = distance; }
            });
            if (best && bestDistance <= Math.max(18, canvas.width / 30)) {
                state.draggedKey = best;
                canvas.setPointerCapture(event.pointerId);
            }
        });
        canvas.addEventListener('pointermove', event => {
            if (!state.draggedKey || !state.prediction) return;
            const point = pointFromEvent(event);
            state.prediction.keypoints[state.draggedKey].x = point.x;
            state.prediction.keypoints[state.draggedKey].y = point.y;
            drawPrediction();
        });
        canvas.addEventListener('pointerup', () => { state.draggedKey = null; });
        canvas.addEventListener('pointercancel', () => { state.draggedKey = null; });
    }

    async function runPreview() {
        const button = byId('posePreviewButton');
        if (button) button.disabled = true;
        text('poseModelStatus', '正在运行姿态关键点预览…');
        try {
            const config = readConfig();
            const response = await fetch(`/api/pose_preview/${encodeURIComponent(state.sessionId)}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    frame: currentFrameIndex(),
                    video: currentVideoIndex(),
                    rois: currentRois(),
                    profile_id: config.pose_model_profile_id,
                    version_id: config.pose_model_version_id,
                }),
            });
            const payload = await response.json();
            if (!response.ok || !payload.success) {
                state.prediction = null;
                syncButtons();
                text('poseModelStatus', `${errorText(payload, '姿态预览失败')} 当前预览继续使用经典视觉。`);
                return;
            }
            state.prediction = normalizePrediction(payload);
            const canvas = byId('previewResult');
            if (canvas?.width && canvas?.height) {
                state.baseCanvasImage = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
            }
            drawPrediction();
            syncButtons();
            text('poseModelStatus', '姿态预览完成。可拖动右侧结果图中的关键点，再保存当前纠错帧。');
        } catch (error) {
            text('poseModelStatus', `姿态预览失败：${error.message}`);
        } finally {
            if (button) button.disabled = false;
        }
    }

    async function createProfile() {
        const name = window.prompt('模型档案名称（例如：动物房A-顶视黑鼠）');
        if (!name) return;
        const response = await fetch('/api/pose_models', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, assays: [state.assay]}),
        });
        const payload = await response.json();
        if (!response.ok || !payload.success) {
            window.alert(errorText(payload, '保存模型档案失败'));
            return;
        }
        state.requestedSelection = {
            profileId: payload.profile.profile_id,
            versionId: payload.profile.latest_version_id,
        };
        await loadProfiles();
        text('poseModelStatus', '已保存零样本基础版本。历史版本不会被覆盖。');
    }

    function addCorrection() {
        if (!state.prediction) return;
        const validation = window.confirm('是否将这一帧加入固定验证集？\n“取消”会加入训练集。');
        state.corrections.push({
            video_ref: currentVideoReference(),
            frame_index: state.prediction.frame,
            frame_size: state.prediction.frameSize,
            coordinate_space: 'pixels',
            split: validation ? 'validation' : 'train',
            keypoints: JSON.parse(JSON.stringify(state.prediction.keypoints)),
        });
        text('poseCorrectionStatus', `待生成新版本：${state.corrections.length} 帧纠错（历史标注会自动继承）。`);
        syncButtons();
    }

    async function createVersion() {
        const profile = selectedProfile();
        const version = selectedVersion();
        if (!profile || !version || !state.corrections.length) return;
        const response = await fetch(`/api/pose_models/${encodeURIComponent(profile.profile_id)}/versions`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                parent_version: version.version_id,
                annotations: state.corrections,
                training_status: 'awaiting_external_training',
            }),
        });
        const payload = await response.json();
        if (!response.ok || !payload.success) {
            window.alert(errorText(payload, '生成新版本失败'));
            return;
        }
        state.corrections = [];
        state.requestedSelection = {profileId: profile.profile_id, versionId: payload.version.version_id};
        await loadProfiles();
        text('poseCorrectionStatus', '');
        text('poseModelStatus', `v${payload.version.version_number} 已保存，状态为等待外部 DLC 训练；旧版本保持不变。`);
        syncButtons();
    }

    function syncButtons() {
        const hasPrediction = Boolean(state.prediction);
        const hasVersion = Boolean(selectedVersion());
        const correctionButton = byId('poseAddCorrectionButton');
        const versionButton = byId('poseCreateVersionButton');
        if (correctionButton) correctionButton.disabled = !hasPrediction || !hasVersion;
        if (versionButton) versionButton.disabled = !hasVersion || !state.corrections.length;
    }

    async function init() {
        state.root = document.querySelector('[data-pose-model-controls]');
        if (!state.root) return;
        state.assay = state.root.dataset.assay || '';
        state.sessionId = state.root.dataset.sessionId || '';
        byId('poseModelProfile')?.addEventListener('change', rebuildVersionOptions);
        byId('poseModelVersion')?.addEventListener('change', syncButtons);
        byId('trackingBackend')?.addEventListener('change', syncBackendUi);
        byId('posePreviewButton')?.addEventListener('click', runPreview);
        byId('poseCreateProfileButton')?.addEventListener('click', createProfile);
        byId('poseAddCorrectionButton')?.addEventListener('click', addCorrection);
        byId('poseCreateVersionButton')?.addEventListener('click', createVersion);
        installCanvasEditing();
        syncBackendUi();
        try {
            let draftConfig = {};
            try {
                const perVideo = initialDraftPayload?.per_video_configs || [];
                draftConfig = perVideo[currentVideoIndex()] || initialDraftPayload || {};
            } catch (_error) {}
            applyConfig(draftConfig);
            await Promise.all([loadProfiles(), loadRuntime()]);
        } catch (error) {
            text('poseModelStatus', `姿态模型档案初始化失败：${error.message}；经典视觉不受影响。`);
        }
    }

    window.MouseScopePoseModels = {readConfig, applyConfig, init};
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
