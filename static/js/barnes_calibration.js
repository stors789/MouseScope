'use strict';

// Global names are also consumed by the shared pose-model controls.
let currentVideoIdx = Number(barnesPage.videoIndex || 0);
let rois = [];
const barnesConfigs = barnesPage.videoInfos.map((_, i) => ({
    ...structuredClone(barnesPage.defaults),
    ...structuredClone(initialDraftPayload?.per_video_configs?.[i] || (i === 0 ? initialDraftPayload : {}) || {}),
}));
let activeArena = 0;
let barnesImage = null;
let frameRequest = 0;
let gesture = null;
const el = id => document.getElementById(id);
const clone = value => structuredClone(value);
const point = p => Array.isArray(p) ? {x: Number(p[0]), y: Number(p[1])} : p;
const currentConfig = () => barnesConfigs[currentVideoIdx];
const selectedArena = () => rois[activeArena];
const optionalNumber = id => el(id).value.trim() === '' ? null : Number(el(id).value);
const message = (value, error = false) => {
    el('barnesStatus').textContent = value;
    el('barnesStatus').className = `alert ${error ? 'alert-danger' : 'alert-light'} barnes-alert`;
};
const apiError = data => data.error?.message || data.error || data.message || '操作失败';

function currentVideoName(index) {
    return barnesPage.videoInfos[index]?.original_name || `Video ${index + 1}`;
}

function newArena() {
    return {name: `Barnes ${rois.length + 1}`, arena_points: [], holes: [], detection_points: [],
        target_hole_id: '', time_config: {mode: 'global', start: null, end: null}};
}

function normalisePoints(roi) {
    roi.arena_points = (roi.arena_points || []).map(point);
    roi.holes = (roi.holes || []).map(h => ({...h, id: String(h.id), point: point(h.point)}));
    roi.detection_points = (roi.detection_points || []).map(point);
    return roi;
}

function pullForm() {
    const cfg = currentConfig();
    document.querySelectorAll('[data-config-field]').forEach(input => {
        cfg[input.dataset.configField] = input.type === 'number' ? Number(input.value) : input.value;
    });
    Object.assign(cfg, window.MouseScopePoseModels.readConfig());
    cfg.generate_video = el('generateVideo').checked;
    const roi = selectedArena();
    if (roi) {
        roi.name = el('roiName').value;
        roi.subject_id = el('subjectId').value || null;
        const subject = barnesPage.projectSubjects.find(s => s.subject_id === roi.subject_id);
        for (const key of ['animal_code', 'group_id', 'group_name']) roi[key] = subject?.[key] || null;
        for (const key of ['arena_diameter_cm', 'hole_radius_cm', 'hole_count']) roi[key] = cfg[key];
        roi.target_hole_id = el('targetHole').value;
        const start = optionalNumber('roiStart'), end = optionalNumber('roiEnd');
        roi.time_config = {mode: start == null && end == null ? 'global' : 'precise', start, end};
        roi.manual_escape_time_s = optionalNumber('escapeTime');
        roi.manual_hole_events = el('manualEventsEnabled').checked ? parseEvents(el('manualEvents').value) : null;
    }
    cfg.rois = clone(rois);
    return cfg;
}

function parseEvents(value) {
    return value.split(/\r?\n/).map(s => s.trim()).filter(Boolean).map((line, i) => {
        const parts = line.split(/[,，\t]/).map(s => s.trim());
        if (parts.length !== 2 || !parts[0] || !parts[1] || !Number.isFinite(Number(parts[0]))) {
            throw new Error(`人工事件第 ${i + 1} 行格式应为：视频秒,洞号`);
        }
        return {time_s: Number(parts[0]), hole_id: parts[1]};
    });
}

function fillForm() {
    const cfg = currentConfig();
    const roi = selectedArena();
    document.querySelectorAll('[data-config-field]').forEach(input => {
        const key = input.dataset.configField;
        input.value = String((['arena_diameter_cm', 'hole_radius_cm', 'hole_count'].includes(key) ? roi?.[key] : null) ?? cfg[key] ?? barnesPage.defaults[key] ?? '');
    });
    window.MouseScopePoseModels.applyConfig(cfg);
    el('generateVideo').checked = cfg.generate_video !== false;
    el('roiName').value = roi?.name || '';
    el('subjectId').value = roi?.subject_id || '';
    el('roiStart').value = roi?.time_config?.start ?? '';
    el('roiEnd').value = roi?.time_config?.end ?? '';
    el('escapeTime').value = roi?.manual_escape_time_s ?? '';
    el('manualEventsEnabled').checked = Array.isArray(roi?.manual_hole_events);
    el('manualEvents').value = (roi?.manual_hole_events || []).map(e => `${e.time_s},${e.hole_id}`).join('\n');
    refreshLists();
}

function refreshLists() {
    el('arenaSelect').replaceChildren(...rois.map((r, i) => new Option(r.name || `Barnes ${i + 1}`, String(i))));
    el('arenaSelect').value = String(activeArena);
    const roi = selectedArena();
    el('targetHole').replaceChildren(new Option('选择目标洞', ''), ...(roi?.holes || []).map(h => new Option(h.id, h.id)));
    el('targetHole').value = roi?.target_hole_id || '';
    el('geometryStatus').textContent = `边界 ${roi?.arena_points?.length || 0} 点 · 洞位 ${roi?.holes?.length || 0}/${document.querySelector('[data-config-field="hole_count"]').value} · 遮罩 ${roi?.detection_points?.length || 0} 点`;
    el('holeList').replaceChildren();
    for (const h of roi?.holes || []) {
        const row = document.createElement('div');
        row.className = 'd-flex align-items-center gap-2 mb-1';
        const input = document.createElement('input');
        input.className = 'form-control form-control-sm'; input.value = h.id;
        input.setAttribute('aria-label', `洞号 ${h.id}`);
        input.addEventListener('change', () => {
            const old = h.id;
            const id = input.value.trim();
            if (!id || roi.holes.some(other => other !== h && other.id === id)) {
                input.value = old; message('洞号不能为空或重复。', true); return;
            }
            h.id = id;
            if (roi.target_hole_id === old) roi.target_hole_id = id;
            currentConfig().rois = clone(rois); refreshLists(); draw();
        });
        const label = document.createElement('span');
        label.textContent = `(${h.point.x.toFixed(0)}, ${h.point.y.toFixed(0)})`;
        row.append(input, label); el('holeList').append(row);
    }
    draw();
}

function loadVideo() {
    rois = clone(currentConfig().rois || []).map(normalisePoints);
    if (!rois.length) rois.push(newArena());
    activeArena = 0;
    const info = barnesPage.videoInfos[currentVideoIdx];
    el('frameSliderDebug').max = Math.max(0, Number(info.total_frames || 1) - 1);
    el('frameSliderDebug').value = 0;
    fillForm(); loadFrame();
}

async function loadFrame() {
    window.MouseScopePoseModels.clearPreview();
    const request = ++frameRequest;
    const frame = Number(el('frameSliderDebug').value);
    el('frameLabel').textContent = String(frame);
    try {
        const response = await fetch(`/api/get_first_frame/${sessionId}?video=${currentVideoIdx}&frame=${frame}`);
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(apiError(data));
        const image = new Image();
        image.onload = () => {
            if (request !== frameRequest) return;
            barnesImage = image;
            el('barnesCanvas').width = image.width; el('barnesCanvas').height = image.height;
            draw(); paintImage('previewOriginal', image); paintImage('previewResult', image);
        };
        image.src = `data:image/jpeg;base64,${data.image}`;
    } catch (error) { message(error.message, true); }
}

function paintImage(id, image) {
    const canvas = el(id); canvas.width = image.width; canvas.height = image.height;
    canvas.getContext('2d').drawImage(image, 0, 0);
}

function draw() {
    const canvas = el('barnesCanvas'), ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (barnesImage) ctx.drawImage(barnesImage, 0, 0);
    const width = Math.max(2, canvas.width / 450);
    ctx.lineWidth = width; ctx.font = `${Math.max(14, canvas.width / 65)}px sans-serif`;
    function polygon(points, color) {
        if (!points.length) return;
        ctx.strokeStyle = color; ctx.beginPath(); ctx.moveTo(points[0].x, points[0].y);
        for (const p of points.slice(1)) ctx.lineTo(p.x, p.y);
        if (points.length > 2) ctx.closePath(); ctx.stroke();
        ctx.fillStyle = color;
        for (const p of points) { ctx.beginPath(); ctx.arc(p.x, p.y, width * 2, 0, Math.PI * 2); ctx.fill(); }
    }
    rois.forEach((r, index) => {
        ctx.globalAlpha = index === activeArena ? 1 : .45;
        polygon(r.arena_points, '#36ccda'); polygon(r.detection_points, '#d284ff');
        const xs = r.arena_points.map(p => p.x), ys = r.arena_points.map(p => p.y);
        const diameter = Number(r.arena_diameter_cm || currentConfig().arena_diameter_cm || 90);
        const rx = Math.max(4, (Math.max(...xs) - Math.min(...xs)) * Number(r.hole_radius_cm || currentConfig().hole_radius_cm || 2) / diameter);
        const ry = Math.max(4, (Math.max(...ys) - Math.min(...ys)) * Number(r.hole_radius_cm || currentConfig().hole_radius_cm || 2) / diameter);
        for (const h of r.holes) {
            ctx.strokeStyle = h.id === r.target_hole_id ? '#ffe04a' : '#ff7757';
            ctx.beginPath(); ctx.ellipse(h.point.x, h.point.y, rx, ry, 0, 0, Math.PI * 2); ctx.stroke();
            ctx.fillStyle = ctx.strokeStyle; ctx.fillText(h.id, h.point.x + 6, h.point.y - 6);
        }
    });
    ctx.globalAlpha = 1;
}

function requestBody() {
    pullForm();
    return {...clone(currentConfig()), video: currentVideoIdx, frame: Number(el('frameSliderDebug').value),
        per_video_configs: clone(barnesConfigs), _selected_video_idx: currentVideoIdx};
}

async function saveConfig() {
    const response = await fetch(`/api/draft/${sessionId}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(requestBody())});
    const data = await response.json();
    if (!response.ok || data.success === false) throw new Error(apiError(data));
    message('配置已保存，包括目标洞、试次、DLC开关和人工复核记录。');
}

async function preview() {
    const button = el('previewButton'); button.disabled = true;
    el('previewStatus').textContent = '正在生成预览…';
    try {
        const body = requestBody();
        const response = await fetch(`/api/debug_preview/${sessionId}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(apiError(data));
        for (const [id, key] of [['previewOriginal', 'original'], ['previewResult', 'result']]) {
            const image = new Image(); image.onload = () => paintImage(id, image);
            image.src = `data:image/jpeg;base64,${data[key]}`;
        }
        selectedArena().reviewed_frame = body.frame;
        selectedArena().roi_reviewed = true;
        el('previewStatus').textContent = `实际后端：${data.backend} · 检出 ${data.detections} 只。鼻尖若缺失，请调整检测或启用DLC并检查模型。`;
    } catch (error) { el('previewStatus').textContent = error.message; message(error.message, true); }
    finally { button.disabled = false; }
}

function canvasPoint(event) {
    const canvas = el('barnesCanvas'), rect = canvas.getBoundingClientRect();
    return {x: Math.max(0, Math.min(canvas.width - 1, (event.clientX - rect.left) * canvas.width / rect.width)),
        y: Math.max(0, Math.min(canvas.height - 1, (event.clientY - rect.top) * canvas.height / rect.height))};
}

function editPoints() {
    const r = selectedArena();
    return r ? [...r.arena_points, ...r.holes.map(h => h.point), ...r.detection_points] : [];
}

function onPointerDown(event) {
    const r = selectedArena(); if (!r || !barnesImage) return;
    const p = canvasPoint(event), mode = el('drawMode').value;
    if (mode === 'ellipse') gesture = {start: p, mode};
    else {
        const radius = 10 * el('barnesCanvas').width / el('barnesCanvas').clientWidth;
        const closest = editPoints().find(q => Math.hypot(q.x - p.x, q.y - p.y) < radius);
        if (closest) gesture = {point: closest, mode: 'drag'};
        else if (mode === 'boundary') r.arena_points.push(p);
        else if (mode === 'mask') r.detection_points.push(p);
        else if (mode === 'holes') {
            const count = Number(document.querySelector('[data-config-field="hole_count"]').value);
            if (r.holes.length >= count) { message('已达到全部洞数；可拖动标记修正。'); return; }
            let id = 1; while (r.holes.some(h => h.id === String(id))) id++;
            r.holes.push({id: String(id), point: p});
        }
    }
    el('barnesCanvas').setPointerCapture(event.pointerId);
    r.roi_reviewed = false; refreshLists();
}

function onPointerMove(event) {
    if (!gesture) return;
    const p = canvasPoint(event);
    if (gesture.mode === 'drag') Object.assign(gesture.point, p);
    else if (gesture.mode === 'ellipse') {
        const a = gesture.start, cx = (p.x + a.x) / 2, cy = (p.y + a.y) / 2;
        selectedArena().arena_points = Array.from({length: 32}, (_, i) => {
            const angle = i * Math.PI / 16;
            return {x: cx + Math.abs(p.x - a.x) / 2 * Math.cos(angle), y: cy + Math.abs(p.y - a.y) / 2 * Math.sin(angle)};
        });
    }
    draw();
}

function safe(action) {
    return async event => { try { await action(event); } catch (error) { message(error.message, true); } };
}

document.addEventListener('DOMContentLoaded', () => {
    el('videoSelect').replaceChildren(...barnesPage.videoInfos.map((_, i) => new Option(currentVideoName(i), String(i))));
    el('videoSelect').value = String(currentVideoIdx);
    el('videoSelect').addEventListener('change', safe(() => { pullForm(); currentVideoIdx = Number(el('videoSelect').value); loadVideo(); }));
    el('arenaSelect').addEventListener('change', safe(() => { const next = Number(el('arenaSelect').value); pullForm(); activeArena = next; fillForm(); }));
    el('addArena').addEventListener('click', safe(() => { pullForm(); rois.push(newArena()); activeArena = rois.length - 1; fillForm(); }));
    el('removeArena').addEventListener('click', () => { rois.splice(activeArena, 1); if (!rois.length) rois.push(newArena()); activeArena = 0; fillForm(); });
    el('targetHole').addEventListener('change', safe(() => { pullForm(); draw(); }));
    for (const id of ['undoPoint', 'clearPoints']) el(id).addEventListener('click', () => {
        const r = selectedArena(), mode = el('drawMode').value;
        const key = mode === 'holes' ? 'holes' : mode === 'mask' ? 'detection_points' : 'arena_points';
        if (id === 'clearPoints') r[key] = []; else r[key].pop();
        r.roi_reviewed = false; refreshLists();
    });
    const canvas = el('barnesCanvas');
    canvas.addEventListener('pointerdown', onPointerDown); canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', () => { gesture = null; refreshLists(); });
    canvas.addEventListener('pointercancel', () => { gesture = null; });
    el('frameSliderDebug').addEventListener('change', loadFrame);
    el('seekFrame').addEventListener('click', () => {
        el('frameSliderDebug').value = Math.round(Number(el('seekSeconds').value) * Number(currentConfig().fps_override || barnesPage.videoInfos[currentVideoIdx].fps || 30)); loadFrame();
    });
    el('saveConfig').addEventListener('click', safe(saveConfig));
    el('previewButton').addEventListener('click', safe(preview));
    el('primaryActionBtn').addEventListener('click', safe(async () => {
        if (barnesPage.workspaceMode) { await saveConfig(); window.location.href = barnesPage.workspaceUrl; return; }
        el('primaryActionBtn').disabled = true;
        try {
            const response = await fetch(`/api/start_analysis/${sessionId}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(requestBody())});
            const data = await response.json();
            if (!response.ok || data.success === false) throw new Error(apiError(data));
            window.location.href = `/monitor/${sessionId}`;
        } finally { el('primaryActionBtn').disabled = false; }
    }));
    loadVideo();
});

// The shared template service stores device geometry. Trial metadata stays local.
window.MouseScopeRoiTemplateAdapter = {
    assay: 'barnes_maze', isBatch: barnesPage.isBatch, videoCount: barnesConfigs.length,
    getPayload: () => {
        pullForm();
        return {rois: clone(rois).map(r => ({...r, detection_points: r.detection_points?.length ? r.detection_points : null})), geometry: Object.fromEntries(['arena_diameter_cm', 'hole_radius_cm', 'hole_count'].map(k => [k, currentConfig()[k]]))};
    },
    getFrameMeta: index => {
        const info = barnesPage.videoInfos[index ?? currentVideoIdx];
        return {width: index == null ? barnesImage?.width || info.width : info.width,
            height: index == null ? barnesImage?.height || info.height : info.height,
            orientation_degrees: info.orientation_degrees || 0};
    },
    applyPayload: (payloads, all) => {
        pullForm();
        const indices = all ? barnesConfigs.map((_, i) => i) : [currentVideoIdx];
        indices.forEach((index, p) => {
            const cfg = barnesConfigs[index], original = cfg.rois || [];
            cfg.rois = payloads[p].rois.map((r, i) => ({...original[i], ...r,
                name: original[i]?.name || r.name, target_hole_id: original[i]?.target_hole_id || '',
                roi_reviewed: false}));
            Object.assign(cfg, payloads[p].geometry);
        });
        loadVideo(); message('设备几何已应用。请逐视频检查位置，核对目标洞和试次。');
    },
};
