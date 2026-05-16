const projectConfig = window.MouseScopeProject || {};
const projectId = projectConfig.projectId || document.querySelector('.project-shell').dataset.projectId;
const projectExperiments = projectConfig.experiments || [];
let projectState = {project: {}, groups: [], subjects: [], videos: [], assignments: [], runs: []};
let uploadedProjectFiles = [];
let projectVideoUploading = false;
let projectRefreshTimer = null;

function currentLang() {
    return typeof getLang === 'function' ? getLang() : (localStorage.getItem('mousescope_lang') || 'zh');
}

function tr(key, fallback = '') {
    const dict = (typeof I18N !== 'undefined' && I18N[currentLang()]) || {};
    return dict[key] ?? fallback;
}

function tf(key, values = {}, fallback = '') {
    return tr(key, fallback).replace(/\{(\w+)\}/g, (_, name) => values[name] ?? '');
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
}

function formatTime(ts) {
    if (!ts) return '-';
    try { return new Date(Number(ts) * 1000).toLocaleString(currentLang() === 'zh' ? 'zh-CN' : 'en-US'); } catch (_) { return '-'; }
}

function statusText(status) {
    const key = `project.status.${status || 'unknown'}`;
    return tr(key, status || tr('project.status.unknown', '未知'));
}

function alertBox(message, type = 'info') {
    const box = document.getElementById('projectAlert');
    box.className = `alert alert-${type}`;
    box.textContent = message;
    box.style.display = message ? 'block' : 'none';
}

function setVideoUploadStatus(message, type = 'muted') {
    const node = document.getElementById('projectVideoUploadStatus');
    if (!node) return;
    node.className = `small mt-2 text-${type}`;
    node.textContent = message || '';
}

function renderUploadedProjectFiles() {
    const list = document.getElementById('projectUploadedFiles');
    if (!list) return;
    list.innerHTML = uploadedProjectFiles.map((file, index) => `
        <span class="uploaded-file-chip" title="${escapeHtml(file.name || file.path)}">
            <i class="fas fa-file-video"></i><span>${escapeHtml(file.name || file.path)}</span>
            <button type="button" class="btn btn-link btn-sm p-0 text-danger" onclick="removeUploadedProjectFile(${index})" aria-label="Remove">
                <i class="fas fa-xmark"></i>
            </button>
        </span>
    `).join('');
}

function removeUploadedProjectFile(index) {
    uploadedProjectFiles.splice(index, 1);
    renderUploadedProjectFiles();
    const count = uploadedProjectFiles.length;
    setVideoUploadStatus(count ? tf('project.videos.uploaded_count', {count}, '已选择 {count} 个视频文件') : '', count ? 'success' : 'muted');
}

function uploadProjectVideoFiles(files) {
    const selected = Array.from(files || []).filter(file => file && file.name);
    if (!selected.length) return;
    const formData = new FormData();
    selected.forEach(file => formData.append('files', file));
    projectVideoUploading = true;
    setVideoUploadStatus(tr('project.videos.uploading', '正在上传视频...'), 'muted');
    fetch('/api/upload_files', {method: 'POST', body: formData})
        .then(r => r.json().then(data => ({ok: r.ok, data})))
        .then(({ok, data}) => {
            if (!ok || !data.success) throw new Error(data.error || tr('project.videos.upload_failed', '视频上传失败'));
            const paths = data.paths || [];
            const names = data.names || [];
            paths.forEach((path, index) => {
                uploadedProjectFiles.push({path, name: names[index] || path});
            });
            renderUploadedProjectFiles();
            setVideoUploadStatus(tf('project.videos.uploaded_count', {count: uploadedProjectFiles.length}, '已选择 {count} 个视频文件'), 'success');
        })
        .catch(error => setVideoUploadStatus(error.message || tr('project.videos.upload_failed', '视频上传失败'), 'danger'))
        .finally(() => { projectVideoUploading = false; });
}

function api(path, options = {}) {
    return fetch(path, options).then(r => r.json()).then(data => {
        if (!data.success) throw new Error(data.error || tr('project.error.default', '操作失败'));
        return data;
    });
}

function showTab(name) {
    document.querySelectorAll('.project-tab').forEach(btn => btn.classList.toggle('active', btn.dataset.tab === name));
    document.querySelectorAll('.section-pane').forEach(pane => pane.classList.toggle('active', pane.id === `pane-${name}`));
}

function groupOptions(selected = '') {
    return `<option value="">${escapeHtml(tr('project.option.ungrouped', '未分组'))}</option>` + projectState.groups.map(group => `
        <option value="${escapeHtml(group.group_id)}" ${group.group_id === selected ? 'selected' : ''}>${escapeHtml(group.name)}</option>
    `).join('');
}

function subjectsForGroup(groupId) {
    return (projectState.subjects || []).filter(subject => String(subject.group_id || '') === String(groupId || ''));
}

function assignmentForVideo(videoId) {
    return (projectState.assignments || []).find(item => String(item.video_id || '') === String(videoId || '')) || null;
}

function experimentOptions(selected = '') {
    const empty = `<option value="">${escapeHtml(tr('project.videos.experiment_none', '未选择实验'))}</option>`;
    return empty + projectExperiments.map(spec => `
        <option value="${escapeHtml(spec.key)}" ${spec.key === selected ? 'selected' : ''}>${escapeHtml(spec.label)}</option>
    `).join('');
}

function isCleanedUploadCache(video) {
    const notes = String(video.notes || video.video_notes || '');
    return String(video.storage_mode || video.video_storage_mode || '').toLowerCase() === 'cleaned' || notes.includes('upload_cache:cleaned');
}

function videoPathStatus(video) {
    if (isCleanedUploadCache(video)) {
        return {cls: 'text-muted', text: tr('project.cache_cleaned', '上传副本已清理')};
    }
    return video.path_exists
        ? {cls: 'text-success', text: tr('project.exists', '存在')}
        : {cls: 'text-danger', text: tr('project.missing', '缺失')};
}

function videoPathText(video) {
    if (isCleanedUploadCache(video)) {
        return tr('project.cache_cleaned.hint', '已保留分析记录；需要重新分析时请重新导入视频');
    }
    return video.project_path || video.source_path || '';
}

function renderStats() {
    const p = projectState.project || {};
    const stats = [
        [tr('project.stat.groups', '组别'), p.group_count || 0],
        [tr('project.stat.subjects', '小鼠'), p.subject_count || 0],
        [tr('project.stat.videos', '视频'), p.video_count || 0],
        [tr('project.stat.assignments', '实验记录'), p.assignment_count || 0],
        [tr('project.stat.completed', '已完成'), p.completed_assignment_count || 0],
    ];
    document.getElementById('statGrid').innerHTML = stats.map(([label, value]) => `<div class="stat-box"><strong>${value}</strong><span>${escapeHtml(label)}</span></div>`).join('');
}

function renderGroups() {
    document.getElementById('groupList').innerHTML = projectState.groups.length
        ? projectState.groups.map(group => {
            const subjectCount = subjectsForGroup(group.group_id).length;
            return `
            <div class="group-card">
                <div class="group-card-head">
                    <div>
                        <div><span class="color-dot me-2" style="background:${escapeHtml(group.color || '#c96442')}"></span><strong>${escapeHtml(group.name)}</strong></div>
                        <div class="mini-note">${escapeHtml(tf('project.groups.subject_count', {count: subjectCount}, '{count} 只小鼠'))}</div>
                    </div>
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteGroup('${escapeHtml(group.group_id)}')" title="${escapeHtml(tr('tasks.action.delete', '删除记录'))}"><i class="fas fa-trash"></i></button>
                </div>
                <div class="group-subject-tools">
                    <div>
                        <label class="form-label small mb-1">${escapeHtml(tr('project.subjects.count_label', '数量'))}</label>
                        <input class="form-control form-control-sm" id="subjectCount-${escapeHtml(group.group_id)}" type="number" min="1" max="500" step="1" value="8">
                    </div>
                    <button class="btn btn-primary btn-sm" type="button" onclick="generateGroupSubjects('${escapeHtml(group.group_id)}')">
                        <i class="fas fa-plus me-1"></i>${escapeHtml(tr('project.subjects.generate_btn', '生成小鼠'))}
                    </button>
                </div>
            </div>
        `}).join('')
        : `<div class="mini-note">${escapeHtml(tr('project.groups.empty', '还没有组别。'))}</div>`;
}

function renderSubjects() {
    document.getElementById('subjectTable').innerHTML = projectState.subjects.length
        ? projectState.subjects.map(subject => `
            <tr>
                <td>
                    <input class="form-control form-control-sm subject-code-input"
                           value="${escapeHtml(subject.animal_code)}"
                           onchange="updateSubjectCode('${escapeHtml(subject.subject_id)}', this.value)">
                </td>
                <td>
                    <select class="form-select form-select-sm subject-select" onchange="updateSubjectGroup('${escapeHtml(subject.subject_id)}', this.value)">
                        ${groupOptions(subject.group_id || '')}
                    </select>
                </td>
                <td class="text-end"><button class="btn btn-sm btn-outline-danger" onclick="deleteSubject('${escapeHtml(subject.subject_id)}')" title="${escapeHtml(tr('tasks.action.delete', '删除记录'))}"><i class="fas fa-trash"></i></button></td>
            </tr>
        `).join('')
        : `<tr><td colspan="3" class="text-muted text-center py-3">${escapeHtml(tr('project.empty.subjects', '还没有小鼠。'))}</td></tr>`;
}

function renderVideos() {
    document.getElementById('videoTable').innerHTML = projectState.videos.length
        ? projectState.videos.map(video => {
            const assignment = assignmentForVideo(video.video_id);
            const pathStatus = videoPathStatus(video);
            const width = video.width || '-';
            const height = video.height || '-';
            const fps = video.fps ? `${Number(video.fps).toFixed(2)} fps` : '-';
            return `
            <div class="video-card">
                <div class="video-card-main">
                    <div class="video-title">${escapeHtml(video.original_name)}</div>
                    <div class="video-path-line">
                        <span class="video-status ${pathStatus.cls}">${escapeHtml(pathStatus.text)}</span>
                        <span class="ms-2">${escapeHtml(videoPathText(video))}</span>
                    </div>
                    <div class="video-meta">
                        <span class="meta-pill"><i class="fas fa-expand"></i>${escapeHtml(`${width} x ${height}`)}</span>
                        <span class="meta-pill"><i class="fas fa-gauge-high"></i>${escapeHtml(fps)}</span>
                    </div>
                </div>
                <div class="video-experiment">
                    <div class="video-experiment-label">${escapeHtml(tr('project.videos.experiment', '实验类型'))}</div>
                    <select class="form-select form-select-sm" onchange="setVideoExperiment('${escapeHtml(video.video_id)}', this.value)">${experimentOptions(assignment ? assignment.experiment_type : '')}</select>
                    ${assignment ? '' : `<div class="experiment-required">${escapeHtml(tr('project.videos.type_required_short', '选择后才可进入项目分析'))}</div>`}
                </div>
                <div class="video-actions">
                    <button class="btn btn-sm btn-outline-danger" type="button" onclick="deleteVideo('${escapeHtml(video.video_id)}')" title="${escapeHtml(tr('project.videos.delete', '删除视频'))}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `}).join('')
        : `<div class="empty-state">${escapeHtml(tr('project.empty.videos', '还没有导入视频。'))}</div>`;
}

function renderAssignments() {
    document.getElementById('assignmentTable').innerHTML = projectState.assignments.length
        ? projectState.assignments.map(item => {
            const sourceAvailable = !!item.path_exists;
            const sourceNote = isCleanedUploadCache(item)
                ? tr('project.cache_cleaned.hint', '已保留分析记录；需要重新分析时请重新导入视频')
                : (!sourceAvailable ? tr('project.missing', '缺失') : '');
            return `
            <div class="assignment-card ${sourceAvailable ? '' : 'is-disabled'}">
                <div>
                    <input type="checkbox" class="form-check-input assignment-check" value="${escapeHtml(item.assignment_id)}" ${sourceAvailable ? '' : 'disabled'} onchange="updateAssignmentSelectionSummary()">
                </div>
                <div><span class="experiment-badge"><i class="fas fa-flask"></i>${escapeHtml(experimentLabel(item.experiment_type))}</span></div>
                <div class="assignment-main">
                    <div class="assignment-title">${escapeHtml(item.original_name)}</div>
                    ${sourceNote ? `<div class="small text-muted mt-1">${escapeHtml(sourceNote)}</div>` : ''}
                </div>
                <div><span class="run-status">${escapeHtml(statusText(item.status || 'pending'))}</span></div>
            </div>
        `}).join('')
        : `<div class="empty-state">${escapeHtml(tr('project.empty.assignments', '还没有视频实验记录。请先在视频库中为视频选择实验类型。'))}</div>`;
    updateAssignmentSelectionSummary();
}

function experimentLabel(type) {
    const spec = projectExperiments.find(item => item.key === type);
    return spec ? spec.label : (type || '-');
}

function assignmentsForRun(run) {
    const ids = new Set((run.assignment_ids || []).map(item => String(item)));
    return (projectState.assignments || []).filter(item => ids.has(String(item.assignment_id || '')));
}

function openRunFolder(projectRunId) {
    api(`/api/projects/${encodeURIComponent(projectId)}/runs/${encodeURIComponent(projectRunId)}/open_folder`, {method: 'POST'})
        .catch(error => alert(error.message || String(error)));
}

function renderRuns(targetId = 'runList', limit = 0) {
    const runs = limit ? projectState.runs.slice(0, limit) : projectState.runs;
    const html = runs.length
        ? `<div class="result-grid">${runs.map(run => {
            const assignments = assignmentsForRun(run);
            const videoNames = assignments.map(item => item.original_name).filter(Boolean);
            const outputDir = run.output_dir || '';
            const files = run.result_files || [];
            return `
                <div class="result-card">
                    <div class="result-card-head">
                        <div>
                            <div class="result-title">${escapeHtml(experimentLabel(run.experiment_type))}</div>
                            <div class="result-meta">
                                <span><i class="fas fa-film me-1"></i>${escapeHtml(videoNames.length ? videoNames.join(' / ') : tf('project.results.video_count', {count: (run.assignment_ids || []).length}, '{count} 个视频'))}</span>
                                <span><i class="fas fa-clock me-1"></i>${escapeHtml(formatTime(run.updated_at))}</span>
                                <span><i class="fas fa-file-lines me-1"></i>${escapeHtml(tf('project.results.file_count', {count: files.length}, '{count} 个结果文件'))}</span>
                            </div>
                        </div>
                        <span class="run-status">${escapeHtml(statusText(run.status || 'unknown'))} · ${Number(run.progress || 0)}%</span>
                    </div>
                    <div class="small text-muted mb-2">${escapeHtml(outputDir || tr('project.table.output', '输出目录'))}</div>
                    <button class="btn btn-outline-primary btn-sm" type="button" ${outputDir ? '' : 'disabled'} onclick="openRunFolder('${escapeHtml(run.project_run_id)}')">
                        <i class="fas fa-folder-open me-1"></i>${escapeHtml(tr('project.results.open_folder', '打开结果目录'))}
                    </button>
                </div>
            `;
        }).join('')}</div>`
        : `<div class="mini-note">${escapeHtml(tr('project.empty.runs', '还没有从项目启动过分析。'))}</div>`;
    document.getElementById(targetId).innerHTML = html;
}

function renderAll() {
    renderStats();
    renderGroups();
    renderSubjects();
    renderVideos();
    renderAssignments();
    renderRuns('runList');
}

function hasActiveProjectWork() {
    const active = new Set(['waiting', 'analyzing', 'running']);
    return (projectState.runs || []).some(run => active.has(String(run.status || '').toLowerCase()))
        || (projectState.assignments || []).some(item => active.has(String(item.status || '').toLowerCase()));
}

function scheduleProjectRefresh() {
    if (projectRefreshTimer) window.clearTimeout(projectRefreshTimer);
    if (!hasActiveProjectWork()) return;
    projectRefreshTimer = window.setTimeout(() => loadProject({silent: true}), 5000);
}

function loadProject(options = {}) {
    api(`/api/projects/${encodeURIComponent(projectId)}`)
        .then(data => {
            projectState = data;
            renderAll();
            if (!options.silent) alertBox('');
            scheduleProjectRefresh();
        })
        .catch(error => alertBox(error.message || String(error), 'danger'));
}

function openProjectFolder() {
    api(`/api/projects/${encodeURIComponent(projectId)}/open_folder`, {method: 'POST'})
        .catch(error => alert(error.message || String(error)));
}

function deleteGroup(groupId) {
    if (!confirm(tr('project.delete_group_confirm', '删除组别不会删除小鼠，但会清空这些小鼠的组别。继续吗？'))) return;
    api(`/api/projects/${encodeURIComponent(projectId)}/groups/${encodeURIComponent(groupId)}`, {method: 'DELETE'}).then(loadProject).catch(error => alert(error.message || String(error)));
}

function deleteSubject(subjectId) {
    if (!confirm(tr('project.delete_subject_confirm', '删除小鼠会把它标记为停用，不会删除已有分析记录。继续吗？'))) return;
    api(`/api/projects/${encodeURIComponent(projectId)}/subjects/${encodeURIComponent(subjectId)}`, {method: 'DELETE'}).then(loadProject).catch(error => alert(error.message || String(error)));
}

function deleteVideo(videoId) {
    const video = (projectState.videos || []).find(item => String(item.video_id || '') === String(videoId || ''));
    const name = video ? (video.original_name || videoId) : videoId;
    if (!confirm(tf(
        'project.delete_video_confirm',
        {name},
        '从项目视频库删除“{name}”？这会移除对应实验记录，但不会删除原始视频或已有结果。'
    ))) return;
    api(`/api/projects/${encodeURIComponent(projectId)}/videos/${encodeURIComponent(videoId)}`, {method: 'DELETE'})
        .then(data => {
            projectState.project = data.project || projectState.project;
            projectState.videos = data.videos || [];
            projectState.assignments = data.assignments || [];
            projectState.runs = data.runs || projectState.runs || [];
            renderAll();
            alertBox(tf('project.delete_video_success', {name}, '已删除视频“{name}”。'), 'success');
        })
        .catch(error => alertBox(error.message || String(error), 'danger'));
}

function generateGroupSubjects(groupId) {
    const group = (projectState.groups || []).find(item => String(item.group_id || '') === String(groupId || ''));
    const input = document.getElementById(`subjectCount-${groupId}`);
    const count = parseInt(input ? input.value : '0', 10);
    if (!Number.isFinite(count) || count < 1 || count > 500) {
        alertBox(tr('project.subjects.count_invalid', '请输入 1-500 之间的小鼠数量。'), 'warning');
        if (input) input.focus();
        return;
    }
    api(`/api/projects/${encodeURIComponent(projectId)}/groups/${encodeURIComponent(groupId)}/subjects/bulk`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({count})
    }).then(data => {
        projectState.subjects = data.subjects || projectState.subjects;
        projectState.project = {...(projectState.project || {}), subject_count: (projectState.subjects || []).length};
        renderAll();
        alertBox(tf(
            'project.subjects.generate_success',
            {group: group ? group.name : '', count: (data.created || []).length},
            '已生成 {count} 只小鼠。'
        ), 'success');
    }).catch(error => alertBox(error.message || String(error), 'danger'));
}

function updateSubjectCode(subjectId, value) {
    const animalCode = String(value || '').trim();
    if (!animalCode) {
        alertBox(tr('project.subjects.code_required', '请输入小鼠编号。'), 'warning');
        loadProject({silent: true});
        return;
    }
    const duplicate = (projectState.subjects || []).find(subject =>
        String(subject.subject_id || '') !== String(subjectId || '')
        && String(subject.animal_code || '').trim() === animalCode
    );
    if (duplicate) {
        alertBox(tf('project.subjects.duplicate_code', {code: animalCode}, '小鼠编号 {code} 已存在，请使用不同编号。'), 'warning');
        loadProject({silent: true});
        return;
    }
    api(`/api/projects/${encodeURIComponent(projectId)}/subjects/${encodeURIComponent(subjectId)}`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({animal_code: animalCode})
    }).then(data => {
        projectState.subjects = data.subjects || projectState.subjects;
        renderAll();
        alertBox('');
    }).catch(error => {
        alertBox(error.message || String(error), 'danger');
        loadProject({silent: true});
    });
}

function updateSubjectGroup(subjectId, groupId) {
    api(`/api/projects/${encodeURIComponent(projectId)}/subjects/${encodeURIComponent(subjectId)}`, {
        method: 'PATCH',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({group_id: groupId || ''})
    }).then(data => {
        projectState.subjects = data.subjects || projectState.subjects;
        renderAll();
        alertBox('');
    }).catch(error => {
        alertBox(error.message || String(error), 'danger');
        loadProject({silent: true});
    });
}

function toggleAllAssignments(checked) {
    document.querySelectorAll('.assignment-check:not(:disabled)').forEach(input => { input.checked = checked; });
    updateAssignmentSelectionSummary();
}

function selectedAssignmentItems() {
    const ids = new Set(Array.from(document.querySelectorAll('.assignment-check:checked:not(:disabled)')).map(input => String(input.value)));
    return (projectState.assignments || []).filter(item => ids.has(String(item.assignment_id || '')));
}

function updateAssignmentSelectionSummary() {
    const node = document.getElementById('assignmentSelectionSummary');
    const selectAll = document.getElementById('selectAllAssignments');
    const checks = Array.from(document.querySelectorAll('.assignment-check:not(:disabled)'));
    const checked = checks.filter(input => input.checked);
    if (selectAll) {
        selectAll.checked = checks.length > 0 && checked.length === checks.length;
        selectAll.indeterminate = checked.length > 0 && checked.length < checks.length;
    }
    if (!node) return;
    const selected = selectedAssignmentItems();
    if (!selected.length) {
        node.textContent = tr('project.analysis.selection_empty', '尚未选择分析记录');
        return;
    }
    const types = Array.from(new Set(selected.map(item => item.experiment_type || '')));
    if (types.length > 1) {
        node.textContent = tf('project.analysis.selection_mixed', {count: selected.length}, '已选择 {count} 条记录；请选择同一种实验类型');
        return;
    }
    node.textContent = tf(
        'project.analysis.selection_ready',
        {count: selected.length, type: experimentLabel(types[0])},
        '已选择 {count} 条 {type} 记录'
    );
}

function setVideoExperiment(videoId, experimentType) {
    const assignment = assignmentForVideo(videoId);
    if (!experimentType) {
        if (!assignment) return;
        api(`/api/projects/${encodeURIComponent(projectId)}/assignments/${encodeURIComponent(assignment.assignment_id)}`, {
            method: 'DELETE'
        }).then(loadProject).catch(error => alertBox(error.message || String(error), 'danger'));
        return;
    }
    const path = assignment
        ? `/api/projects/${encodeURIComponent(projectId)}/assignments/${encodeURIComponent(assignment.assignment_id)}`
        : `/api/projects/${encodeURIComponent(projectId)}/assignments`;
    const method = assignment ? 'PATCH' : 'POST';
    const payload = assignment
        ? {experiment_type: experimentType}
        : {video_id: videoId, experiment_type: experimentType};
    api(path, {
        method,
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    }).then(loadProject).catch(error => alertBox(error.message || String(error), 'danger'));
}

function startSelectedAssignments() {
    const selected = selectedAssignmentItems();
    const ids = selected.map(item => item.assignment_id);
    if (!ids.length) { alert(tr('project.select_assignments', '请先选择要分析的视频实验记录。')); return; }
    const types = Array.from(new Set(selected.map(item => item.experiment_type || '')));
    if (types.length > 1) {
        alert(tr('project.analysis.mixed_types', '一次项目分析只能选择同一种实验类型，请取消多余记录后再开始。'));
        return;
    }
    api(`/api/projects/${encodeURIComponent(projectId)}/analysis/start`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({assignment_ids: ids})
    }).then(data => {
        window.location.href = data.redirect_url;
    }).catch(error => alert(error.message || String(error)));
}

document.getElementById('groupForm').addEventListener('submit', event => {
    event.preventDefault();
    api(`/api/projects/${encodeURIComponent(projectId)}/groups`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            name: document.getElementById('groupName').value.trim(),
            color: '',
            notes: ''
        })
    }).then(() => {
        event.target.reset();
        loadProject();
    }).catch(error => alert(error.message || String(error)));
});

document.getElementById('videoImportForm').addEventListener('submit', event => {
    event.preventDefault();
    if (projectVideoUploading) {
        alert(tr('project.videos.wait_upload', '视频还在上传，请稍等。'));
        return;
    }
    const localPaths = document.getElementById('videoPaths').value.trim();
    if (!localPaths && !uploadedProjectFiles.length) {
        alert(tr('project.videos.no_input', '请先拖拽/选择视频，或输入至少一个本机视频路径。'));
        return;
    }
    api(`/api/projects/${encodeURIComponent(projectId)}/videos/import`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            local_paths: localPaths,
            uploaded_files: uploadedProjectFiles,
            storage_mode: 'linked'
        })
    }).then(data => {
        const importedCount = (data.imported || []).length;
        const errorCount = (data.errors || []).length;
        document.getElementById('videoPaths').value = '';
        uploadedProjectFiles = [];
        renderUploadedProjectFiles();
        setVideoUploadStatus('', 'muted');
        loadProject();
        if (errorCount) {
            alertBox(tf('project.import.partial', {imported: importedCount, errors: errorCount}, '已导入 {imported} 个视频，{errors} 个失败。'), 'warning');
        } else {
            alertBox(tf('project.import.success', {imported: importedCount}, '已导入 {imported} 个视频。'), 'success');
        }
    }).catch(error => alert(error.message || String(error)));
});

function initProjectVideoDropZone() {
    const dropZone = document.getElementById('projectVideoDropZone');
    const input = document.getElementById('projectVideoFileInput');
    const chooseBtn = document.getElementById('chooseProjectVideosBtn');
    if (!dropZone || !input || !chooseBtn) return;
    chooseBtn.addEventListener('click', () => input.click());
    input.addEventListener('change', event => {
        uploadProjectVideoFiles(event.target.files);
        input.value = '';
    });
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, event => {
            event.preventDefault();
            event.stopPropagation();
        });
    });
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'));
    });
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'));
    });
    dropZone.addEventListener('drop', event => uploadProjectVideoFiles(event.dataTransfer.files));
}

document.addEventListener('mousescope:langchange', () => {
    renderAll();
    renderUploadedProjectFiles();
    if (uploadedProjectFiles.length) {
        setVideoUploadStatus(tf('project.videos.uploaded_count', {count: uploadedProjectFiles.length}, '已选择 {count} 个视频文件'), 'success');
    }
});
document.addEventListener('DOMContentLoaded', () => {
    initProjectVideoDropZone();
    loadProject();
});

MouseScopeTour.register('project_home', {
    version: 'v1',
    steps: [
        {target: '[data-tour-id="project-tab-subjects"]', titleKey: 'tour.project.subjects.title', bodyKey: 'tour.project.subjects.body'},
        {target: '[data-tour-id="project-groups"]', titleKey: 'tour.project.groups.title', bodyKey: 'tour.project.groups.body'},
        {target: '[data-tour-id="project-subjects"]', titleKey: 'tour.project.animals.title', bodyKey: 'tour.project.animals.body'},
        {target: '[data-tour-id="project-tab-videos"]', titleKey: 'tour.project.videos.title', bodyKey: 'tour.project.videos.body'},
        {target: '[data-tour-id="project-video-drop"]', showTab: 'videos', titleKey: 'tour.project.drop.title', bodyKey: 'tour.project.drop.body'},
        {target: '[data-tour-id="project-video-type-guide"]', showTab: 'videos', titleKey: 'tour.project.experiment.title', bodyKey: 'tour.project.experiment.body'},
        {target: '[data-tour-id="project-video-table"]', showTab: 'videos', titleKey: 'tour.project.videos.title', bodyKey: 'tour.project.videos.body'},
        {target: '[data-tour-id="project-analysis-guide"]', showTab: 'analysis', titleKey: 'tour.project.analysis.title', bodyKey: 'tour.project.analysis.body'},
        {target: '[data-tour-id="project-start-analysis"]', showTab: 'analysis', titleKey: 'tour.project.analysis.title', bodyKey: 'tour.project.analysis.body'},
        {target: '[data-tour-id="project-results"]', showTab: 'results', titleKey: 'tour.project.results.title', bodyKey: 'tour.project.results.body'}
    ]
});
