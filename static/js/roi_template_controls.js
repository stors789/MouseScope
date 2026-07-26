(function () {
    'use strict';

    const clone = value => JSON.parse(JSON.stringify(value));
    const language = () => (typeof window.getLang === 'function' && window.getLang() === 'en') ? 'en' : 'zh';
    const text = (zh, en) => language() === 'en' ? en : zh;
    const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[char]);

    function frameMeta(adapter, index) {
        const raw = adapter.getFrameMeta(index) || {};
        return {
            width: Math.max(1, Number(raw.width) || 1),
            height: Math.max(1, Number(raw.height) || 1),
            orientation_degrees: Number(raw.orientation_degrees || 0) || 0,
        };
    }

    function materialize(value, template, target, parentKey = '') {
        if (Array.isArray(value)) return value.map(item => materialize(item, template, target, parentKey));
        if (!value || typeof value !== 'object') return value;
        const source = template.source_frame || {};
        const legacyUnknown = template.coordinate_space === 'legacy_pixels' && source.unknown;
        if (Object.prototype.hasOwnProperty.call(value, 'x') && Object.prototype.hasOwnProperty.call(value, 'y')) {
            if (legacyUnknown) return {...value, x: Number(value.x), y: Number(value.y)};
            return {...value, x: Number(value.x) * target.width, y: Number(value.y) * target.height};
        }
        const out = {};
        Object.entries(value).forEach(([key, nested]) => {
            if (template.assay === 'water_maze' && key === 'radius_x_px') {
                out[key] = Number(nested) * target.width;
            } else if (template.assay === 'water_maze' && key === 'radius_y_px') {
                out[key] = Number(nested) * target.height;
            } else {
                out[key] = materialize(nested, template, target, key);
            }
        });
        return out;
    }

    function warnings(template, targets) {
        const source = template.source_frame || {};
        if (template.coordinate_space === 'legacy_pixels' || source.unknown) {
            return [text(
                '这是旧版迁移模板，缺少源画面尺寸；将保持原像素坐标，请应用后逐项检查。',
                'This migrated template has no source frame size. Original pixels are preserved; inspect every ROI after applying.'
            )];
        }
        const result = [];
        targets.forEach(target => {
            if (Number(source.orientation_degrees || 0) !== Number(target.orientation_degrees || 0)) {
                result.push(text('模板与目标视频的画面方向不同。', 'Template and target video orientations differ.'));
            }
            const sourceRatio = Number(source.aspect_ratio || (Number(source.width) / Number(source.height)));
            const targetRatio = target.width / target.height;
            if (Math.abs(sourceRatio - targetRatio) / Math.max(sourceRatio, 1e-9) > .02) {
                result.push(text(
                    '模板与目标视频宽高比不同；会按宽高分别缩放，请检查对齐。',
                    'Template and target aspect ratios differ. Coordinates scale independently; inspect alignment.'
                ));
            }
        });
        return [...new Set(result)];
    }

    function mount() {
        const adapter = window.MouseScopeRoiTemplateAdapter;
        if (!adapter || document.getElementById('sharedRoiTemplatePanel') || document.getElementById('roiTemplatePanel')) return;
        const list = document.querySelector('#roiList, .fst-card-list, .mwm-list, #arenaList, #cylinderList');
        if (!list || !list.parentElement) return;
        const panel = document.createElement('div');
        panel.id = 'sharedRoiTemplatePanel';
        panel.className = 'card border-0 shadow-sm mb-3';
        panel.innerHTML = `
          <div class="card-body">
            <div class="fw-bold">${text('设备 ROI 模板', 'Device ROI Templates')}</div>
            <div class="small text-secondary mb-2">${text('只保存设备区域、检测遮罩与物理尺寸；不会保存小鼠、时间段或识别参数。', 'Saves apparatus zones, detection masks, and physical dimensions only—not animals, timing, or detection tuning.')}</div>
            <div class="input-group mb-2">
              <input id="sharedRoiTemplateName" class="form-control" maxlength="80" placeholder="${text('模板名称', 'Template name')}">
              <button id="sharedRoiTemplateSave" class="btn btn-outline-primary" type="button">${text('保存当前 ROI', 'Save current ROI')}</button>
            </div>
            <div class="input-group mb-2">
              <select id="sharedRoiTemplateSelect" class="form-select"><option value="">${text('正在读取模板…', 'Loading templates…')}</option></select>
              <button id="sharedRoiTemplateApply" class="btn btn-outline-success" type="button" disabled>${text('应用', 'Apply')}</button>
              <button id="sharedRoiTemplateDelete" class="btn btn-outline-danger" type="button" disabled aria-label="${text('删除模板', 'Delete template')}"><i class="fas fa-trash"></i></button>
            </div>
            <label id="sharedRoiTemplateAllWrap" class="form-check-label small text-secondary">
              <input id="sharedRoiTemplateAll" class="form-check-input me-1" type="checkbox">
              ${text('应用到本批次全部视频', 'Apply to every video in this batch')}
            </label>
            <div id="sharedRoiTemplateStatus" class="small mt-2"></div>
          </div>`;
        list.parentElement.insertBefore(panel, list);
        const select = panel.querySelector('#sharedRoiTemplateSelect');
        const status = panel.querySelector('#sharedRoiTemplateStatus');
        panel.querySelector('#sharedRoiTemplateAllWrap').style.display = adapter.isBatch ? '' : 'none';
        let templates = [];
        const setStatus = (message, error = false) => {
            status.textContent = message || '';
            status.className = `small mt-2 ${error ? 'text-danger' : 'text-success'}`;
        };
        const render = selected => {
            select.innerHTML = templates.length
                ? `<option value="">${text('选择已保存模板', 'Select a saved template')}</option>` + templates.map(item =>
                    `<option value="${escapeHtml(item.id)}" ${item.id === selected ? 'selected' : ''}>${escapeHtml(item.name)} · ${(item.rois || []).length} ROI</option>`
                ).join('')
                : `<option value="">${text('还没有保存模板', 'No saved templates')}</option>`;
            const enabled = Boolean(select.value);
            panel.querySelector('#sharedRoiTemplateApply').disabled = !enabled;
            panel.querySelector('#sharedRoiTemplateDelete').disabled = !enabled;
        };
        const load = async selected => {
            const response = await fetch(`/api/roi_templates?assay=${encodeURIComponent(adapter.assay)}`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.message || data.error || text('读取模板失败。', 'Failed to load templates.'));
            templates = Array.isArray(data.templates) ? data.templates : [];
            render(selected);
        };
        select.addEventListener('change', () => render(select.value));
        panel.querySelector('#sharedRoiTemplateSave').addEventListener('click', async () => {
            const nameInput = panel.querySelector('#sharedRoiTemplateName');
            const name = nameInput.value.trim();
            if (!name) { nameInput.focus(); return; }
            try {
                const payload = adapter.getPayload();
                if (!payload || !(payload.rois || []).length) throw new Error(text('请先完成至少一个 ROI。', 'Finish at least one ROI first.'));
                const response = await fetch('/api/roi_templates', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        assay: adapter.assay,
                        name,
                        rois: payload.rois,
                        geometry: payload.geometry || {},
                        source_frame: frameMeta(adapter),
                    }),
                });
                const data = await response.json();
                if (!response.ok || !data.success) throw new Error(data.message || data.error || text('保存失败。', 'Save failed.'));
                nameInput.value = '';
                await load(data.template.id);
                setStatus(text('模板已保存。', 'Template saved.'));
            } catch (error) { setStatus(error.message, true); }
        });
        panel.querySelector('#sharedRoiTemplateApply').addEventListener('click', () => {
            const template = templates.find(item => item.id === select.value);
            if (!template) return;
            const all = Boolean(panel.querySelector('#sharedRoiTemplateAll').checked && adapter.isBatch);
            const count = all ? Number(adapter.videoCount || 1) : 1;
            const targets = Array.from({length: count}, (_, index) => frameMeta(adapter, all ? index : undefined));
            const warningText = warnings(template, targets);
            if (warningText.length && !window.confirm(`${warningText.join('\n\n')}\n\n${text('仍要继续吗？', 'Continue?')}`)) return;
            if (!window.confirm(text('应用模板会替换现有 ROI，是否继续？', 'Applying the template replaces existing ROIs. Continue?'))) return;
            const payloads = targets.map(target => ({
                rois: materialize(clone(template.rois || []), template, target),
                geometry: clone(template.geometry || {}),
            }));
            adapter.applyPayload(payloads, all);
            setStatus(text('模板已应用，请检查区域对齐。', 'Template applied. Inspect ROI alignment.'));
        });
        panel.querySelector('#sharedRoiTemplateDelete').addEventListener('click', async () => {
            if (!select.value || !window.confirm(text('确定删除这个模板？', 'Delete this template?'))) return;
            try {
                const response = await fetch(`/api/roi_templates/${encodeURIComponent(select.value)}?assay=${encodeURIComponent(adapter.assay)}`, {method: 'DELETE'});
                const data = await response.json();
                if (!response.ok || !data.success) throw new Error(data.message || data.error || text('删除失败。', 'Delete failed.'));
                await load();
                setStatus(text('模板已删除。', 'Template deleted.'));
            } catch (error) { setStatus(error.message, true); }
        });
        load().catch(error => setStatus(error.message, true));
    }

    document.addEventListener('DOMContentLoaded', mount);
    document.addEventListener('mousescope:langchange', () => {
        // The page reloads language-dependent calibration labels already; keeping
        // entered names and selections is safer than rebuilding the live panel.
    });
})();
