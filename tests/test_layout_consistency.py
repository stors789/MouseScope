from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"


class LayoutConsistencyTest(unittest.TestCase):
    def test_page_templates_do_not_start_with_utf8_bom(self):
        for path in TEMPLATES.glob("*.html"):
            self.assertFalse(
                path.read_bytes().startswith(b"\xef\xbb\xbf"),
                f"UTF-8 BOM before Jinja markup creates a visible blank line: {path.name}",
            )

    def test_license_page_does_not_request_task_badge_before_acceptance(self):
        source = (TEMPLATES / "base.html").read_text(encoding="utf-8")
        badge_body = source.split("const badge = document.getElementById('navTaskBadge')", 1)[1]
        self.assertIn("if (!localStorage.getItem('mousescope_license_accepted'))", badge_body)
        self.assertIn("if (!serverAccepted) return;", badge_body)
        self.assertIn("window.MouseScopeOnboarding.licenseAccepted", badge_body)
        self.assertLess(
            badge_body.index("if (!localStorage.getItem('mousescope_license_accepted'))"),
            badge_body.index("fetch('/api/tasks?summary=1')"),
        )

    def test_quick_analysis_uses_standard_page_width(self):
        source = (TEMPLATES / "upload.html").read_text(encoding="utf-8")
        self.assertRegex(source, r"\.upload-shell\s*\{[^}]*max-width:\s*1440px")
        self.assertNotRegex(source, r"\.upload-shell\s*\{[^}]*max-width:\s*1120px")

    def test_quick_analysis_defers_runtime_settings_to_analysis_config(self):
        upload = (TEMPLATES / "upload.html").read_text(encoding="utf-8")
        config = (TEMPLATES / "batch_workspace.html").read_text(encoding="utf-8")
        for field in ("start_seconds", "end_seconds", "fps_override", "output_dir"):
            self.assertNotIn(f'name="{field}"', upload)
        self.assertIn('data-tour-page="analysis-config"', config)
        self.assertIn('id="video-start-${card.index}"', config)
        self.assertIn('id="video-end-${card.index}"', config)
        self.assertIn('id="video-fps-${card.index}"', config)
        self.assertIn('id="video-orientation-${card.index}"', config)

        pages_route = (ROOT / "routes" / "pages.py").read_text(encoding="utf-8")
        project_route = (ROOT / "routes" / "project_api.py").read_text(encoding="utf-8")
        self.assertIn("redirect_url = url_for('analysis_config', session_id=session_id)", pages_route)
        self.assertIn('redirect_url = url_for("analysis_config", session_id=session_id)', project_route)

    def test_start_analysis_saves_visible_video_windows_first(self):
        source = (TEMPLATES / "batch_workspace.html").read_text(encoding="utf-8")
        start_body = source.split("async function startBatchAnalysis()", 1)[1].split("async function saveVideoSettings", 1)[0]
        self.assertIn("await saveAllVisibleVideoSettings()", start_body)
        self.assertLess(
            start_body.index("await saveAllVisibleVideoSettings()"),
            start_body.index("await refreshBatchState()"),
        )
        self.assertIn("const payloads = (latestBatchState.cards || []).map", source)
        self.assertIn("if (isBatchStarting || !beginWorkspaceMutation()) return", start_body)
        self.assertIn("isBatchStarting = true", start_body)
        before_save = start_body.split("await saveAllVisibleVideoSettings()", 1)[0]
        self.assertNotIn("refreshUI()", before_save)
        self.assertIn("renderSummary(latestBatchState)", before_save)

    def test_multi_video_settings_are_preflighted_before_any_save(self):
        source = (TEMPLATES / "batch_workspace.html").read_text(encoding="utf-8")
        save_all = source.split("async function saveAllVisibleVideoSettings()", 1)[1].split(
            "async function openCalibration", 1
        )[0]
        bulk = source.split("async function applySettingsToAll()", 1)[1].split(
            "async function copyPreviousConfig", 1
        )[0]
        self.assertIn("preflightVideoSettingsPayload", save_all)
        self.assertLess(save_all.index("preparedPayloads.push"), save_all.index("submitVideoSettingsPayload"))
        self.assertIn("for (const payload of payloads)", bulk)
        self.assertLess(bulk.index("preflightVideoSettingsPayload"), bulk.index("let latestData"))

    def test_card_rerenders_preserve_unsaved_video_inputs(self):
        source = (TEMPLATES / "batch_workspace.html").read_text(encoding="utf-8")
        self.assertIn("function captureVisibleVideoInputs()", source)
        self.assertIn("function restoreVisibleVideoInputs(captured)", source)
        self.assertIn("function syncVisibleVideoInputsFromState()", source)
        self.assertIn("if (!document.getElementById(`video-start-${card.index}`)) continue", source)
        self.assertIn("document.addEventListener('DOMContentLoaded', () => {", source)
        self.assertIn("refreshUI(false)", source)
        self.assertIn("window.addEventListener('pageshow'", source)
        self.assertIn("videoInputsReadyForPreservation = true", source)
        self.assertIn("if (refresh) refreshUI(true)", source)
        self.assertIn("'mousescope:langchange', () => refreshUI(true)", source)

    def test_analysis_configuration_writes_share_one_page_level_lock(self):
        source = (TEMPLATES / "batch_workspace.html").read_text(encoding="utf-8")
        self.assertIn("let workspaceMutationActive = false", source)
        self.assertIn("function beginWorkspaceMutation()", source)
        self.assertIn("function endWorkspaceMutation()", source)
        for function_name in (
            "startBatchAnalysis",
            "saveVideoSettings",
            "openCalibration",
            "applySettingsToAll",
            "copyPreviousConfig",
            "copyFirstReadyToPending",
            "resetVideoConfig",
        ):
            body = source.split(f"async function {function_name}", 1)[1].split("\nasync function ", 1)[0]
            self.assertIn("beginWorkspaceMutation()", body, function_name)

    def test_project_analysis_start_has_a_synchronous_double_click_guard(self):
        source = (ROOT / "static" / "js" / "project_home.js").read_text(encoding="utf-8")
        template = (TEMPLATES / "project_home.html").read_text(encoding="utf-8")
        body = source.split("async function startSelectedAssignments()", 1)[1].split(
            "document.getElementById('groupForm')", 1
        )[0]
        self.assertIn("if (projectAnalysisStarting) return", body)
        self.assertLess(body.index("projectAnalysisStarting = true"), body.index("await api("))
        self.assertIn('id="projectStartAnalysisBtn"', template)

    def test_backend_start_reservation_precedes_all_run_side_effects(self):
        source = (ROOT / "routes" / "api.py").read_text(encoding="utf-8")
        body = source.split("def start_analysis(session_id):", 1)[1].split("def get_status(session_id):", 1)[0]
        reservation = body.index("analysis_sessions.reserve_start(session_id)")
        self.assertLess(reservation, body.index("analysis_sessions.save_draft("))
        self.assertLess(reservation, body.index("prepare_session_run_output("))
        self.assertLess(reservation, body.index("write_run_snapshot("))
        self.assertIn("analysis_sessions.commit_reserved_analysis(session_id, analysis)", body)

    def test_all_monitor_pages_share_resilient_status_fetching(self):
        base = (TEMPLATES / "base.html").read_text(encoding="utf-8")
        helper = (ROOT / "static" / "js" / "monitor_status.js").read_text(encoding="utf-8")
        self.assertIn("js/monitor_status.js", base)
        self.assertIn("const activeRequests = new Map()", helper)
        self.assertIn("response.status >= 400 && response.status < 500", helper)
        self.assertIn("while (true)", helper)
        for path in TEMPLATES.glob("monitor*.html"):
            source = path.read_text(encoding="utf-8")
            uses_helper = "MouseScopeMonitor.fetchStatus" in source or "MouseScopeMonitor.routeFetch" in source
            self.assertTrue(uses_helper, path.name)

    def test_all_calibration_primary_actions_have_a_shared_double_click_guard(self):
        base = (TEMPLATES / "base.html").read_text(encoding="utf-8")
        guard = (ROOT / "static" / "js" / "analysis_start_guard.js").read_text(encoding="utf-8")
        self.assertIn("js/analysis_start_guard.js", base)
        self.assertIn("analysisStartPending", guard)
        self.assertIn("stopImmediatePropagation", guard)
        for path in TEMPLATES.glob("calibrate*.html"):
            source = path.read_text(encoding="utf-8")
            self.assertIn('id="primaryActionBtn"', source, path.name)

    def test_opening_roi_saves_every_visible_video_window(self):
        source = (TEMPLATES / "batch_workspace.html").read_text(encoding="utf-8")
        open_body = source.split("async function openCalibration", 1)[1].split(
            "function normalizedOrientation", 1
        )[0]
        self.assertIn("await saveAllVisibleVideoSettings()", open_body)
        self.assertNotIn("await saveVideoSettings(index)", open_body)

    def test_global_setting_pages_share_the_same_hero_and_icon_structure(self):
        for filename in ("visual_style.html", "experiment_defaults.html"):
            source = (TEMPLATES / filename).read_text(encoding="utf-8")
            content = source.split("{% block content %}", 1)[1]
            hero_start = content.index('class="app-page-hero settings-page-hero"')
            hero_end = content.index("</section>", hero_start)
            tabs = content.index('data-tour-id="global-settings-tabs"')
            self.assertLess(hero_start, tabs, f"tabs precede hero in {filename}")
            self.assertLess(tabs, hero_end, f"tabs are outside hero in {filename}")
            self.assertIn('class="settings-kicker"', content[hero_start:hero_end])
            self.assertIn('class="app-page-icon"', content[hero_start:hero_end])

        defaults = (TEMPLATES / "experiment_defaults.html").read_text(encoding="utf-8")
        self.assertNotIn('class="defaults-icon"', defaults)
        self.assertNotIn('class="defaults-heading"', defaults)

        theme = (ROOT / "static" / "css" / "tech-theme.css").read_text(encoding="utf-8")
        self.assertRegex(
            theme,
            r"\.settings-page-hero\s+\.app-page-title-group\s*\{[^}]*align-items:\s*center",
        )
        self.assertRegex(
            theme,
            r"\.settings-page-hero\s+\.app-page-title-group\s*\{[^}]*min-height:\s*130px",
        )

    def test_experiment_default_boolean_label_tracks_checkbox_state(self):
        source = (TEMPLATES / "experiment_defaults.html").read_text(encoding="utf-8")
        self.assertIn("document.getElementById('groups').addEventListener('change'", source)
        self.assertIn("input.checked?(lang()==='en'?'Enabled':'启用')", source)
        self.assertIn("(lang()==='en'?'Disabled':'关闭')", source)

    def test_experiment_defaults_waits_for_schema_before_language_render(self):
        source = (TEMPLATES / "experiment_defaults.html").read_text(encoding="utf-8")
        render_body = source.split("function renderModule()", 1)[1].split(
            "function selectModule", 1
        )[0]
        self.assertIn("const modules=schema.modules||{}", render_body)
        self.assertIn("if(!moduleKeys.length)return", render_body)

    def test_experiment_defaults_cannot_save_before_loading_or_during_another_write(self):
        source = (TEMPLATES / "experiment_defaults.html").read_text(encoding="utf-8")
        self.assertIn('id="saveBtn" disabled', source)
        self.assertIn('id="resetBtn" disabled', source)
        self.assertIn("if(!defaultsLoaded||defaultsMutationActive)return", source)
        self.assertIn("let defaultsLoadGeneration = 0", source)


if __name__ == "__main__":
    unittest.main()
