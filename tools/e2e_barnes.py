"""Browser acceptance against an isolated running MouseScope server.

Usage: python tools/e2e_barnes.py --url http://127.0.0.1:8093 --output DIR
Requires Playwright and a local Edge installation; uses generated videos only.
"""

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8093")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--edge", default="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
    )
    parser.add_argument(
        "--chromium",
        action="store_true",
        help="Use the Chromium installed by Playwright instead of Edge",
    )
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    video = out / "barnes-a.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 320)
    )
    assert writer.isOpened()
    for index in range(40):
        frame = np.full((320, 320, 3), 235, np.uint8)
        cv2.ellipse(frame, (140 + index, 160), (18, 10), 0, 0, 360, (15, 15, 15), -1)
        cv2.line(frame, (122 + index, 160), (103 + index, 165), (15, 15, 15), 3)
        writer.write(frame)
    writer.release()
    second = out / "barnes-b.avi"
    second.write_bytes(video.read_bytes())
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True, **({} if args.chromium else dict(executable_path=args.edge))
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 1080}, accept_downloads=True
        )
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("dialog", lambda dialog: dialog.accept())

        def dismiss_tour():
            page.wait_for_timeout(800)
            skip = page.locator('[data-tour-action="skip"]')
            if skip.is_visible():
                skip.click()

        try:
            page.goto(args.url + "/license?next=/upload")
            page.wait_for_load_state("networkidle")
            # Inspect the rendered page before using the existing upload controls.
            assert page.locator("#agreeCheck").is_visible()
            page.locator("#agreeCheck").check()
            page.locator("#licenseAcceptBtn").click()
            page.wait_for_url("**/upload")
            page.wait_for_load_state("networkidle")
            dismiss_tour()
            page.locator("#experimentType").select_option("barnes_maze")
            page.locator("#localPaths").fill(str(video) + "\n" + str(second))
            page.locator("#validatePathsBtn").click()
            page.locator("#configSection").wait_for(state="visible")
            page.locator("#submitBtn").click()
            page.wait_for_url("**/analysis-config/*")
            page.wait_for_load_state("networkidle")
            sid = page.url.rsplit("/", 1)[-1]
            dismiss_tour()
            assert page.locator("#videoCardList article").count() == 2
            page.locator(
                'article[data-video-index="0"] a[data-tour-id="analysis-config-roi"]'
            ).click()
            page.wait_for_load_state("networkidle")
            page.wait_for_function(
                "typeof barnesImage !== 'undefined' && barnesImage?.width === 320"
            )
            method = page.locator('[data-config-field="classical_method"]')
            assert method.input_value() == "local_otsu_v1"
            assert (
                page.locator('[data-config-field="body_erosion_kernel"]').input_value()
                == "3"
            )
            assert (
                page.locator(
                    '[data-config-field="head_confidence_threshold"]'
                ).input_value()
                == "0.4"
            )
            assert page.locator("#poseConfidenceThreshold").input_value() == "0.4"
            method.select_option("legacy")
            assert page.locator("[data-classical-modern]").first.is_hidden()
            assert page.locator("[data-classical-legacy]").first.is_visible()
            method.select_option("local_otsu_v1")
            page.locator('[data-config-field="body_erosion_iter"]').fill("2")
            page.locator('[data-config-field="hole_count"]').fill("4")
            page.locator('[data-config-field="bg_method"]').select_option("threshold")
            page.locator("#drawMode").select_option("ellipse")
            canvas = page.locator("#barnesCanvas")
            canvas.scroll_into_view_if_needed()
            box = canvas.bounding_box()

            def at(x, y):
                return dict(
                    x=box["x"] + box["width"] * x / 320,
                    y=box["y"] + box["height"] * y / 320,
                )

            a, b = at(25, 40), at(295, 280)
            page.mouse.move(**a)
            page.mouse.down()
            page.mouse.move(**b, steps=8)
            page.mouse.up()
            page.locator("#drawMode").select_option("holes")
            for x, y in [(265, 160), (160, 255), (55, 160), (160, 65)]:
                canvas.click(
                    position={"x": box["width"] * x / 320, "y": box["height"] * y / 320}
                )
            assert page.locator("#holeList input").count() == 4
            page.locator("#targetHole").select_option("1")
            page.locator("#roiName").fill("Mouse <A>")
            page.locator("#escapeTime").fill("2.8")
            page.locator("#manualEventsEnabled").check()
            page.locator("#manualEvents").fill("0.4,2\n1.0,2\n2.0,1")
            page.locator("#previewButton").click()
            page.wait_for_function(
                "document.querySelector('#previewStatus').textContent.includes('classical')"
            )
            page.locator("#saveConfig").click()
            page.wait_for_function(
                "document.querySelector('#barnesStatus').textContent.includes('配置已保存')"
            )
            page.reload()
            page.wait_for_load_state("networkidle")
            page.wait_for_function("barnesImage?.width === 320")
            assert page.locator("#holeList input").count() == 4
            assert page.locator("#escapeTime").input_value() == "2.8"
            assert (
                page.locator('[data-config-field="classical_method"]').input_value()
                == "local_otsu_v1"
            )
            assert (
                page.locator('[data-config-field="body_erosion_iter"]').input_value()
                == "2"
            )
            page.locator('[data-config-field="body_erosion_iter"]').fill("1")
            assert page.locator("#manualEventsEnabled").is_checked()
            page.locator("#trackingBackend").check()
            assert page.locator("#classicalDetectionControls").is_hidden()
            page.locator("#saveConfig").click()
            page.wait_for_function(
                "document.querySelector('#barnesStatus').textContent.includes('配置已保存')"
            )
            page.reload()
            page.wait_for_load_state("networkidle")
            page.wait_for_function("barnesImage?.width === 320")
            assert page.locator("#trackingBackend").is_checked()
            page.locator("#trackingBackend").uncheck()
            page.locator("#sharedRoiTemplateName").fill("Barnes E2E apparatus")
            with page.expect_response(
                lambda r: "/api/roi_templates" in r.url and r.request.method == "POST"
            ) as saved:
                page.locator("#sharedRoiTemplateSave").click()
            data = saved.value.json()
            assert data["success"], data
            template = data["template"]
            template_id = template["id"]
            assert "target_hole_id" not in template["rois"][0]
            assert "manual_escape_time_s" not in template["rois"][0]
            page.locator("#targetHole").select_option("2")
            page.locator("#escapeTime").fill("2.5")
            page.locator("#sharedRoiTemplateSelect").select_option(template_id)
            page.locator("#sharedRoiTemplateApply").click()
            assert page.locator("#targetHole").input_value() == "2"
            assert page.locator("#escapeTime").input_value() == "2.5"
            page.locator("#targetHole").select_option("1")
            page.locator("#escapeTime").fill("2.8")
            page.screenshot(path=str(out / "01_calibration.png"), full_page=True)
            page.locator("#primaryActionBtn").click()
            page.wait_for_url("**/analysis-config/*")
            page.locator(
                'article[data-video-index="1"] a[data-tour-id="analysis-config-roi"]'
            ).click()
            page.wait_for_load_state("networkidle")
            page.wait_for_function("barnesImage?.width === 320")
            page.locator("#sharedRoiTemplateSelect").select_option(template_id)
            page.locator("#sharedRoiTemplateApply").click()
            assert page.locator("#targetHole").input_value() == ""
            page.locator("#targetHole").select_option("3")
            page.locator('[data-config-field="trial_type"]').select_option("probe")
            page.locator('[data-config-field="bg_method"]').select_option("threshold")
            page.locator("#generateVideo").uncheck()
            page.locator("#primaryActionBtn").click()
            page.wait_for_url("**/analysis-config/*")
            page.locator("#startBatchBtn").wait_for(state="visible")
            page.wait_for_function("!document.querySelector('#startBatchBtn').disabled")
            page.locator("#startBatchBtn").click()
            page.wait_for_url("**/monitor/*")
            page.wait_for_load_state("networkidle")
            deadline = time.time() + 60
            while time.time() < deadline:
                status = context.request.get(args.url + f"/api/status/{sid}").json()
                if status["status"] in ["completed", "error", "stopped"]:
                    break
                page.wait_for_timeout(300)
            assert status["status"] == "completed", status
            page.locator("#downloadSection").wait_for(state="visible")
            page.screenshot(path=str(out / "02_results.png"), full_page=True)
            with page.expect_download() as downloaded:
                page.locator("#downloadLinks a").first.click(no_wait_after=True)
            package = out / "results.zip"
            downloaded.value.save_as(str(package))
            with zipfile.ZipFile(package) as archive:
                summaries = [
                    name for name in archive.namelist() if name.endswith("_summary.csv")
                ]
                assert len(summaries) == 2, archive.namelist()
                results = [
                    next(
                        csv.DictReader(
                            io.StringIO(archive.read(name).decode("utf-8-sig"))
                        )
                    )
                    for name in summaries
                ]
                acquisition = next(
                    row for row in results if row["trial_type"] == "acquisition"
                )
                probe = next(row for row in results if row["trial_type"] == "probe")
                assert (
                    acquisition["target_hole_id"] == "1"
                    and probe["target_hole_id"] == "3"
                )
                assert acquisition["hole_visit_sequence"] == "2-2-1"
                assert float(acquisition["total_latency_s"]) == 2.8
                assert (
                    probe["total_latency_s"] == ""
                    and probe["escape_status"] == "not_applicable"
                )
                assert any(
                    name.endswith("_analyzed.mp4") for name in archive.namelist()
                )
            assert not errors, errors
            (out / "acceptance.json").write_text(
                json.dumps(
                    dict(
                        status="passed",
                        session_id=sid,
                        summaries=results,
                        page_errors=errors,
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                "Barnes browser acceptance passed: calibration, previews, DLC persistence, geometry-only templates, per-video targets, batch analysis, annotation and ZIP exports."
            )
        except Exception:
            page.screenshot(path=str(out / "failure.png"), full_page=True)
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
