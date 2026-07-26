"""Compatibility imports for third-party code using the old Y-maze module."""

from services.roi_templates import (  # noqa: F401
    RoiTemplateStore,
    RoiTemplateValidationError,
    canonical_assay,
    compatibility_warnings,
    materialize_template,
    normalize_template_payload as _normalize_template_payload,
)


def _with_inferred_source_frame(payload):
    candidate = dict(payload or {})
    if "source_frame" in candidate:
        return candidate
    points = [
        point
        for roi in candidate.get("rois") or []
        for group in list((roi.get("zones") or {}).values())
        + [roi.get("detection_points") or []]
        for point in group or []
    ]
    candidate["source_frame"] = {
        "width": max(1, int(max([float(p.get("x") or 0) for p in points] or [1]) + 0.999)),
        "height": max(1, int(max([float(p.get("y") or 0) for p in points] or [1]) + 0.999)),
    }
    return candidate


def normalize_template_payload(payload):
    """Preserve the legacy one-argument Y-maze normalization API."""
    return _normalize_template_payload("y_maze", _with_inferred_source_frame(payload))


class YMazeRoiTemplateStore(RoiTemplateStore):
    def __init__(self, path):
        super().__init__(path)

    def list_templates(self):
        templates = super().list_templates("y_maze")
        return [
            materialize_template(template, template.get("source_frame") or {"width": 1, "height": 1})
            for template in templates
        ]

    def save_template(self, name, payload):
        return super().save_template("y_maze", name, _with_inferred_source_frame(payload))

    def delete_template(self, template_id):
        return super().delete_template("y_maze", template_id)
