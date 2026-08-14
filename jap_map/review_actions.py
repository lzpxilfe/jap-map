"""One-keystroke review actions for the editable proposal-review queue."""

from __future__ import annotations

from qgis.core import QgsVectorLayer


REVIEW_LAYER_PROPERTY = "historical_map_tools/review_queue"
REVIEW_STATUSES = {
    "contour": "contour",
    "text": "text",
    "road_river": "road_river",
    "unsure": "unsure",
}


def _message(iface, level: str, message: str) -> None:
    bar = iface.messageBar()
    method = getattr(bar, f"push{level}", None)
    if method:
        method("Historical Map Tools", message)


def classify_selected_proposals(iface, status: str) -> int:
    """Persist a status for all selected queue features and return their count."""
    if status not in REVIEW_STATUSES:
        raise ValueError(f"Unknown review status: {status}")
    layer = iface.activeLayer()
    if not isinstance(layer, QgsVectorLayer) or not layer.customProperty(REVIEW_LAYER_PROPERTY, False):
        _message(iface, "Warning", "Select features in 'Quick review queue — development only' first.")
        return 0
    feature_ids = layer.selectedFeatureIds()
    if not feature_ids:
        _message(iface, "Warning", "Select one or more proposals first.")
        return 0
    field_index = layer.fields().indexOf("review_status")
    if field_index < 0:
        raise RuntimeError("Review queue has no review_status field")
    if not layer.isEditable() and not layer.startEditing():
        raise RuntimeError("Could not start editing the review queue")
    for feature_id in feature_ids:
        if not layer.changeAttributeValue(feature_id, field_index, REVIEW_STATUSES[status]):
            layer.rollBack()
            raise RuntimeError("Could not update a selected review proposal")
    if not layer.commitChanges():
        raise RuntimeError("Could not save review statuses")
    layer.triggerRepaint()
    _message(iface, "Success", f"Marked {len(feature_ids)} proposal(s) as {status}.")
    return len(feature_ids)
