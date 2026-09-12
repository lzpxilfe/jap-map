"""QGIS Processing entry point for review-only Ink proposal extraction."""

from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QImage
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsWkbTypes,
)

from histcontour_core.ink import ARCHAEOTRACE_UPSTREAM_COMMIT, INK_BACKEND_ID, InkCenterlineSettings, ink_centerline_candidates
from histcontour_core.onnx_scorer import OnnxScorer, OnnxScorerUnavailable
from histcontour_core.provenance import INK_ADAPTER_VERSION, execution_id, polyline_geometry_id, sha256_file
from histcontour_core.segment_review import LogisticModel, geometry_features
from histcontour_core.vectorization import DEFAULT_DIAGONAL_POLICY, SKELETON_GRAPH_VERSION, skeleton_to_pixel_line_proposals


class ExtractInkProposalsAlgorithm(QgsProcessingAlgorithm):
    RASTER, MINIMUM_LENGTH, MODEL, MINIMUM_SCORE, OUTPUT = "RASTER", "MINIMUM_LENGTH", "MODEL", "MINIMUM_SCORE", "OUTPUT"

    def name(self): return "extractinkproposals"
    def displayName(self): return "Extract Ink Proposals / Ink 선분 추출·검토"
    def group(self): return "Historical Map Tools"
    def groupId(self): return "historicalmaptools"
    def createInstance(self): return ExtractInkProposalsAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.RASTER, "Source raster"))
        self.addParameter(QgsProcessingParameterNumber(self.MINIMUM_LENGTH, "Minimum segment length (source pixels)", QgsProcessingParameterNumber.Double, 18.0, False, 2.0, 2048.0))
        self.addParameter(QgsProcessingParameterFile(self.MODEL, "Optional trained contour-score model JSON", behavior=QgsProcessingParameterFile.File, optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.MINIMUM_SCORE, "Minimum contour score (0 keeps all; model required above 0)", QgsProcessingParameterNumber.Double, 0.0, False, 0.0, 1.0))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Ink proposals (review only)", QgsProcessing.TypeVectorLine))

    @staticmethod
    def _array(layer):
        try:
            import numpy as np
        except ImportError as error:
            raise QgsProcessingException("Ink extraction needs NumPy in the QGIS Python environment") from error
        image = QImage(layer.source().split("|")[0])
        if image.isNull():
            raise QgsProcessingException("Ink extraction currently needs a readable local raster image")
        image = image.convertToFormat(QImage.Format.Format_Grayscale8)
        values = np.frombuffer(image.constBits().asstring(image.sizeInBytes()), dtype=np.uint8)
        return values.reshape(image.height(), image.bytesPerLine())[:, :image.width()].copy()

    @staticmethod
    def _map_points(extent, width, height, points):
        return [QgsPointXY(extent.xMinimum() + (x + 0.5) * extent.width() / width, extent.yMaximum() - (y + 0.5) * extent.height() / height) for x, y in points]

    @staticmethod
    def _score_model(path):
        if not path:
            return None
        if path.lower().endswith(".onnx"):
            try:
                return "onnx", OnnxScorer(path)
            except (OnnxScorerUnavailable, ValueError) as error:
                raise QgsProcessingException(f"ONNX contour-score model is incompatible: {error}") from error
        try:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
            if report.get("status") != "trained":
                raise ValueError("model report is not trained")
            return "logistic", LogisticModel.from_dict(report["model"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
            raise QgsProcessingException(f"Contour-score model is incompatible: {error}") from error

    @staticmethod
    def _context_features(source, points):
        import numpy as np
        samples = []
        for first, second in zip(points, points[1:]):
            count = max(1, int(math.ceil(math.hypot(second[0] - first[0], second[1] - first[1]))))
            samples.extend((first[0] + step / count * (second[0] - first[0]), first[1] + step / count * (second[1] - first[1])) for step in range(count))
        samples.append(points[-1])
        height, width = source.shape
        xs = np.clip(np.rint([point[0] for point in samples]).astype(int), 0, width - 1)
        ys = np.clip(np.rint([point[1] for point in samples]).astype(int), 0, height - 1)
        centre_x, centre_y = int(round(sum(point[0] for point in samples) / len(samples))), int(round(sum(point[1] for point in samples) / len(samples)))
        def crop(radius):
            return source[max(0, centre_y - radius):min(height, centre_y + radius + 1), max(0, centre_x - radius):min(width, centre_x + radius + 1)]
        near, context = crop(16), crop(48)
        return {
            "line_darkness": float((1.0 - source[ys, xs].astype(np.float32) / 255.0).mean()),
            "dark_fraction_near": float((near <= 96).mean()),
            "dark_fraction_context": float((context <= 96).mean()),
            "mid_fraction_context": float((context <= 176).mean()),
            "context_std": float(context.astype(np.float32).std() / 255.0),
        }

    def processAlgorithm(self, parameters, context, feedback):
        layer = self.parameterAsRasterLayer(parameters, self.RASTER, context)
        if layer is None:
            raise QgsProcessingException("Could not open source raster")
        minimum_length = self.parameterAsDouble(parameters, self.MINIMUM_LENGTH, context)
        model_path = self.parameterAsFile(parameters, self.MODEL, context)
        model = self._score_model(model_path)
        minimum_score = self.parameterAsDouble(parameters, self.MINIMUM_SCORE, context)
        if not math.isfinite(minimum_score) or not 0 <= minimum_score <= 1:
            raise QgsProcessingException("Minimum contour score must be in [0, 1]")
        if minimum_score > 0 and model is None:
            raise QgsProcessingException("A contour-score model is required for score filtering")
        model_digest = sha256_file(model_path) if model is not None else None
        source = self._array(layer)
        evidence = ink_centerline_candidates(source)
        proposals = skeleton_to_pixel_line_proposals(evidence.centerline, evidence.center_score, minimum_length_px=minimum_length, simplify_tolerance_px=0.75, proposal_prefix="ink-line")
        width, height, extent = layer.width(), layer.height(), layer.extent()
        source_path = Path(layer.source().split("|")[0])
        raster_digest = sha256_file(source_path) if source_path.is_file() else hashlib.sha256(layer.source().encode("utf-8")).hexdigest()
        run_id = execution_id(raster_sha256=raster_digest, backend=INK_BACKEND_ID, upstream_commit=ARCHAEOTRACE_UPSTREAM_COMMIT, settings={"ink": InkCenterlineSettings().__dict__}, vectorization={"minimum_length_px": minimum_length, "simplify_tolerance_px": 0.75, "graph_version": SKELETON_GRAPH_VERSION, "diagonal_policy": DEFAULT_DIAGONAL_POLICY})
        fields = [
            QgsField("proposal_id", QMetaType.Type.QString), QgsField("segment_uid", QMetaType.Type.QString),
            QgsField("ink_run_id", QMetaType.Type.QString), QgsField("backend", QMetaType.Type.QString),
            QgsField("upstream_commit", QMetaType.Type.QString), QgsField("adapter_version", QMetaType.Type.QString),
            QgsField("source_raster_sha256", QMetaType.Type.QString),
            QgsField("pixel_length", QMetaType.Type.Double), QgsField("ink_support", QMetaType.Type.Double),
            QgsField("direction_coherence", QMetaType.Type.Double), QgsField("contour_score", QMetaType.Type.Double),
            QgsField("score_threshold", QMetaType.Type.Double), QgsField("score_model_sha256", QMetaType.Type.QString),
            QgsField("review_status", QMetaType.Type.QString),
        ]
        sink, destination = self.parameterAsSink(parameters, self.OUTPUT, context, QgsFields(fields), QgsWkbTypes.LineString, layer.crs())
        if sink is None:
            raise QgsProcessingException("Could not create Ink proposal output")
        retained = 0
        for index, proposal in enumerate(proposals):
            if feedback.isCanceled():
                break
            points = proposal.points
            coherence = sum(float(evidence.coherence[int(round(y)), int(round(x))]) for x, y in points) / len(points)
            score = None
            if model is not None:
                kind, scorer = model
                score = scorer.probability(source, points) if kind == "onnx" else scorer.probability({**geometry_features(points), **self._context_features(source, points)})
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise QgsProcessingException("Contour model produced an invalid score")
            feedback.setProgress(100 * (index + 1) / max(1, len(proposals)))
            if score is not None and score < minimum_score:
                continue
            geometry_id = polyline_geometry_id(points)
            output = QgsFeature(QgsFields(fields))
            output.setGeometry(QgsGeometry.fromPolylineXY(self._map_points(extent, width, height, points)))
            ink_support = sum(float(evidence.support_score[int(round(y)), int(round(x))]) for x, y in points) / len(points)
            output.setAttributes([proposal.proposal_id, f"{run_id}:{geometry_id}", run_id, INK_BACKEND_ID, ARCHAEOTRACE_UPSTREAM_COMMIT, INK_ADAPTER_VERSION, raster_digest, proposal.pixel_length, ink_support, coherence, score, minimum_score, model_digest, "unreviewed"])
            sink.addFeature(output)
            retained += 1
        feedback.pushInfo(f"Retained {retained}/{len(proposals)} visible-linework proposals; score threshold {minimum_score}.")
        feedback.pushInfo("Ink support is line evidence. contour_score is review-only and is not final contour geometry.")
        return {self.OUTPUT: destination}
