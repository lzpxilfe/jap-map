"""Processing algorithm that measures the no-training contour baseline."""

import json

from qgis.PyQt.QtGui import QImage
from qgis.core import QgsProcessingAlgorithm, QgsProcessingException, QgsProcessingParameterFile, QgsProcessingParameterFileDestination, QgsProcessingParameterNumber, QgsProcessingParameterString

from histcontour_core.contours import extract_visible_contours, generate_link_candidates
from histcontour_core.models import MapProfile
from histcontour_core.pilot import BaselineMetrics, make_baseline_report


class AssessContourBaselineAlgorithm(QgsProcessingAlgorithm):
    IMAGE, PROFILE, SHEET_ID, MAX_DIMENSION, OUTPUT = "IMAGE", "PROFILE", "SHEET_ID", "MAX_DIMENSION", "OUTPUT"

    def name(self): return "assesscontourbaseline"
    def displayName(self): return "Assess Contour Baseline / 등고선 기준선 측정"
    def group(self): return "Historical Map Tools"
    def groupId(self): return "historicalmaptools"
    def createInstance(self): return AssessContourBaselineAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile(self.IMAGE, "Original scan", behavior=QgsProcessingParameterFile.File))
        self.addParameter(QgsProcessingParameterFile(self.PROFILE, "Map profile JSON", behavior=QgsProcessingParameterFile.File))
        self.addParameter(QgsProcessingParameterString(self.SHEET_ID, "Sheet ID", defaultValue="unregistered-pilot"))
        self.addParameter(QgsProcessingParameterNumber(self.MAX_DIMENSION, "Maximum analysis dimension (pixels)", QgsProcessingParameterNumber.Integer, 2048, False, 256, 8192))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, "Baseline report JSON", "JSON files (*.json)"))

    @staticmethod
    def _pixels(image):
        return [[(image.pixelColor(x, y).red(), image.pixelColor(x, y).green(), image.pixelColor(x, y).blue()) for x in range(image.width())] for y in range(image.height())]

    def processAlgorithm(self, parameters, context, feedback):
        image = QImage(self.parameterAsFile(parameters, self.IMAGE, context))
        if image.isNull(): raise QgsProcessingException("Original scan could not be read")
        try:
            with open(self.parameterAsFile(parameters, self.PROFILE, context), encoding="utf-8") as handle: profile = MapProfile.from_dict(json.load(handle))
            if profile.contour_rgb is None: raise ValueError("Map profile requires contour_rgb")
        except (OSError, ValueError) as error:
            raise QgsProcessingException(str(error)) from error
        maximum = self.parameterAsInt(parameters, self.MAX_DIMENSION, context)
        original_width, original_height = image.width(), image.height()
        scale = min(1.0, maximum / max(original_width, original_height))
        if scale < 1.0:
            image = image.scaled(round(original_width * scale), round(original_height * scale))
        lines = extract_visible_contours(self._pixels(image), profile.contour_rgb, profile.color_distance, profile.min_component_pixels)
        candidates = generate_link_candidates(lines)
        metrics = BaselineMetrics.from_results(image.width(), image.height(), lines, candidates)
        report = make_baseline_report(self.parameterAsString(parameters, self.SHEET_ID, context), profile.profile_id, metrics, scale)
        output = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        with open(output, "w", encoding="utf-8") as handle: json.dump(report, handle, ensure_ascii=False, indent=2)
        feedback.pushInfo(f"Measured {metrics.contour_line_count} lines and {metrics.candidate_count} review candidates at scale {scale:.3f}.")
        return {self.OUTPUT: output}
