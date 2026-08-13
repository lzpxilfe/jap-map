"""Processing output for visible contours and conservative link proposals."""

import json

from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QImage
from qgis.core import QgsFeature, QgsField, QgsGeometry, QgsPointXY, QgsProcessing, QgsProcessingAlgorithm, QgsProcessingException, QgsProcessingParameterFeatureSink, QgsProcessingParameterFile, QgsWkbTypes

from histcontour_core.contours import extract_visible_contours, generate_link_candidates
from histcontour_core.models import MapProfile
from histcontour_core.registration import SheetRegistration


class ExtractContoursAlgorithm(QgsProcessingAlgorithm):
    IMAGE, REGISTRATION, PROFILE, OUTPUT, CANDIDATES = "IMAGE", "REGISTRATION", "PROFILE", "OUTPUT", "CANDIDATES"
    def name(self): return "extractcontours"
    def displayName(self): return "Extract Visible Contours / 가시 등고선 추출"
    def group(self): return "Historical Map Tools"
    def groupId(self): return "historicalmaptools"
    def createInstance(self): return ExtractContoursAlgorithm()
    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile(self.IMAGE, "Original scan", behavior=QgsProcessingParameterFile.File))
        self.addParameter(QgsProcessingParameterFile(self.REGISTRATION, "Registration JSON", behavior=QgsProcessingParameterFile.File))
        self.addParameter(QgsProcessingParameterFile(self.PROFILE, "Map profile JSON", behavior=QgsProcessingParameterFile.File))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Visible contours (GeoPackage or other vector output)", QgsProcessing.TypeVectorLine))
        self.addParameter(QgsProcessingParameterFeatureSink(self.CANDIDATES, "Reconnection proposals (review before approval)", QgsProcessing.TypeVectorLine))
    @staticmethod
    def _pixels(image):
        return [[(image.pixelColor(x, y).red(), image.pixelColor(x, y).green(), image.pixelColor(x, y).blue()) for x in range(image.width())] for y in range(image.height())]
    def processAlgorithm(self, parameters, context, feedback):
        image = QImage(self.parameterAsFile(parameters, self.IMAGE, context))
        if image.isNull(): raise QgsProcessingException("Original scan could not be read")
        try:
            registration = SheetRegistration.read_json(self.parameterAsFile(parameters, self.REGISTRATION, context))
            with open(self.parameterAsFile(parameters, self.PROFILE, context), encoding="utf-8") as handle: profile = MapProfile.from_dict(json.load(handle))
            if profile.contour_rgb is None: raise ValueError("Map profile requires contour_rgb")
            lines = extract_visible_contours(self._pixels(image), profile.contour_rgb, profile.color_distance, profile.min_component_pixels)
        except (OSError, ValueError) as error: raise QgsProcessingException(str(error)) from error
        fields = [QgsField("contour_id", QMetaType.Type.QString), QgsField("sheet_id", QMetaType.Type.QString), QgsField("segment_kind", QMetaType.Type.QString), QgsField("seg_conf", QMetaType.Type.Double), QgsField("link_conf", QMetaType.Type.Double), QgsField("qc_status", QMetaType.Type.QString), QgsField("registration_rmse", QMetaType.Type.Double)]
        sink, destination = self.parameterAsSink(parameters, self.OUTPUT, context, fields, QgsWkbTypes.LineString, registration.crs_authid)
        if sink is None: raise QgsProcessingException("Could not create contour output")
        candidate_fields = [QgsField("candidate_id", QMetaType.Type.QString), QgsField("sheet_id", QMetaType.Type.QString), QgsField("source_lines", QMetaType.Type.QString), QgsField("cost", QMetaType.Type.Double), QgsField("link_conf", QMetaType.Type.Double), QgsField("status", QMetaType.Type.QString), QgsField("registration_rmse", QMetaType.Type.Double)]
        candidate_sink, candidate_destination = self.parameterAsSink(parameters, self.CANDIDATES, context, candidate_fields, QgsWkbTypes.LineString, registration.crs_authid)
        if candidate_sink is None: raise QgsProcessingException("Could not create candidate output")
        for index, line in enumerate(lines):
            if feedback.isCanceled(): break
            feature = QgsFeature(fields); feature.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(*registration.transform(x, y).as_tuple()) for x, y in line.points])); feature.setAttributes([line.line_id, registration.sheet_id, "visible", line.confidence, None, "unreviewed", registration.rmse]); sink.addFeature(feature)
            feedback.setProgress(100 * (index + 1) / max(len(lines), 1))
        for candidate in generate_link_candidates(lines):
            feature = QgsFeature(candidate_fields); feature.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(*registration.transform(x, y).as_tuple()) for x, y in candidate.points])); feature.setAttributes([candidate.candidate_id, registration.sheet_id, ",".join(candidate.source_line_ids), candidate.cost, candidate.link_confidence, candidate.status, registration.rmse]); candidate_sink.addFeature(feature)
        return {self.OUTPUT: destination, self.CANDIDATES: candidate_destination}
