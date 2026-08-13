"""Turn explicitly approved proposal features into completion segments."""

from qgis.PyQt.QtCore import QMetaType
from qgis.core import QgsFeature, QgsField, QgsProcessing, QgsProcessingAlgorithm, QgsProcessingException, QgsProcessingParameterFeatureSink, QgsProcessingParameterFeatureSource, QgsWkbTypes


class ApproveContourLinksAlgorithm(QgsProcessingAlgorithm):
    CANDIDATES, OUTPUT = "CANDIDATES", "OUTPUT"
    def name(self): return "approvecontourlinks"
    def displayName(self): return "Create Approved Contour Completions / 승인된 등고선 연결 만들기"
    def group(self): return "Historical Map Tools"
    def groupId(self): return "historicalmaptools"
    def createInstance(self): return ApproveContourLinksAlgorithm()
    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.CANDIDATES, "Reviewed reconnection proposals (status=approved)", [QgsProcessing.TypeVectorLine]))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Approved completion segments", QgsProcessing.TypeVectorLine))
    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.CANDIDATES, context)
        if source is None: raise QgsProcessingException("Could not read candidate layer")
        fields = [QgsField("completion_id", QMetaType.Type.QString), QgsField("sheet_id", QMetaType.Type.QString), QgsField("source_lines", QMetaType.Type.QString), QgsField("segment_kind", QMetaType.Type.QString), QgsField("link_conf", QMetaType.Type.Double), QgsField("qc_status", QMetaType.Type.QString)]
        sink, destination = self.parameterAsSink(parameters, self.OUTPUT, context, fields, QgsWkbTypes.LineString, source.sourceCrs())
        if sink is None: raise QgsProcessingException("Could not create completion output")
        for feature in source.getFeatures():
            if str(feature["status"]).lower() != "approved": continue
            output = QgsFeature(fields); output.setGeometry(feature.geometry()); output.setAttributes([feature["candidate_id"], feature["sheet_id"], feature["source_lines"], "approved_completion", feature["link_conf"], "approved"]); sink.addFeature(output)
        return {self.OUTPUT: destination}
