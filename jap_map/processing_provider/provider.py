from qgis.core import QgsProcessingProvider

from .register_map import RegisterMapAlgorithm
from .extract_contours import ExtractContoursAlgorithm
from .approve_links import ApproveContourLinksAlgorithm


class HistoricalMapToolsProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(RegisterMapAlgorithm())
        self.addAlgorithm(ExtractContoursAlgorithm())
        self.addAlgorithm(ApproveContourLinksAlgorithm())

    def id(self):
        return "historicalmaptools"

    def name(self):
        return "Historical Map Tools"

    def longName(self):
        return "Historical Map Tools / 역사 지도 도구"
