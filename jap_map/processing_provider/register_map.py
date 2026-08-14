"""Processing algorithm for a four-corner projective registration file."""

from qgis.PyQt.QtGui import QImage
from qgis.core import QgsProcessingAlgorithm, QgsProcessingException, QgsProcessingParameterFile, QgsProcessingParameterFileDestination, QgsProcessingParameterString

from histcontour_core.registration import GroundControlPoint, SheetRegistration


class RegisterMapAlgorithm(QgsProcessingAlgorithm):
    IMAGE, SHEET_ID, CRS, CORNERS, PIXEL_CORNERS, OUTPUT = "IMAGE", "SHEET_ID", "CRS", "CORNERS", "PIXEL_CORNERS", "OUTPUT"

    def name(self): return "registermap"
    def displayName(self): return "Register Map / 지도 맞추기"
    def group(self): return "Historical Map Tools"
    def groupId(self): return "historicalmaptools"
    def createInstance(self): return RegisterMapAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile(self.IMAGE, "Original scan", behavior=QgsProcessingParameterFile.File))
        self.addParameter(QgsProcessingParameterString(self.SHEET_ID, "Sheet ID"))
        self.addParameter(QgsProcessingParameterString(self.CRS, "Target CRS authid (for example EPSG:4326)"))
        self.addParameter(QgsProcessingParameterString(self.CORNERS, "NW;NE;SE;SW map corners as x,y;x,y;x,y;x,y"))
        self.addParameter(QgsProcessingParameterString(self.PIXEL_CORNERS, "Optional printed-map pixel corners as NW;NE;SE;SW x,y pairs (blank uses image edges)", optional=True))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, "Registration JSON", "JSON files (*.json)"))

    def processAlgorithm(self, parameters, context, feedback):
        image_path = self.parameterAsFile(parameters, self.IMAGE, context)
        image = QImage(image_path)
        if image.isNull(): raise QgsProcessingException("Original scan could not be read")
        try:
            values = [tuple(float(value) for value in corner.split(",")) for corner in self.parameterAsString(parameters, self.CORNERS, context).split(";")]
            if len(values) != 4 or any(len(value) != 2 for value in values): raise ValueError
            width, height = image.width(), image.height()
            pixel_text = self.parameterAsString(parameters, self.PIXEL_CORNERS, context).strip()
            pixels = [tuple(float(value) for value in corner.split(",")) for corner in pixel_text.split(";")] if pixel_text else [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
            if len(pixels) != 4 or any(len(value) != 2 for value in pixels): raise ValueError
            gcps = [GroundControlPoint(pixel[0], pixel[1], map_point[0], map_point[1], label) for pixel, map_point, label in zip(pixels, values, ("NW", "NE", "SE", "SW"))]
            registration = SheetRegistration.create(self.parameterAsString(parameters, self.SHEET_ID, context), image_path, width, height, self.parameterAsString(parameters, self.CRS, context), gcps)
            output = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
            registration.write_json(output)
        except ValueError as error:
            raise QgsProcessingException("CORNERS and PIXEL_CORNERS must be four x,y pairs in NW;NE;SE;SW order") from error
        return {self.OUTPUT: output}
