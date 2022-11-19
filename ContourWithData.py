MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

class ContourWithData:

    npaContour = None
    boundingRect = None
    intRectX = 0
    intRectY = 0
    intRectWidth = 0
    intRectHeight = 0
    fltArea = 0.0

    def calculateRectTopLeftPointAndWidthAndHeight(self):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):
        # ratio = self.intRectWidth / self.intRectHeight
        # if 1 <= ratio <= 3.5:  # Only select contour with defined ratio
        #     if self.intRectHeight / self.fltArea >= 0.3:
        #         return False
        #     else:
        #         return True
        if self.fltArea < MIN_CONTOUR_AREA :
            return False
        else:
            return True