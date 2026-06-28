"""
Segmentor: Separate fruit from background
Producing binary mask where white pixels = fruits, black pixels = background

3 Steps:
1. HSV conversion - convert preprocessed BGR image into HSV color space to separate color(Hue) from brightness (Value)
2. HSV thresholding - pixels within the banana hue range are kept(white), others discarded(black)
3. Morphological cleanup - small holes and noise in the mask are removed using opening and closing operations
"""

import cv2
import numpy as np
import config

class segmentor():
    def __init__(self):
        self.kernel = cv2.getStructuringElement(
            #creates elliptical kernel as fruits shapes are curved
            cv2.MORPH_ELLIPSE,
            config.MORPH_KERNEL_SIZE
        )
        self.min_area = config.MIN_CONTOUR_AREA

    def create_mask(self, image):
        #convert to HSV, stable color representation under lighting changes
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #build mask from all HSV ranges defined in fruit profile
        profile = config.FRUIT_PROFILES[config.ACTIVE_FRUIT]
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for (lower, upper) in profile["hsv_ranges"]:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask |= cv2.inRange(hsv, lower, upper)  #combine all ranges

        #Morphological opening(removes small noise outside fruit)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        #Morphological closing(fills small holes inside fruit)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        return mask

    def get_contours(self, mask):
        #find all contours in the binary mask
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,  #outer contours only
            cv2.CHAIN_APPROX_SIMPLE  #compress redundant points
        )

        #filter out contours smaller than minimum area threshold
        valid = [c for c in contours if cv2.contourArea(c) > self.min_area]

        return valid


#Testing
if __name__ == "__main__":
    import os
    from preprocessor import Preprocessor

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base, "data", "raw", "banana_ripe.jpg")

    img = cv2.imread(img_path)
    p = Preprocessor()
    preprocessed = p.preprocess(img)

    s = segmentor()
    mask = s.create_mask(preprocessed)
    contours = s.get_contours(mask)

    # Draw contours on original image
    output = img.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    print(f"Contours found: {len(contours)}")
    cv2.imshow("Mask", mask)
    cv2.imshow("Contours", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()