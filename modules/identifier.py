"""
No deep learning involve, identification is done by matching the detected
contours shape and color profile against entries in FRUIT_PROFILES inside config.py

Two Features:
1. Circularity - measures how round the shape is,
         Formula: C = (4 pi A)/ P^2
  For better understanding
  Given a standard Circle = 1,0
  Banana(elongated) = low value, close to 0.0
  Apple, Orange (round) = high value, close to 1.0

2. Average Hue - mean hue value of pixels inside the mask compared against each fruit known hue range in config.py
"""
import cv2
import numpy as np
import config

class Identifier:
    def __init__(self):
        self.profiles = config.FRUIT_PROFILES
        self.active   = config.ACTIVE_FRUIT

    def identify(self, image, mask, contour):
        #calculate circularity of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        #calculate average Hue inside the masked region
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_channel = hsv[:, :, 0]
        avg_hue = float(np.mean(hue_channel[mask == 255]))

        #match against active fruit profile
        for (lower, upper) in self.profiles[self.active]["hsv_ranges"]:
            if lower[0] <= avg_hue <= upper[0]:
                return self.active, avg_hue, circularity

        #return active fruit, single fruit mode
        return self.active, avg_hue, circularity


#Testing
if __name__ == "__main__":
    import os
    from preprocessor import Preprocessor
    from segmentor import segmentor

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base, "data", "raw", "banana_ripe.jpg")

    img = cv2.imread(img_path)
    p   = Preprocessor()
    s   = segmentor()
    idf = Identifier()

    preprocessed = p.preprocess(img)
    mask         = s.create_mask(preprocessed)
    contours     = s.get_contours(mask)

    fruit, avg_hue, circularity = idf.identify(img, mask, contours[0])

    print(f"Fruit      : {fruit}")
    print(f"Avg Hue    : {avg_hue:.2f}")
    print(f"Circularity: {circularity:.4f}")