import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import config

class FeatureExtractor:
    def __inti__(self):
        self.active = config. ACTIVE_FRUIT

    def extract_colour(self, image, mask):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:,:,0]
        sat = hsv[:,:,1]

        #read masked pixels only
        hue_pixels = hue[mask==255].astype(float)
        sat_pixels = sat[mask == 255].astype(float)

        hue_mean = np.mean(hue_pixels)
        hue_std = np.std(hue_pixels)
        #fixed, np.cbrt handles negative cube roots correctly
        hue_skew = float(np.cbrt(np.mean((hue_pixels - hue_mean) ** 3))) if len(hue_pixels) > 0 else 0
        sat_mean = np.mean(sat_pixels)

        return {
            "hue_mean" : hue_mean,
            "hue_std"  : hue_std,
            "hue_skew" : hue_skew,
            "sat_mean" : sat_mean
        }

    def extract_blemish(self, image, mask):
        #convert to LAB, L channel captures dark spots clearly
        """
        Why L channel but5 not Hue for blemish?
        - brown spot on banana do not have consistent Hue
        - they can appear dark yellow, brown , near-black

        What they always share is being darker than healthy skin.
        Lightness is therefore a more reliable blemish detector than Hue.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        #apply mask (fruit pixels only)
        l_masked = l_channel.copy()
        l_masked[mask == 0] = 255  #set background to white (non-blemish)
        #threshold dark pixels inside fruit, blemishes are darker than skin
        _, blemish_mask = cv2.threshold(l_masked, 80, 255, cv2.THRESH_BINARY_INV)

        #calculate blemish ratio
        fruit_pixels = np.sum(mask == 255)
        blemish_pixels = np.sum(blemish_mask == 255)
        blemish_ratio = blemish_pixels / fruit_pixels if fruit_pixels > 0 else 0

        return {
            "blemish_ratio": blemish_ratio,
            "blemish_mask": blemish_mask
        }

    def extract_texture(self, image, mask):
        #convert to grayscale- GLCM works on single channel
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #apply mask,fruit pixels only
        gray_masked = gray.copy()
        gray_masked[mask == 0] = 0

        #compute GLCM, distance=1, angles=0° and 90°
        glcm = graycomatrix(
            gray_masked,
            distances=[1],
            angles=[0, np.pi / 2],
            levels=256,
            symmetric=True,
            normed=True
        )

        # Extract four texture descriptors
        contrast = graycoprops(glcm, "contrast").mean()
        homogeneity = graycoprops(glcm, "homogeneity").mean()
        energy = graycoprops(glcm, "energy").mean()
        correlation = graycoprops(glcm, "correlation").mean()

        return {
            "contrast": contrast,
            "homogeneity": homogeneity,
            "energy": energy,
            "correlation": correlation
        }

    def extract(self, image, mask):
        colour = self.extract_colour(image, mask)
        blemish = self.extract_blemish(image, mask)
        texture = self.extract_texture(image, mask)
        return {**colour, **blemish, **texture}

#Testing
if __name__ == "__main__":
    import os
    from preprocessor import Preprocessor
    from segmentor import segmentor

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base, "data", "raw", "banana_ripe.jpg")

    img = cv2.imread(img_path)
    p = Preprocessor()
    s = segmentor()
    fe = FeatureExtractor()
    preprocessed = p.preprocess(img)
    mask = s.create_mask(preprocessed)
    features = fe.extract(preprocessed, mask)

    for key, value in features.items():
        if key != "blemish_mask":  #skip printing the mask array
            print(f"{key:15}: {value:.4f}")