import config

class Grader:
    def __init__(self):
        self.profile  = config.FRUIT_PROFILES[config.ACTIVE_FRUIT]
        self.thresholds = self.profile["ripeness_thresholds"]

        # Overripe detection thresholds
        self.overripe_sat_limit= 150   # below this = desaturated/brown
        self.overripe_blemish_limit= 0.10  # above this = heavily blemished

    def grade(self, features):
        hue_mean = features["hue_mean"]
        sat_mean = features["sat_mean"]
        blemish_ratio = features["blemish_ratio"]

        #Pass 1: check overripe first (brown = low sat + high blemish)
        if sat_mean < self.overripe_sat_limit or blemish_ratio > self.overripe_blemish_limit:
            return "Overripe"

        #Pass 2: classify by Hue range
        unripe_low, unripe_high = self.thresholds["unripe"]
        ripe_low, ripe_high = self.thresholds["ripe"]

        if unripe_low <= hue_mean <= unripe_high:
            return "Unripe"
        elif ripe_low <= hue_mean <= ripe_high:
            return "Ripe"

        #fallback, hue outside all defined ranges
        return "Unknown"

#Testing
if __name__ == "__main__":
    import os
    import cv2
    from preprocessor import Preprocessor
    from segmentor import segmentor
    from feature_extractor import FeatureExtractor

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #test all three images
    images   = {
        "Unripe": "banana_unripe.jpg",
        "Ripe": "banana_ripe.jpg",
        "Overripe": "banana_overripe.jpg"
    }
    p  = Preprocessor()
    s  = segmentor()
    fe = FeatureExtractor()
    g  = Grader()

    for expected, filename in images.items():
        img_path = os.path.join(base, "data", "raw", filename)
        img = cv2.imread(img_path)
        preprocessed = p.preprocess(img)
        mask = s.create_mask(preprocessed)
        features = fe.extract(preprocessed, mask)
        result = g.grade(features)

        status = "[Correct]" if result == expected else "[Incorrect]"
        print(f"{status} Expected: {expected:10} | Got: {result:10} | Hue: {features['hue_mean']:.2f} | Sat: {features['sat_mean']:.2f} | Blemish: {features['blemish_ratio']:.4f}")