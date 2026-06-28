import cv2
import os
import config
from modules.preprocessor import Preprocessor
from modules.segmentor import segmentor
from modules.identifier import Identifier
from modules.feature_extractor import FeatureExtractor
from modules.grader import Grader

def run_pipeline(img_path):
    #load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return None

    #step 1: Preprocess
    p = Preprocessor()
    preprocessed = p.preprocess(img)

    #step 2: Segment
    s = segmentor()
    mask = s.create_mask(preprocessed)
    contours = s.get_contours(mask)

    if len(contours) == 0:
        print("Error: No fruit detected in image")
        return None

    #step 3: Identify
    idf = Identifier()
    fruit, avg_hue, circular = idf.identify(img, mask, contours[0])

    #step 4: Extract features
    fe = FeatureExtractor()
    features = fe.extract(preprocessed, mask)

    #step 5: Grade
    g = Grader()
    result = g.grade(features)

    return {
        "fruit": fruit,
        "ripeness" : result,
        "hue_mean": features["hue_mean"],
        "sat_mean" : features["sat_mean"],
        "blemish" : features["blemish_ratio"],
        "circularity": circular
    }

#Testing
if __name__ == "__main__":
    test_images = {
        "Unripe"  : "data/raw/banana_unripe.jpg",
        "Ripe"    : "data/raw/banana_ripe.jpg",
        "Overripe": "data/raw/banana_overripe.jpg"
    }

    print("=" * 60)
    print("  Fruit Ripeness Assessment System — Integration Test")
    print("=" * 60)

    for expected, path in test_images.items():
        output = run_pipeline(path)
        if output:
            status = "[Correct]" if output["ripeness"] == expected else "[Incorrect]"
            print(f"\n{status} Image    : {path}")
            print(f"   Fruit    : {output['fruit']}")
            print(f"   Expected : {expected}")
            print(f"   Got      : {output['ripeness']}")
            print(f"   Hue      : {output['hue_mean']:.2f}")
            print(f"   Sat      : {output['sat_mean']:.2f}")
            print(f"   Blemish  : {output['blemish']:.4f}")
            print(f"   Circular : {output['circularity']:.4f}")

    print("\n" + "=" * 60)

"""
temporary debug stage
    import numpy as np
    from modules.preprocessor import Preprocessor
    from modules.segmentor import segmentor

    debug_img  = cv2.imread("data/raw/banana_overripe.jpg")
    p          = Preprocessor()
    s          = segmentor()
    pre        = p.preprocess(debug_img)
    mask       = s.create_mask(pre)
    contours   = s.get_contours(mask)

    print(f"White pixels in mask : {np.sum(mask == 255)}")
    print(f"Contours found       : {len(contours)}")
    cv2.imshow("Overripe Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""