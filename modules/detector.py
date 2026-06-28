import cv2
import numpy as np
import config
from modules.preprocessor import Preprocessor
from modules.segmentor import segmentor
from modules.feature_extractor import FeatureExtractor
from modules.grader import Grader

class Detector:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.segmentor = segmentor()
        self.extractor = FeatureExtractor()
        self.grader = Grader()

        #color coding for each ripeness label
        self.label_colours = {
            "Unripe": (0, 255, 0),# green
            "Ripe": (0, 255, 255),# yellow
            "Overripe": (0, 0, 255),# red
            "Unknown": (255, 255, 255) # white
        }

    def detect(self, frame):
        #preprocess frame once,reused for all fruit profiles
        preprocessed = self.preprocessor.preprocess(frame)
        annotated = frame.copy()
        results = []

        #loop through every fruit profile independently
        for fruit_name, profile in config.FRUIT_PROFILES.items():

            #temporarily set active fruit to current profile
            original_active = config.ACTIVE_FRUIT
            config.ACTIVE_FRUIT = fruit_name

            #create mask and find contours for this fruit
            mask = self.segmentor.create_mask(preprocessed)
            contours = self.segmentor.get_contours(mask)

            #restore active fruit
            config.ACTIVE_FRUIT = original_active

            #process each detected contour
            for contour in contours:
                features = self.extractor.extract(preprocessed, mask)
                ripeness = self.grader.grade(features)

                #get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                colour = self.label_colours.get(ripeness, (255, 255, 255))

                #draw bounding box and label
                cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)
                cv2.putText(
                    annotated,
                    f"{fruit_name} | {ripeness}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colour, 2
                )

                results.append({
                    "fruit": fruit_name,
                    "ripeness": ripeness,
                    "hue": features["hue_mean"],
                    "blemish": features["blemish_ratio"]
                })

        return annotated, results