import cv2
import config

class Preprocessor:
    def __init__(self): #load setting from config
        self.blur_kernel = config.GAUSSIAN_BLUR_KERNEL
        self.clahe = cv2.createCLAHE(
            #prepares CLAHE object ready for use
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID
        )

    def preprocess(self, image):
        # Step 1:Gaussian blur (removes noise before color analysis)
        blurred = cv2.GaussianBlur(image, self.blur_kernel, 0)

        # Step 2:Convert to LAB (CLAHE works on luminance channel only)
        """use LAB instead of BGR as CLAHE enhances contrast by redistributing pixel intensities.
        If applied directly to BGR, it shifts all three color channels independently, causing color distortion.
        
        By converting to LAB first;
        - CLAHE apply to lightness (L) only
        - Color information (A and B) remain unlocked
        - Fruit color stays accurate for HSV thresholding later 
        """
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)

        # Step 3:Apply CLAHE to L channel only (enhances contrast without shifting color)
        l_enhanced = self.clahe.apply(l)

        # Step 4:Merge back and convert to BGR
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

# Testing
if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base, "data", "raw", "banana_ripe.jpg")

    img = cv2.imread(img_path)
    p = Preprocessor()
    result = p.preprocess(img)
    cv2.imshow("Original", img)
    cv2.imshow("Preprocessed", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()