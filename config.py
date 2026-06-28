#config.py is the single source of truth for the entire pipeline. Every module reads from here
RAW_IMAGE_DIR = "data/raw"
OUTPUT_DIR = "outputs"

#Active fruit for this session
ACTIVE_FRUIT = "banana"

# Fruit profiles = add new fruits here without touching any other module
FRUIT_PROFILES = {
    "banana": {
        "hsv_ranges": [
            # (lower, upper) — covers yellow to green range
            ([20, 80, 80],[35, 255, 255]),# yellow (ripe)
            ([35, 80, 80],[85, 255, 255]),# green  (unripe)
        ],
        "ripeness_thresholds": {
            # Based on average Hue of masked fruit
            "unripe":(35, 85),   # green hue range
            "ripe":(20, 35),   # yellow hue range
            "overripe":(0,  20),   # brown/dark — low hue, low saturation
        }
    }
}

# Preprocessing settings
GAUSSIAN_BLUR_KERNEL = (5, 5) # must be odd numbers
CLAHE_CLIP_LIMIT = 2.0 # contrast enhancement strength
CLAHE_TILE_GRID = (8, 8) # image divided into 8x8 tiles for CLAHE

# Segmentation settings
MORPH_KERNEL_SIZE = (5, 5) # size of structuring element for noise cleanup
MIN_CONTOUR_AREA = 5000 # ignore contours smaller than this (filters noise)