





from PIL import Image
import os

interp = Image.BILINEAR

import sys
print("Running {}".format(sys.argv[0]))

test_pre = sys.argv[1]
test_post = sys.argv[2]


import torch
import numpy as np
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import matplotlib.pyplot as plt

# Load damage model.
cfg_damage = get_cfg()
DAMAGE_MODEL_CONFIG = "./configs/xview/joint-11.yaml"
cfg_damage.merge_from_file(DAMAGE_MODEL_CONFIG)
# Load damage checkpoint.
cfg_damage.MODEL.WEIGHTS = os.path.join("model_weights.pth")
cfg_damage.MODEL.DEVICE = "cpu"
# cfg_damage.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set the testing threshold for this model
predictor_damage = DefaultPredictor(cfg_damage)

# Load the images.
image = cv2.imread(test_post)
pre_image = cv2.imread(test_pre)

def visualize_semantic_segmentation(output_tensor):
    # Define label colors
    label_colors = [
        (0, 0, 0),       # 0=background
        (0, 255, 0),     # no damage (or just 'building' for localization) (green)
        (255, 255, 0),   # minor damage (yellow)
        (255, 128, 0),   # major damage (orange)
        (255, 0, 0),     # destroyed (red)
        (127, 127, 127)  # Unlabeled (gray)
    ]

    # Map pixel values to RGB colors
    output_rgb = np.zeros((output_tensor.shape[0], output_tensor.shape[1], 3), dtype=np.uint8)
    for label_id, color in enumerate(label_colors):
        output_rgb[output_tensor == label_id] = color

    return output_rgb

# Perform damage prediction and visualization for each quadrant
quadrants = []
for y_start, y_end in [(0, 512), (512, 1024)]:
    for x_start, x_end in [(0, 512), (512, 1024)]:
        temp_image = image[y_start:y_end, x_start:x_end]
        temp_pre_image = pre_image[y_start:y_end, x_start:x_end]
        outputs = predictor_damage(temp_image, temp_pre_image)
        output = outputs["sem_seg"].argmax(dim=0).cpu().numpy()
        quadrants.append(output)

# Construct the complete image from quadrants
temp = np.zeros((1024, 1024)).astype(int)
for i, (y_start, y_end) in enumerate([(0, 512), (512, 1024)]):
    for j, (x_start, x_end) in enumerate([(0, 512), (512, 1024)]):
        temp[y_start:y_end, x_start:x_end] = quadrants[i * 2 + j]

# Visualize the semantic segmentation
output_image = visualize_semantic_segmentation(temp)

# Convert the output image to BGR format (required by OpenCV)
output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

# Overlay the output mask on the original post-disaster image
overlay = cv2.addWeighted(image, 0.5, output_image_bgr, 0.5, 0)

# Write the overlay image to file
cv2.imwrite(test_post, overlay)
