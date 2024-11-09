# input one big image, clip it, predict them and merge the results 

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from PIL import Image
import yaml
import gdal

from datetime import datetime

#Root dir of the project
ROOT_DIR = os.path.abspath("../..")

#import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualizemod

# Directory to save logs and trained model
MODEL_DIR = "E:/package/code/Mask_RCNN/pre-logs/shapes20200622T0951"

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_shapes_0033.h5")  #_0040.h5?

# Directory of images to run detection on
IMAGE_DIR = r'E:\data\test\18\test_3band'

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 types of yardangs  修改
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = "square"
    #IMAGE_MIN_DIM = 512
    #IMAGE_MAX_DIM =2048
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64,128,256)  # anchor side in pixels
    
    ###!!在较大范围进行检测时要对对多可探测instance数目 默认只有100!!!!!
    DETECTION_MAX_INSTANCES = 500
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 500
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG','long-ridge', 'mesa', 'whaleback']

area_perc = 0.5
ImgPath = r''
ResultPath = r''
RepetitiveLength = int((1 - math.sqrt(area_perc))*256/2)

testtime = []
starttime = datatime.now()

im_width, im_height, im_bands, im_data, im_






















