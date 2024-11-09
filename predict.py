# -*- coding: utf-8 -*-
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

from datetime import datetime
 
# Root directory of the project
ROOT_DIR = os.getcwd()
#ROOT_DIR = os.path.abspath("../..")
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualizemod

'''
fileID = sys.argv[1]

if (len(sys.argv)!=2):
    print('the input params should be equal to 1, namely the image_id')
    sys.exit(1)
'''

ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = "../logs/shapes20200622T0951"

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_shapes_0027.h5") #mask_rcnn_shapes_0040.h5

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "../dataset/test_data/pic")
print("IMAGE_DIR is:", IMAGE_DIR)


class YardangConfig(Config):

    # Give the configuration a recognizable name
    NAME = "yardang"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 types of yardang
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    RPN_ANCHOR_SCALES = (16,32,64,128,256)  # anchor side in pixels
    
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_MAX_INSTANCES = 1500
    
    TRAIN_ROIS_PER_IMAGE = 4000
    STEPS_PER_EPOCH = 300
    VALIDATION_STEPS = 50

class YardangDataset(utils.Dataset):
    # Count of the instances(objects)
    def get_obj_index(self, image):
        n = np.max(image)
        return n
 
    # Parse the yaml file obtained from LabelMe 
    # and get corresponding labels for each layer of the mask
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels
 
    # Override draw_mask
    def draw_mask(self, num_obj, mask, image,image_id):
        #print("draw_mask-->",image_id)
        #print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        #print("info-->",info)
        #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    #print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask
 
    # Override load_image
    def load_image(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        
        # Add classes
        self.add_class("yardang", 1, "long-ridge")
        self.add_class("yardang", 2, "mesa")
        self.add_class("yardang", 3, "whaleback")

        for i in range(count):
            # image height & width
 
            filestr = imglist[i].split(".")[0]
            #print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            #print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            #filestr = filestr.split("_")[1]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
            print(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")

            #self.add_image
            super.add_image("yardang", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # Override load_mask
    def load_mask(self, image_id):
        """Generate instance masks for yardangs of the given image ID.
        """
        global iter_num
        print("image_id",image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img,image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
 
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("long-ridge") != -1:
                labels_form.append("long-ridge")
            elif labels[i].find("mesa")!=-1:
                labels_form.append("mesa")
            elif labels[i].find("whaleback")!=-1:
                labels_form.append("whaleback")
            
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

class InferenceConfig(YardangConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG','long-ridge', 'mesa', 'whaleback']

'''
resultpath = IMAGE_DIR + '\\binary'
if not os.path.exists(resultpath):
    os.mkdir(resultpath)
'''

#if only show boundary (set alpha=0 in apply_mask in mrcnn.visualize)

count = os.listdir(IMAGE_DIR)

for i in range(0,len(count)):
    path = os.path.join(IMAGE_DIR, count[i])
    if os.path.isfile(path):
        file_names = next(os.walk(IMAGE_DIR))[2]
        image = skimage.io.imread(os.path.join(IMAGE_DIR, count[i]))
        a=datetime.now()
        # Run detection
        results = model.detect([image], verbose=1)
        b=datetime.now()
        print("lasting time", (b - a).seconds)
        r = results[0]
        visualize.display_instances(count[i],image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])
        
        # visualize.display_binary(resultpath,count[i], image, r['rois'], r['masks'], r['class_ids'])

# Calculate mAP（iou threshold can be set in utils）
APs=[]

test_dataset_root_path="dataset/test_data/"
test_img_folder = test_dataset_root_path + "pic"
test_mask_folder = test_dataset_root_path + "cv2_mask"
test_imglist=os.listdir(test_img_folder)
test_count=len(test_imglist)

dataset_test = YardangDataset()
dataset_test.load_image(test_count, test_img_folder, test_mask_folder, test_imglist,test_dataset_root_path)
dataset_test.prepare()

image_ids = dataset_test.image_ids
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_test, config,image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    
    # Show difference
    #visualize.display_differences(image,gt_bbox, gt_class_id, gt_mask,r['rois'],r['class_ids'],r['scores'],r['masks'],class_names)
    
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,r["rois"], r["class_ids"], r["scores"], r['masks'])
    
    print(AP)
    APs.append(AP)
    
print("mAP: ", np.mean(APs))