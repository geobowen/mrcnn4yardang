# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
#import utils
from PIL import Image
import yaml
import imgaug

 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Root directory of the project
ROOT_DIR = os.getcwd()
#ROOT_DIR = os.path.abspath("../..")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
iter_num=0

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib,utils
from mrcnn import visualize
from mrcnn.model import log
 
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
 
 
class YardangConfig(Config):
    
    """Configuration for training on the yardang dataset.
    Derives from the base Config class and overrides values specific
    to the yardang dataset.
    """
    
    BACKBONE = "resnet101"
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
    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16,32,64,128,256)  # anchor side in pixels
    
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 300
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 120
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20
 
 
config = YardangConfig()
config.display()
 
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
 
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax
 

dataset_root_path="../dataset/train_data/"
img_floder = dataset_root_path + "pic"
mask_floder = dataset_root_path + "cv2_mask"
val_dataset_root_path="../dataset/val_data/"
val_img_floder = val_dataset_root_path + "pic"
val_mask_floder = val_dataset_root_path + "cv2_mask"

imglist = os.listdir(img_floder)
count = len(imglist)

val_imglist=os.listdir(val_img_floder)
val_count=len(val_imglist)
 
# Prepare train and val data
dataset_train = YardangDataset()
dataset_train.load_image(count, img_floder, mask_floder, imglist,dataset_root_path)
dataset_train.prepare()
#print("dataset_train-->",dataset_train._image_ids)

dataset_val = YardangDataset()
dataset_val.load_image(val_count, val_img_floder, val_mask_floder, val_imglist,val_dataset_root_path)
dataset_val.prepare()
#print("dataset_val-->",dataset_val._image_ids)
 
# Load and display random samples
#image_ids = np.random.choice(dataset_train.image_ids, 4)
#for image_id in image_ids:
#    image = dataset_train.load_image(image_id)
#    mask, class_ids = dataset_train.load_mask(image_id)
#    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
 
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
 
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last
 
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

augmentation=imgaug.augmenters.Sometimes(0.5,imgaug.augmenters.OneOf([
                imgaug.augmenters.Fliplr(1), 
                imgaug.augmenters.Flipud(1), 
                imgaug.augmenters.Affine(rotate=(270)),
                imgaug.augmenters.Affine(rotate=(180)), 
                imgaug.augmenters.Affine(rotate=(90))]))
 
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE*2,
            epochs=5,
            layers='heads',
            augmentation=augmentation)


# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
            layers="all",
            augmentation=augmentation)