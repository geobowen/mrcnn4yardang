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
import pandas as pd


from datetime import datetime
 
# Root directory of the project
#ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.abspath("../..")
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utilsmod
import mrcnn.model as modellib
from mrcnn import visualizemod
 
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# from samples.coco import coco
 
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
#MODEL_DIR="E:/package/code/Mask_RCNN/logs/shapes20200621T1132"


# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "shapes20200622T0951/mask_rcnn_shapes_0040.h5")  # 修改
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utilsmod.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")
 
# Directory of images to run detection on
#IMAGE_DIR = os.path.join(ROOT_DIR, "images")
#IMAGE_DIR = os.path.join(os.getcwd(),'test_data/pic_with_alpha_removed')#os.path.join(os.getcwd(),'test_data/pic_with_alpha_removed') #"E:/package/code/Mask_RCNN/samples/new-yardang/test_data/pic"
#IMAGE_DIR = 'G:/data/alpha_removed_ori_data/ava'

IMAGE_DIR = os.path.join(os.getcwd(),'test_data_mod/pic_with_alpha_removed')
#IMAGE_DIR = os.path.join(os.getcwd(),'ex_data_onebyone/pic_ori')

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
    NUM_CLASSES = 1 + 3  # background + 3 shapes  修改
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = "none"
    #IMAGE_MIN_DIM = 512 
    #IMAGE_MAX_DIM = 512
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels
    
    
    ###!!在较大范围进行检测时要对对多可探测instance数目 默认只有100!!!!!
    DETECTION_MAX_INSTANCES = 500
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 500
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

class DrugDataset(utilsmod.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n
 
    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels
 
    # 重新写draw_mask
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
 
    # 重新写load_shapes，里面包含自己的类别,可以任意添加
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes,可通过这种方式扩展多个物体
        self.add_class("shapes", 1, "long-ridge")
        self.add_class("shapes", 2, "mesa")
        self.add_class("shapes", 3, "whaleback")

        for i in range(count):
            # 获取图片宽和高
 
            filestr = imglist[i].split(".")[0]
            #print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            #print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            #filestr = filestr.split("_")[1]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
            print(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
 
            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)
 
    # 重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
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
                # print "box"
                labels_form.append("long-ridge")
            elif labels[i].find("mesa")!=-1:
                #print "column"
                labels_form.append("mesa")
            elif labels[i].find("whaleback")!=-1:
                #print "package"
                labels_form.append("whaleback")
            
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)
 
 
# import train_tongue
# class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
 
config = InferenceConfig()
 
#model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
 
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG','long-ridge', 'mesa', 'whaleback']  # 修改

#只画出边界（visualizemod中的apply_mask中的alpha=0）
'''
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
        visualizemod.display_instances(count[i],image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
'''

#计算mAP（utils中的iou可以设置）

APs=[]
IoUs=[]
gt_instances = 0
pred_instances =0



#test_dataset_root_path="ex_data_onebyone/"
test_dataset_root_path="test_data_mod/"
test_img_floder = test_dataset_root_path + "pic_with_alpha_removed" 
#test_img_floder = test_dataset_root_path + "pic_ori" 
test_mask_floder = test_dataset_root_path + "cv2_mask"
test_imglist=os.listdir(test_img_floder)
test_count=len(test_imglist)

dataset_test = DrugDataset()
dataset_test.load_shapes(test_count, test_img_floder, test_mask_floder, test_imglist,test_dataset_root_path)
dataset_test.prepare()
image_ids = dataset_test.image_ids

gt_tot = np.array([])
pred_tot = np.array([])
mAP_ = []

for image_id in dataset_test.image_ids:
#image_id = random.choice(dataset.image_ids)
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_test, config, image_id, use_mini_mask=False)
    info = dataset_test.image_info[image_id]
    #print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                           #dataset.image_reference(image_id)))
    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    #ax = get_ax(1)
    r = results[0]
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                #dataset.class_names, r['scores'], ax=ax,
                                #title="Predictions")
    #log("gt_class_id", gt_class_id)
    #log("gt_bbox", gt_bbox)
    #log("gt_mask", gt_mask)
    gt, pred = utilsmod.gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
    gt_tot = np.append(gt_tot, gt)
    pred_tot = np.append(pred_tot, pred)
    #precision_, recall_, AP_ = utils.compute_precision_recall_map(gt_tot, pred_tot)
    AP_, precision_, recall_, overlap_ = utilsmod.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          r['rois'], r['class_ids'], r['scores'], r['masks'])
    print("the actual len of the gt vect is : ", len(gt_tot))
    print("the actual len of the pred vect is : ", len(pred_tot))
    mAP_.append(AP_)
    print("Average precision of this image")
    print("The actual mean average precision for the whole images (matterport methode) ", sum(mAP_)/len(mAP_))
    
gt_tot=gt_tot.astype(int)
pred_tot=pred_tot.astype(int)
#save the vectors of gt and pred
save_dir = "ourput"
gt_pred_tot_json = {"gt_tot" : gt_tot, "pred_tot" : pred_tot}
df = pd.DataFrame(gt_pred_tot_json)
if not os.path.exists(save_dir):
    os.makedir(save_dir)
    
df.to_json(os.path.join(save_dir,"gt_pred_test.json"))

#print the confusion matrix :

tp, fp, fn = utilsmod._test_data_class(dataset_test.class_names, gt_tot, pred_tot)

ap,mrec,mprec=utilsmod.voc_ap(tp, fp, fn)

print(ap)

#plot precision recall curve : 

utilsmod.plot_precision_recall(ap, mprec, mrec)











'''#计算mAP （range）
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_test, config,image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    
    

    #Compute IOU
    
    pred_masks = results[0]['masks']

    if pred_masks.shape[0] == 512:
            pred_mask_sum = np.sum(pred_masks, axis=-1)
            pred_mask_bool = pred_mask_sum.clip(max=1)

            # Flatten GT masks
            gt_mask_sum = np.sum(gt_mask, axis=-1)
            gt_mask_bool=gt_mask_sum.clip(max=1)

            # Calculate intersect and union:
            iou_mask = pred_mask_bool+gt_mask_bool

            intercept = (iou_mask == 2).sum()
            union = (iou_mask != 0).sum()

            iou = intercept/union
            pred_instances += pred_masks.shape[2]
    else:
            iou = 0
            print("iou else loop")
        
        #print("IoU = "+str(iou))
    IoUs.append(iou)
        #precisions_out.append(np.mean(precisions))
        #recalls_out.append(np.mean(recalls))
        #gt_instances += gt_mask.shape[2]
    print(IoUs)

    
    #Compute AP Range
    utilsmod.compute_ap_range(gt_bbox, gt_class_id, gt_mask,r["rois"], r["class_ids"], r["scores"], r['masks'])
    
    #Show difference
    #visualizemod.display_differences(image,gt_bbox, gt_class_id, gt_mask,r['rois'],r['class_ids'],r['scores'],r['masks'],class_names)
    
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utilsmod.compute_ap(gt_bbox, gt_class_id, gt_mask,r["rois"], r["class_ids"], r["scores"], r['masks'],iou_threshold=0.75)
    
    print(AP)
    APs.append(AP)
    
print("mAP: ", np.mean(APs))
'''






'''
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]#????
#image = skimage.io.imread("./images/ScrewYJ0090.jpg")  # 修改测试图片
image = skimage.io.imread("E:/package/code/Mask_RCNN/samples/yardang/test_data/pic/L1_o59new.tif")
 
a = datetime.now()
# Run detection
results = model.detect([image], verbose=1)
b = datetime.now()
# Visualize results
print("shijian", (b - a).seconds)
r = results[0]
visualize-mod.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
'''