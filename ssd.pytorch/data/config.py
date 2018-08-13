#coding=utf-8
# config.py
import logging
import os.path

# gets home dir cross platform
# home = os.path.expanduser("~")
ddir = os.path.join("/home/hyer/datasets/OCR/SSD_k1")
# ddir = "/home/nd/datasets/ssd_k1/k1_scan"
# class_name_file = "./data/k1.txt"  # k1.txt when run eval.py no background, 注意xml里面的name字段是否都在里面
class_name_file = "./data/k1.names"  # k1.txt when run eval.py no background, 注意xml里面的name字段是否都在里面

# ddir = os.path.join("/home/hyer/datasets/OCR/SSD_OCR/k6")
# ddir = os.path.join("/home/hyer/datasets/chinese/voc_data_200")
# ddir = os.path.join("/home/hyer/datasets/VOC/VOCdevkit")

# note: if you used our download scripts, this should be right
VOCroot = ddir # path to VOCdevkit root dir

# cls_num = 82 # no background
#
class_names = []
# for i in range(cls_num):
    # class_names.append(str(i))
with open(class_name_file, "r") as f:
    data = f.readlines()
    for li in data:
        name = li.strip()
        class_names.append(name)

class_names = tuple(class_names)
logging.info("Class names: " + str(class_names))
# class_names = (  # always index 0
#     'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor')



# default batch size
BATCHES = 32
# data reshuffled at every epoch
SHUFFLE = True
# number of subprocesses to use for data loading
WORKERS = 4


#SSD300 CONFIGS
# newer version: use additional conv11_2 layer as last layer before multibox layers
v2 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1], # feature map size
    'min_dim' : 300,
    'steps' : [8, 16, 32, 64, 100, 300],
    'min_sizes' : [30, 60, 111, 162, 213, 264],
    'max_sizes' : [60, 111, 162, 213, 264, 315],
    # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
    #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'name' : 'v2',
}

# use average pooling layer as last layer before multibox layers
v1 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],
    'min_dim' : 300,
    'steps' : [8, 16, 32, 64, 100, 300],
    'min_sizes' : [30, 60, 114, 168, 222, 276],
    'max_sizes' : [-1, 114, 168, 222, 276, 330],
    # 'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'aspect_ratios' : [[1,1,2,1/2],[1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3],
                        [1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3]],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'name' : 'v1',
}


COCO_mobile_300 = {
    'feature_maps' : [19, 10, 5, 3, 2, 1],
    'min_dim' : 300,
    'steps' : [16, 32, 64, 100, 150, 300],
    'min_sizes' : [45, 90, 135, 180, 225, 270],
    'max_sizes' : [90, 135, 180, 225, 270, 315],
    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'name' : 'COCO_mobile_300',
}

VOC_mobile_300 = {  # should modified the SSD base network forward only to (conv12 dw) or change conv12 dw stride=1
    'feature_maps' : [38, 19, 10, 5, 3, 1],   # feature map size, -> 8732 default box
    'min_dim' : 300,
    'steps' : [8, 16, 32, 64, 100, 300],
    'min_sizes' : [30, 105, 150, 195, 240, 284],
    'max_sizes' : [60, 150, 195, 240, 284, 300],
    # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
    #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'name' : 'VOC_mobile_300',
}

VOC_mobile_300_all = {
    'feature_maps' : [38, 10, 5, 3, 2, 1],  # 6600 default box
    'min_dim' : 300,
    'steps' : [8, 16, 32, 64, 100, 300],
    'min_sizes' : [30, 105, 150, 195, 240, 284],
    'max_sizes' : [60, 150, 195, 240, 284, 300],
    # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
    #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'name' : 'VOC_mobile_300_all',
}

VOC_mobile_300_19105321 = {
    'feature_maps' : [19, 10, 5, 3, 2, 1],   #  2268
    'min_dim' : 300,
    'steps' : [16, 32, 64, 100, 150, 300],
    'min_sizes' : [45, 90, 135, 180, 225, 270],
    'max_sizes' : [90, 135, 180, 225, 270, 315],
    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'name' : 'VOC_mobile_300_19105321',
}