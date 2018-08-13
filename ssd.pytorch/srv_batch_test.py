"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""
from utils.tools import load_class_names

print("######################")
print()
print("MUST running with python3.6 !!!!!!!!!")
print()
print("######################")

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot, config, logging, base_transform
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


use_cuda = True
if use_cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# YEAR = '2007'
devkit_path = VOCroot
dataset_mean = (128.0, 128.0, 128.0)
set_type = 'test'

num_classes = len(VOC_CLASSES) + 1  # +1 background
conf_thres = 0.5
nms_thresh = 0.25
input_size = 300
ovthresh = 0.4
top_k = 100
label_file = "/home/hyer/workspace/algo/Detection/SSD/ssd.pytorch/data/k1.txt"
# weightfile = '/home/hyer/workspace/algo/Detection/SSD/ssd.pytorch/models/mobv1-ssd-k1/T16184/VOC_mobile_300_19105321_10000.pth'
class_names = load_class_names(label_file)
version = config.VOC_mobile_300_19105321
trained_model = '/home/hyer/workspace/algo/Detection/SSD/ssd.pytorch/models/mobv1-ssd-k1/add_0/VOC_mobile_300_19105321_40000.pth'
net = build_ssd('test', version, size=300, num_classes=num_classes, bkg_label=0, top_k=top_k, conf_thresh=conf_thres,
                nms_thresh=nms_thresh)  # initialize SSD
net.load_weights(trained_model)
net.eval()
print('==> Finished loading model!')

# # load data
# print("==> load data.")
# annopath = os.path.join(VOCroot, 'Annotations', '%s.xml')
# imgpath = os.path.join(VOCroot, 'JPEGImages', '%s.jpg')
# imgsetpath = os.path.join(VOCroot, 'ImageSets', 'Main', '{:s}.txt')
# eval_result_txt = os.path.join(devkit_path, "eval", 'det_txts')
# cachedir = os.path.join(devkit_path, "eval", 'annotations_cache')
# save_folder = os.path.join(VOCroot, "eval")

#
# logging.info("==> save eval result txt of each class to the directory of: ", eval_result_txt)
# if not os.path.exists(eval_result_txt):
#     os.makedirs(eval_result_txt)
#
# if not os.path.exists(cachedir):
#     os.makedirs(cachedir)
# logging.info("==> save annotation cache file to the directory of: ", cachedir)
#
# output_dir = get_output_dir(save_folder, "PR_curves")
# logging.info("==> save detections.pkl to the directory of: ", output_dir)

# comp_id = "comp4_"
transform = BaseTransform(input_size, dataset_mean)
# dataset = VOCDetection(VOCroot, [(set_type)], transform, AnnotationTransform())
if use_cuda:
    net = net.cuda()


def main(net):
    logging.info("==> do detect on every image with model reference.")

    img_list_file = "/home/hyer/datasets/OCR/ssd_k1_test.txt"
    with open(img_list_file, "r") as f:
        data = f.readlines()

    img_list = []
    for li in data:
        img_list.append(li.strip())


    batch_size = 4
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    for bs_idx in range(int(np.ceil(len(img_list)/batch_size))):
        xs = []
        # im, gt, h, w = dataset.pull_item(i)
        # img_cv2 = cv2.imread("/home/hyer/datasets/OCR/scan_k1/b7f44061-6d4c-11e8-ad39-480fcf43d407.jpg")
        img_paths = img_list[batch_size*bs_idx : batch_size*(bs_idx+1)]
        for img_path in img_paths:
            # img_cv2 = cv2.imread(input("imgPath: "))
            img_cv2 = cv2.imread(img_path)

            img_transformed = base_transform(img_cv2, 300, dataset_mean)
            img_transformed = img_transformed[:, :, (2, 1, 0)]
            img_tensor = torch.from_numpy(img_transformed).permute(2, 0, 1)

            xs.append(img_tensor)
            continue

        # x = Variable(im.unsqueeze(0))
        images_tensor = torch.stack(xs, 0)
        images = Variable(images_tensor.cuda())

        t1 = time.time()
        detections = net(images).data
        t2 = time.time()
        box_num = num_classes*top_k*5
        # detect_time = _t['im_detect'].toc(average=False)
        boxes_batch = []

        for bs_idx in range(batch_size):
            boxes = []
            currentAxis = plt.gca()

            # image = cv2.imread("/home/hyer/Pictures/2334.jpg")
            img_cv2 = cv2.imread(img_paths[bs_idx])
            rgb_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            # plt.figure("result")
            colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
            plt.imshow(rgb_image)  # plot the image for matplotlib
            currentAxis = plt.gca()

            print(rgb_image.shape[1::-1], rgb_image.shape[1::-1])

            scale = torch.Tensor([rgb_image.shape[1::-1], rgb_image.shape[1::-1]])

            # dets = detections[bs_idx*box_num: (bs_idx+1)*box_num][bs_idx].view(1, num_classes, 200, 5)
            draw = False
            dets = detections[bs_idx].view(1, num_classes, top_k, 5)
            for i in range(dets.size(1)):
                j = 0
                while dets[0, i, j, 0] >= conf_thres:
                    score = dets[0, i, j, 0]
                    label_name = class_names[i - 1]
                    display_txt = '%s: %.2f' % (label_name, score)
                    print(display_txt)

                    pt = (dets[0, i, j, 1:] * scale).cpu().numpy()  # detections的第四维是5个数，表示[cls_conf, x1, y1, x2, y2]
                    coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                    color = colors[i]
                    if draw:
                        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
                    j += 1
                    # print("i, j: ", i, j)
                    boxes.append((label_name, score, pt[0], pt[1], pt[2], pt[3]))

            boxes_batch.append(boxes)
            # plt.figure(figsize=(10, 10))
            # plt.imshow(rgb_image)
            if draw:
                plt.show()
                plt.savefig("./test.jpg")
        print("batch forward time: ", t2 - t1, "fusion time: ", time.time() - t2, "total time: ", time.time() - t1)


if __name__ == '__main__':
    # cudnn.benchmark = True
    # evaluation
    main(net)
