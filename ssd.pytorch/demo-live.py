#coding=utf-8

import os
import sys

import time

# from eval import Timer
from PIL import ImageDraw, ImageFont, Image

from eval import Timer
import copy

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.serialization import load_lua
import numpy as np
import cv2
from data import config

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def uni2chi(x):
    output = str("\\"+x).decode("unicode_escape")
    return output

from ssd import build_ssd

# from models import build_ssd as build_ssd_v1 # uncomment for older pool6 mode
version = config.VOC_mobile_300_19105321
conf_thres = 0.9
use_cuda = True
num_cls = 201

font = ImageFont.truetype(
            font='/home/hyer/workspace/algo/Detection/SSD/ssd.pytorch/data/simheittf.ttf',
            # size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
            size=np.floor(30).astype('int32'),
            encoding='utf-8')

class_name_file = "/home/hyer/datasets/chinese/voc_data_200/code_200.txt"
#
class_names = []
with open(class_name_file, "r") as f:
    label_data = f.readlines()
    for li in label_data:
        name = li.strip()
        class_names.append(uni2chi(name))

# cudnn.benchmark = True # maybe faster
net = build_ssd('test', version, size=300, num_classes=num_cls, bkg_label=0, top_k=100, conf_thresh=0.1, nms_thresh=0.45)  # initialize SSD
net.load_weights('/home/hyer/workspace/algo/Detection/SSD/ssd.pytorch/models/VOC_mobile_300_19105321_5000-cls201.pth')
net.eval()  # important!!!
if use_cuda:
    net = net.cuda()

from matplotlib import pyplot as plt

_t = {'im_detect': Timer(), 'misc': Timer()}

# -*- coding: UTF-8 -*-
import cv2

capture = cv2.VideoCapture(0)
if (capture.isOpened() == False):
    print("[ERROR] Failed to Open Camera!")

rect_x_start = 300
rect_y_start = 240
rect_w = 114
rect_h = 114

def get_color(c, x, max_val):
    ratio = float(x) / max_val * 5
    i = int(np.math.floor(ratio))
    j = int(np.math.ceil(ratio))
    ratio = ratio - i
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
    return int(r * 255)

while True:
    ret, img = capture.read()
    rectImg = copy.copy(img[rect_y_start: rect_y_start + rect_w, rect_x_start:rect_x_start + rect_w])

    cv2.rectangle(img, (rect_x_start, rect_y_start), (rect_x_start + rect_w, rect_y_start+rect_h), (200, 200,0), 2, lineType=16)

    # video.write(img)
    # cv2.imshow('Video', img)
    # cv2.imshow('patch', rectImg)


    rgb_image = cv2.cvtColor(rectImg, cv2.COLOR_BGR2RGB)
    # View the sampled input image before transform

    x = cv2.resize(rgb_image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    # plt.imshow(x)
    x = torch.from_numpy(x).permute(2, 0, 1)

    # SSD forward
    xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
    if use_cuda:
        xx = xx.cuda()

    _t['im_detect'].tic()
    detections = net(xx).data
    detect_time = _t['im_detect'].toc(average=False)
    # print("Forward time: ", detect_time)

    from data import VOC_CLASSES as labels

    # plt.figure(figsize=(2, 2))
    colors = plt.cm.hsv(np.linspace(0, 1, num_cls)).tolist()
    # plt.imshow(rgb_image)  # plot the image for matplotlib
    # currentAxis = plt.gca()
    ret_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(ret_image)

    # scale each detection back up to the image
    scale = torch.Tensor([rgb_image.shape[1::-1], rgb_image.shape[1::-1]])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= conf_thres:
            score = detections[0, i, j, 0]
            label_name = class_names[i - 1]

            # label_name = labels[i - 1]
            display_txt = label_name + ': %.2f' % (score)
            print(label_name)

            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            print(pt)
            # color = colors[i]

            x1 = int(round((pt[0] - pt[2] / 2.0)))
            y1 = int(round((pt[1] - pt[3] / 2.0)))
            x2 = int(round((pt[0] + pt[2] / 2.0)))
            y2 = int(round((pt[1] + pt[3] / 2.0)))

            red = int(round(colors[i][0]*255))
            green = int(round(colors[i][1]*255))
            blue = int(round(colors[i][2]*255))

            rgb = (red, green, blue)
            print(rgb)

            draw = ImageDraw.Draw(pil_img)
            draw.rectangle([(rect_x_start + x1, rect_y_start + y1), (rect_x_start +x2, rect_y_start + y2)])
            draw.text((rect_x_start + x1, rect_y_start + y1-25), label_name, font=font)
            # cv2.rectangle(img,  (rect_x_start + x1, rect_y_start + y1), (rect_x_start +x2, rect_y_start + y2), rgb, 2)
            # cv2.putText(img, display_txt, (rect_x_start + x1, rect_y_start + y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)
            j += 1
            # ret_image.save('ret.jpg')
            # break
    im = np.array(pil_img)

    cv2.imshow('det result', im)
    cv2.waitKey(1)

    # plt.show()

capture.release()

cv2.destroyAllWindows()