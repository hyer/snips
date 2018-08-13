import matplotlib.pyplot as plt
import uuid
import numpy as np
import time
import torch

import tornado.ioloop
import tornado.web
import tornado.httpserver
from torch.autograd import Variable
from tornado.options import define, options
import json


from PIL import Image, ImageDraw
from colorama import Fore, Back, Style

import cv2

from data import config
from ssd import build_ssd
from utils.dir_opt import Dir
from utils.tools import load_class_names

label_file = "/home/hyer/workspace/algo/Detection/SSD/ssd.pytorch/data/k1.txt"
weightfile = '/home/hyer/workspace/algo/Detection/SSD/ssd.pytorch/models/mobv1-ssd-k1/add_0/VOC_mobile_300_19105321_40000.pth'
class_names = load_class_names(label_file)

# from models import build_ssd as build_ssd_v1 # uncomment for older pool6 mode
version = config.VOC_mobile_300_19105321
conf_thres = 0.5
nms_thresh = 0.1
num_cls = 13 # include background
use_cuda = True

# cudnn.benchmark = True # maybe faster
net = build_ssd('test', version, size=300, num_classes=num_cls, bkg_label=0, top_k=100, conf_thresh=conf_thres, nms_thresh=nms_thresh)  # initialize SSD
net.load_weights(weightfile)
#
# torch.save(net.half().state_dict(), "test.pth")
for param in net.parameters():
    param.requires_grad = False

print(Fore.RED + "Please runing with python3.6!."),
print(Style.RESET_ALL)

if use_cuda:
    print(Fore.RED + "Using GPU."),
    print(Style.RESET_ALL)
    net = net.cuda()
else:
    print(Fore.GREEN + "Using CPU."),
    print(Style.RESET_ALL)

net.eval()  # important!!!

namesfile = label_file
print('Loading weights from %s... Done!' % (weightfile))

# image = cv2.imread('./data/example.jpg', cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
# %matplotlib inline
from matplotlib import pyplot as plt
from data import VOCDetection, VOCroot, AnnotationTransform

# here we specify year (07 or 12) and dataset ('test', 'val', 'train')
# testset = VOCDetection(VOCroot, [('2007', 'val')], None, AnnotationTransform())
# img_id = 60


dirop = Dir()

img_paths = dirop.getPaths("/home/hyer/Desktop/test_img/tts", ".jpg")
save_root = "/home/hyer/datasets/OCR/result_k1-ssd"

for img_path in img_paths:
# while True:
#     img_path = input("Img path: ")
    # img_id = input("img id: ")
    # img_path = "/home/hyer/datasets/chinese/voc_data_200/JPEGImages/" + img_id + ".jpg"
    print("img_path: ", img_path)
    img_name = img_path.split("/")[-1]
    image = cv2.imread(img_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # View the sampled input image before transform

    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (128.0, 128.0, 128.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    # plt.imshow(x)
    x = torch.from_numpy(x).permute(2, 0, 1)

    # SSD forward
    xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
    if use_cuda:
        xx = xx.cuda()

    # _t['im_detect'].tic()
    detections = net(xx).data
    # detect_time = _t['im_detect'].toc(average=False)
    # print("Forward time: ", detect_time)

    # from data import VOC_CLASSES as labels

    plt.figure()
    colors = plt.cm.hsv(np.linspace(0, 1, num_cls)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    print(rgb_image.shape[1::-1], rgb_image.shape[1::-1])

    # scale each detection back up to the image
    scale = torch.Tensor([rgb_image.shape[1::-1], rgb_image.shape[1::-1]])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= conf_thres:
            score = detections[0, i, j, 0]
            label_name = class_names[i - 1]
            display_txt = '%s: %.2f' % (label_name, score)
            print(display_txt)

            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()  # detections的第四维是5个数，表示[cls_conf, x1, y1, x2, y2]
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
            j += 1

    # plt.figure(figsize=(10, 10))
    # plt.imshow(rgb_image)
    # plt.show()
    plt.savefig(save_root + "/" + img_name)
