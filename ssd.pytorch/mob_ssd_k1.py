#coding=utf-8

import time
import numpy as np
import torch

import cv2

from colorama import Fore, Back, Style
from torch.autograd import Variable

from data import config
from layers.box_utils import jaccard
from ssd import build_ssd


class SSD_mob_k1():
    def __init__(self, version, weightfile, label_file, img_size=300, conf_thres=0.2, nms_thres=0.25, num_cls=13, use_cuda=True):
        # from models import build_ssd as build_ssd_v1 # uncomment for older pool6 mode
        version = version
        conf_thres = conf_thres
        nms_thresh = nms_thres
        num_cls = num_cls  # include background

        self.use_cuda = use_cuda

        # cudnn.benchmark = True # maybe faster
        self.net = build_ssd('test', version, size=img_size, num_classes=num_cls, bkg_label=0, top_k=100, conf_thresh=conf_thres,
                        nms_thresh=nms_thresh)  # initialize SSD
        self.net.load_weights(weightfile)
        #
        self.class_names = self.load_class_names(label_file)

        # torch.save(self.net.half().state_dict(), "test.pth")
        for param in self.net.parameters():
            param.requires_grad = False

        print(Fore.RED + "Please runing with python3.6!."),
        print(Style.RESET_ALL)

        if use_cuda:
            print(Fore.RED + "Using GPU."),
            print(Style.RESET_ALL)
            self.net = self.net.cuda()
        else:
            print(Fore.GREEN + "Using CPU."),
            print(Style.RESET_ALL)

        self.net.eval()  # important!!!

        namesfile = label_file
        print('Loading weights from %s... Done!' % (weightfile))

    def load_class_names(self, namesfile):
        class_names = []
        with open(namesfile, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        return class_names


    def detect(self, imgfile):
        image = cv2.imread(imgfile)
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (128.0, 128.0, 128.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)

        # SSD forward
        xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
        if self.use_cuda:
            xx = xx.cuda()

        t1 = time.time()
        detections = self.net(xx).data
        print("exec time: ", time.time() - t1)

        boxes = []
        labels = []
        scores = []
        # scale = torch.Tensor([rgb_image.shape[1::-1], rgb_image.shape[1::-1]])
        scale = torch.Tensor([image.shape[1::-1], image.shape[1::-1]])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= conf_thres:
                score = detections[0, i, j, 0]
                label_name = self.class_names[i - 1]
                display_txt = '%s: %.2f' % (label_name, score)
                # print(display_txt)

                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()  # detections的第四维是5个数，表示[cls_conf, x1, y1, x2, y2]
                coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                boxes.append([pt[0], pt[1], pt[2], pt[3]])
                labels.append(label_name)
                scores.append(score)
                # color = colors[i]
                # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
                j += 1

        return boxes, labels, scores


    def sort_bbox(self):
        pass

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def filter_inter_IOU(boxes, labels, scores):
    '''
    discard boxes that have IOU > other box
    :param boxes:
    :param labels:
    :param scores:
    :return:
    '''
    dicard_boxes = []
    dicard_boxes_idx = []
    for i, box in enumerate(boxes):
        for j, box_o in enumerate(boxes):
            if (bbox_iou(box, box_o) > 0.8 and scores[i] < scores[j]):
                dicard_boxes.append((box, labels[i], scores[i]))
                dicard_boxes_idx.append(i)
    return dicard_boxes, dicard_boxes_idx


if __name__ == '__main__':
    label_file = "/home/hyer/workspace/algo/Detection/SSD/ssd.pytorch/data/k1.txt"
    weightfile = '/home/hyer/workspace/algo/Detection/SSD/ssd.pytorch/models/mobv1-ssd-k1/T16184/VOC_mobile_300_19105321_110000.pth'

    # from models import build_ssd as build_ssd_v1 # uncomment for older pool6 mode
    version = config.VOC_mobile_300_19105321
    conf_thres = 0.2
    nms_thresh = 0.25
    num_cls = 13  # include background
    use_cuda = True

    m = SSD_mob_k1(version, weightfile, label_file, img_size=300, conf_thres=conf_thres, nms_thres=nms_thresh, num_cls=num_cls, use_cuda=True)

    img_path = '/home/hyer/datasets/OCR/scan_k1/roi_13.jpg'

    boxes, labels, scores = m.detect(img_path)

    dicard_boxes, dicard_boxes_idx = filter_inter_IOU(boxes, labels, scores)
    final_boxes = []
    final_labels = []
    for i, box in enumerate(boxes):
        if i not in dicard_boxes_idx:
            final_boxes.append(box)
            final_labels.append(labels[i])

    print(boxes, labels)
    print("========= final boxes ========")
    print(final_boxes, final_labels)
    print("ok")