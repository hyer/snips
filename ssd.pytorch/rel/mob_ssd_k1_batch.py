#coding=utf-8
import os
import time
import numpy as np
import torch

import cv2

from colorama import Fore, Back, Style
from torch.autograd import Variable

from data import config, base_transform
from layers.box_utils import jaccard
from ssd import build_ssd


class SSD_mob_k1_batch():
    def __init__(self, version, weightfile, label_file, img_size=300, conf_thres=0.2, nms_thres=0.25, num_cls=13, top_k=100, use_cuda=True):
        # from models import build_ssd as build_ssd_v1 # uncomment for older pool6 mode
        version = version
        conf_thres = conf_thres
        nms_thresh = nms_thres
        self.num_cls = num_cls  # include background
        self.top_k = top_k

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


    def detect(self, img_cv2_list, batch_size=1, conf_thres=0.25, top_k=200):
        boxes_all = []
        for bs_idx in range(int(np.ceil(len(img_cv2_list) / batch_size))):
            xs = []
            # img_paths = img_list[batch_size * bs_idx: batch_size * (bs_idx + 1)]
            # for img_path in img_paths:
            #     img_cv2 = cv2.imread(img_path)
            #
            #     img_transformed = base_transform(img_cv2, 300, (128.0, 128.0, 128.0))
            #     img_transformed = img_transformed[:, :, (2, 1, 0)]
            #     img_tensor = torch.from_numpy(img_transformed).permute(2, 0, 1)
            #
            #     xs.append(img_tensor)
            #     continue
            imgs = img_cv2_list[batch_size * bs_idx: batch_size * (bs_idx + 1)]
            for img in imgs:
                img_cv2 = img

                img_transformed = base_transform(img_cv2, 300, (128.0, 128.0, 128.0))
                img_transformed = img_transformed[:, :, (2, 1, 0)]
                img_tensor = torch.from_numpy(img_transformed).permute(2, 0, 1)

                xs.append(img_tensor)
                continue

            # x = Variable(im.unsqueeze(0))
            images_tensor = torch.stack(xs, 0)
            images = Variable(images_tensor.cuda())

            t1 = time.time()
            detections = self.net(images).data
            t2 = time.time()
            box_num = self.num_cls * self.top_k * 5
            # detect_time = _t['im_detect'].toc(average=False)
            boxes_batch = []

            for bs_idx in range(len(imgs)):
                boxes = []
                # print("len img_paths: ", len(img_paths))
                # print("detections detections: ", len(detections))
                # if os.path.exists(imgs[bs_idx]):
                img_cv2 = imgs[bs_idx]
                rgb_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

                # print(rgb_image.shape[1::-1], rgb_image.shape[1::-1])

                scale = torch.Tensor([rgb_image.shape[1::-1], rgb_image.shape[1::-1]])

                # dets = detections[bs_idx*box_num: (bs_idx+1)*box_num][bs_idx].view(1, num_classes, 200, 5)

                dets = detections[bs_idx].view(1, self.num_cls, top_k, 5)
                for i in range(dets.size(1)):
                    j = 0
                    while dets[0, i, j, 0] >= conf_thres:
                        score = dets[0, i, j, 0]
                        label_name = self.class_names[i - 1]
                        display_txt = '%s: %.2f' % (label_name, score)
                        # print(display_txt)

                        pt = (dets[0, i, j,
                              1:] * scale).cpu().numpy()  # detections的第四维是5个数，表示[cls_conf, x1, y1, x2, y2]
                        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                        j += 1
                        # print("i, j: ", i, j)
                        boxes.append((label_name, score, pt[0], pt[1], pt[2], pt[3]))

                boxes_all.append(boxes)
            print("batch forward time: ", t2 - t1, "fusion time: ", time.time() - t2, "total time: ", time.time() - t1)
        return boxes_all

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


def main():
    label_file = "./data/k1.txt"
    weightfile = './models/VOC_mobile_300_19105321-add0.pth'

    # from models import build_ssd as build_ssd_v1 # uncomment for older pool6 mode
    version = config.VOC_mobile_300_19105321
    conf_thres = 0.5
    nms_thresh = 0.25
    num_cls = 13  # include background
    input_size = 300
    ovthresh = 0.4
    top_k = 100
    use_cuda = True

    m = SSD_mob_k1_batch(version, weightfile, label_file, img_size=300, conf_thres=conf_thres, nms_thres=nms_thresh,
                         num_cls=num_cls, use_cuda=True)

    img_list_file = "./data/ssd_k1_test.txt"
    with open(img_list_file, "r") as f:
        data = f.readlines()

    img_paths = []
    for li in data:
        img_paths.append(li.strip())

    img_cv2_list = []
    for i in range(len(img_paths)):
        img_cv2_list.append(cv2.imread(img_paths[i]))

    t = time.time()
    dets = m.detect(img_cv2_list, 4, top_k=top_k)
    print("all exec time: ", time.time() - t)
    # dicard_boxes, dicard_boxes_idx = filter_inter_IOU(boxes, labels)
    # final_boxes = []
    # final_labels = []
    # for i, box in enumerate(boxes):
    #     if i not in dicard_boxes_idx:
    #         final_boxes.append(box)
    #         final_labels.append(labels[i])
    #
    # print(boxes, labels)
    print("========= final boxes ========")
    # print(final_boxes, final_labels)
    print("ok")


if __name__ == '__main__':
    main()