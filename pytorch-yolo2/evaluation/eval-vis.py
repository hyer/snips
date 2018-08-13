# coding=utf-8

'''
统计precision，recall，并保存测试图片。
'''
from __future__ import print_function
import sys

import time

import torch
from torchvision import datasets, transforms
from torch.autograd import Variable

import dataset
import random
import math
from utils import *
from cfg import parse_cfg
from darknet import Darknet

import cv2

# Training settings
datacfg = "cfg/voc_ocr.data"
cfgfile_1 = "cfg/yolo-voc-ocr-eval.cfg"
weightfile = "backup/800x800/000060.weights"
namesfile = "/home/hyer/workspace/algo/OCR/Math_Exp/OCR_Tools/charmap.txt"
save_detected_image = "/home/hyer/datasets/OCR/SSD_OCR/yolo2-test-result-800x800"
save_detect = False

data_options = read_data_cfg(datacfg)
net_options = parse_cfg(cfgfile_1)[0]

trainlist = data_options['train']
testlist = data_options['valid']
gpus = data_options['gpus']  # e.g. 0,1,2,3
num_workers = int(data_options['num_workers'])

batch_size = int(net_options['batch'])

# Train parameters
use_cuda = True
seed = 22222
eps = 1e-5

# Test parameters
conf_thresh = 0.25
nms_thresh = 0.4
iou_thresh = 0.5

###############
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

model = Darknet(cfgfile_1)
model.print_network()

init_width = model.width
init_height = model.height

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(testlist, shape=(init_width, init_height),
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]), train=False),
        batch_size=batch_size, shuffle=False, **kwargs)

if use_cuda:
    model = torch.nn.DataParallel(model).cuda()


def test():
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    model.eval()
    num_classes = model.module.num_classes
    anchors = model.module.anchors
    num_anchors = model.module.num_anchors
    total = 0.0
    proposals = 0.0
    correct = 0.0
    null_count = 0

    for batch_idx, (image_paths, data, target) in enumerate(test_loader):
        logging("test batch_idx = %d" % batch_idx)
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data).data
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        for out_idx in range(output.size(0)):
            boxes = all_boxes[out_idx]  # 检测出的所有boxes
            boxes = nms(boxes, nms_thresh)  # 进行NMS后的所有boxes
            if len(boxes) == 0:
                null_count += 1

            truths = target[out_idx].view(-1, 5)
            num_gts = truths_length(truths)

            total = total + num_gts

            for j in range(len(boxes)):
                if boxes[j][4] > conf_thresh:
                    proposals = proposals + 1

            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                #this_detected = False
                for j in range(len(boxes)):  # 预测的每个bbox都要和gt的每个bbox进行比对
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    best_iou = max(iou, best_iou)
                    if best_iou > iou_thresh and boxes[j][6] == box_gt[6]:  # 标签正确并且预测的最大iou大于设置的阈值
                        correct = correct + 1
                        #this_detected = True
                        break
                #if this_detected == True:
                #    correct += 1
                        # print(i, correct)
            # save detected images
            if save_detect:
                class_names = load_class_names(namesfile)
                save_name = save_detected_image + "/" + image_paths[out_idx].split("/")[-1]
                plot_boxes(Image.open(image_paths[out_idx]).convert('RGB'), boxes, savename=save_name, class_names=class_names)

    print("correct/proposals/total: %f / %f / %f" %(correct, proposals, total))

    precision = 1.0 * correct / (proposals + eps)
    recall = 1.0 * correct / (total + eps)
    fscore = 2.0 * precision * recall / (precision + recall + eps)
    logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))
    logging("null_count: %d" % null_count)


if __name__ == '__main__':
    model.module.load_weights(weightfile)
    logging('evaluating ... %s' % (weightfile))
    logging("batch size: %d" % batch_size)
    test()
