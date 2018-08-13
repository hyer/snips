# coding=utf-8

'''
目标检测评测需要comp4_det_test_*.txt
'''
from __future__ import print_function

import pickle
import sys
sys.path.append("../")
import time

import torch
from torchvision import datasets, transforms
from torch.autograd import Variable

# import dataset
import random
import math

import dataset
from eval_util import voc_eval
from utils import *
from cfg import parse_cfg
from darknet import Darknet

import cv2

# Training settings
datacfg = "../cfg/k3.data"
cfgfile_1 = "../cfg/yolo-voc-ocr-eval.cfg"
weightfile = "../backup/k3_800x800/000625.weights"
namesfile = "../data/k3.names"
save_detected_image = "/home/nd/datasets/k3/k3-yolo2-eval-result"
neti_width = 800
neti_height = 800
save_detect = False
# Test parameters
det_conf_thresh = 0.25 # 注意，这个阈值会影响漏检率，也就是PR曲线的横轴最多可以到1.0，但是如果阈值太大，
# 则PR曲线的recall可能只能到某个小于1.0的值，甚至是0.2之类的。但是如果始终检测不出来，怎么调阈值都无法让recall到1.0附近
nms_thresh = 0.4
iou_thresh = 0.5

eval_out_dir = "/home/nd/datasets/k3/k3-yolo2-eval-result"  # save det txt files.
if os.path.exists(eval_out_dir):
    shutil.rmtree(eval_out_dir)
    os.mkdir(eval_out_dir)
# output_dir = get_output_dir(eval_out_dir)
# det_file = os.path.join(eval_out_dir, 'detections.pkl')


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
ds = dataset.listDataset(testlist, shape=(init_width, init_height),
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]), train=False)

test_loader = torch.utils.data.DataLoader(ds,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          **kwargs)

if use_cuda:
    model = torch.nn.DataParallel(model).cuda()


def test():
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    model.eval()
    num_classes = model.module.num_classes
    num_samples = ds.nSamples
    anchors = model.module.anchors
    num_anchors = model.module.num_anchors
    total = 0.0
    proposals = 0.0
    correct = 0.0
    null_count = 0

    all_boxes = [[[] for _ in range(num_samples)]  # 21x4952: class_num x test_sample_num
             for _ in range(num_classes + 1)]

    test_sample_idx = 0

    for batch_idx, (image_paths, data, target) in enumerate(test_loader):
        # if test_sample_idx > 10:break
        logging("test batch_idx = %d" % batch_idx)

        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data).data
        det_boxes = get_region_boxes(output, det_conf_thresh, num_classes, anchors, num_anchors)
        for out_idx in range(output.size(0)):
            boxes = det_boxes[out_idx]  # 检测出的所有boxes
            boxes = nms(boxes, nms_thresh)  # 进行NMS后的所有boxes

            for i in range(len(boxes)):
                box = boxes[i]
                x1 = (box[0] - box[2]/2.0) * neti_width
                y1 = (box[1] - box[3]/2.0) * neti_height
                x2 = (box[0] + box[2]/2.0) * neti_width
                y2 = (box[1] + box[3]/2.0) * neti_height

                if len(box) >= 7:
                    cls_conf = box[5]
                    cls_id = box[6]
                    # a = [x1, y1, x2, y2, cls_conf]
                    with open(os.path.join(eval_out_dir, "comp4_det_test_"+str(cls_id) + ".txt"), "a") as f:
                        img_name = image_paths[out_idx].split("/")[-1].split(".")[0]
                        f.write("%s %.6f %.6f %.6f %.6f %.6f\n"%(img_name, cls_conf, x1, y1, x2, y2))

                    # if all_boxes[cls_id][test_sample_idx] == []:
                    #     all_boxes[cls_id][test_sample_idx].append(a)
                    # else:
                    #     all_boxes[cls_id][test_sample_idx] = np.vstack((all_boxes[cls_id][test_sample_idx], a))


            # save detected images
            if save_detect:
                class_names = load_class_names(namesfile)
                save_name = save_detected_image + "/" + image_paths[out_idx].split("/")[-1]
                plot_boxes(Image.open(image_paths[out_idx]).convert('RGB'), boxes, savename=save_name, class_names=class_names)

            test_sample_idx += 1

    # logging("Writing to pkl file.")
    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    logging("test done.")
    # voc_eval("/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/eval_dir/txts/comp4_det_test_141.txt",
    #          "/home/hyer/datasets/OCR/SSD_OCR/OCR_yolo2_no_frac_minus/Annotations/%s",
    #          "/home/hyer/datasets/OCR/SSD_OCR/OCR_yolo2_no_frac_minus/ImageSets/Main/test.txt",
    #          "1",
    #          "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/eval_dir/cache_dir")



if __name__ == '__main__':
    model.module.load_weights(weightfile)
    logging('evaluating ... %s' % (weightfile))
    logging("batch size: %d" % batch_size)
    test()
