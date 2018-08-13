# coding=utf-8

from __future__ import print_function
import sys
# if len(sys.argv) != 4:
#     print('Usage:')
#     print('python train.py datacfg cfgfile weightfile')
#     exit()

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
# from torch.autograd import Variable

import dataset
# import random
# import math
from YOLO2_M import MobileNet, YOLO2_M
from utils import *
from cfg import parse_cfg
# from region_loss import RegionLoss
from darknet import Darknet
# from models.tiny_yolo import TinyYoloNet
from colorama import Fore, Back, Style

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
datacfg = "cfg/k6.data"
cfgfile = "./cfg/mobilenet_yolo_voc.cfg"
weightfile = "models/mobilenet_feature.pth"
trained_weightfile = "./backup/k6_mob-416x800/000046.weights"

data_options = read_data_cfg(datacfg)
net_options = parse_cfg(cfgfile)[0]

trainlist = data_options['train']
testlist = data_options['valid']
backupdir = data_options['backup']
nsamples = file_lines(trainlist)
gpus = data_options['gpus']  # e.g. 0,1,2,3
ngpus = len(gpus.split(','))
num_workers = int(data_options['num_workers'])

batch_size = int(net_options['batch'])
max_batches = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum = float(net_options['momentum'])
decay = float(net_options['decay'])
steps = [float(step) for step in net_options['steps'].split(',')]
scales = [float(scale) for scale in net_options['scales'].split(',')]
show_inter = int(net_options['show_inter'])

# Train parameters
max_epochs = 1000  # 手动设置
use_cuda = True
seed = int(time.time())
eps = 1e-5
save_interval = 2  # epoches

# Test parameters
conf_thresh = 0.25
nms_thresh = 0.4
iou_thresh = 0.5
resume_train = False  # 是否从之前的模型恢复训练（保存的weight文件里面会记录之前的参数和epoch数等信息）

use_visdom = True
visdom_env = "YOLO2-Mobilenet-k6"

cudnn.benchmark = True

###############
import visdom

if use_visdom:
    viz = visdom.Visdom(env=visdom_env)
print("Total train samples = ", nsamples)

torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    print(Fore.YELLOW + "Visible GPUs: " + gpus), print(Style.RESET_ALL)
    torch.cuda.manual_seed(seed)

base = MobileNet()
model = YOLO2_M(cfgfile, base)
region_loss = model.loss

if resume_train:
    print("RRResume training from previous model.")
    model = torch.load(trained_weightfile)
else:
    print('Train from scrach, loading base network...')
    pretrained_weights = torch.load(weightfile)
    model.base.load_state_dict(pretrained_weights)

model.print_network()

region_loss.seen = model.seen
processed_batches = model.seen / batch_size

init_width = model.width
init_height = model.height
init_epoch = model.seen / nsamples

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(testlist, shape=(init_width, init_height),
                        shuffle=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]), train=False),
    batch_size=batch_size, shuffle=False, **kwargs)

if use_cuda:
    if ngpus > 1:
        print(Fore.YELLOW + "Using MULTI GPUs: " + str(ngpus)), print(Style.RESET_ALL)
        model = torch.nn.DataParallel(model).cuda()
        print(Fore.YELLOW + "Deploy model on GPUs. done."), print(Style.RESET_ALL)
    else:
        print(Fore.YELLOW + "Using SINGLE GPUs: " + str(ngpus)), print(Style.RESET_ALL)
        model = model.cuda()

params_dict = dict(model.named_parameters())
params = []
for key, value in params_dict.items():
    if key.find('.bn') >= 0 or key.find('.bias') >= 0:
        params += [{'params': [value], 'weight_decay': 0.0}]
    else:
        params += [{'params': [value], 'weight_decay': decay * batch_size}]
optimizer = optim.SGD(model.parameters(), lr=learning_rate / batch_size, momentum=momentum, dampening=0,
                      weight_decay=decay * batch_size)


def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / batch_size
    return lr

if use_visdom:
    # initialize visdom loss plot
    epoch_lot = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel='epoch',
            ylabel='prf',
            title='Test result',
            legend=["precision", "recall", "fscore"]
        )
    )

    lot = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel='iter',
            ylabel='loss',
            title='Losses',
            legend=["loss_conf", "loss_cls", "loss"]
        )
    )

    lr_lot = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,)).cpu(),
        opts=dict(
            xlabel='batch',
            ylabel='lr',
            title='Iteration YOLO2-OCR learning rate',
            legend=['lr']
        )
    )


def train(epoch):
    global processed_batches, viz, lr_lot

    t0 = time.time()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, shape=(init_width, init_height),
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]),
                            train=True,
                            seen=cur_model.seen,
                            batch_size=batch_size,
                            num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)

    lr = adjust_learning_rate(optimizer, processed_batches)
    # print(Fore.MAGENTA)
    # print('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
    # print(Style.RESET_ALL)
    batch_steps = len(train_loader.dataset) / batch_size

    model.train()
    t1 = time.time()
    avg_time = torch.zeros(9)
    for batch_idx, (_, data, target) in enumerate(train_loader):
        if batch_idx % show_inter == 0:
            # print(Fore.YELLOW)
            print("epoch: %d/%d, batch: %d/%d" % (epoch, max_epochs, batch_idx, batch_steps))
            # print(Style.RESET_ALL)
        t2 = time.time()
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1
        # if (batch_idx+1) % dot_interval == 0:
        #    sys.stdout.write('.')

        if use_cuda:
            data = data.cuda()
            # target= target.cuda()
        t3 = time.time()
        data, target = Variable(data), Variable(target)
        t4 = time.time()
        optimizer.zero_grad()
        t5 = time.time()
        output = model(data)
        t6 = time.time()
        region_loss.seen = region_loss.seen + data.data.size(0)
        loss = region_loss(output, target, batch_idx, show_inter)
        t7 = time.time()
        loss.backward()
        t8 = time.time()
        optimizer.step()
        t9 = time.time()
        if False and batch_idx > 1:
            avg_time[0] = avg_time[0] + (t2 - t1)
            avg_time[1] = avg_time[1] + (t3 - t2)
            avg_time[2] = avg_time[2] + (t4 - t3)
            avg_time[3] = avg_time[3] + (t5 - t4)
            avg_time[4] = avg_time[4] + (t6 - t5)
            avg_time[5] = avg_time[5] + (t7 - t6)
            avg_time[6] = avg_time[6] + (t8 - t7)
            avg_time[7] = avg_time[7] + (t9 - t8)
            avg_time[8] = avg_time[8] + (t9 - t1)
            print('-------------------------------')
            print('       load data : %f' % (avg_time[0] / (batch_idx)))
            print('     cpu to cuda : %f' % (avg_time[1] / (batch_idx)))
            print('cuda to variable : %f' % (avg_time[2] / (batch_idx)))
            print('       zero_grad : %f' % (avg_time[3] / (batch_idx)))
            print(' forward feature : %f' % (avg_time[4] / (batch_idx)))
            print('    forward loss : %f' % (avg_time[5] / (batch_idx)))
            print('        backward : %f' % (avg_time[6] / (batch_idx)))
            print('            step : %f' % (avg_time[7] / (batch_idx)))
            print('           total : %f' % (avg_time[8] / (batch_idx)))
        t1 = time.time()
        if use_visdom:
            viz.line(
                X=torch.ones((1,)).cpu() * epoch,
                Y=torch.FloatTensor([lr]).cpu(),
                win=lr_lot,
                update='append'
            )
            viz.line(
                X=torch.ones((1, 3)).cpu() * processed_batches,
                Y=torch.FloatTensor([region_loss.loss_conf, region_loss.loss_cls, region_loss.loss]).unsqueeze(0).cpu(),
                win=lot,
                update='append'
            )

    print('')
    t1 = time.time()
    logging('training with %f samples/s' % (len(train_loader.dataset) / (t1 - t0)))
    if (epoch + 1) % save_interval == 0:
        if not os.path.exists(backupdir):
            os.makedirs(backupdir)
        logging('save weights to %s/%06d.weights' % (backupdir, epoch + 1))
        cur_model.seen = (epoch + 1) * len(train_loader.dataset)
        torch.save(cur_model.state_dict(), '%s/%06d.weights' % (backupdir, epoch + 1))
        # cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch + 1))


def test(epoch, use_visdom=True):
    global viz, epoch_lot
    logging("test epoch: %d" % epoch)

    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    model.eval()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    num_classes = cur_model.num_classes
    anchors = cur_model.anchors
    num_anchors = cur_model.num_anchors
    total = 0.0
    proposals = 0.0
    correct = 0.0

    for batch_idx, (_, data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data).data
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)

            total = total + num_gts

            for i in range(len(boxes)):
                if boxes[i][4] > conf_thresh:
                    proposals = proposals + 1

            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    best_iou = max(iou, best_iou)
                    if best_iou > iou_thresh and boxes[j][6] == box_gt[6]:  # 标签正确并且预测的最大iou大于设置的阈值
                        correct = correct + 1
                        # this_detected = True
                        break

    precision = 1.0 * correct / (proposals + eps)
    recall = 1.0 * correct / (total + eps)
    fscore = 2.0 * precision * recall / (precision + recall + eps)
    logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))

    if use_visdom:
        logging("vis test.")
        viz.line(
            X=torch.ones((1, 3)).cpu()*epoch,
            Y=torch.FloatTensor([precision, recall, fscore]).unsqueeze(0).cpu(),
            win=epoch_lot,
            update='append'
        )




evaluate = False
if evaluate:
    logging('evaluating ...')
    test(0)
else:
    print(init_epoch, max_epochs)
    for epoch in range(init_epoch, max_epochs):
        train(epoch)
        test(epoch)
