#coding=utf-8
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import v2, COCO_mobile_300
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, version, size, base, extras, head, num_classes, bkg_label=0, top_k=200, conf_thresh=0.01, nms_thresh=0.45):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.priorbox = PriorBox(version)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.base = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)  # 第一个参数为连接到Detection Layer的第一个卷积层的通道数， 虽有一个38x38大小的feature map的通道数
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, bkg_label, top_k, conf_thresh, nms_thresh)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        t1 = time.time()
        # apply MobileNet-v1 up to conv11 relu
        for k in range(12):
            x = self.base[k](x)  # 1x256x38x38

        s = self.L2Norm(x)
        sources.append(s)

        # apply MobileNet-v1 up to conv13
        for k in range(12, len(self.base)):
            x = self.base[k](x)
        sources.append(x)

        t2 = time.time()
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # print("TIME base network: ", t2- t1)
        # print("TIME extra network: ", time.time() - t1)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers



def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        # Conv2dDepthwise(inp, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

# MobileNet v1
def MobileNet():
    layers = []

    layers += [conv_bn(3, 32, 2)] # conv0

    layers += [conv_dw(32, 64, 1)]  # [conv1 dw /s1] + [conv1 /s1], conv1

    layers += [conv_dw(64, 128, 2)]

    layers += [conv_dw(128, 128, 1)]

    layers += [conv_dw(128, 256, 2)]

    layers += [conv_dw(256, 256, 1)]

    layers += [conv_dw(256, 512, 2)]

    layers += [conv_dw(512, 512, 1)] # conv7
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]

    layers += [conv_dw(512, 1024, 2)]

    layers += [conv_dw(1024, 1024, 1)] # conv13

    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(base_network, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []

    base_network_source = [-4, -1]  # MobileNet的conv11和conv13连接到Detection Layer
    for k, v in enumerate(base_network_source):
        loc_layers += [nn.Conv2d(base_network[v][3].out_channels, cfg[k] * 4, kernel_size=3, padding=1)] # *4是box坐标
        conf_layers += [nn.Conv2d(base_network[v][3].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)] # *num_classes是每个box对应的各个类别的得分

    for k, v in enumerate(extra_layers[1::2], 2):  # 8个extra layer的4个layer连接到Detection Layer
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]

    return base_network, extra_layers, (loc_layers, conf_layers)

# def multibox(size, base, extra_layers, cfg, num_classes):
#     loc_layers = []
#     conf_layers = []
#     base_net = [-2, -1]
#     for k, v in enumerate(base_net):
#         if k == 0:
#             loc_layers += [nn.Conv2d(512, cfg[k] * 4, kernel_size=1, padding=0)]
#             conf_layers += [nn.Conv2d(512, cfg[k] * num_classes, kernel_size=1, padding=0)]
#         else:
#             loc_layers += [nn.Conv2d(1024, cfg[k] * 4, kernel_size=1, padding=0)]
#             conf_layers += [nn.Conv2d(1024, cfg[k] * num_classes, kernel_size=1, padding=0)]
#     # i = 2.0
#     # indicator = 0
#     # if size == 300:
#     #     indicator = 1
#     # else:
#     #     print("Error: Sorry only RFB300_mobile is supported!")
#     #     return
#     #
#     # for k, v in enumerate(extra_layers):
#     #     if k < indicator or k % 2 == 0:
#     #         loc_layers += [nn.Conv2d(v.out_channels, cfg[i] * 4, kernel_size=1, padding=0)]
#     #         conf_layers += [nn.Conv2d(v.out_channels, cfg[i] * num_classes, kernel_size=1, padding=0)]
#     #         i += 1
#     return base, extra_layers, (loc_layers, conf_layers)

#
# base = {
#     '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
#             512, 512, 512],
#     '512': [],
# }
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 64, 'S', 128],  # ref: Caffe impl: https://github.com/chuanqi305/MobileNet-SSD
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
}


def build_ssd(phase, version, size=300, num_classes=21, bkg_label=0, top_k=200, conf_thresh=0.01, nms_thresh=0.45):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only SSD300 is supported currently!")
        return
    base_, extras_, head_ = multibox(MobileNet(),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)],
                                     num_classes)
    return SSD(phase, version, size, base_, extras_, head_, num_classes, bkg_label=bkg_label, top_k=top_k, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
