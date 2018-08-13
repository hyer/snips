# -*- coding:utf-8 -*-
# import cv2



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
from utils.tools import load_class_names

port = 9800
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd

label_file = "/home/hyer/workspace/algo/Detection/SSD/ssd.pytorch/data/k1.txt"
weightfile = '/home/hyer/workspace/algo/Detection/SSD/ssd.pytorch/models/mobv1-ssd-k1/add_0/VOC_mobile_300_19105321-add0.pth'
class_names = load_class_names(label_file)

# from models import build_ssd as build_ssd_v1 # uncomment for older pool6 mode
version = config.VOC_mobile_300_19105321
conf_thres = 0.2
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



def detect(imgfile):
    image = cv2.imread(imgfile)
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

    t1 = time.time()
    detections = net(xx).data
    print("exec time: ", time.time() - t1)
    # detect_time = _t['im_detect'].toc(average=False)
    # print("Forward time: ", detect_time)

    # from data import VOC_CLASSES as labels

    # # plt.figure()
    # colors = plt.cm.hsv(np.linspace(0, 1, num_cls)).tolist()
    # plt.imshow(rgb_image)  # plot the image for matplotlib
    # currentAxis = plt.gca()
    #
    # print(rgb_image.shape[1::-1], rgb_image.shape[1::-1])
    #
    # # scale each detection back up to the image
    # scale = torch.Tensor([rgb_image.shape[1::-1], rgb_image.shape[1::-1]])
    # for i in range(detections.size(1)):
    #     j = 0
    #     while detections[0, i, j, 0] >= conf_thres:
    #         score = detections[0, i, j, 0]
    #         label_name = class_names[i - 1]
    #         display_txt = '%s: %.2f' % (label_name, score)
    #         print(display_txt)
    #
    #         pt = (detections[0, i, j, 1:] * scale).cpu().numpy()  # detections的第四维是5个数，表示[cls_conf, x1, y1, x2, y2]
    #         coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
    #         color = colors[i]
    #         currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    #         currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
    #         j += 1
    #
    # # plt.figure(figsize=(10, 10))
    # # plt.imshow(rgb_image)
    # plt.show()

    return detections

def norm_data(list_allpts):
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    flip_y = -1

    duration = 0

    xs = []
    ys = []
    for pt in list_allpts:
        if pt[0] != -10000 and pt[1] != -10000:
            xs.append(int(pt[0]))
            ys.append(flip_y * int(pt[1]))

    np_xs = np.array(xs).argsort()
    np_ys = np.array(ys).argsort()

    min_x = xs[np_xs[0]]
    min_y = ys[np_ys[0]]
    max_x = xs[np_xs[-1]]
    max_y = ys[np_ys[-1]]
    print("max:", max_x-min_x, max_y-min_y)

    list_out = []
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    # ratio = 120.0 / max(w, h)
    # if ratio > 1.2:
    # ratio = 1.0/2.78
    ratio = 1.0
    scale = 400.0 / h

    for pt in list_allpts:
        if pt[0] == -10000:
        # if pt[0] == -10000 and pt[1] == -10000:
            list_out.append(np.array([-10000, -10000]))
            duration += 1
            continue
        x = pt[0] - min_x - w / 2
        y = flip_y * pt[1] - min_y - h / 2
        x = int(x * scale)
        y = int(y * scale)
        list_out.append(np.array([x, y]))
        duration += 1
    return list_out, duration


def get_canv_bbox(pts_list, boxes, canv_H=400, canv_W=800, im_size=800, pad_size=200):
    xs = []
    ys = []
    for pt in pts_list:
        if pt[0] != -10000 and pt[1] != -10000:
            xs.append(int(pt[0]))
            ys.append(int(pt[1]))

    np_xs = np.array(xs).argsort()
    np_ys = np.array(ys).argsort()

    min_x = xs[np_xs[0]]
    min_y = ys[np_ys[0]]
    max_x = xs[np_xs[-1]]
    max_y = ys[np_ys[-1]]

    trace_w = max_x - min_x + 1
    trace_h = max_y - min_y + 1
    if trace_w < 0 or trace_h < 0:
        assert ("pts error.")

    canv_boxes = []
    for box in boxes:
        im_x1 = box[0]
        im_y1 = box[1]
        im_x2 = box[2]
        im_y2 = box[3]

        canv_x1 = min_x + im_x1 * float((max(trace_w, trace_h) + pad_size)) / im_size - float((
                    max(trace_w, trace_h) + pad_size - trace_w)) / 2 -3

        canv_y1 = min_y + im_y1 * float((max(trace_w, trace_h) + pad_size)) / im_size - float((
                    max(trace_w, trace_h) + pad_size - trace_h)) / 2 -3

        canv_x2 = min_x + im_x2 * float((max(trace_w, trace_h) + pad_size)) / im_size - float((
                    max(trace_w, trace_h) + pad_size - trace_w)) / 2 +5

        canv_y2 = min_y + im_y2 * float((max(trace_w, trace_h) + pad_size)) / im_size - float((
                    max(trace_w, trace_h) + pad_size - trace_h)) / 2 +5
        canv_boxes.append([canv_x1, canv_y1, canv_x2, canv_y2])

    return canv_boxes


def get_canv_bbox_limit(pts_list, boxes, scale, canv_H=400, canv_W=800, im_w=800):
    xs = []
    ys = []
    for pt in pts_list:
        if pt[0] != -10000 and pt[1] != -10000:
            xs.append(int(pt[0]))
            ys.append(int(pt[1]))

    np_xs = np.array(xs).argsort()
    np_ys = np.array(ys).argsort()

    min_x = xs[np_xs[0]]
    min_y = ys[np_ys[0]]
    max_x = xs[np_xs[-1]]
    max_y = ys[np_ys[-1]]
    formule_w = (max_x - min_x)
    formule_h = (max_y - min_y)
    f_c_x = min_x + formule_w / 2
    f_c_y = min_y + formule_h / 2

    trace_w = max_x - min_x + 1
    trace_h = max_y - min_y + 1
    if trace_w < 0 or trace_h < 0:
        assert ("pts error.")

    canv_boxes = []
    for box in boxes:
        im_x1 = box[0]
        im_y1 = box[1]
        im_x2 = box[2]
        im_y2 = box[3]

        canv_x1 = int((im_x1 - canv_W/2)/scale + canv_W/2 +f_c_x - canv_W/2) -3

        canv_y1 = int((im_y1 - im_w/2)/scale + im_w/2 - canv_H/2 +f_c_y - canv_H/2) -3

        canv_x2 = int((im_x2 - canv_W/2)/scale + canv_W/2 +f_c_x - canv_W/2) +5

        canv_y2 = int((im_y2 - im_w/2)/scale + im_w/2 - canv_H/2 + f_c_y - canv_H/2) +5
        canv_boxes.append([canv_x1, canv_y1, canv_x2, canv_y2])

    return canv_boxes


define("port", default=port, help='run a test')
class MainHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        """
        避免浏览器请求跨域问题,每次请求都会自动调用这个函数
        Returns:

        """
        # print "setting headers!!!"
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')


    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self, *args, **kwargs):
        imgfile = self.request.files.get('up_img')
        save_file_name = time.strftime('%Y-%m-%d %H%M%S', time.localtime(time.time())) + ".jpg"
        with open('/home/hyer/data/g1_test_imgs/' + save_file_name, 'wb') as f:
            f.write(imgfile[0]['body'])

        img_path = '/home/hyer/data/g1_test_imgs/' + save_file_name
        t1 = time.time()
        detections = detect(img_path)
        t2 = time.time()

        plt.figure("result")
        image = cv2.imread(img_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        colors = plt.cm.hsv(np.linspace(0, 1, num_cls)).tolist()
        plt.imshow(rgb_image)  # plot the image for matplotlib
        # plt.show()
        currentAxis = plt.gca()

        print(rgb_image.shape[1::-1], rgb_image.shape[1::-1])

        # scale each detection back up to the image
        draw = 1
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
                if draw:
                    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                    currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
                j += 1

        # plt.figure(figsize=(10, 10))
        # plt.imshow(rgb_image)
        plt.show()
        print("batch time: ", t2 - t1, "fusion time: ", time.time() - t2)
        # ret_lables = []
        # for sym in ret_syms:
        #     ret_lables.append([sym])
        #
        # for k, sym in enumerate(ret_syms):
        #     print("#", k, sym, ret_boxes[k])
        #
        # ret_img = cv2.imread(savename)
        # ret_img_tight = cv2.imread(savename_tight)

        response = {}
        # response["success"] = 0
        # response["correct"] = True
        # if use_limit:
        #     response["canv_boxes"] = get_canv_bbox_limit(pts_list, ret_boxes,
        #                                              scale=scale)  # [[10, 20, 100, 100], [100, 200, 500, 300]]
        # else:
        #     response["canv_boxes"] = get_canv_bbox(pts_list, ret_boxes)
        # response["latex"] = "DEBUG"
        self.finish(json.dumps(response))
        #
        # if 1:
        #     cv2.imshow("det", ret_img)
        #     # cv2.imshow("det_tight", ret_img_tight)
        #     cv2.moveWindow('det', 10, 10)
        #     # cv2.moveWindow('det_tight', 810, 10)
        #     cv2.waitKey(3000)
        #
        #     cv2.destroyAllWindows()
        #     cv2.waitKey(1)
        #     cv2.waitKey(1)
        #     cv2.waitKey(1)
        #     cv2.waitKey(1)
        #     cv2.waitKey(1)
        #     cv2.waitKey(1)

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self, *args, **kwargs):
        url = self.get_argument('url')
        self.write(url )
        print(url)



if __name__ == "__main__":
    application = tornado.web.Application([
        (r"/fr_g1/", MainHandler),
    ])
    application.listen(port)
    print("Srv started at %d." % port)
    tornado.ioloop.IOLoop.instance().start()
