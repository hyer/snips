# -*- coding:utf-8 -*-
# import cv2
import uuid

from YOLO2_M import MobileNet, YOLO2_M
from crohme_inkml_result import stroke_in_box, get_stroke_hw, boxes_tighted

import tornado.web
import tornado.httpserver
from tornado.options import define
import json

from tools.gen_image import draw, draw_limit, draw_custom

from utils import *
from colorama import Fore, Back, Style

import cv2

port = 8989
cfgfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/cfg/yolo-voc-ocr-test.cfg"
weightfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/backup/k6/yolo2-mob-k6/yolo2-mob-k6-416x800.pth"  # 416x800
label_file = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/data/k6.txt"
# weightfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/backup/k3_800x800_finetune/fcrohme/000366.weights"
# label_file = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/data/k3.txt"
in_box_thres = 0.6
conf_thresh = 0.25
nms_thresh = 0.3

base = MobileNet()
m = YOLO2_M(cfgfile, base)
m.load_state_dict(torch.load(weightfile))

use_cuda = True
# if use_cuda:
#     m = torch.load(weightfile)
# else:
#     m = torch.load(weightfile, map_location = lambda storage, loc: storage)

# torch.save(m.state_dict(), "ym-k6.weights")

# static_dict = m.state_dict()
# cpu_model_dict = {}
# for key, val in static_dict.items():
#     cpu_model_dict[key] = val.cpu()

for param in m.parameters():
    param.requires_grad = False

namesfile = label_file
# print('Loading weights from %s... Done!' % (weightfile))


if use_cuda:
    print(Fore.RED + "Using GPU."),
    print(Style.RESET_ALL)
    m.cuda()
else:
    print(Fore.GREEN + "Using CPU."),
    print(Style.RESET_ALL)


def detect(imgfile):
    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))

    # for i in range(2):
    start = time.time()
    boxes = do_detect(m, sized, conf_thresh, nms_thresh, use_cuda)  # forward
    # print boxes
    finish = time.time()
    # if i == 1:
    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    savename, ret_boxes, ret_syms, ret_cls_conf = plot_boxes(img, boxes, 'predictions.jpg', class_names)
    return savename, ret_boxes, ret_syms, ret_cls_conf

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
    print "max:", max_x-min_x, max_y-min_y

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

def get_stroke_box_pair(formule_traces_resized, ret_boxes):
    '''
    每个笔画（索引）对应的box索引
    Args:
        formule_traces_resized:
        ret_boxes:

    Returns:

    '''
    stroke_boxes = []
    for stroke_idx, stroke in enumerate(formule_traces_resized):
        box_inter = []
        for box_idx, box in enumerate(ret_boxes):
            if stroke_in_box(stroke, box, in_box_thres):
                box_inter.append(box_idx)
        # if len(box_inter) > 0:
        stroke_boxes.append(box_inter)
        del box_inter

    print stroke_boxes

    #@2
    stroke_boxes_filterd = []
    for stroke_idx, box_idxs in enumerate(stroke_boxes):
        if len(box_idxs) == 0:
            stroke_boxes_filterd.append(None) # 该笔画不属于任何box
            continue
        if len(box_idxs) > 1:
            stroke_h, stroke_w = get_stroke_hw(formule_traces_resized[stroke_idx])
            box_hs= []
            box_ws= []
            for box_id in box_idxs:
                box_hs.append(abs(ret_boxes[box_id][1] -ret_boxes[box_id][3]))
                box_ws.append(abs(ret_boxes[box_id][0] -ret_boxes[box_id][2]))

            if stroke_h >= stroke_w:
                id = np.argmax(float(stroke_h)/np.array(box_hs))
                stroke_boxes_filterd.append(box_idxs[id])
                continue
            else:
                id = np.argmax(float(stroke_w)/np.array(box_ws))
                stroke_boxes_filterd.append(box_idxs[id])
                continue
        stroke_boxes_filterd.append(box_idxs[0])

    return stroke_boxes_filterd

def get_boxes_stks(boxes, stroke_boxes_filterd):
    '''
    每个box对应的所有笔画id
    Args:
        boxes:
        stroke_boxes_filterd:

    Returns:

    '''
    box_num = len(boxes)
    boxes_stks = []
    # for box_idx in boxes:
    for box_idx in range(box_num):
        def find(x, y):
            return [ a for a in range(len(y)) if y[a] == x]

        stk_idxs = find(box_idx, stroke_boxes_filterd)
        if len(stk_idxs) > 0:
            # print stk_idxs
            boxes_stks.append(stk_idxs)

    return boxes_stks

def getNewBox(rect, list_id):
    minx = 100000000
    miny = 100000000
    maxx = -100000000
    maxy = -100000000
    for i in list_id:
        if i < 0 or i > len(rect):
            continue
        if rect[i][0] < minx:
            minx = rect[i][0]
        if rect[i][1] < miny:
            miny = rect[i][1]
        if rect[i][2] > maxx:
            maxx = rect[i][2]
        if rect[i][3] > maxy:
            maxy = rect[i][3]
    return [minx, miny, maxx, maxy]

def combine_sqrt(labels, rects):
    list_sqrtfrt = []
    list_sqrtend = []
    list_labels = []
    list_rects = []
    for i in xrange(len(labels)):
        if labels[i][0] not in ['\\sqrt_frt', '\\sqrt_end']:
            list_labels.append(labels[i])
            list_rects.append(rects[i])
        elif labels[i][0] == '\\sqrt_frt':
            list_sqrtfrt.append(rects[i])
        else:
            list_sqrtend.append(rects[i])
    for i in xrange(len(list_sqrtfrt)):
        mindis = 100000001
        matchid = -1
        for j in xrange(len(list_sqrtend)):
            dis = (list_sqrtend[j][0] - list_sqrtfrt[i][0]) * (list_sqrtend[j][0] - list_sqrtfrt[i][0])
            dis += (list_sqrtend[j][1] - list_sqrtfrt[i][1]) * (list_sqrtend[j][1] - list_sqrtfrt[i][1])
            if dis < mindis:
                mindis = dis
                matchid = j
        if matchid >= 0:
            combinerect = getNewBox([list_sqrtfrt[i], list_sqrtend[matchid]], [0, 1])
            list_labels.append(['\\sqrt'])
            #list_sqrtfrt[i][2] = (list_sqrtfrt[i][0] * 2 + list_sqrtfrt[i][2]) / 3
            list_rects.append(combinerect)
    return list_labels, list_rects


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
        data = json.loads(self.request.body)
        # print data
        traces = data["pt_dat"]
        img_name = "pad_formule.jpg"
        print "#################### PREDICT ########################"
        print "collecting traces..."

        pts_list = []

        save_path = "/home/hyer/workspace/algo/OCR/Math_Exp/test_data/test_ocr_trace/" + str(uuid.uuid1()) +  ".trace"
        #pyperclip.copy(save_path)
        with open(save_path, "w") as fin:
            for stroke in traces:
                for pts in stroke:
                    pts_list.append([pts[0], pts[1]])
                pts_list.append([-10000, -10000])
            norm_list, _ = norm_data(pts_list)

            for temp in norm_list:
                fin.write(str(temp[0]) + " " + str(temp[1]) + "\n")


        # _, _, _, formule_traces_resized = draw(traces, image_size=800, formule_pad=200)
        # use_limit = False
        # _, formule_traces_resized, scale, use_limit = draw_limit(traces)
        img_cus, formule_traces_resized, scale, use_limit = draw_custom(traces, image_h=416, image_w=800, image_name=img_name)
        print "Draw image done."
        print "stroke count:", formule_traces_resized

        savename, ret_boxes, ret_syms, ret_cls_conf = detect(img_name)

        ret_lables = []
        for sym in ret_syms:
            ret_lables.append([sym])

        print "merge sqrt."
        ret_syms, ret_boxes = combine_sqrt(ret_lables, ret_boxes)

        for k, sym in enumerate(ret_syms):
            print "#", k, sym, ret_boxes[k]

        stroke_boxes_filterd = get_stroke_box_pair(formule_traces_resized, ret_boxes)
        print "stroke with box ids:", stroke_boxes_filterd
        boxes_stks = get_boxes_stks(ret_boxes, stroke_boxes_filterd)
        print "boxes with stroke ids: ", boxes_stks

        boxes_tight = boxes_tighted(boxes_stks, formule_traces_resized)
        print "boxes tight", boxes_tight

        class_names = load_class_names(namesfile)
        img = Image.open(img_name).convert('RGB')
        savename_tight = plot_boxes_tight(img, boxes_tight, ret_syms, ret_cls_conf, 'predictions_tight.jpg', class_names)

        ret_img = cv2.imread(savename)
        # ret_img_tight = cv2.imread(savename_tight)
        #
        response = {}
        response["success"] = 0
        response["correct"] = True
        if use_limit:
            response["canv_boxes"] = get_canv_bbox_limit(pts_list, ret_boxes,
                                                     scale=scale)  # [[10, 20, 100, 100], [100, 200, 500, 300]]
        else:
            response["canv_boxes"] = get_canv_bbox(pts_list, ret_boxes)
        response["latex"] = "DEBUG"
        self.finish(json.dumps(response))
        #
        if 1:
            cv2.imshow("det", ret_img)
            # cv2.imshow("det_tight", ret_img_tight)
            cv2.moveWindow('det', 10, 10)
            # cv2.moveWindow('det_tight', 810, 10)
            cv2.waitKey(3000)

            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.waitKey(1)
            cv2.waitKey(1)
            cv2.waitKey(1)
            cv2.waitKey(1)
            cv2.waitKey(1)




application = tornado.web.Application([
    (r"/math_recog/", MainHandler),

])

if __name__ == "__main__":
    application.listen(port)
    print "Srv started at %d." % port
    tornado.ioloop.IOLoop.instance().start()
