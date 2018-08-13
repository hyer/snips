import copy
import math
import torch

from PIL import Image, ImageDraw, ImageFont
from darknet import Darknet
from tools.gen_image import draw
from utils import nms, get_region_boxes, load_class_names
import numpy as np

class YOLO2_OCR():
    def __init__(self, cfg_file, weight_file, label_file, use_cuda=True):
        self.label_file = label_file
        self.m = Darknet(cfg_file)
        self.m.print_network()
        self.m.load_weights(weight_file)
        print('Loading weights from %s... Done!' % (weight_file))

        if use_cuda:
            print("Using GPU."),
            self.m.cuda()
        else:
            print("Using CPU."),


    def reference(self, trace_seq, use_cuda=True):
        '''

        Args:
            trace_seq: [300, 200, 300, 400, -10000, -10000, 800, 200, 800, 400, -10000, -10000, ...]
            use_cuda:

        Returns:

        '''

        traces = self.process_trace(trace_seq)
        draw(traces, image_size=800)

        imgfile = "pad_formule.jpg"
        img = Image.open(imgfile).convert('RGB')
        sized = img.resize((self.m.width, self.m.height))

        # for i in range(2):
        # start = time.time()
        boxes = self.do_detect(self.m, sized, 0.5, 0.4, use_cuda)  # forward
        # finish = time.time()
        # if i == 1:
        # print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

        class_names = load_class_names(self.label_file)
        dets = self.detect_labels(img, boxes, class_names)


        ret_labels = [] # sym list
        ret_boxes = []  # sym box list
        ret_cls_conf = [] # sym class confidence
        for i in range(len(dets)):
            ret_labels.append(dets[i][-1])
            ret_cls_conf.append(dets[i][-2])
            ret_boxes.append(dets[i][:-2])

        return ret_labels, ret_cls_conf, ret_boxes
        # return dets


    def detect_labels(self, img, boxes, class_names):
        dets = []
        width = img.width
        height = img.height

        for i in range(len(boxes)):
            box = boxes[i]
            x1 = (box[0] - box[2]/2.0) * width
            y1 = (box[1] - box[3]/2.0) * height
            x2 = (box[0] + box[2]/2.0) * width
            y2 = (box[1] + box[3]/2.0) * height

            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                print('%s: %f' % (class_names[cls_id], cls_conf))
                dets.append([x1, y1, x2, y2, cls_conf, class_names[cls_id]])

        return dets


    def do_detect(self, model, img, conf_thresh, nms_thresh, use_cuda=True):
        model.eval()
        # t0 = time.time()

        if isinstance(img, Image.Image):
            width = img.width
            height = img.height
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
            img = img.view(1, 3, height, width)
            img = img.float().div(255.0)
        elif type(img) == np.ndarray: # cv2 image
            img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
        else:
            print("unknow image type")
            exit(-1)

        # t1 = time.time()

        if use_cuda:
            img = img.cuda()
        img = torch.autograd.Variable(img)
        # t2 = time.time()

        output = model(img)
        output = output.data
        #for j in range(100):
        #    sys.stdout.write('%f ' % (output.storage()[j]))
        #print('')
        # output = output.cpu()
        # t3 = time.time()
        boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors, model.num_anchors, use_cuda=use_cuda)[0]
        #for j in range(len(boxes)):
        #    print(boxes[j])
        # t4 = time.time()

        boxes = nms(boxes, nms_thresh)
        # t5 = time.time()

        # if True:
        #     print('-----------------------------------')
        #     print(' image to tensor : %f' % (t1 - t0))
        #     print('  tensor to cuda : %f' % (t2 - t1))
        #     print('         predict : %f' % (t3 - t2))
        #     print('get_region_boxes : %f' % (t4 - t3))
        #     print('             nms : %f' % (t5 - t4))
        #     print('           total : %f' % (t5 - t0))
        #     print('-----------------------------------')
        return boxes

    def process_trace(self, trace_seq):
        traces = []
        stroke = []
        for i in range(1, len(trace_seq), 2):
            x = trace_seq[i-1]
            y = trace_seq[i]
            if x != -10000:
                stroke.append([x, y])
            else:
                traces.append(copy.copy(stroke))
                stroke = []

        return traces

if __name__ == '__main__':
    cfgfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/cfg/yolo-voc-ocr-test.cfg"
    weightfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/backup/split_sqrt_800x800_final/000050.weights"
    label_file = "data/charmap_split_sqrt.txt"

    model = YOLO2_OCR(cfgfile, weightfile, label_file)

    # pt_seq = [300, 200, 300, 400, -10000, -10000, 800, 200, 800, 400, -10000, -10000]

    trace_file = "/home/hyer/Downloads/99U_Downloads/2020-math-math-2018-01-17-17-59-25-064794-192fa79a-fb6d-11e7-b9c5-9c5c8e8f37ca.txt"
    with open(trace_file, "r") as f:
        data = f.readlines()

    pt_seq = []
    for li in data:
        line = li.strip()
        # if line != "-10000 -10000\n":
        pt_seq.append(int(float(line.split()[0])))
        pt_seq.append(int(float(line.split()[1])))

    print model.reference(pt_seq)