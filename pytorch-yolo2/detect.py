import copy
import sys
import time


from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from tools.gen_image import draw
from utils import *
from darknet import Darknet
import cv2

conf_thresh = 0.25
nms_thresh = 0.3
iou_thresh = 0.5

# cfgfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/cfg/yolo-voc-ocr-test.cfg"
# weightfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/backup/800x800_distin_fix_frac/000040.weights"
# label_file = "/home/hyer/workspace/algo/OCR/Math_Exp/OCR_Tools/charmap.txt"

cfgfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/cfg/yolo-voc-ocr-test.cfg"
weightfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/backup/k3_800x800_finetune/fcrohme/000366" \
             ".weights"
label_file = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/data/k3.txt"

m = Darknet(cfgfile)
m.print_network()
m.load_weights(weightfile)
namesfile = label_file
print('Loading weights from %s... Done!' % (weightfile))

use_cuda = 1
if use_cuda:
    print "Using CUDA."
    m.cuda()


# def detect(cfgfile, weightfile, imgfile, label_file):
def detect(imgfile):
    # m = Darknet(cfgfile)
    #
    # m.print_network()
    # m.load_weights(weightfile)
    # print('Loading weights from %s... Done!' % (weightfile))
    #
    # # if m.num_classes == 20:
    # #     namesfile = 'data/voc.names'
    # # elif m.num_classes == 80:
    # #     namesfile = 'data/coco.names'
    # # else:
    # namesfile = label_file
    #
    # use_cuda = 1
    # if use_cuda:
    #     m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, conf_thresh, nms_thresh, use_cuda)  # forward
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    savename, _, _, _ = plot_boxes(img, boxes, 'predictions.jpg', class_names)
    print boxes
    return savename

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def process_trace(trace_seq):
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
    # if len(sys.argv) == 4:
    # imgfile = "/home/hyer/datasets/OCR/SSD_OCR/SSD_OCR_3/JPEGImages/000011.jpg"
    while 1:

        img_path = raw_input("img_path: ")
        # # img_id = raw_input("img_id: ")
        # # img_path = "/home/hyer/datasets/OCR/SSD_OCR/VOC_OCR_distin/JPEGImages/" + img_id + ".jpg"
        #
        # imgfile = "pad_formule.jpg"
        img = cv2.imread(img_path)
        cv2.imshow("src", img)

        # cfgfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/cfg/yolo-voc-ocr-test.cfg"
        # weightfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/backup/000010.weights"
        # label_file = "/home/hyer/workspace/algo/OCR/Math_Exp/OCR_Tools/charmap.txt"


        savename = detect(img_path)
        ret_img = cv2.imread(savename)
        cv2.imshow("det", ret_img)
        cv2.waitKey(0) # close window when a key press is detected
        # cv2.destroyWindow('det')
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)

    ### test trace file.
    # trace_file = "/home/hyer/Downloads/99U_Downloads/2020-math-math-2018-01-17-18-00-05-489438-3147f76a-fb6d-11e7-b9c5-9c5c8e8f37ca.txt"
    # with open(trace_file, "r") as f:
    #     data = f.readlines()
    #
    # pt_seq = []
    # for li in data:
    #     line = li.strip()
    #     # if line != "-10000 -10000\n":
    #     pt_seq.append(int(float(line.split()[0])))
    #     pt_seq.append(int(float(line.split()[1])))
    # traces = process_trace(pt_seq)
    # draw(traces, image_size=800)
    #
    # savename = detect("pad_formule.jpg")
    # ret_img = cv2.imread(savename)
    # cv2.imshow("det", ret_img)
    # cv2.waitKey(0) # close window when a key press is detected
    # # cv2.destroyWindow('det')
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # cv2.waitKey(1)
    # cv2.waitKey(1)
    # cv2.waitKey(1)
    # cv2.waitKey(1)
    # cv2.waitKey(1)