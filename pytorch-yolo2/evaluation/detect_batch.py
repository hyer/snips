import copy
import sys
import time


from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from tools.dir_opt import Dir
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
weightfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/backup/k6/000130-add5.weights"
label_file = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/data/k6.txt"

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
    # print boxes
    return savename, boxes

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
    save_root = "/home/hyer/Desktop/test_trace_txt_imgaes"
    dirop = Dir()

    result_file = "result.txt"
    with open(result_file, "w") as fout:
        for img_path in dirop.getPaths(save_root, "jpg"):
            # img_path = raw_input("img_path: ")
            gt_label = img_path.split("/")[-1].split("#")[0]
            # # img_id = raw_input("img_id: ")
            # # img_path = "/home/hyer/datasets/OCR/SSD_OCR/VOC_OCR_distin/JPEGImages/" + img_id + ".jpg"
            #
            # imgfile = "pad_formule.jpg"
            img = cv2.imread(img_path)
            cv2.imshow("src", img)

            # cfgfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/cfg/yolo-voc-ocr-test.cfg"
            # weightfile = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/backup/000010.weights"
            # label_file = "/home/hyer/workspace/algo/OCR/Math_Exp/OCR_Tools/charmap.txt"


            savename, boxes = detect(img_path)
            if len(boxes) > 0:
                # boxes = np.sort(np.array(boxes), axis=0)
                idxs = np.lexsort(np.array(boxes)[:,::-1].T)
                labels = []
                for box_idx in idxs:
                    labels.append(boxes[box_idx][-1])
                    fout.write(str(boxes[box_idx][-1]) + "_")
                fout.write(" " + gt_label + "\n")
            else:
                fout.write(" " + gt_label + "\n")

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

