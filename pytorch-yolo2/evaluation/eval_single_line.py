import cv2

from tools.dir_opt import Dir
from tools.gen_image import draw_limit
from utils import load_class_names


def read_txt(filename):
    label = filename.split("/")[-1].split("#")[0]
    with open(filename, "r") as f:
        data = f.readlines()

    traces = []
    # trace = []
    for li in data:
        line = li.strip()
        if line[-1] == "-1":
            continue
        traces.append(line)

    print "xxx"
    # img_limit, formule_traces_resized, scale, use_limit = draw_limit(traces)

    stroke = []
    all_traces = []
    for trace in traces:
        t = trace.split(":")[-1]
        t1 = t.split()
        t_label = []
        for i in range(len(t1)/2):
            if int(t1[i*2]) == -4321:
                all_traces.append(t_label)
                t_label = []
                continue
            t_label.append([int(t1[i*2]), int(t1[i*2+1]), -1, -1])

    return all_traces, label

if __name__ == '__main__':
    # save_root = "/home/hyer/Desktop/test_trace_txt_imgaes"
    # dirop = Dir()
    #
    # for file in dirop.getPaths("/home/hyer/Desktop/test_trace_txt", "txt"):
    #     imgname = file.split("/")[-1].split(".")[0]
    #     traces, label = read_txt(file)
    #     img_limit, formule_traces_resized, scale, use_limit = draw_limit(traces)
    #     cv2.imwrite("/home/hyer/Desktop/test_trace_txt_imgaes/" + imgname + ".jpg", img_limit)
    #     # cv2.waitKey()
    class_names = load_class_names("/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/data/k6.names")
    label_names = load_class_names("/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/data/charmap_split_sqrt.txt")

    ret_file = "/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/evaluation/result.txt"
    with open(ret_file, "r") as f:
        data = f.readlines()

    correct = 0
    for li in data:
        line = li.strip()
        if len(line.split(" ")) > 1:
            pred = line.split(" ")[0]
            pred = pred.split("_")[:-1]
            pred_t = []
            for i in range(len(pred)):
                pred_t.append(class_names[int(pred[i])])
            pred = "_".join(pred_t)


            gt = line.split(" ")[1]
            if pred == gt:
                correct += 1
            else:
                pred = pred.split("_")
                pred_t = []
                for i in range(len(pred)):
                    pred_t.append(label_names[int(pred[i])])
                print pred_t

                gt = gt.split("_")
                gt_l = []
                for i in range(len(gt)):
                    gt_l.append(label_names[int(gt[i])])
                print gt_l

                print "-----------------"

                # print(pred, " ", gt)
    print("precision: ", float(correct)/len(data))
