# coding=utf-8
'''
某个类别的AP，PR曲线并作图
'''
import os
import sys

from eval_voc_tool import voc_eval
from utils.tools import load_class_names

sys.path.append("../")
import matplotlib.pyplot as plt

if __name__ == '__main__':
    VOC_OCR_ROOT = "/home/hyer/data/k1_scan"
    annot_path = os.path.join(VOC_OCR_ROOT, "Annotations/%s.xml")
    test_file_list = os.path.join(VOC_OCR_ROOT, "ImageSets/Main/test.txt")
    cache_dir = VOC_OCR_ROOT + "/cache_dir"
    label_file = "./data/k1.txt"
    ovthresh = 0.5
    use_07_metric = False
    if use_07_metric: print("use_07_metric !")

    # cls_id = 146
    name = input("name in xml:")

    class_names = load_class_names(label_file)
    # cls_name = class_names[name]  # \sqrt_frt: 146_2
    # print("Class Name: {}".format(cls_name))

    det_txt = VOC_OCR_ROOT + "/eval/det_txts/comp4_det_test_%s.txt" % name

    rec, prec, ap = voc_eval(det_txt,
                             annot_path,
                             test_file_list,
                             name,
                             cache_dir,
                             ovthresh,
                             use_07_metric=use_07_metric)

    print("Last value of RECALL / PRECISION / AP: %.4f / %.4f / %.4f" %(rec[-1], prec[-1], ap))

    plt.plot(rec, prec, '-', color="orange", linewidth=3)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.title("class: %s, AP = %.3f" % (name, ap))
    plt.grid()

    plt.show()
