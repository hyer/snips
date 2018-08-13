#coding=utf-8
'''
某个类别的AP，PR曲线并作图
'''
import os
import sys

sys.path.append("../")

from utils import load_class_names
from evaluation.eval_util import voc_eval
import matplotlib.pyplot as plt

if __name__ == '__main__':
    VOC_OCR_ROOT = "/home/nd/datasets/test_dataset_voc"
    annot_path = os.path.join(VOC_OCR_ROOT, "Annotations/%s.xml")
    test_file_list = os.path.join(VOC_OCR_ROOT, "ImageSets/Main/test.txt")
    cache_dir = VOC_OCR_ROOT + "/cache_dir"
    name_file = "../data/voc_ocr_split_sqrt.names"
    namesfile = ""
    ovthresh = 0.5,
    use_07_metric=False
    if use_07_metric: print "use_07_metric !"

    # cls_id = 146
    cls_id = raw_input("cls_id: ")

    class_names = load_class_names(name_file)
    cls_name = class_names[int(cls_id)]  # \sqrt_frt: 146_2
    print("Class Name: {}".format(cls_name))

    det_txt = VOC_OCR_ROOT + "/eval_dir/800x800-e=50/comp4_det_test_%s.txt" %cls_id
    rec, prec, ap = voc_eval(det_txt,
                             annot_path,
                             test_file_list,
                             cls_name,
                             cache_dir,
                             ovthresh,
                             use_07_metric=use_07_metric,
                             load_cached_anno=False)

    print rec, prec, ap

    plt.plot(rec, prec, '-', color="orange", linewidth=3)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.title("class: %s, AP = %.3f" %(cls_name, ap))
    plt.grid()

    plt.show()

