#coding=utf-8
'''
计算各个类别的AP和mAP，并绘制PR曲线
'''
import os
import pickle
import sys
sys.path.append("../")

from evaluation.eval_util import voc_eval
import matplotlib.pyplot as plt
import numpy as np

from utils import load_class_names

if __name__ == '__main__':
    VOC_OCR_ROOT = "/home/nd/datasets/k3"
    name_file = "../data/k3.names"
    class_names = load_class_names(name_file)
    output_dir = os.path.join(VOC_OCR_ROOT, "/k3-yolo2-eval-result-PR")  # root to save PR curves.
    ap_result_file = output_dir + "/" + "ap_results_distin_det_conf=0.25.txt"
    det_txt_root = VOC_OCR_ROOT + "/k3-yolo2-eval-result"
    annot_path = os.path.join(VOC_OCR_ROOT, "Annotations/%s.xml")
    test_file_list = os.path.join(VOC_OCR_ROOT, "ImageSets/Main/test.txt")
    cache_dir = det_txt_root
    class_num = len(class_names)

    ovthresh = 0.5,
    use_07_metric=False
    if use_07_metric: print "use_07_metric !"
    gen_cache_annot = True

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    fr = open(ap_result_file, "w")
    fr.write("label_id  label_name  AP\n")
    aps = []

    for cls in range(class_num):
        # if cls <113:continue
        cls_id = str(cls)
        cls_name = class_names[cls]
        print "Process cls:", cls_name
        det_txt = det_txt_root + "/comp4_det_test_%s.txt" %cls_id
        print "Process det file: ", det_txt
        if not os.path.exists(det_txt):
            print "!!!!!!!! NO EXIST det file: ", det_txt
            continue

        if gen_cache_annot:  # :
            rec, prec, ap = voc_eval(det_txt,
                                     annot_path,
                                     test_file_list,
                                     cls_name,
                                     cache_dir,
                                     ovthresh,
                                     use_07_metric=use_07_metric,
                                     load_cached_anno=False)
            gen_cache_annot = False
        else:
            rec, prec, ap = voc_eval(det_txt,
                                     annot_path,
                                     test_file_list,
                                     cls_name,
                                     cache_dir,
                                     ovthresh,
                                     use_07_metric=use_07_metric,
                                     load_cached_anno=True)

        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls_name + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

        # print rec, prec, ap

        plt.plot(rec, prec, '-', color="orange", linewidth=3)
        plt.title("class: %s, AP = %.4f" %(cls_name, ap))
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        plt.grid()

        class_name = class_names[cls]
        plt.savefig(os.path.join(output_dir, cls_name + "=" + class_name + '_PRCurve_AP=%.4f.png'%ap), format='png')
        plt.close()
        fr.write(cls_name + "\t" + class_name + "\t" + str(ap) + "\n")
        # plt.show()

    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')
    fr.close()


