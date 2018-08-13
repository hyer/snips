import os

import sys

dataset = "/home/nd/datasets/k6"
# dataset = "/home/hyer/datasets/OCR/SSD_OCR/VOC_OCR_distin"
print "dataset:", dataset
anno_xml_root = dataset + "/Annotations"

test_file = dataset + "/ImageSets/Main/trainval.txt"
count = 0
label_idx = raw_input("label idx: ")

with open(test_file, 'r') as fi:
    data = fi.readlines()

for li in data:
    img_id = li.strip().split("/")[-1].split(".")[0]
    with open(dataset + "/labels" + "/" + img_id + ".txt", 'r') as f:
        dat = f.readlines()
        for lid in dat:
            label = lid.strip().split(" ")
            if label[0] == label_idx:
                count += 1
print count