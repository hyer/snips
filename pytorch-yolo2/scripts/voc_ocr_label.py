# coding=utf-8

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

from utils import load_class_names

VOC_OCR_DATA = "/home/hyer/datasets/OCR/SSD_OCR/k6"
# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets = ['trainval', 'test']
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# classes = []
# for i in range(146):
#     classes.append(str(i))
# classes.append("146_1")  # sqrt part 1, idx = 146, labels目录下是类别的索引，从0开始，不包含背景类别
# classes.append("146_2")  # sqrt part 2, idx = 147
# classes.append("147")  # \frac, idx = 148
namesfile = "./data/k6.names"
class_names = load_class_names(namesfile)

def get_sym_map(map_file):
    with open(map_file, 'r') as f:
        data = f.readlines()
        syms = []

        for i in range(len(data)):
            sym = data[i].split("\n")[0]
            syms.append(sym)
    return syms


# classes = get_sym_map("/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/data/charmap_$.txt")

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open(VOC_OCR_DATA + '/Annotations/%s.xml' % image_id)
    out_file = open(VOC_OCR_DATA + '/labels/%s.txt' % image_id, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in class_names or int(difficult) == 1:
            print cls + "not in class_names !!!"
            continue
        cls_id = class_names.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()

for image_set in sets:
    # if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
    #     os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open(VOC_OCR_DATA + '/ImageSets/Main/%s.txt' % image_set).read().strip().split()
    # list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        image_id = image_id.split("/")[-1].split(".")[0]
        # list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        convert_annotation(image_id)
    # list_file.close()
