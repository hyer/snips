#coding=utf-8

'''
数据集按比例划分
'''
import os
import random
from math import floor
import sys
sys.path.append("../")
from utils.Dir import Dir

data_root = "/home/nd/datasets/flower/flower_531_crop_256"
save_root = "/home/nd/datasets/flower/flower_531_crop_256-train"
train_dir = save_root + "/train"
val_dir = save_root + "/val"

if not os.path.exists(save_root):
    os.mkdir(save_root)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(val_dir):
    os.mkdir(val_dir)

train_ratio = 0.9

dir_op = Dir()

data_dirs = dir_op.getNamesFromDir(data_root)
num_cls = len(data_dirs)
print "Total class num: ", num_cls

count = 0
for dir in data_dirs:
    index = 0
    if count % 100 == 0:
        print "processing: {}/{}".format(count, num_cls)

    img_paths = dir_op.getImagePaths(data_root + "/" + dir)
    random.shuffle(img_paths)
    num_img = len(img_paths)


    train_num = floor(num_img*train_ratio)
    if train_num < 1 or (num_img - train_num) < 1:
        count += 1
        continue

    if not os.path.exists(train_dir + "/" + dir):
        os.mkdir(train_dir + "/" + dir)
    if not os.path.exists(val_dir + "/" + dir):
        os.mkdir(val_dir + "/" + dir)

    for img_path in img_paths:
        img_name = img_path.split("/")[-1]
        if index < train_num:
            dir_op.copyFiles(img_path, train_dir + "/" + dir + "/" + img_name)
        else:
            dir_op.copyFiles(img_path, val_dir + "/" + dir + "/" + img_name)
        index += 1
    count += 1