import os

# traindir = "/home/nd/datasets/flower/flower_531_crop_256-train/train"
# cls_dirs = os.listdir(traindir)
# print "dirs num: ", len(cls_dirs)
#
name_file = "/home/hyer/datasets/FlowerDataBase/label_cid_word_530.txt"
with open(name_file, "r") as f:
    data = f.readlines()

with open("cls_name_legacy.txt", "w") as fo:
    label_dirs = []
    for li in data:
        line = li.strip().split()
        dir_name = line[1]
        fo.write(dir_name + "\n")
        label_dirs.append(dir_name)
    print "name num: ", len(label_dirs)



#
# for dir in cls_dirs:
#     if dir not in label_dirs:
#         print dir
#
# print "xxxx"
#
#
# for dir in label_dirs:
#     if dir not in cls_dirs:
#         print dir



# ################ list dir to txt ########################
# from utils.Dir import Dir
#
# dir_op = Dir()
#
# dirs = dir_op.getNamesFromDir("/home/nd/datasets/flower/flower_531_crop_256-train/train")
#
# import natsort
# dirs = natsort.natsorted(dirs)
#
# with open("cls_label.txt", "w") as fi:
#     for dir in dirs:
#         fi.write(dir + "\n")


