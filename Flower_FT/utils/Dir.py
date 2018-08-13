import uuid

import cv2
import numpy as np
import os
import glob
import random
import shutil

import time


class Dir():
    def __init__(self):
        pass

    def getNamesFromDir(self, rootDir):
        '''
        Get directory list in subdirectory.
        :param rootDir:
        :return:
        '''
        names = []
        for fn in glob.glob(rootDir+"/*"):
            if os.path.isdir(fn):
                names.append(fn.split("/")[-1])
        return names


    def getImagePaths(self, rootDir):
        '''
        Get *.jpg image list in specified directory.
        :param rootDir:
        :return: abs path of images.
        '''
        paths = []
        for fn in glob.glob(rootDir+"/*.jpg"):
            paths.append(fn)
        return paths

    def check_file(self, file_name):
        if not os.path.isfile(file_name):
            print '[ERROR] File "%s" is not found.' % file_name
            exit(-101)

    def check_dir(self, dir):
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)

    def file_count(self, rootDir):
        file_num = 0
        sub_dirs = self.getNamesFromDir(rootDir)
        for dir in sub_dirs:
            file_paths = self.getImagePaths(rootDir + '/' + dir)
            file_num += len(file_paths)

        print "Total file number:%d in %s"%(file_num, rootDir)

        return file_num

    def categorize_image_by_id(self, image_root, new_root):
        '''
        categorize images of different person in the same directory into distinct directory of specified id.
        :param image_root: contain images of different person.
        :return:
        '''
        if os.path.exists(new_root):
            shutil.rmtree(new_root)
        os.mkdir(new_root)

        image_list = self.getImagePaths(image_root)
        index = 1
        for image_name in image_list:
            src_name = image_name.split('/')[-1]
            id = image_name.split('/')[-1].split('-')[1]
            if not os.path.exists(new_root + '/' + id):
                os.mkdir(new_root + '/' + id)
                index = 1
            # shutil.copy(image_name, new_root + '/' + str(id) + '/' + str(uuid.uuid1()) + '.jpg')
            shutil.copy(image_name, new_root + '/' + str(id) + '/' + src_name)

    def find_dir(self, rootDir, count):
        dirs = []
        sub_dirs = self.getNamesFromDir(rootDir)
        for dir in sub_dirs:
            file_paths = self.getImagePaths(rootDir + '/' + dir)
            file_num = len(file_paths)
            if file_num < count:
                dirs.append(dir)

        print "dirs that contains less than %d files:" % count
        print dirs


    def copyFiles(self, src, dst):
        '''
        Warning: If src end with '\n', the os.path.isfile() check will get False! Using the split() function to get rid
        of this beat.
        :param src:
        :param dst:
        :return:
        '''
        if os.path.isfile(src):
            shutil.copy(src, dst)
        else:
            print "[ERROR] %s is not a file! Please check the path. May be the is \\n at the end of path string." % src
            exit(-1)


    def moveFiles(self, src, dst):
        '''
        Warning: If src end with '\n', the os.path.isfile() check will get False! Using the split() function to get rid
        of this beat.
        :param src:
        :param dst:
        :return:
        '''
        if os.path.isfile(src):
            shutil.move(src, dst)
        else:
            print "[ERROR] %s is not a file! Please check the path. May be the is \\n at the end of path string." % src
            exit(-1)


    def create_dir(self, src_root, dst_root):
        self.check_dir(dst_root)
        dirs = self.getNamesFromDir(src_root)
        for dir in dirs:
            os.mkdir(dst_root + '/' + dir)


    def select_image(self, src_root, dst_root, num):
        self.check_dir(dst_root)
        dirs = self.getNamesFromDir(src_root)
        for dir in dirs:
            os.mkdir(dst_root + '/' + dir)
            images = self.getImagePaths(src_root + '/' + dir)
            if len(images) <= num:
                for img in images:
                    self.copyFiles(img, dst_root + '/' + dir)
            else:
                for i in range(num):
                    self.copyFiles(images[i], dst_root + '/' + dir)
        print "Copy Done."


    def select_image_by_id(self, src_root_a, src_root_b, dst_root):
        '''
        num_a > num_b, dst_root = a & b;
        Copy directories of b that also contained by a into new directory.
        :param src_root_a:
        :param src_root_b:
        :param dst_root:
        :param num:
        :return:
        '''
        self.check_dir(dst_root)
        dirs_a = self.getNamesFromDir(src_root_a)
        dirs_b = self.getNamesFromDir(src_root_b)
        for dir in dirs_b:
            print "Copying:", dir
            if dir in dirs_a:
                os.mkdir(dst_root + '/' + dir)
                images = self.getImagePaths(src_root_b + '/' + dir)
                for img in images:
                    self.copyFiles(img, dst_root + '/' + dir)

                shutil.rmtree(src_root_a + '/' + dir)
        print "Copy Done."


    def move_image(self, src_root, dst_root, num):
        self.check_dir(dst_root)
        dirs = self.getNamesFromDir(src_root)
        for dir in dirs:
            os.mkdir(dst_root + '/' + dir)
            images = self.getImagePaths(src_root + '/' + dir)
            if len(images) <= num:
                for img in images:
                    self.moveFiles(img, dst_root + '/' + dir)
            else:
                for i in range(num):
                    self.moveFiles(images[i], dst_root + '/' + dir)
        print "Move Done."

    def change_backend(self, root):
        files = os.listdir(root)
        for file in files:
            self.copyFiles(root + "/" + file, root + "/" + file + ".jpg")
            os.remove(root + "/" + file)

    def get_single_image(self, src_root, dst_root):
        self.check_dir(dst_root)
        dirs = self.getNamesFromDir(src_root)
        for dir in dirs:
            # os.mkdir(dst_root + '/' + dir)
            images = self.getImagePaths(src_root + '/' + dir)
            self.copyFiles(images[0], dst_root + '/' + dir + ".jpg")

    def del_empty_dir(self, root):
        directories = self.getNamesFromDir(root)
        for directory in directories:
            if len(os.listdir(root + '/' + directory)) == 0:
                os.rmdir(root + '/' + directory)

    def batch_rename(self, file_root):
        dirs = self.getNamesFromDir(file_root)
        for directory in dirs:
            files = self.getImagePaths(file_root + '/' + directory)
            i = 1
            for file in files:
                os.rename(file, file_root + '/' + directory + '/' + 'img_' + str(i) + '.jpg')
                i += 1

def main():
    dir = Dir()
    # total_file_num = dir.file_count('/home/hyer/datasets/face/intell_99image_merge/merged_256x256_100')
    # print total_file_num

    # dir.categorize_image_by_id("/home/hyer/datasets/ndface_data/face_100007",
    #                            "/home/hyer/datasets/ndface_data/face_100007_dirs")

    # dir.find_dir('/home/hyer/datasets/face/intell_door_register_20160316', 3)
    # dir.create_dir("/home/hyer/datasets/face/99image_256x256", "/home/hyer/datasets/face/99image_register")
    # dir.move_image("/home/hyer/datasets/face/intell_99image_merge/merged-256x256-test_47", "/home/hyer/datasets/face/intell_99image_merge/merged_256x256-register_47", 3)
    # dir.change_backend("")
    # print os.path.isfile("/media/hyer/000B241E000F3AED/faceData/2016-03-24_new/36a727d8-1377-11e6-af1b-ecb1d74c581e.jpg")
    # dir.get_signle_image("/home/hyer/datasets/face/99image_intell_door_register", "/home/hyer/datasets/face/99image_signle")
    # dir.check_dir("./temp")

    src_root = '/home/hyer/workspace/business/Face/water_bar_face_collection/images'
    dst_root = '/home/hyer/datasets/ar_100'

    ids = dir.getNamesFromDir(src_root)
    print ids[0]
    count = 0
    for id in ids:
        count += 1
        path = src_root + '/' + id
        images = dir.getImagePaths(path)
        # dir.check_dir(dst_root + '/' + id)
        # for img in images:
        #     dir.copyFiles(img, dst_root + '/' + id)
        dir.copyFiles(images[0], dst_root+"/" + time.strftime('%Y%m%d_%H%M%S_') + str(time.clock()) +".jpg")
        # if count == 100:
        #     break
    # dir.del_empty_dir(dst_root)

    # dir.select_image_by_id("/home/hyer/datasets/ndface_data/face_100007_dirs_1",
    #                        "/home/hyer/datasets/ndface_data/xiaoshan_132.22_dirs",
    #                        "/home/hyer/datasets/ndface_data/face_100007_dirs_1_in_xiaoshan")

    # dir.batch_rename('/home/hyer/datasets/test_3D_faces/gall_multi')

    # images = dir.getImagePaths("/home/hyer/datasets/ndface_data/xiaoshan_132.22")
    # # print images
    # xiaoshan_id_list = []
    # for img in images:
    #     xiaoshan_id_list.append(img.split('/')[-1].split('_')[0])
    # print xiaoshan_id_list



if __name__ == '__main__':
    main()