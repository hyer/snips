# coding=utf-8
#
import os
import sys
from random import random

import cv2
import argparse
import numpy as np


def get_trace(f_file_name, save_trace_file):
    '''
    :param f_file_name: stroke file name.
    :return: trace返回公式中所有采样轨迹点的坐标值
    '''
    tf = open(save_trace_file, 'w')
    with open(f_file_name, 'r') as f:
        data = f.readlines()

    formule_x_min = 0.0
    formule_y_min = 0.0
    formule_x_max = 0.0
    formule_y_max = 0.0
    pick_flag = 0
    pick_first_cord = True

    for i, line in enumerate(data):
        if line == ".PEN_UP\n":
            pick_flag = 0
            tf.write(str(-10000) + " " + str(-10000) + "\n")

        if pick_flag == 1:
            x = line.split('\n')[0].split(" ")[0]
            y = line.split('\n')[0].split(" ")[1]
            x1 = int(round(float(x)))
            y1 = int(round(float(y)))
            tf.write(str(x1) + " " + str(y1) + "\n")

            if pick_first_cord:
                formule_x_min = float(x)
                formule_y_min = float(y)
                pick_first_cord = False
            if float(x) < formule_x_min: formule_x_min = float(x)
            if float(x) > formule_x_max: formule_x_max = float(x)
            if float(y) < formule_y_min: formule_y_min = float(y)
            if float(y) > formule_y_max: formule_y_max = float(y)

        if line == ".PEN_DOWN\n":
            pick_flag = 1
    tf.close()
    return formule_x_min, formule_x_max, formule_y_min, formule_y_max


def formule_size(trace_file_path, coord_list):
    xs = []
    ys = []
    if not coord_list:
        with open(trace_file_path, "r") as f:
            data = f.readlines()
        for line in data:
            x = int(line.split("\n")[0].split(" ")[0])
            y = int(line.split("\n")[0].split(" ")[1])
            if x == -10000:
                continue
            xs.append(x)
            ys.append(y)

    else:
        for list_sym in coord_list:
            for line in list_sym:
                x = int(list(line)[0])
                y = int(list(line)[1])
                xs.append(x)
                ys.append(y)

    xs_min = min(xs)
    xs_max = max(xs)
    ys_min = min(ys)
    ys_max = max(ys)
    formule_w = xs_max - xs_min
    formule_h = ys_max - ys_min

    return formule_w, formule_h, xs_min, ys_min


def draw(coord_list, trace_file_path=None, image_name=None, image_size=800, formule_pad=200, line_width=3):
    xs = []
    ys = []
    formule_traces = []
    formule_w, formule_h, xs_min, ys_min = formule_size(trace_file_path, coord_list)
    form_size = max(formule_w, formule_h) + formule_pad
    scale = float(format(float(image_size) / float(form_size), '.5f'))
    print("Scale img to network with scale:", scale)
    img = np.zeros((image_size, image_size), np.uint8) * 255
    padding_w = form_size - formule_w
    padding_h = form_size - formule_h

    if not coord_list:
        with open(trace_file_path, "r") as f:
            data = f.readlines()
        for line in data:
            x = int(line.split("\n")[0].split(" ")[0])
            # x_1 = x + formule_w/2 + padding/2
            x_1 = int(round((x - xs_min + padding_w / 2) * scale))
            print(x_1)
            y = int(line.split("\n")[0].split(" ")[1])
            # y_1 = -y + formule_h/2 + padding/2
            y_1 = int(round((y - ys_min + padding_h / 2) * scale))

            if x == -10000:
                for i in range(len(xs) - 1):
                    cv2.line(img, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), 255, line_width)
                xs = []
                ys = []
                continue
            xs.append(x_1)
            ys.append(y_1)

    else:
        for list_sym in coord_list:
            stroke = []
            for line in list_sym:
                x = int(list(line)[0])
                x_1 = int(round((x - xs_min + padding_w / 2) * scale))
                y = int(list(line)[1])
                y_1 = int(round((y - ys_min + padding_h / 2) * scale))
                xs.append(x_1)
                ys.append(y_1)
                stroke.append((x_1, y_1))
            for i in range(len(xs) - 1):
                cv2.line(img, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), 255, line_width)
            xs = []
            ys = []
            formule_traces.append(stroke)
            del stroke

    if not image_name:
        cv2.imwrite("pad_formule.jpg", img)
    return img, padding_w, padding_h, formule_traces


def draw_limit(coord_list, trace_file_path=None, image_size=800, canvas_h=400, canvas_w=800, line_width=3, pad_x = 50,
               image_name=None):
    '''
    有效的公式区域方尽可能放缩到800x800的图像中间的400x800的长条区域
    瘦高的公式 f_h > f_w：scale= f_h /canvas_h， canvas_h=400
    行的长公式 f_h < f_w：scale= f_h /canvas_w
    训练集重新生成800x400的图片，
    Args:
        coord_list:
        trace_file_path:
        image_name:
        image_size:
        formule_pad:
        line_width:
        pad_x: 水平补边

    Returns:

    '''
    xs = []
    ys = []
    formule_traces = []
    use_limit = True
    formule_w, formule_h, xs_min, ys_min = formule_size(trace_file_path, coord_list)
    if formule_h < 25: # h < 10pixel, then keep the size
        _, _, _, formule_traces_resized = draw(coord_list, image_size=800, formule_pad=200)
        use_limit = False
        return _, formule_traces, _, use_limit

    f_c_x = xs_min + formule_w / 2
    f_c_y = ys_min + formule_h / 2
    shift_x = f_c_x - canvas_w / 2
    shift_y = f_c_y - canvas_h / 2

    if formule_h > formule_w:
        scale = canvas_h / float(max(formule_w, formule_h))
        print("Scale img to h")
    else:
        scale = (canvas_w - pad_x) / float(max(formule_w, formule_h))
        if formule_h * scale > canvas_h:
            scale = canvas_h / float(max(formule_w, formule_h))
        print("Scale img to w")

    img = np.zeros((image_size, image_size), np.uint8) * 255

    for list_sym in coord_list:
        stroke = []
        for line in list_sym:
            x = int(list(line)[0])
            x_1 = int((round(x - shift_x) - canvas_w / 2) * scale + canvas_w / 2)
            y = int(list(line)[1])
            y_1 = int((round(y - shift_y + canvas_h/2) - image_size / 2) * scale + image_size / 2)
            # y_1 = int(round((y - shift_y + canvas_h/2)))
            xs.append(x_1)
            ys.append(y_1)
            stroke.append((x_1, y_1))
        for i in range(len(xs) - 1):
            cv2.line(img, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), 255, line_width)
        xs = []
        ys = []
        formule_traces.append(stroke)
        del stroke

    if not image_name:
        cv2.imwrite("pad_formule.jpg", img)
    return img, formule_traces, scale, use_limit


def draw_custom(coord_list, trace_file_path=None, image_h=416, image_w=800, canvas_h=400, canvas_w=800, line_width=3, pad = 100,
               image_name=None):
    '''

    Args:
        coord_list:
        trace_file_path:
        image_name:
        image_size:
        formule_pad:
        line_width:
        pad: 有效区域外左右上下的pad

    Returns:

    '''
    xs = []
    ys = []
    formule_traces = []
    use_limit = True
    formule_w, formule_h, xs_min, ys_min = formule_size(trace_file_path, coord_list)
    # if formule_h < 25: # h < 10pixel, then keep the size
    #     _, _, _, formule_traces_resized = draw(coord_list, image_size=800, formule_pad=200)
    #     use_limit = False
    #     return _, formule_traces, _, use_limit

    f_c_x = xs_min + formule_w / 2
    f_c_y = ys_min + formule_h / 2
    shift_x = f_c_x - canvas_w / 2
    shift_y = f_c_y - canvas_h / 2

    scale_h = float(formule_h) / (image_h - 2 * pad)
    scale_w = float(formule_w) / (image_w - 2 * pad)

    if formule_h > formule_w:
        print("Scale img to h")
        scale = scale_h
    else:
        scale = scale_w
        h_temp = formule_h / scale
        if h_temp > image_h: # 如果按宽缩放，轨迹点高度超出图片，需要将缩放后的轨迹点高度再次压缩到图像高度以内
            print("#"),
            scale *= h_temp / (image_h - 2 * pad)
        print("Scale img to w")


    # if formule_h > image_h or formule_w > image_w:
    #     scale = 1.0 / scale

    img = np.zeros((image_h, image_w), np.uint8) * 255

    for list_sym in coord_list:
        stroke = []
        for line in list_sym:
            x = int(list(line)[0])
            x_1 = int(round(x - shift_x - canvas_w / 2) / scale + image_w / 2)
            y = int(list(line)[1])
            # y_1 = int(round(y - shift_y - canvas_h/2) * scale + image_h/2) + pad
            y_1 = int(round(y - shift_y - canvas_h / 2) / scale + image_h / 2)
            xs.append(x_1)
            ys.append(y_1)
            stroke.append((x_1, y_1))
        for i in range(len(xs) - 1):
            cv2.line(img, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), 255, line_width)
        xs = []
        ys = []
        formule_traces.append(stroke)
        del stroke

    if not image_name:
        cv2.imwrite("pad_formule_custom.jpg", img)
    else:
        cv2.imwrite(image_name, img)
    return img, formule_traces, scale, use_limit


def main(args):
    if args.f_file_name != None:
        formule_x_min, formule_x_max, formule_y_min, formule_y_max = get_trace(args.f_file_name,
                                                                               args.save_trace_file)
        img, padding_w, padding_h, _ = draw(args.save_trace_file, args.coord_list, args.image_name,
                                            args.image_size, args.formule_pad, args.line_width)
    else:
        img, padding_w, padding_h, _ = draw(args.save_trace_file, args.coord_list, args.image_name,
                                            args.image_size, args.formule_pad, args.line_width)

    cv2.imwrite(args.image_name, img)


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=800)
    parser.add_argument('--formule_pad', type=int, default=200)
    parser.add_argument('--line_width', type=int, default=2)
    parser.add_argument('--image_resize', default=300, type=int)
    parser.add_argument('--coord_list', default=None)
    parser.add_argument('--image_name', type=str, default='formule.jpg')
    parser.add_argument('--f_file_name',
                        default="/home/nd/project/ND_OCR/fp/ohwfp001-160809-ef-1608091119-u2002545310far-0000.dat")
    parser.add_argument('--save_trace_file',
                        default="/home/nd/project/ND_OCR/fp_image/ohwfp001-160809-ef-1608091119-u2002545310far-0000_0.dat")
    return parser.parse_args()


if __name__ == '__main__':
    # main(parse_args())
    trace_file = "/home/hyer/Downloads/99U_Downloads/2020-math-math-2018-01-11-14-39-20-348936-2756e28e-f69a-11e7-b189-9c5c8e8f37ca.txt"
    with open(trace_file) as f:
        data = f.readlines()
        formule_traces = []
        stroke = []
        for line in data:
            if line != "-10000 -10000\n":
                x = int(line.strip().split(" ")[0])
                y = int(line.strip().split(" ")[1])
                stroke.append((x, y))
            else:
                formule_traces.append(stroke)
                del stroke
                stroke = []

        img, padding_w, padding_h, formule_traces = draw_custom(formule_traces, image_name="traces.jpg")
        cv2.imwrite("traces.jpg", img)
        cv2.imshow("img", img)
        cv2.waitKey()

# f_file_name = "/home/nd/project/ND_OCR/fp/ohwfp001-160809-ef-1608091119-u2002545310far-0000.dat"
# save_trace_file = "/home/nd/project/ND_OCR/fp_image/ohwfp001-160809-ef-1608091119-u2002545310far-0000_0.dat"
# formule_x_min, formule_x_max, formule_y_min, formule_y_max = get_trace(f_file_name, save_trace_file)
# image_name = "suhuijia.jpg"
# image_size = 800
# formule_pad = 200
# line_width = 2
# img, padding_w, padding_h = draw(save_trace_file, image_name, image_size, formule_pad, line_width)
# cv2.imwrite(image_name, img)
