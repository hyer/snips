#coding=utf-8
'''
@1. 每个stroke属于哪些boxes； stroke_id:[1, 2]：每个stroke的轨迹点在各个box里面的占比超过阈值：80%（判断点是否在矩形框里面）
@2. 每个stroke所属的bbox超过1，则这些box里面选一个最可能的box，规则：该笔画的长边占各个box对应边的比例最大的，则属于该box；

consider: 漏检，误检，根号，交叉区域大
'''
import numpy as np


in_box_thres = 0.8

def dot_in_box(pt, box):
    '''
    判断坐标点是否在box里面
    Args:
        pt:
        box:

    Returns:

    '''
    if box[0] < pt[0] < box[2] and box[1] < pt[1] < box[3]:
        return True
    else:
        return False

def stroke_in_box(stroke, box, thres):
    '''
    判断笔画是否在box里面
    Args:
        stroke:
        box:
        thres:

    Returns:

    '''
    count = 0
    for pt in stroke:
        if dot_in_box(pt, box):
            count += 1
    if float(count)/len(stroke) >= thres:
        return True
    else:
        return False

def get_stroke_hw(stroke):
    stk_np = np.array(stroke)
    stroke_w = stk_np[:,0].max() - stk_np[:,0].min()
    stroke_h = stk_np[:,1].max() - stk_np[:,1].min()
    return stroke_h, stroke_w

def get_box_tight(traces, stroke_ids):
    stk_x1s = []
    stk_y1s = []
    stk_x2s = []
    stk_y2s = []

    for stk_id in stroke_ids:
        strk = traces[stk_id]
        stk_np = np.array(strk)
        stk_x1s.append(stk_np[:,0].min())
        stk_y1s.append(stk_np[:,1].min())
        stk_x2s.append(stk_np[:,0].max())
        stk_y2s.append(stk_np[:,1].max())

    return [min(stk_x1s)-2, min(stk_y1s)-2, max(stk_x2s)+2, max(stk_y2s)+2]


def boxes_tighted(boxes_stks, traces):
    boxes_tight = []
    for stks in boxes_stks:
        box_cord = get_box_tight(traces, stks)
        boxes_tight.append(box_cord)

    return boxes_tight

if __name__ == '__main__':
    boxes = [[374.3851852416992, 353.8599166870117, 479.2382278442383, 485.3978958129883],
            # [200.05406951904297, 338.37522888183594, 365.5789871215821, 491.2609405517578],
            [504.3486785888672, 295.7617874145507, 603.9088897705079, 492.8516769409179]]

    traces = [[(287, 354), (273, 346), (265, 348), (252, 358), (231, 385), (212, 429), (208, 469), (219, 492), (244, 490), (271, 460), (290, 423), (300, 394), (300, 390), (294, 402), (292, 427), (302, 442), (323, 448), (348, 442), (354, 435)], [(383, 421), (394, 417), (417, 412), (442, 408), (460, 402), (471, 396), (469, 392)], [(431, 360), (429, 373), (429, 410), (431, 433), (435, 462), (435, 477), (437, 477), (435, 477)], [(515, 308), (515, 306), (515, 312), (515, 333), (517, 365), (521, 404), (525, 435), (525, 452), (525, 456), (523, 448), (527, 427), (535, 406), (552, 398), (569, 402), (583, 417), (592, 435), (590, 458), (571, 481), (548, 490), (535, 485), (533, 473), (535, 479), (535, 460)]]

    stroke_boxes = [] # 每个笔画属于哪些box(存的是box的索引id）

    #@1
    for stroke_idx, stroke in enumerate(traces):
        box_inter = []
        for box_idx, box in enumerate(boxes):
            if stroke_in_box(stroke, box, in_box_thres):
                box_inter.append(box_idx)
        # if len(box_inter) > 0:
        stroke_boxes.append(box_inter)
        del box_inter

    print "stroke box id:", stroke_boxes

    #@2
    stroke_boxes_filterd = []
    for stroke_idx, box_idxs in enumerate(stroke_boxes):
        if len(box_idxs) == 0:
            stroke_boxes_filterd.append(None)
            continue
        if len(box_idxs) > 1:
            stroke_h, stroke_w = get_stroke_hw(traces[stroke_idx])
            box_hs= []
            box_ws= []
            for box_id in box_idxs:
                box_hs.append(abs(boxes[box_id][1] -boxes[box_id][3]))
                box_ws.append(abs(boxes[box_id][0] -boxes[box_id][2]))

            if stroke_h >= stroke_w:
                id = np.argmax(float(stroke_h)/np.array(box_hs))
                stroke_boxes_filterd.append(box_idxs[id])
                continue
            else:
                id = np.argmax(float(stroke_w)/np.array(box_ws))
                stroke_boxes_filterd.append(box_idxs[id])
                continue
        stroke_boxes_filterd.append(box_idxs[0])

    # @3 合并属于同一个box的笔画，生成tight box
    box_num = len(boxes)
    boxes_stks = []
    # for box_idx in boxes:
    for box_idx in range(box_num):
        def find(x, y):
            return [ a for a in range(len(y)) if y[a] == x]

        stk_idxs = find(box_idx, stroke_boxes_filterd)
        if len(stk_idxs) > 0:
            # print stk_idxs
            boxes_stks.append(stk_idxs)

    print "box_stks:", boxes_stks

    boxes_tight = boxes_tighted(boxes_stks, traces)
    print "boxes tight", boxes_tight