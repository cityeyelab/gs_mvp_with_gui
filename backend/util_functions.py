import cv2
import numpy as np

def get_bbox_conf(data):
    thr = 5
    if 0<= data < thr:
        result = 1
    elif thr< data <=2*thr:
        result = 2 + -1*(1/thr)*data
    else:
        result = 0
    return result


def cal_min_IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    
    if box1_area == 0 or box2_area == 0:
        iou = 0
    else:
        iou = inter / min(box1_area, box2_area)

    return iou



def cal_IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    
    if box1_area == 0 or box2_area == 0:
        iou = 0
    else:
        iou = inter / (box1_area + box2_area - inter)

    return iou


def add_lst_vec(lst1, lst2):
    lst1 = lst1.copy()
    lst2 = lst2.copy()
    return [(lst1[i]+lst2[i]) for i in range(0, len(lst1))]

def subtract_lst_vec(lst1, lst2):
    lst1 = lst1.copy()
    lst2 = lst2.copy()
    return [(lst1[i]-lst2[i]) for i in range(0, len(lst1))]

def mult_const_lst_vec(lst, const):
    lst = lst.copy()
    return [(const*lst[i]) for i in range(0, len(lst))]

def get_center_pt(box_info):
    return [(box_info[0]+box_info[2])/2, (box_info[1]+box_info[3])/2]


wh_ratio_x_thr_a1 = 0.7
y2_portion_a1 = 0.8
slope_a1 = -2*(10*y2_portion_a1-1)
y_intercept_a1 = 2*10*y2_portion_a1 - 1
def get_center_pt_a1(box_info):
    width = box_info[2] - box_info[0]
    height = box_info[3] - box_info[1]
    wh_ratio = height/width
    if wh_ratio < wh_ratio_x_thr_a1:
        ct_pt = [(box_info[0]+box_info[2])/2, ((1- y2_portion_a1)*box_info[1] + y2_portion_a1*box_info[3])]
    elif wh_ratio_x_thr_a1 < wh_ratio <= 1.0:
        cal_num = slope_a1*wh_ratio + y_intercept_a1
        ct_pt = [(box_info[0]+box_info[2])/2, (1*box_info[1] + cal_num*box_info[3])/(1 + cal_num)]
    else:
        ct_pt = [(box_info[0]+box_info[2])/2, (box_info[1] + box_info[3])/2]
    return ct_pt


area3_y2_portion = 0.9
def get_center_pt_a3(box_info):
    ct_pt = [(box_info[0]+box_info[2])/2, ((1- area3_y2_portion)*box_info[1] + area3_y2_portion*box_info[3])]
    return ct_pt

wh_ratio_x_thr_a4 = 0.7
y2_portion_a4 = 0.8
slope_a4 = -2*(10*y2_portion_a4-1)
y_intercept_a4 = 2*10*y2_portion_a4 - 1
def get_center_pt_a4(box_info):
    width = box_info[2] - box_info[0]
    height = box_info[3] - box_info[1]
    wh_ratio = height/width
    if wh_ratio < wh_ratio_x_thr_a4:
        ct_pt = [(box_info[0]+box_info[2])/2, ((1- y2_portion_a4)*box_info[1] + y2_portion_a4*box_info[3])]
    elif wh_ratio_x_thr_a4 < wh_ratio <= 1.0:
        cal_num = slope_a4*wh_ratio + y_intercept_a4
        ct_pt = [(box_info[0]+box_info[2])/2, (1*box_info[1] + cal_num*box_info[3])/(1 + cal_num)]
    else:
        ct_pt = [(box_info[0]+box_info[2])/2, (box_info[1] + box_info[3])/2]
    return ct_pt

trk_fn_get_ct_pt_lst = [get_center_pt_a1, get_center_pt_a3, get_center_pt_a4]

def filter_out_low_conf(dets, conf_thr):
    return [dets[j] for j in range(0, len(dets))  if (dets[j][4] >  conf_thr)] 





def merge_multiple_dets(dets):
    dets = dets.copy()
    len_dets = len(dets)
    conf_list = [dets[i][4] for i in range(0, len_dets)]
    weights_denominator = sum(conf_list)
    weights_list = [dets[i][4]/weights_denominator for i in range(0, len_dets)]

    
    x1 = sum([ weights_list[i] * dets[i][0] for i in range(0, len_dets)])
    y1 = sum([ weights_list[i] * dets[i][1] for i in range(0, len_dets)])
    x2 = sum([ weights_list[i] * dets[i][2] for i in range(0, len_dets)])
    y2 = sum([ weights_list[i] * dets[i][3] for i in range(0, len_dets)])
    
    conf =  weights_denominator/len_dets

    max_conf_idx = conf_list.index(max(conf_list))
    cls = dets[max_conf_idx][5]

    new_det= [x1, y1, x2, y2, conf, cls]

    return new_det



def check_bbox_inclusion(bbox1, bbox2):
    result = False
    result1 = False
    result2 = False

    bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2, = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    bbox2_x1, bbox2_y1, bbox2_x2, bbox2_y2, = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w1 = bbox1_x2 - bbox1_x1
    h1 = bbox1_y2 - bbox1_y1
    w2 = bbox2_x2 - bbox2_x1
    h2 = bbox2_y2 - bbox2_y1

    wh_ratio1 = h1/w1
    wh_ratio2 = h2/w2

    ratio_ratio = wh_ratio1 / wh_ratio2

    ratio_const = 0.2
    if 1. - ratio_const  < ratio_ratio < 1. + ratio_const:
        result1 = True

    result = False
    box1_area = (bbox1_x2 - bbox1_x1 + 1) * (bbox1_y2 - bbox1_y1 + 1)
    box2_area = (bbox2_x2 - bbox2_x1 + 1) * (bbox2_y2 - bbox2_y1 + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(bbox1_x1, bbox2_x1)
    y1 = max(bbox1_y1, bbox2_y1)
    x2 = min(bbox1_x2, bbox2_x2)
    y2 = min(bbox1_y2, bbox2_y2)

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h


    if box1_area == 0 or box2_area == 0:
        iou = 0
    else:
        iou = inter / min(box1_area, box2_area)


    box_area_ratio = min(box1_area, box2_area) / max(box1_area, box2_area)

    if iou > 0.85 and box_area_ratio > 0.65:
        result2 = True

    if result1 and result2:
        result = True
    
    return result




def eliminate_dup(frame_det):
    this_frame = frame_det
    frame_del_lst = []

    for j in range(0, len(this_frame)-1):
        this_bbox = this_frame[j]

        this_cnt = 0
        this_dup_list = []
        dets_list = [this_bbox]
        for k in range(j+1, len(this_frame)):
            that_bbox = this_frame[k]
            inclusion = check_bbox_inclusion(this_bbox, that_bbox)
            if inclusion:
                this_dup_list.append([k, that_bbox])
                dets_list.append(that_bbox)
                if k not in frame_del_lst:
                    frame_del_lst.append(k)

                this_cnt += 1

        if this_cnt > 0 :
            new_det = merge_multiple_dets(dets_list)
            this_frame[j] = new_det.copy()

    frame_del_lst.sort()
    for m in range(0, len(frame_del_lst)):
        target_idx = frame_del_lst[len(frame_del_lst) - m - 1]
        del this_frame[target_idx]

    # return 



def xyxy_to_xywh(xyxy):
    xyxy = xyxy.copy()
    x1 = xyxy[0]
    y1 = xyxy[1]
    x2 = xyxy[2]
    y2 = xyxy[3]

    center_pt = get_center_pt(xyxy)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    return [center_pt[0], center_pt[1], w, h]

def xywh_to_xyxy(xywh):
    xywh = xywh.copy()
    x = xywh[0]
    y = xywh[1]
    w = xywh[2]
    h = xywh[3]

    x1 = ( x - w/2 )
    y1 = ( y - h/2 )
    x2 = ( x + w/2 )
    y2 = ( y + h/2 )

    return [x1, y1, x2, y2]



def cvt_cls_list_to_info_list(cls_list):
    ret = []
    for cls in cls_list:
        ret.append(cls.det_info_on_path)
    return ret

def subtract_img(img1, img2): #color img input
    img1 = img1.astype(np.int16)
    img2 = img2.astype(np.int16)
    img = np.abs(img1 - img2)
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img



def magnify_bbox(bbox, ratio):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    ct_pt = get_center_pt(bbox)

    diag_vec = subtract_lst_vec(ct_pt, [x1, y1])

    new_x1, new_y1 = subtract_lst_vec(ct_pt, [ratio*diag_vec[0], ratio*diag_vec[1]])
    new_x2, new_y2 = add_lst_vec(ct_pt, [ratio*diag_vec[0], ratio*diag_vec[1]])

    return [new_x1, new_y1, new_x2, new_y2]