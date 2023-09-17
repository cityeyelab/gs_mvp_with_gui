# from yolov5 import detect
import cv2
from collections import deque
import gc
from datetime import datetime
import numpy as np
from yolov5.detect_que_orig import inference
import pickle
import multiprocessing
# from queue import Queue
import time
# import threading
from multiprocessing import Queue
from math import sqrt

img_que = deque()


# image_que = Queue(200)
# result_que = Queue(200)


# cap = cv2.VideoCapture('intermediate_version/test.mp4')

# opt = parse_opt()
# detect.main(opt)
# detect.main(test_image)



# total_frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# cm_x = int(width/2)
# cm_y = int(height/2)

# primaray_batch_size = total_frame_cnt

# print('total frame cnt = ', total_frame_cnt)
# print('width = ' , width)

# scale_factor_x = int(width/640)
# scale_factor_y = int(height/640)

# frame_cnt = 0

# font =  cv2.FONT_HERSHEY_PLAIN



# need_inferenct = False
# # from yolov5.detect import inference
# # template = np.zeros((1080 + 100, 1920 + 200, 3), dtype=np.uint8)

# if need_inferenct:

#     start_time = datetime.now()

#     frame_cnt = 0

#     while True:

#         if frame_cnt == primaray_batch_size:
#             break
        
#         ret, frame = cap.read()

#         # template = np.zeros((1080 + 100, 1920 + 200, 3), dtype=np.uint8)

#         if not ret:
#             break

#         # template[ 50 : 1080 + 50, 100 : 1920 + 100, :] = frame


#         # inference_instance = inference()
#         # result = inference_instance.run(frame)



#         img_que.append(frame)

#         frame_cnt += 1

#     ##################################

    

#     inference_instance = inference()


#     result = inference_instance.run(img_que)
#     names = inference_instance.names

#     end_time = datetime.now()
#     elapsed_time = end_time - start_time
#     print('elapsed time : ', elapsed_time)
#     print('names = ', names)
#     print('result = ' , result)
#     print('len result = ' , len(result))

#     with open('result_rev1', 'wb') as f:
#         pickle.dump(result, f)
#     # np.save('result_rev1', result)
























# path_str = 'intermediate_version/test.mp4'

# path_str = 'test.mp4'

# path_str = 'sejong_night.mp4'
# path_str = 'sejong_test.mp4'
# path_str = 'test4.mp4'
# path_str = 'check1.mp4'

# print('this path = ' , path)

# path_str = path
image_que = Queue(200)
result_que = Queue(200)




# cap = cv2.VideoCapture(path_str + '.mp4')

# total_frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



# fps = cap.get(cv2.CAP_PROP_FPS)
# # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# # delay = round(1000/fps)
# out = cv2.VideoWriter('test4_combined.mp4', fourcc, fps, (vid_width, vid_height))

# print('total frame cnt = '  , total_frame_cnt)

cm_x = int(1920/2)
cm_y = int(1080/2)



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

# def get_center_pt(box_info):
#     return (int((box_info[0]+box_info[2])/2), int((box_info[1]+box_info[2])/2))

def get_center_pt(box_info):
    return [(box_info[0]+box_info[2])/2, (box_info[1]+box_info[3])/2]


def filter_out_low_conf(dets, conf_thr):
    # filtered_out_list = [dets[j] for j in range(0, len(dets))  if (dets[j][4] <= conf_thr)]
    # if len(filtered_out_list) > 0:
    #     print('filtered out flist = ', filtered_out_list)

    # dets = [[dets[i][j] for j in range(0, len(dets[i]))  if (dets[j][4] >  conf_thr)] for i in range(0, len(dets))]
    return [dets[j] for j in range(0, len(dets))  if (dets[j][4] >  conf_thr)] 


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
    # box = (x1, y1, x2, y2)
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
        # iou = inter / min(box1_area, box2_area)
        iou = inter / min(box1_area, box2_area)


    box_area_ratio = min(box1_area, box2_area) / max(box1_area, box2_area)

    if iou > 0.85 and box_area_ratio > 0.65:
        result2 = True

    if result1 and result2:
        result = True
    
    return result




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



def eliminate_dup(frame_det):
    this_frame = frame_det
    frame_del_lst = []

    for j in range(0, len(this_frame)-1):
        this_bbox = this_frame[j]
        # cls1 = this_bbox[5]
        this_cnt = 0
        this_dup_list = []
        dets_list = [this_bbox]
        for k in range(j+1, len(this_frame)):
            that_bbox = this_frame[k]
            inclusion = check_bbox_inclusion(this_bbox, that_bbox)
            if inclusion:
                this_dup_list.append([k, that_bbox])
                dets_list.append(that_bbox)
                # print('i, j, k = ' , i, ' ',  j, ' ',k )
                if k not in frame_del_lst:
                    frame_del_lst.append(k)

                this_cnt += 1
                # if this_cnt > 1:
                #     print('********************')
                #     print('3 or more bbox are dup!!')
                #     print('frame_cnt = ' , i)
                #     print('det1 = ' , j)
                #     print('det2 = ', k)
                #     print('info = ' , incl_det_list[-1])
                #     print('info = ' , incl_det_list[-2])
                #     print('********************')

        # new_det = merge_two_det(this_bbox, that_bbox)

        # new_det = merge_multiple_dets(dets_list)
        # incl_det_list.append([i, j, this_dup_list, new_det]) #frame_cnt, det_cnt, [dup list], new_det

        if this_cnt > 0 :
            new_det = merge_multiple_dets(dets_list)
            # incl_det_list.append([i, j, this_dup_list, new_det]) #frame_cnt, det_cnt, [dup list], new_det


            this_frame[j] = new_det.copy()
            # new_results[j].append(new_det)

                # new_results[i][0].pop(j)
                # new_results[i][0].pop(k)
                # new_results.append(new_det)
                
                # this_frame[j] = new_det
                # this_frame.pop(k)
    frame_del_lst.sort()
    for m in range(0, len(frame_del_lst)):
        target_idx = frame_del_lst[len(frame_del_lst) - m - 1]
        del this_frame[target_idx]

    # return 

## 멀리 있다가 가까워지는거

## 이상한 놈 블락커

## 영상기반 motion detector -> 영상 필요

def smooth_list(lst, window_size):
    lst = lst.copy()
    len_lst = len(lst)
    # divider = 2 * window_size + 1
    result_lst = []
    for i in range(0, len_lst):
        sliced = lst[ max(0, (i-window_size))   :  min(len_lst, (i + window_size + 1 )) ].copy()
        result = sum(sliced)/len(sliced)
        result_lst.append(result)
    return result_lst


def local_minima_criterion(lst):
    len_lst = len(lst)
    result = False
    for i in range(0, len_lst-2):
        new = lst[i+2] - lst[i+1]
        former = lst[i+1] - lst[i]
        if new >=0 and former < 0:
            result = True
    return result


def get_avg_box(box_lst):
    len_boxes = len(box_lst)
    # for i in range(0, len_boxes):
    x1 = sum([box_lst[i][0] for i in range(0, len_boxes)])/len_boxes
    y1 = sum([box_lst[i][1] for i in range(0, len_boxes)])/len_boxes
    x2 = sum([box_lst[i][2] for i in range(0, len_boxes)])/len_boxes
    y2 = sum([box_lst[i][3] for i in range(0, len_boxes)])/len_boxes

    return [x1, y1, x2, y2]





class OcclusionMatchData():
    def __init__(self, id) -> None:
        self.id = id
        self.host_id = None
        self.guest_ids = []
        self.candidate_ids = []
        self.online = True
        


class DetectionInfo():
    # _id = 0

    def __init__(self, id) -> None:
        self.det_info_on_path = []
        self.det_info_on_path_raw = []
        self.distance_from_cm = []
        self.id = id # private arg may be better
        self.avg_dia_vec = [0, 0]
        self.avg_vel_vec = [0, 0]
        self.latest_estimated_box = [0., 0., 0., 0.]
        self.delegate_lst = []

    def __repr__(self):
        return (self.__class__.__qualname__ + f"(id={self.id!r})")
    
    def latest_box_wh_ratio(self):
        latest_box = self.det_info_on_path_raw[-1][2]
        x1 = latest_box[0]
        y1 = latest_box[1]
        x2 = latest_box[2]
        y2 = latest_box[3]
        return (y2 - y1)/(x2 - x1)

    def latest_box_area(self):
        latest_box = self.det_info_on_path_raw[-1][2]
        x1 = latest_box[0]
        y1 = latest_box[1]
        x2 = latest_box[2]
        y2 = latest_box[3]
        return (x2 - x1) * (y2 - y1)


    
    def get_id(self):
        return self.id
    
    def get_avg_dia_vec(self):
        # if len(self.det_info_on_path) > 0:
        # print(self.det_info_on_path)
        targets = self.det_info_on_path[-3:]
        dia_vec_list = []
        for i in range(0, len(targets)):
            target = targets[i][2][0:4]
            dia_vec = np.array([target[2], target[3]]) - np.array([target[0], target[1]])
            dia_vec_list.append(dia_vec)
        # print('dia vec list = ' , dia_vec_list)
        self.avg_dia_vec = np.mean(dia_vec_list, axis=0).tolist()
        # print('avg dia vec = ' , avg_dia_vec)
        # avg_dia_vec = np.mean()

    def after_occ_box(self):
        last_three_boxes = self.det_info_on_path[-3:]
        last_three_boxes = [last_three_boxes[i][2][0:4] for i in range(0, len(last_three_boxes))]
        result_box = get_avg_box(last_three_boxes)
        ct_pt_result = get_center_pt(result_box)
        ct_pt_last = get_center_pt(self.det_info_on_path[-1][2])
        drt_vec = subtract_lst_vec(ct_pt_last, ct_pt_result)
        result1 = add_lst_vec(result_box[0:2], drt_vec)
        result2 = add_lst_vec(result_box[2:4], drt_vec)
        
        return result1 + result2

    def update_avg_vel_vec(self):
        #가속도적용
        targets = self.det_info_on_path[-4:].copy()
        x0, y0 = (targets[0][2][0]+targets[0][2][2])/2 , (targets[0][2][1]+targets[0][2][3])/2
        xn, yn = (targets[-1][2][0]+targets[-1][2][2])/2 , (targets[-1][2][1]+targets[-1][2][3])/2
        len_targets = len(targets)

        if len(targets) > 1:
            x_n1, y_n1 = (targets[-2][2][0]+targets[-2][2][2])/2 , (targets[-2][2][1]+targets[-2][2][3])/2
            x_1, y_1 = (targets[1][2][0]+targets[1][2][2])/2 , (targets[1][2][1]+targets[1][2][3])/2
            acc = [ 0.5*((xn-x_n1-x_1+x0)/(len_targets-1))    , 0.5*((yn-y_n1-y_1+y0)/(len_targets-1)) ]
            # acc = [0, 0]
        else:
            acc = [0, 0]
        
        vx, vy = (xn - x0)/len_targets , (yn - y0)/len_targets
        # if abs(vx) > 200:
        #     print('vx overflow')
        #     print('targets = ' , targets)
        # # self.avg_vel_vec = [vx, vy]
        self.avg_vel_vec = add_lst_vec([vx, vy], acc)

    def append_cm_distance(self):
        last_box = self.det_info_on_path[-1][2][0:4]
        x1 = last_box[0]
        y1 = last_box[1]
        x2 = last_box[2]
        y2 = last_box[3]
        new_pos = [(x1+x2)/2, (y1+y2)/2]
        dist = sqrt( (cm_x - new_pos[0] )**2 + (cm_y - new_pos[1])**2 )
        self.distance_from_cm.append(dist)

    def criterion_cm_complete(self):
        dists = self.distance_from_cm.copy()
        len_dists = len(dists)
        window_size = int(len_dists/4)
        smoothed_list = smooth_list(dists, window_size)
        smoothed_list = smooth_list(dists, 1)
        return local_minima_criterion(smoothed_list)




    def get_line_distance_start_to_end(self):
        targets = self.det_info_on_path
        start = targets[0][2][0:4]
        end = targets[-1][2][0:4]
        start_pt = ((start[0]+start[2])/2, (start[1]+start[3])/2)
        end_pt = ((end[0]+end[2])/2, (end[1]+end[3])/2)
        dist = sqrt((start_pt[0] - end_pt[0])**2 + (start_pt[1] - end_pt[1])**2)
        return dist
    
    def estimate_pos(self, cur_frame_cnt):

        # 높이에 따른 비박스 크기 변화 적용
        vel_vec = self.avg_vel_vec
        # vel_vec = [0, 0]
        h_diff = -1*vel_vec[1]
        h_diff_const = (1-(h_diff/(2*1080)))
        # h_diff_const = 1
        last_box = self.det_info_on_path[-1][2][0:4]
        last_box_xywh = xyxy_to_xywh(last_box)
        last_box_center = last_box_xywh[0:2]
        last_box_dia_vec_half = mult_const_lst_vec(last_box_xywh[2:4], 0.5)
        scaled_last_box_dia_vec_half = mult_const_lst_vec(last_box_dia_vec_half, h_diff_const)
        frame_diff = cur_frame_cnt - self.det_info_on_path[-1][0]
        estimated_center = add_lst_vec(last_box_center , mult_const_lst_vec(vel_vec, frame_diff))
        estimated_x1y1 = subtract_lst_vec(estimated_center, scaled_last_box_dia_vec_half)
        estimated_x2y2 = add_lst_vec(estimated_center, scaled_last_box_dia_vec_half)
        estimated_pos = [estimated_x1y1[0], estimated_x1y1[1], estimated_x2y2[0], estimated_x2y2[1]]
        return estimated_pos


    def append_info(self, additional_info):

        # self.get_avg_dia_vec()
        additional_info = additional_info.copy()
        self.det_info_on_path_raw.append(additional_info)

        # additional_info_raw = [frame_cnt, index, det_info]
        if len(self.det_info_on_path) > 0:
            # print(self.det_info_on_path[-1][2][0:4])
            # print(additional_info[2][0:4])
            # additional_info[2][0:4] = np.mean([self.det_info_on_path[-1][2][0:4], additional_info[2][0:4]], axis=0)
            
            alpha = 0.4#formers
            beta = 0.6 #new commer
            gamma = 0.4 #est box
            # additional_info[2][0:4] = alpha * self.det_info_on_path[-1][2][0:4] + beta * additional_info[2][0:4]

            additional_info_box_dia_vec = subtract_lst_vec([additional_info[2][2], additional_info[2][3]],
                                                        [additional_info[2][0], additional_info[2][1]])
            additional_info_box_dia_vec_half = mult_const_lst_vec(additional_info_box_dia_vec, 0.5)
            cp_x, cp_y = get_center_pt(additional_info[2][0:4])
            center_of_latest_est_box = get_center_pt(self.latest_estimated_box)
            # cp_x, cp_y = get_center_pt([cp_x, cp_y] + center_of_latest_est_box)
            center_point = add_lst_vec(mult_const_lst_vec([cp_x, cp_y], 0.6), mult_const_lst_vec(center_of_latest_est_box, 0.4))
            self.get_avg_dia_vec()
            avg_dia_vec_half = mult_const_lst_vec(self.avg_dia_vec, 0.5)
            vec_2_added = ( add_lst_vec( mult_const_lst_vec(avg_dia_vec_half, alpha),
                                                            mult_const_lst_vec(additional_info_box_dia_vec_half, beta) ) )
            new_box_top_left = subtract_lst_vec([center_point[0], center_point[1]], vec_2_added)
            new_box_bottom_right = add_lst_vec([center_point[0], center_point[1]], vec_2_added )

            additional_info[2][0:4] = [new_box_top_left[0], new_box_top_left[1], new_box_bottom_right[0], new_box_bottom_right[1]]
        self.det_info_on_path.append(additional_info)
        self.update_avg_vel_vec()
        self.append_cm_distance()
        latest_frame_num = additional_info[0]
        self.latest_estimated_box = self.estimate_pos(latest_frame_num + 1)


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

### lists ###
under_examination = []
missing_objects = []
unidentified_objects = []
complete = []
# complete_list = []
all_view_list = []
abandon_list = []



def cvt_cls_list_to_info_list(cls_list):
    ret = []
    for cls in cls_list:
        ret.append(cls.det_info_on_path)
    return ret


unique_id_cnt = 0



def mapping_iou(input_data):

    # print('mapping iou data = ' , input_data)
    
    data = input_data.copy()

    if type(data) == list:
        data = np.array(data)
    
    iou_thr = 0.10

    data[np.where(data < iou_thr)] = 0


    iou_map = {}

    h, w = data.shape

    remaining_starter_idx = [k for k in range(0, h)]
    remaining_target_idx = [k for k in range(0, w)] 

    while True:
    
        starter_idx, target_idx = np.unravel_index(data.argmax(), data.shape)

        iou_map[starter_idx] = target_idx

        data[starter_idx, :] = 0
        data[:, target_idx] = 0

        remaining_starter_idx.remove(starter_idx)
        remaining_target_idx.remove(target_idx)

        if np.sum(data) == 0:
            break
    
    return iou_map, remaining_starter_idx, remaining_target_idx



font =  cv2.FONT_HERSHEY_PLAIN



total_start_time = time.time()

total_obj_cnt = 0
complete_cnt = 0
missing_cnt = 0
abandon_cnt = 0

need_view_display = True


def subtract_img(img1, img2): #color img input
    img1 = img1.astype(np.int16)
    img2 = img2.astype(np.int16)
    img = np.abs(img1 - img2)
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img



def cal_inter_box(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_box = [x1, y1, x2, y2]

    if x2 > x1 and y2 > y1:
        inter_box_area = (inter_box[2] - inter_box[0] + 1) * (inter_box[3] - inter_box[1] + 1)

        if inter_box_area/box1_area > 0.5 or inter_box_area/box2_area > 0.5:
            inter_box = [2, 2, 1, 1] #nothing



    return inter_box




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


def tracker(image_que, result_que, proc_num):

    
    # cap = cv2.VideoCapture(path + '.mp4')

    # total_frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # # fps = cap.get(cv2.CAP_PROP_FPS)
    # # print('fps = ' , fps)
    # fps = 30
    # # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # # delay = round(1000/fps)
    # out = cv2.VideoWriter(path +'_output2.mp4', fourcc, fps, (vid_width, vid_height))



    frame_cnt = 0


# for frame_cnt in range(0, len(results)):
    while True:
        dets = result_que.get()
        if dets == None:
            # out.release()
            print('tracker None input break')
            break
        # dets = filter_out_low_conf(dets, 0.25)
        # eliminate_dup(dets)

        # print('dets = ', dets)

        
        # ret, frame = cap.read()
        frame = image_que.get()
        # print('frame from img que = ', frame)
        # if type(frame) != np.ndarray and frame == None:
        #     out.release()
        #     print('tracker not ret break')
        #     break

        frame_cnt += 1


        # all_view_list.append({'frame_cnt': frame_cnt})


        if frame_cnt%10 == 0:
            print('frame cnt = ' , frame_cnt)


        cv2.putText(frame, str(frame_cnt), (80, 80), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        for data in dets:
            det = data
            p_x1, p_y1 = int(det[0]), int(det[1])
            p_x2, p_y2 = int(det[2]), int(det[3])
            frame = cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (255, 0, 0), 2)

        cv2.namedWindow('frame'+str(proc_num), cv2.WINDOW_NORMAL)
        cv2.imshow('frame'+str(proc_num), frame)
        # if frame_cnt > 800:
        #     wk = 0
        # else:
        #     wk = 1
        if cv2.waitKey(1) == 27:
            break
        
        # cv2.imshow('frame', result_frame)
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(100) == 27:``
        
        # if frame_cnt <= 1368:
        # out.write(frame)  
            
        ######################################################################
        

        #############################################################
        
        
        # if ret:

    print('video output released!')
    cv2.destroyAllWindows()
    # out.release()
    print('tracker breaked!')




def video_load(image_que, image_que2, path):
    # cap_loader = cv2.VideoCapture(path + '.mp4')
    # cap_loader = cv2.VideoCapture('rtsp://admin:self1004@@118.37.223.147:8522/live/main8')
    cap_loader = cv2.VideoCapture(path)
    # cap_loader.set(cv2.CAP_PROP_POS_FRAMES, 1600)
    video_frame_cnt = 0
    while True:
        
        # print('data load')
        _, _ = cap_loader.read()
        # _, _ = cap_loader.read()
        ret, frame = cap_loader.read()

        # video_frame_cnt += 1
        # if video_frame_cnt == 60:
        #     image_que.put(None)
        #     cap_loader.release()
        #     print('loader break')
        #     break


        if not ret:
            image_que.put(None)
            image_que2.put(None)
            cap_loader.release()
            print('loader break')
            break
        if ret:
            image_que.put(frame)
            image_que2.put(frame)

    print('loader breaked!!!')


# def yolo_inference(image_que, result_que):
#     inference_instance = inference()
#     y_s = time.time()
#     inference_instance.run(image_que, result_que)
#     print('yolo inference breaked!')
#     y_e = time.time()

#     y_elapsed_time = y_e - y_s
#     print('yolo elapsed time = ', y_elapsed_time)
#     # while True:
#     #     result = inference_instance.run(image_que, result_que)
#     #     print('result = ' , result)


def yolo_inference(image_que, result_que):
    inference_instance = inference()
    y_s = time.time()
    inference_instance.run(image_que, result_que)
    print('yolo inference breaked!')
    y_e = time.time()

    y_elapsed_time = y_e - y_s
    print('yolo elapsed time = ', y_elapsed_time)

need_write = False

# paths = ['intermediate_version/gs_new_city/camera1/2023-08-04 11_57_07.000',
#         'intermediate_version/gs_new_city/camera1/2023-08-04 13_07_07.000',
#         'intermediate_version/gs_new_city/camera2/2023-08-04 11_57_08.000',
#         'intermediate_version/gs_new_city/camera2/2023-08-04 13_07_08.000',
#         'intermediate_version/gs_new_city/camera8/2023-08-04 11_57_08.000',
#         'intermediate_version/gs_new_city/camera8/2023-08-04 13_07_08.000',
#         'intermediate_version/bc_okgil/camera7/2023-08-02 17_07_08.000' ,
#         'intermediate_version/bc_okgil/camera7/2023-08-02 17_27_08.000',
#         'intermediate_version/bc_okgil/camera8/2023-08-02 17_07_09.000',
#         'intermediate_version/bc_okgil/camera8/2023-08-02 17_27_08.000',
#         'intermediate_version/bc_okgil/camera9/2023-08-02 17_07_09.000',
#         'intermediate_version/bc_okgil/camera9/2023-08-02 17_27_08.000']

paths = ['rtsp://admin:self1004@@118.37.223.147:8522/live/main6', 
         'rtsp://admin:self1004@@118.37.223.147:8522/live/main7', 
         'rtsp://admin:self1004@@118.37.223.147:8522/live/main8',
         'rtsp://admin:self1004@@118.37.223.147:8522/live/main9']

# paths = ['rtsp://admin:self1004@@118.37.223.147:8522/live/main8']

# paths = ['intermediate_version/gs_new_city/camera2/2023-08-04 13_07_08.000',
#         'intermediate_version/gs_new_city/camera8/2023-08-04 11_57_08.000',
#         'intermediate_version/gs_new_city/camera8/2023-08-04 13_07_08.000',
#         'intermediate_version/bc_okgil/camera7/2023-08-02 17_07_08.000' ,
#         'intermediate_version/bc_okgil/camera7/2023-08-02 17_27_08.000',
#         'intermediate_version/bc_okgil/camera8/2023-08-02 17_07_09.000',
#         'intermediate_version/bc_okgil/camera8/2023-08-02 17_27_08.000',
#         'intermediate_version/bc_okgil/camera9/2023-08-02 17_07_09.000',
#         'intermediate_version/bc_okgil/camera9/2023-08-02 17_27_08.000']

if __name__ == '__main__':

    image_que_lst_proc = []
    image_que_lst_draw = []
    result_que_lst = []
    video_loader_lst = []
    yolo_inference_lst = []
    tracking_proc_lst = []
    yolo_instance_lst = []

    # for i in range(0, len(paths)):
    #     yolo_instance_lst.append(inference())

    for i in range(0, len(paths)):
        # print('check')
        image_que_lst_proc.append(Queue(2000))
        image_que_lst_draw.append(Queue(2000))
        result_que_lst.append(Queue(2000))
    # for i in range(0, len(paths)):
        video_loader_lst.append(multiprocessing.Process(target=video_load, args=(image_que_lst_proc[i], image_que_lst_draw[i], paths[i]), daemon=False))
        yolo_inference_lst.append(multiprocessing.Process(target=yolo_inference, args=(image_que_lst_proc[i], result_que_lst[i]), daemon=False))
        tracking_proc_lst.append(multiprocessing.Process(target=tracker, args=(image_que_lst_draw[i], result_que_lst[i], i), daemon=False))

    
    



    # for i in range(0, len(paths)):
        
        # print('check!')

    # video_loader_thread_lst = []


    for i in range(0, len(paths)):
        video_loader_lst[i].start()
        yolo_inference_lst[i].start()
        tracking_proc_lst[i].start()
        # print('check!')

    for i in range(0, len(paths)):
        video_loader_lst[i].join()
        yolo_inference_lst[i].join()
        tracking_proc_lst[i].join()

    # image_que = Queue(200)
    # image_que2 = Queue(200)
    # result_que = Queue(200)

    # video_loader_thread = multiprocessing.Process(target=video_load, args=(image_que, image_que2, path), daemon=False)
    # yolo_inference_thread = multiprocessing.Process(target=yolo_inference, args=(image_que, result_que,), daemon=False)
    # tracking_thread = multiprocessing.Process(target=tracker, args=(image_que2, result_que, path), daemon=False)

    # video_loader_thread.start()
    # yolo_inference_thread.start()
    # tracking_thread.start()
    # video_loader_thread.join()
    # yolo_inference_thread.join()
    # tracking_thread.join()

    for i in range(0, len(paths)):
        video_loader_lst[i].close()
        yolo_inference_lst[i].close()
        tracking_proc_lst[i].close()

    print('main proc end')
