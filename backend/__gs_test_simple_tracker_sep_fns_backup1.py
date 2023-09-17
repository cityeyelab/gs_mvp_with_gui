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

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"






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




font =  cv2.FONT_HERSHEY_PLAIN



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

touch_template = np.load('maps/2023-08-27_135132.npy')


# area4_map_lst = []
# area4_map_strings_lst = ['maps/2023-08-27_135132.npy']
# for i in range(0, len(area4_map_strings_lst)):
#     map_string = area4_map_strings_lst[i]
#     map = np.load(map_string)
#     area4_map_lst.append(map)

# all_template = [area4_map_lst]

area4_global_inout_map = np.load('area4_global_inout_map.npy')
area4_car_wash_waiting_map = np.load('area4_car_wash_waiting_map.npy')
area4_electric_vehicle_charging_map = np.load('area4_electric_vehicle_charging_map.npy')
area4_car_interior_washing_map = np.load('area4_car_interior_washing_map.npy')

area4_global_inout_map_img =np.zeros((1080, 1920, 3), np.uint8)
area4_global_inout_map_non_false_idx  = np.where(area4_global_inout_map==True)
area4_global_inout_map_img[area4_global_inout_map_non_false_idx] = (0,255,0)

area4_car_wash_waiting_map_img = np.zeros((1080, 1920, 3), np.uint8)
area4_car_wash_waiting_map_non_false_idx  = np.where(area4_car_wash_waiting_map==True)
area4_car_wash_waiting_map_img[area4_car_wash_waiting_map_non_false_idx] = (0,255,0)


area4_electric_vehicle_charging_map_img = np.zeros((1080, 1920, 3), np.uint8)
area4_electric_vehicle_charging_map_non_false_idx  = np.where(area4_electric_vehicle_charging_map==True)
area4_electric_vehicle_charging_map_img[area4_electric_vehicle_charging_map_non_false_idx] = (0,255,0)

area4_car_interior_washing_map_img = np.zeros((1080, 1920, 3), np.uint8)
area4_car_interior_washing_map_non_false_idx  = np.where(area4_car_interior_washing_map==True)
area4_car_interior_washing_map_img[area4_car_interior_washing_map_non_false_idx] = (0,255,0)


# area4_global_inout_map = np.load('area4_global_inout_map.npy')
# area4_global_inout_map_img =np.zeros((1080, 1920, 3), np.uint8)
# non_false_idx  = np.where(area4_global_inout_map==True)
# area4_global_inout_map_img[non_false_idx] = (255,0,0)






def tracker_area4(det_result_que, trk_result_que, draw_proc_result_que, proc_num):

    
    center_points_lst = []
    frame_cnt = 0
    car_cnt = 0

    area4_view_global_map = False
    area4_view_car_wash_waiting_map = False
    area4_view_car_interior_wash_map = False
    area4_view_electric_map = False
    area4_slope = 0.22451456310679613
    previous_put_data = {}

# for frame_cnt in range(0, len(results)):
    while True:
        dets = det_result_que.get()
        
        if dets == None:
            # out.release()
            print('tracker None input break!')
            break
        dets = filter_out_low_conf(dets, 0.25)
        eliminate_dup(dets)
        
        # print('frame cnt = ' , frame_cnt)
        # print('dets = ', dets)

        center_points_lst_orig = center_points_lst.copy()
        for data in center_points_lst_orig:
            data_orig = data
            center_point = data[0]
            for det in dets:
                det = magnify_bbox(det, 0.75)
                p_x1, p_y1 = det[0], det[1]
                p_x2, p_y2 = det[2], det[3]

                if (p_x1 < center_point[0] < p_x2) and (p_y1 < center_point[1] < p_y2):
                    # print('center points list = ' , center_points_lst)
                    # print('this center point = ', center_point)
                    # print('this det = ' , det[0:4])
                    # print('this det ct pt = ' , get_center_pt(det[0:4]))
                   
                    try:
                        # center_points_lst.remove(center_point)
                        center_points_lst.remove(data_orig)
                    except:
                        # print('remove fail')
                        # break
                        pass
        
        # print('before ct pts = ' , center_points_lst)
        for det in dets:
            ct_pt = get_center_pt(det[0:4]) # [x, y]
            # print('ct_pt = ', ct_pt) q
            if area4_global_inout_map[int(ct_pt[1]), int(ct_pt[0])] == True:
                center_points_lst.append([ct_pt, det[0:4], 0])
        # print('after ct pts = ' , center_points_lst)

        # car_cnt = len(center_points_lst)
        # dets_length = len(dets)
        
        # # center_points_lst.clear()
        # for det in dets:
        #     ct_pt = get_center_pt(det[0:4]) # [x, y]
        #     # print('ct_pt = ', ct_pt)
            
        #     center_points_lst.append(ct_pt)

        glb_in_cnt = 0
        car_wash_waiting_cnt = 0
        electric_vehicle_charging_waiting_cnt = 0
        car_interior_washing_waiting_cnt = 0

        put_data = {}
        pos_data = []

        for data in center_points_lst_orig:
            data[2] += 1

            conf = get_bbox_conf(data[2])
            if conf <= 0 :
                try:
                    center_points_lst.remove(data)
                except:
                    # print('remove fail 2')
                    pass

            center_point = data[0]
            if area4_global_inout_map[int(center_point[1]), int(center_point[0])] == True:
                glb_in_cnt += 1

                if area4_car_wash_waiting_map[int(center_point[1]), int(center_point[0])] == True:
                    pos_data.append((center_point[1] - area4_slope *(center_point[0])))
                    car_wash_waiting_cnt += 1
                elif area4_car_interior_washing_map[int(center_point[1]), int(center_point[0])] == True:
                    car_interior_washing_waiting_cnt += 1
                elif area4_electric_vehicle_charging_map[int(center_point[1]), int(center_point[0])] == True:
                    electric_vehicle_charging_waiting_cnt += 1


        put_data['pos_data'] = pos_data
        put_data.update({'area':4 , 'global_cnt': glb_in_cnt, 'car_washing_waiting_cnt': car_wash_waiting_cnt, 'car_interior_washing_waiting_cnt':car_interior_washing_waiting_cnt, 'electric_vehicle_charging_waiting_cnt':electric_vehicle_charging_waiting_cnt})
        if previous_put_data != put_data:
        # if True:
            trk_result_que.put(put_data)

        previous_put_data = put_data
        draw_put_data = {'cnt_lst': [glb_in_cnt, car_wash_waiting_cnt, electric_vehicle_charging_waiting_cnt, car_interior_washing_waiting_cnt], 'center_points_lst': center_points_lst, 'dets': dets }
        draw_proc_result_que.put(draw_put_data)
        
        # ret, frame = cap.read()
        # frame = image_que.get()
        # print('frame from img que = ', frame)
        # if type(frame) != np.ndarray and frame == None:
        #     out.release()
        #     print('tracker not ret break')
        #     break

        frame_cnt += 1



        # if frame_cnt%50 == 0:
        #     print('frame cnt = ' , frame_cnt)
        # if frame_cnt%50 == 0:    
        #     que_size = result_que.qsize()
        #     # Queue.qsize
        #     print('q size = ' , que_size)
        if det_result_que.qsize() > 10:
            print('Q size = ' , det_result_que.qsize())
            if det_result_que.full():
                print('Queue is full!!')





        ########################## draw #################################
        # cv2.putText(frame, 'frame cnt: '+str(frame_cnt), (80, 80), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # # cv2.putText(frame, 'car cnt: '+str(car_cnt), (80, 120), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # # cv2.putText(frame, 'dets cnt: '+str(dets_length), (80, 160), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

        # # glb_in_cnt = 0
        # # car_wash_waiting_cnt = 0
        # # electric_vehicle_charging_waiting_cnt = 0
        # # car_interior_washing_waiting_cnt = 0
        # cv2.putText(frame, 'car_num: '+str(glb_in_cnt), (80, 120), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, 'car_washing_waiting_cnt: '+str(car_wash_waiting_cnt), (80, 160), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, 'elec_vehicle_charging_cnt: '+str(electric_vehicle_charging_waiting_cnt), (80, 200), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, 'car_interior_washing_cnt: '+str(car_interior_washing_waiting_cnt), (80, 240), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # for data in dets:
        #     det = data
        #     p_x1, p_y1 = int(det[0]), int(det[1])
        #     p_x2, p_y2 = int(det[2]), int(det[3])
        #     frame = cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (255, 0, 0), 2)
        #     # det = magnify_bbox(data, 0.75)
        #     # p_x1, p_y1 = int(det[0]), int(det[1])
        #     # p_x2, p_y2 = int(det[2]), int(det[3])
        #     # frame = cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (155, 0, 0), 2)


        # # y = slope * x + k
        # for data in center_points_lst:
        #     center_point = data[0]
        #     cv2.line(frame, (int(center_point[0]), int(center_point[1])), (int(center_point[0]), int(center_point[1])), (0,255,0), thickness=12, lineType=None, shift=None)
        #     cv2.putText(frame, 'pos:: '+str(round((center_point[1] - area4_slope *(center_point[0])), 2) ), (int(center_point[0]), int(center_point[1]+10)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #     cv2.putText(frame, 'conf: '+str(get_bbox_conf(data[2])), (int(center_point[0]), int(center_point[1]+30)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # if area4_view_global_map:
        #     frame[area4_global_inout_map_non_false_idx] = cv2.addWeighted(frame, 0.5, area4_global_inout_map_img, 0.5, 0)[area4_global_inout_map_non_false_idx] 
        # elif area4_view_car_wash_waiting_map:
        #     frame[area4_car_wash_waiting_map_non_false_idx] = cv2.addWeighted(frame, 0.5, area4_car_wash_waiting_map_img, 0.5, 0)[area4_car_wash_waiting_map_non_false_idx] 
        # elif area4_view_electric_map:
        #     frame[area4_electric_vehicle_charging_map_non_false_idx] = cv2.addWeighted(frame, 0.5, area4_electric_vehicle_charging_map_img, 0.5, 0)[area4_electric_vehicle_charging_map_non_false_idx]
        # elif area4_view_car_interior_wash_map:
        #     frame[area4_car_interior_washing_map_non_false_idx]= cv2.addWeighted(frame, 0.5, area4_car_interior_washing_map_img, 0.5, 0)[area4_car_interior_washing_map_non_false_idx]

        # cv2.namedWindow('frame'+str(proc_num), cv2.WINDOW_NORMAL)
        # cv2.imshow('frame'+str(proc_num), frame)
        # # if frame_cnt > 800:
        # #     wk = 0
        # # else:
        # #     wk = 1

        # key  = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     break
        # elif key == ord('0'):
        #     area4_view_global_map = False
        #     area4_view_car_wash_waiting_map = False
        #     area4_view_car_interior_wash_map = False
        #     area4_view_electric_map = False
        # elif key == ord('1'):
        #     area4_view_global_map = True
        #     area4_view_car_wash_waiting_map = False
        #     area4_view_car_interior_wash_map = False
        #     area4_view_electric_map = False
        # elif key == ord('2'):
        #     area4_view_global_map = False
        #     area4_view_car_wash_waiting_map = True
        #     area4_view_car_interior_wash_map = False
        #     area4_view_electric_map = False
        # elif key == ord('3'):
        #     area4_view_global_map = False
        #     area4_view_car_wash_waiting_map = False
        #     area4_view_car_interior_wash_map = True
        #     area4_view_electric_map = False
        # elif key == ord('4'):
        #     area4_view_global_map = False
        #     area4_view_car_wash_waiting_map = False
        #     area4_view_car_interior_wash_map = False
        #     area4_view_electric_map = True


        ########################## draw #################################


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













area3_global_inout_map = np.load('area3_global_inout_map.npy')
area3_car_wash_waiting_map = np.load('area3_car_wash_waiting_map.npy')

area3_global_inout_map_img =np.zeros((1080, 1920, 3), np.uint8)
area3_global_inout_map_non_false_idx  = np.where(area3_global_inout_map==True)
area3_global_inout_map_img[area3_global_inout_map_non_false_idx] = (0,255,0)

area3_car_wash_waiting_map_img = np.zeros((1080, 1920, 3), np.uint8)
area3_car_wash_waiting_map_non_false_idx  = np.where(area3_car_wash_waiting_map==True)
area3_car_wash_waiting_map_img[area3_car_wash_waiting_map_non_false_idx] = (0,255,0)



def tracker_area3(det_result_que, trk_result_que, draw_proc_result_que, proc_num):

    
    center_points_lst = []
    frame_cnt = 0
    car_cnt = 0

    area3_view_global_map = False
    area3_view_car_wash_waiting_map = False
    area3_slope = 0.657103825136612
    previous_put_data = {}


# for frame_cnt in range(0, len(results)):
    while True:
        dets = det_result_que.get()
        
        if dets == None:
            # out.release()
            print('tracker None input break!')
            break
        dets = filter_out_low_conf(dets, 0.25)
        eliminate_dup(dets)
        
        # print('frame cnt = ' , frame_cnt)
        # print('dets = ', dets)

        center_points_lst_orig = center_points_lst.copy()
        for data in center_points_lst_orig:
            data_orig = data
            center_point = data[0]
            for det in dets:
                det = magnify_bbox(det, 0.75)
                p_x1, p_y1 = det[0], det[1]
                p_x2, p_y2 = det[2], det[3]

                if (p_x1 < center_point[0] < p_x2) and (p_y1 < center_point[1] < p_y2):
                    # print('center points list = ' , center_points_lst)
                    # print('this center point = ', center_point)
                    # print('this det = ' , det[0:4])
                    # print('this det ct pt = ' , get_center_pt(det[0:4]))
                   
                    try:
                        # center_points_lst.remove(center_point)
                        center_points_lst.remove(data_orig)
                    except:
                        # print('remove fail')
                        # break
                        pass
        
        # print('before ct pts = ' , center_points_lst)
        for det in dets:
            ct_pt = get_center_pt(det[0:4]) # [x, y]
            # print('ct_pt = ', ct_pt) q
            if area4_global_inout_map[int(ct_pt[1]), int(ct_pt[0])] == True:
                center_points_lst.append([ct_pt, det[0:4], 0])
        # print('after ct pts = ' , center_points_lst)

        # car_cnt = len(center_points_lst)
        # dets_length = len(dets)
        
        # # center_points_lst.clear()
        # for det in dets:
        #     ct_pt = get_center_pt(det[0:4]) # [x, y]
        #     # print('ct_pt = ', ct_pt)
            
        #     center_points_lst.append(ct_pt)

        glb_in_cnt = 0
        car_wash_waiting_cnt = 0

        put_data = {}
        pos_data = []
        for data in center_points_lst_orig:
            data[2] += 1

            conf = get_bbox_conf(data[2])
            if conf <= 0 :
                try:
                    center_points_lst.remove(data)
                except:
                    # print('remove fail 2')
                    pass

            center_point = data[0]
            if area3_global_inout_map[int(center_point[1]), int(center_point[0])] == True:
                glb_in_cnt += 1

                if area3_car_wash_waiting_map[int(center_point[1]), int(center_point[0])] == True:
                    pos_data.append((center_point[1] - area3_slope *(center_point[0])))
                    car_wash_waiting_cnt += 1

        put_data['pos_data'] = pos_data
        put_data.update({'area':3 , 'global_cnt': glb_in_cnt, 'car_washing_waiting_cnt': car_wash_waiting_cnt})
        if previous_put_data != put_data:
        # if True:
            trk_result_que.put(put_data)

        previous_put_data = put_data

        draw_put_data = {'cnt_lst': [glb_in_cnt, car_wash_waiting_cnt], 'center_points_lst': center_points_lst, 'dets': dets }
        draw_proc_result_que.put(draw_put_data)
        
        # ret, frame = cap.read()
        # frame = image_que.get()
        # print('frame from img que = ', frame)
        # if type(frame) != np.ndarray and frame == None:
        #     out.release()
        #     print('tracker not ret break')
        #     break

        frame_cnt += 1



        # if frame_cnt%50 == 0:
        #     print('frame cnt = ' , frame_cnt)
        # if frame_cnt%50 == 0:    
        #     que_size = result_que.qsize()
        #     # Queue.qsize
        #     print('q size = ' , que_size)
        if det_result_que.qsize() > 10:
            print('Q size = ' , det_result_que.qsize())
            if det_result_que.full():
                print('Queue is full!!')






        ########################## draw #################################

        # cv2.putText(frame, 'frame cnt: '+str(frame_cnt), (80, 80), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # # cv2.putText(frame, 'car cnt: '+str(car_cnt), (80, 120), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # # cv2.putText(frame, 'dets cnt: '+str(dets_length), (80, 160), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

        # # glb_in_cnt = 0
        # # car_wash_waiting_cnt = 0
        # # electric_vehicle_charging_waiting_cnt = 0
        # # car_interior_washing_waiting_cnt = 0
        # cv2.putText(frame, 'car_num: '+str(glb_in_cnt), (80, 120), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, 'car_washing_waiting_cnt: '+str(car_wash_waiting_cnt), (80, 160), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # for data in dets:
        #     det = data
        #     p_x1, p_y1 = int(det[0]), int(det[1])
        #     p_x2, p_y2 = int(det[2]), int(det[3])
        #     frame = cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (255, 0, 0), 2)
        #     # det = magnify_bbox(data, 0.75)
        #     # p_x1, p_y1 = int(det[0]), int(det[1])
        #     # p_x2, p_y2 = int(det[2]), int(det[3])
        #     # frame = cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (155, 0, 0), 2)


        # # y = slope * x + k
        # for data in center_points_lst:
        #     center_point = data[0]
        #     cv2.line(frame, (int(center_point[0]), int(center_point[1])), (int(center_point[0]), int(center_point[1])), (0,255,0), thickness=12, lineType=None, shift=None)
        #     cv2.putText(frame, 'pos:: '+str(round((center_point[1] - area3_slope *(center_point[0])), 2) ), (int(center_point[0]), int(center_point[1]+10)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #     cv2.putText(frame, 'conf: '+str(get_bbox_conf(data[2])), (int(center_point[0]), int(center_point[1]+30)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # if area3_view_global_map:
        #     frame[area3_global_inout_map_non_false_idx] = cv2.addWeighted(frame, 0.5, area3_global_inout_map_img, 0.5, 0)[area3_global_inout_map_non_false_idx] 
        # elif area3_view_car_wash_waiting_map:
        #     frame[area3_car_wash_waiting_map_non_false_idx] = cv2.addWeighted(frame, 0.5, area3_car_wash_waiting_map_img, 0.5, 0)[area3_car_wash_waiting_map_non_false_idx] 
    
        # cv2.namedWindow('frame'+str(proc_num), cv2.WINDOW_NORMAL)
        # cv2.imshow('frame'+str(proc_num), frame)
        # # if frame_cnt > 800:
        # #     wk = 0
        # # else:
        # #     wk = 1

        # key  = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     break
        # elif key == ord('0'):
        #     area3_view_global_map = False
        #     area3_view_car_wash_waiting_map = False
        # elif key == ord('1'):
        #     area3_view_global_map = True
        #     area3_view_car_wash_waiting_map = False
        # elif key == ord('2'):
        #     area3_view_global_map = False
        #     area3_view_car_wash_waiting_map = True
        

        ########################## draw #################################



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











area1_global_inout_map = np.load('area1_global_inout_map.npy')
area1_car_wash_waiting_map = np.load('area1_car_wash_waiting_map.npy')
area1_place0_map = np.load('area1_place0_map.npy')

area1_global_inout_map_img =np.zeros((1080, 1920, 3), np.uint8)
area1_global_inout_map_non_false_idx  = np.where(area1_global_inout_map==True)
area1_global_inout_map_img[area1_global_inout_map_non_false_idx] = (0,255,0)

area1_car_wash_waiting_map_img = np.zeros((1080, 1920, 3), np.uint8)
area1_car_wash_waiting_map_non_false_idx  = np.where(area1_car_wash_waiting_map==True)
area1_car_wash_waiting_map_img[area1_car_wash_waiting_map_non_false_idx] = (0,255,0)


area1_place0_map_img = np.zeros((1080, 1920, 3), np.uint8)
area1_place0_map_non_false_idx  = np.where(area1_place0_map==True)
area1_place0_map_img[area1_place0_map_non_false_idx] = (0,255,0)


def tracker_area1(det_result_que, trk_result_que, draw_proc_result_que, proc_num):

    
    center_points_lst = []
    frame_cnt = 0
    car_cnt = 0

    area1_view_global_map = False
    area1_view_car_wash_waiting_map = False
    area1_view_place0_map = False
    area1_slope = -0.5353805073431241

    previous_put_data = {}

    while True:
        dets = det_result_que.get()
        
        if dets == None:
            # out.release()
            print('tracker None input break!')
            break
        dets = filter_out_low_conf(dets, 0.25)
        eliminate_dup(dets)
        
        # print('frame cnt = ' , frame_cnt)
        # print('dets = ', dets)

        put_data = {}

        center_points_lst_orig = center_points_lst.copy()
        for data in center_points_lst_orig:
            data_orig = data
            center_point = data[0]
            for det in dets:
                det = magnify_bbox(det, 0.75)
                p_x1, p_y1 = det[0], det[1]
                p_x2, p_y2 = det[2], det[3]

                if (p_x1 < center_point[0] < p_x2) and (p_y1 < center_point[1] < p_y2):
                    # print('center points list = ' , center_points_lst)
                    # print('this center point = ', center_point)
                    # print('this det = ' , det[0:4])
                    # print('this det ct pt = ' , get_center_pt(det[0:4]))
                   
                    try:
                        # center_points_lst.remove(center_point)
                        center_points_lst.remove(data_orig)
                    except:
                        # print('remove fail')
                        # break
                        pass
        
        # print('before ct pts = ' , center_points_lst)
        for det in dets:
            ct_pt = get_center_pt(det[0:4]) # [x, y]
            # print('ct_pt = ', ct_pt) q
            if area4_global_inout_map[int(ct_pt[1]), int(ct_pt[0])] == True:
                center_points_lst.append([ct_pt, det[0:4], 0])
        # print('after ct pts = ' , center_points_lst)

        # car_cnt = len(center_points_lst)
        # dets_length = len(dets)
        
        # # center_points_lst.clear()
        # for det in dets:
        #     ct_pt = get_center_pt(det[0:4]) # [x, y]
        #     # print('ct_pt = ', ct_pt)
            
        #     center_points_lst.append(ct_pt)

        glb_in_cnt = 0
        car_wash_waiting_cnt = 0
        place0_cnt = 0

        pos_data = []
        for data in center_points_lst_orig:
            data[2] += 1

            conf = get_bbox_conf(data[2])
            if conf <= 0 :
                try:
                    center_points_lst.remove(data)
                except:
                    # print('remove fail 2')
                    pass

            center_point = data[0]
            if area1_global_inout_map[int(center_point[1]), int(center_point[0])] == True:
                glb_in_cnt += 1

                if area1_car_wash_waiting_map[int(center_point[1]), int(center_point[0])] == True:
                    pos_data.append((center_point[1] - area1_slope *(center_point[0])))
                    car_wash_waiting_cnt += 1
                elif area1_place0_map[int(center_point[1]), int(center_point[0])] == True:
                    place0_cnt +=1

        put_data['pos_data'] = pos_data 
        put_data.update({'area':1 , 'global_cnt': glb_in_cnt, 'car_washing_waiting_cnt': car_wash_waiting_cnt, 'place0_cnt': place0_cnt})
        if previous_put_data != put_data:
        # if True:
            trk_result_que.put(put_data)

        previous_put_data = put_data
        
        # cnt_lst, ct_pt_lst, dets
        draw_put_data = {'cnt_lst': [glb_in_cnt, car_wash_waiting_cnt, place0_cnt], 'center_points_lst': center_points_lst, 'dets': dets }
        draw_proc_result_que.put(draw_put_data)
        # ret, frame = cap.read()
        # frame = image_que.get()
        # print('frame from img que = ', frame)
        # if type(frame) != np.ndarray and frame == None:
        #     out.release()
        #     print('tracker not ret break')
        #     break


        # img_q dets gcenter_points_lst lb_in_cnt car_wash_waiting_cnt place0_cnt -> draw data

        frame_cnt += 1



        # if frame_cnt%50 == 0:
        #     print('frame cnt = ' , frame_cnt)
        # if frame_cnt%50 == 0:    
        #     que_size = result_que.qsize()
        #     # Queue.qsize
        #     print('q size = ' , que_size)
        if det_result_que.qsize() > 10:
            print('Q size = ' , det_result_que.qsize())
            if det_result_que.full():
                print('Queue is full!!')


        ########################## draw #################################

        # cv2.putText(frame, 'frame cnt: '+str(frame_cnt), (80, 80), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # # cv2.putText(frame, 'car cnt: '+str(car_cnt), (80, 120), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # # cv2.putText(frame, 'dets cnt: '+str(dets_length), (80, 160), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

        # # glb_in_cnt = 0
        # # car_wash_waiting_cnt = 0
        # # electric_vehicle_charging_waiting_cnt = 0
        # # car_interior_washing_waiting_cnt = 0
        # cv2.putText(frame, 'car_num: '+str(glb_in_cnt), (80, 120), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, 'car_washing_waiting_cnt: '+str(car_wash_waiting_cnt), (80, 160), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, 'place0_cnt: '+str(place0_cnt), (80, 200), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # for data in dets:
        #     det = data
        #     p_x1, p_y1 = int(det[0]), int(det[1])
        #     p_x2, p_y2 = int(det[2]), int(det[3])
        #     frame = cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (255, 0, 0), 2)
        #     # det = magnify_bbox(data, 0.75)
        #     # p_x1, p_y1 = int(det[0]), int(det[1])
        #     # p_x2, p_y2 = int(det[2]), int(det[3])
        #     # frame = cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (155, 0, 0), 2)


        # # y = slope * x + k
        # for data in center_points_lst:
        #     center_point = data[0]
        #     cv2.line(frame, (int(center_point[0]), int(center_point[1])), (int(center_point[0]), int(center_point[1])), (0,255,0), thickness=12, lineType=None, shift=None)
        #     cv2.putText(frame, 'pos:: '+str(round((center_point[1] - area1_slope *(center_point[0])), 2) ), (int(center_point[0]), int(center_point[1]+10)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #     cv2.putText(frame, 'conf: '+str(get_bbox_conf(data[2])), (int(center_point[0]), int(center_point[1]+30)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # if area1_view_global_map:
        #     frame[area1_global_inout_map_non_false_idx] = cv2.addWeighted(frame, 0.5, area1_global_inout_map_img, 0.5, 0)[area1_global_inout_map_non_false_idx] 
        # elif area1_view_car_wash_waiting_map:
        #     frame[area1_car_wash_waiting_map_non_false_idx] = cv2.addWeighted(frame, 0.5, area1_car_wash_waiting_map_img, 0.5, 0)[area1_car_wash_waiting_map_non_false_idx] 
        # elif area1_view_place0_map:
        #     frame[area1_place0_map_non_false_idx] = cv2.addWeighted(frame, 0.5, area1_place0_map_img, 0.5, 0)[area1_place0_map_non_false_idx]
        
        # cv2.namedWindow('frame'+str(proc_num), cv2.WINDOW_NORMAL)
        # cv2.imshow('frame'+str(proc_num), frame)
        # # if frame_cnt > 800:
        # #     wk = 0
        # # else:
        # #     wk = 1

        

        # key  = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     break
        # elif key == ord('0'):
        #     area1_view_global_map = False
        #     area1_view_car_wash_waiting_map = False
        #     area1_view_place0_map = False
        # elif key == ord('1'):
        #     area1_view_global_map = True
        #     area1_view_car_wash_waiting_map = False
        #     area1_view_place0_map = False
        # elif key == ord('2'):
        #     area1_view_global_map = False
        #     area1_view_car_wash_waiting_map = True
        #     area1_view_place0_map = False
        # elif key == ord('3'):
        #     area1_view_global_map = False
        #     area1_view_car_wash_waiting_map = False
        #     area1_view_place0_map = True
        
        ########################## draw #################################


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


def visualize(img_q, proc_result_q, area_num_idx):
    frame_cnt = 0
    

    # proc_result = proc_result_q.get()
    # dets = proc_result[]
    # center_points_lst = proc_result[]
    
    area_num_lst = [1, 3, 4]
    area_num = area_num_lst[area_num_idx]
    
    # cnt_lst, center_points_lst, 
    if area_num == 1:
        glb_in_cnt = 0
        car_wash_waiting_cnt = 0
        place0_cnt = 0
        cnts_lst = [glb_in_cnt, car_wash_waiting_cnt, place0_cnt]
        cnts_lst_str = ['global in', 'car wash wait', 'place0']
        area1_view_global_map = False
        area1_view_car_wash_waiting_map = False
        area1_view_place0_map = False
        display_bool_lst = [area1_view_global_map, area1_view_car_wash_waiting_map, area1_view_place0_map]
        slope = -0.5353805073431241
        non_false_idx_lst = [area1_global_inout_map_non_false_idx, area1_car_wash_waiting_map_non_false_idx, area1_place0_map_non_false_idx]
        map_imgs = [area1_global_inout_map_img, area1_car_wash_waiting_map_img, area1_place0_map_img]
        zone_name_strings = ['Global In Out', 'Car Wash Waiting', 'Place0']
    elif area_num == 3:
        glb_in_cnt = 0
        car_wash_waiting_cnt = 0
        cnts_lst = [glb_in_cnt, car_wash_waiting_cnt]
        cnts_lst_str = ['global in', 'car wash wait']
        area3_view_global_map = False
        area3_view_car_wash_waiting_map = False
        display_bool_lst = [area3_view_global_map, area3_view_car_wash_waiting_map]
        slope = 0.657103825136612
        non_false_idx_lst = [area3_global_inout_map_non_false_idx, area3_car_wash_waiting_map_non_false_idx]
        map_imgs = [area3_global_inout_map_img, area3_car_wash_waiting_map_img]
        zone_name_strings = ['Global In Out', 'Car Wash Waiting']
    elif area_num == 4:
        glb_in_cnt = 0
        car_wash_waiting_cnt = 0
        electric_vehicle_charging_waiting_cnt = 0
        car_interior_washing_waiting_cnt = 0
        cnts_lst = [glb_in_cnt, car_wash_waiting_cnt, electric_vehicle_charging_waiting_cnt, car_interior_washing_waiting_cnt]
        cnts_lst_str = ['global in', 'car wash wait', 'elctric charging', 'car interior wash']
        area4_view_global_map = False
        area4_view_car_wash_waiting_map = False
        area4_view_electric_charging_map = False
        area4_view_car_interior_wash_map = False
        display_bool_lst = [area4_view_global_map, area4_view_car_wash_waiting_map, area4_view_electric_charging_map, area4_view_car_interior_wash_map]
        slope = 0.22451456310679613
        non_false_idx_lst = [area4_global_inout_map_non_false_idx, area4_car_wash_waiting_map_non_false_idx, area4_electric_vehicle_charging_map_non_false_idx, area4_car_interior_washing_map_non_false_idx]
        map_imgs = [area4_global_inout_map_img, area4_car_wash_waiting_map_img, area4_electric_vehicle_charging_map_img, area4_car_interior_washing_map_img]
        zone_name_strings = ['Global In Out', 'Car Wash Waiting', 'Electric Charging Zone', 'Car Interior Washing Zone']
    # img_q dets center_points_lst lb_in_cnt car_wash_waiting_cnt place0_cnt


    available_key_lst = [ord(str(i)) for i in range(0, len(display_bool_lst)+1)]
    displayed_zone_name = 'None'

    while True:
        proc_result = proc_result_q.get()
        dets = proc_result['dets']
        center_points_lst = proc_result['center_points_lst']
        for i in range(0, len(cnts_lst)):
            # cnt = 
            # print('area num = ' , area_num)
            # print('cnts lst = ' , cnts_lst)
            # print('proc_result[cnt_lst] = ' , proc_result['cnt_lst'])
            cnts_lst[i] = proc_result['cnt_lst'][i]
        frame = img_q.get()


        # draw_put_data = {'cnt_lst': [glb_in_cnt, car_wash_waiting_cnt, electric_vehicle_charging_waiting_cnt, car_interior_washing_waiting_cnt],
        # 'center_points_lst': center_points_lst, 'dets': dets }
        
        cv2.putText(frame, 'frame cnt: '+str(frame_cnt), (80, 80), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, 'car cnt: '+str(car_cnt), (80, 120), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, 'dets cnt: '+str(dets_length), (80, 160), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # cnt_lst, ct_pt_lst, dets
        # glb_in_cnt = 0
        # car_wash_waiting_cnt = 0
        # electric_vehicle_charging_waiting_cnt = 0
        # car_interior_washing_waiting_cnt = 0
        for cnts_lst_idx in range(0, len(cnts_lst)):
            cnt_str = cnts_lst_str[cnts_lst_idx]
            cnt_value = cnts_lst[cnts_lst_idx]
            cv2.putText(frame, cnt_str + ': ' + str(cnt_value), (80, 120 + 35*cnts_lst_idx), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, 'car_num: '+str(glb_in_cnt), (80, 120), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, 'car_washing_waiting_cnt: '+str(car_wash_waiting_cnt), (80, 160), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, 'place0_cnt: '+str(place0_cnt), (80, 200), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        
        
        for data in dets:
            det = data
            p_x1, p_y1 = int(det[0]), int(det[1])
            p_x2, p_y2 = int(det[2]), int(det[3])
            frame = cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (255, 0, 0), 2)
            # det = magnify_bbox(data, 0.75)
            # p_x1, p_y1 = int(det[0]), int(det[1])
            # p_x2, p_y2 = int(det[2]), int(det[3])
            # frame = cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (155, 0, 0), 2)


        # y = slope * x + k
        for data in center_points_lst:
            center_point = data[0]
            cv2.line(frame, (int(center_point[0]), int(center_point[1])), (int(center_point[0]), int(center_point[1])), (0,255,0), thickness=12, lineType=None, shift=None)
            cv2.putText(frame, 'pos:: '+str(round((center_point[1] - slope *(center_point[0])), 2) ), (int(center_point[0]), int(center_point[1]+10)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'conf: '+str(get_bbox_conf(data[2])), (int(center_point[0]), int(center_point[1]+30)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        for i in range(0, len(non_false_idx_lst)):
            non_false_idx = non_false_idx_lst[i]
            map_img = map_imgs[i]
            need_display = display_bool_lst[i]
            if need_display:
                frame[non_false_idx] = cv2.addWeighted(frame, 0.5, map_img, 0.5, 0)[non_false_idx]
                break

        # if area1_view_global_map:
        #     frame[area1_global_inout_map_non_false_idx] = cv2.addWeighted(frame, 0.5, area1_global_inout_map_img, 0.5, 0)[area1_global_inout_map_non_false_idx] 
        # elif area1_view_car_wash_waiting_map:
        #     frame[area1_car_wash_waiting_map_non_false_idx] = cv2.addWeighted(frame, 0.5, area1_car_wash_waiting_map_img, 0.5, 0)[area1_car_wash_waiting_map_non_false_idx] 
        # elif area1_view_place0_map:
        #     frame[area1_place0_map_non_false_idx] = cv2.addWeighted(frame, 0.5, area1_place0_map_img, 0.5, 0)[area1_place0_map_non_false_idx]
        
        
        # if frame_cnt > 800:
        #     wk = 0
        # else:
        #     wk = 1

        
        

        key  = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key != 255:
            # print('wait key = ', key)
            # len_bools = len(display_bool_lst)
            # bool_idxs = range(0, len_bools+1)
            # if key == ord('0'):
            #     for bool in display_bool_lst:
            #         bool = False
            # else:
            # available_key_lst = [ord(str(i)) for i in range(0, len(display_bool_lst))]
            if key in available_key_lst:
                for i in range(0, len(display_bool_lst)):
                    displayed_zone_name = 'None'
                    display_bool_lst[i] = False
                for i in range(0, len(display_bool_lst)):
                    if key == ord(str(i+1)):
                        displayed_zone_name = str(zone_name_strings[i])
                        display_bool_lst[i] = True
                        break
        
        cv2.putText(frame, 'Displayed Zone: ' + displayed_zone_name, (80, 120 + 35*(len(cnts_lst))), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        

        cv2.namedWindow('frame: area'+str(area_num), cv2.WINDOW_NORMAL)
        cv2.imshow('frame: area'+str(area_num), frame)


        # elif key == ord('0'):
        #     area1_view_global_map = False
        #     area1_view_car_wash_waiting_map = False
        #     area1_view_place0_map = False
        # elif key == ord('1'):
        #     area1_view_global_map = True
        #     area1_view_car_wash_waiting_map = False
        #     area1_view_place0_map = False
        # elif key == ord('2'):
        #     area1_view_global_map = False
        #     area1_view_car_wash_waiting_map = True
        #     area1_view_place0_map = False
        # elif key == ord('3'):
        #     area1_view_global_map = False
        #     area1_view_car_wash_waiting_map = False
        #     area1_view_place0_map = True
        
        frame_cnt += 1
    
    cv2.destroyAllWindows()
    


def get_bbox_conf(data):
    thr = 5
    if 0<= data < thr:
        result = 1
    elif thr< data <=2*thr:
        result = 2 + -1*(1/thr)*data
    else:
        result = 0
    return result


def video_load(image_que, image_que2, path):
    # cap_loader = cv2.VideoCapture(path + '.mp4')
    # cap_loader = cv2.VideoCapture('rtsp://admin:self1004@@118.37.223.147:8522/live/main8')
    cap_loader = cv2.VideoCapture(path)
    # cap_loader.set(cv2.CAP_PROP_POS_FRAMES, 1600)
    video_frame_cnt = 0
    while True:
        
        # print('data load')
        # _, _ = cap_loader.read()
        _, _ = cap_loader.read()
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



def yolo_inference(image_que, result_que):
    inference_instance = inference()
    y_s = time.time()
    inference_instance.run(image_que, result_que)
    print('yolo inference breaked!')
    y_e = time.time()

    y_elapsed_time = y_e - y_s
    print('yolo elapsed time = ', y_elapsed_time)

need_write = False


lst_of_trk_fns = [tracker_area1, tracker_area3, tracker_area4]



def control_center(que1, que2, que3):
    # print('cc check!')
    area1_global_cnt = 0
    area1_car_wash_waiting_cnt = 0
    area1_place0_cnt = 0
    area1_pos = []

    area3_global_cnt = 0
    area3_car_wash_waiting_cnt = 0
    area3_pos = []

    area4_global_cnt = 0
    area4_car_wash_waiting_cnt = 0
    area4_electric_vehicle_charging_cnt = 0
    area4_car_interior_wash_cnt = 0
    area4_pos = []

    while True:
        time.sleep(0.02)
        # print('cc check2')
        if not que1.empty(): # area1
            qdata1 = que1.get()
            # qdata1 =  {'pos_data': [], 'area': 1, 'global_cnt': 0, 'car_washing_waiting_cnt': 0, 'place0_cnt': 0}
            area1_global_cnt = qdata1['global_cnt']
            area1_car_wash_waiting_cnt = qdata1['car_washing_waiting_cnt']
            area1_place0_cnt = qdata1['place0_cnt']
            area1_pos = qdata1['pos_data']
            print('qdata1 = ', qdata1)

        if not que2.empty(): # area3
            qdata2 = que2.get()
            area3_global_cnt = qdata2['global_cnt']
            area3_car_wash_waiting_cnt = qdata2['car_washing_waiting_cnt']
            area3_pos = qdata2['pos_data']
            # qdata2 =  {'pos_data': [], 'area': 3, 'global_cnt': 0, 'car_washing_waiting_cnt': 0}
            print('qdata2 = ' , qdata2)

        if not que3.empty(): #area4
            qdata3 = que3.get()
            # qdata3 =  {'pos_data': [], 'area': 4, 'global_cnt': 3, 'car_washing_waiting_cnt': 0,
            #               'car_interior_washing_waiting_cnt': 2, 'electric_vehicle_charging_waiting_cnt': 0}
            area4_global_cnt = qdata3['global_cnt']
            area4_car_wash_waiting_cnt = qdata3['car_washing_waiting_cnt']
            area4_car_interior_wash_cnt = qdata3['car_interior_washing_waiting_cnt']
            area4_electric_vehicle_charging_cnt = qdata3['electric_vehicle_charging_waiting_cnt']
            area4_pos = qdata3['pos_data']
            print('qdata3 = ' , qdata3)
        
        total_cnt = area1_global_cnt + area3_global_cnt + area4_global_cnt
        car_wash_cnt = area1_car_wash_waiting_cnt + area3_car_wash_waiting_cnt + area4_car_wash_waiting_cnt
        electric_charging_waiting_cnt = area4_electric_vehicle_charging_cnt
        car_interior_wash_cnt = area4_car_interior_wash_cnt
        place0_cnt = area1_place0_cnt
        
    

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

#main6 = area3, main7 = area1, main8 = area4

paths = ['rtsp://admin:self1004@@118.37.223.147:8522/live/main7',
         'rtsp://admin:self1004@@118.37.223.147:8522/live/main6', 
         'rtsp://admin:self1004@@118.37.223.147:8522/live/main8']

# paths = ['rtsp://admin:self1004@@118.37.223.147:8522/live/main6', 
#          'rtsp://admin:self1004@@118.37.223.147:8522/live/main7', 
#          'rtsp://admin:self1004@@118.37.223.147:8522/live/main8',
#          'rtsp://admin:self1004@@118.37.223.147:8522/live/main9']

# paths = ['rtsp://admin:self1004@@118.37.223.147:8522/live/main8']

# paths = ['rtsp://admin:self1004@@118.37.223.147:8522/live/main8',
#          'rtsp://admin:self1004@@118.37.223.147:8522/live/main8',
#          'rtsp://admin:self1004@@118.37.223.147:8522/live/main8']


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
    det_result_que_lst = []
    trk_result_que_lst = []
    video_loader_lst = []
    yolo_inference_lst = []
    tracking_proc_lst = []
    yolo_instance_lst = []
    draw_proc_lst = []
    draw_proc_result_que_lst = []


    for i in range(0, len(paths)):
        # print('check')
        image_que_lst_proc.append(Queue(200))
        image_que_lst_draw.append(Queue(200))
        det_result_que_lst.append(Queue(200))
        trk_result_que_lst.append(Queue(200))
        draw_proc_result_que_lst.append(Queue(200))
    # for i in range(0, len(paths)):
        video_loader_lst.append(multiprocessing.Process(target=video_load, args=(image_que_lst_proc[i], image_que_lst_draw[i], paths[i]), daemon=False))
        yolo_inference_lst.append(multiprocessing.Process(target=yolo_inference, args=(image_que_lst_proc[i], det_result_que_lst[i]), daemon=False))
        tracking_proc_lst.append(multiprocessing.Process(target=lst_of_trk_fns[i], args=(det_result_que_lst[i], trk_result_que_lst[i], draw_proc_result_que_lst[i], i), daemon=False))
        draw_proc_lst.append(multiprocessing.Process(target=visualize, args=(image_que_lst_draw[i], draw_proc_result_que_lst[i], i,), daemon=False))
    
    post_proc = multiprocessing.Process(target=control_center, args=(trk_result_que_lst[0], trk_result_que_lst[1], trk_result_que_lst[2]), daemon=False)
    # 1 -> area1, 0 -> area3, 2 -> area4


    for i in range(0, len(paths)):
        video_loader_lst[i].start()
        yolo_inference_lst[i].start()
        tracking_proc_lst[i].start()
        draw_proc_lst[i].start()
        # print('check!')
    post_proc.start()

    for i in range(0, len(paths)):
        video_loader_lst[i].join()
        yolo_inference_lst[i].join()
        tracking_proc_lst[i].join()
        draw_proc_lst[i].join()
    post_proc.join()

    for i in range(0, len(paths)):
        video_loader_lst[i].close()
        yolo_inference_lst[i].close()
        tracking_proc_lst[i].close()
        draw_proc_lst[i].close()
    post_proc.close()

    print('main proc end')
