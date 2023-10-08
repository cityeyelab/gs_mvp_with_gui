# import numpy as np
# from dataclasses import dataclass, field
# from typing import List

from ..util_functions import filter_out_low_conf, eliminate_dup, magnify_bbox, get_center_pt, get_bbox_conf
from ..map_vars import area1_global_inout_map, area1_car_wash_waiting_map, area1_place0_map

# import sys

# @dataclass()
# class CarDetInfo:
#     area: int
#     id: int
#     age: int = 0
#     events: List = field(default_factory=list)


def tracker_area1(op_flag, det_result_que, trk_result_que, draw_proc_result_que, visualize_bp_que, exit_event, proc_num):

    center_points_lst = []
    frame_cnt = 0

    area1_slope = -0.5353805073431241
    previous_put_data = {}

    while True:
        if exit_event.is_set():
            break
        
        dets = det_result_que.get()
        if type(dets) == type(None):
            trk_result_que.put(None)
            visualize_bp_que.put(None)
            # sys.exit()
            break
        
        # if dets == None:
        #     print('tracker None input break!')
        #     break
        dets = filter_out_low_conf(dets, 0.25)
        eliminate_dup(dets)
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
                    try:
                        center_points_lst.remove(data_orig)
                    except:
                        pass

        for det in dets:
            ct_pt = get_center_pt(det[0:4]) # [x, y]
            if area1_global_inout_map[int(ct_pt[1]), int(ct_pt[0])] == True:
                center_points_lst.append([ct_pt, det[0:4], 0])

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
                    pass

            center_point = data[0]
            if area1_global_inout_map[int(center_point[1]), int(center_point[0])] == True:
                glb_in_cnt += 1

                if area1_car_wash_waiting_map[int(center_point[1]), int(center_point[0])] == True:
                    pos_data.append(round((center_point[1] - area1_slope *(center_point[0])), 2))
                    car_wash_waiting_cnt += 1
                elif area1_place0_map[int(center_point[1]), int(center_point[0])] == True:
                    place0_cnt +=1

        put_data['pos_data'] = pos_data 
        put_data.update({'area':1 , 'global_cnt': glb_in_cnt, 'car_washing_waiting_cnt': car_wash_waiting_cnt, 'place0_cnt': place0_cnt})
        if previous_put_data != put_data:
            trk_result_que.put(put_data)

        previous_put_data = put_data

        bp_data = []
        for pts in center_points_lst:
            bp_data.append(pts[0])
        # bp_data = center_points_lst
        visualize_bp_que.put(bp_data)

        if not draw_proc_result_que.full():
            draw_put_data = {'cnt_lst': [glb_in_cnt, car_wash_waiting_cnt, place0_cnt], 'center_points_lst': center_points_lst, 'dets': dets }
            draw_proc_result_que.put(draw_put_data)

        frame_cnt += 1

        if det_result_que.qsize() > 10:
            print('Q size = ' , det_result_que.qsize())
            if det_result_que.full():
                print('Queue is full!!')
    print('area1 tracker end')