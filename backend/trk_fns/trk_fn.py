from ..util_functions import filter_out_low_conf, eliminate_dup, magnify_bbox, get_center_pt, get_bbox_conf
from ..map_vars import area1_global_inout_map, area1_car_wash_waiting_map, area1_place0_map
from .trk_fns_data import TrackingVariables, area1_data_dict, area3_data_dict, area4_data_dict

# slope: float
# global_inout_map: map
# non_global_maps: list of maps
# maps_names: list of strings

# slope = -0.5353805073431241
# maps_names = ['car_wash_waiting', 'place0']
# non_global_maps = [area1_car_wash_waiting_map, area1_place0_map]


data_dicts_lst = [area1_data_dict, area3_data_dict, area4_data_dict]

def tracker(op_flag, det_result_que, trk_result_que, draw_proc_result_que, visualize_bp_que, exit_event, proc_num):

    #data load
    # proc_num_to_area_num = [1, 3, 4]
    data_dict = data_dicts_lst[proc_num]
    tracking_varialbes = TrackingVariables(data_dict)
    
    area_number = tracking_varialbes.area_number
    slope = tracking_varialbes.slope
    glb_inout_map = tracking_varialbes.glb_inout_map
    non_global_maps_names = tracking_varialbes.non_global_maps_names
    non_global_maps = tracking_varialbes.non_global_maps
    zone_cnt_vars = [0 for i in range(0, len(non_global_maps_names))]
    
    #initialize
    center_points_lst = []
    frame_cnt = 0
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
            if glb_inout_map[int(ct_pt[1]), int(ct_pt[0])] == True: # *
                center_points_lst.append([ct_pt, det[0:4], 0])

        glb_in_cnt = 0 # *
        
        # car_wash_waiting_cnt = 0 # *
        # place0_cnt = 0 # *
        zone_cnt_vars[:] = [0 for _ in range(0, len(non_global_maps_names))]

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
            if glb_inout_map[int(center_point[1]), int(center_point[0])] == True: # *
                glb_in_cnt += 1
                
                for i in range(0, len(non_global_maps)):
                    map = non_global_maps[i]
                    if map[int(center_point[1]), int(center_point[0])] == True: # *
                        zone_cnt_vars[i] += 1
                        pos_data.append(round((center_point[1] - slope *(center_point[0])), 2))
                        break
                # if area1_car_wash_waiting_map[int(center_point[1]), int(center_point[0])] == True: # *
                #     pos_data.append(round((center_point[1] - slope *(center_point[0])), 2))
                #     car_wash_waiting_cnt += 1
                # elif area1_place0_map[int(center_point[1]), int(center_point[0])] == True: # *
                #     place0_cnt +=1

        put_data['pos_data'] = pos_data # *
        # put_data.update({'area':1 , 'global_cnt': glb_in_cnt, 'car_washing_waiting_cnt': car_wash_waiting_cnt, 'place0_cnt': place0_cnt}) # *
        data_to_be_updated = {'area': area_number, 'global_cnt': glb_in_cnt}
        for i in range(0, len(non_global_maps_names)):
            map_string = non_global_maps_names[i]
            data_to_be_updated.update({map_string: zone_cnt_vars[i]})
        put_data.update(data_to_be_updated)
        if previous_put_data != put_data:
            trk_result_que.put(put_data)

        previous_put_data = put_data

        bp_data = []
        for pts in center_points_lst:
            bp_data.append(pts[0])
        # bp_data = center_points_lst
        visualize_bp_que.put(bp_data)

        if not draw_proc_result_que.full():
            draw_put_data = {'cnt_lst': [glb_in_cnt] + zone_cnt_vars, 'center_points_lst': center_points_lst, 'dets': dets }
            draw_proc_result_que.put(draw_put_data)

        frame_cnt += 1

        if det_result_que.qsize() > 10:
            print('Q size = ' , det_result_que.qsize())
            if det_result_que.full():
                print('Queue is full!!')
    print('area1 tracker end')




# def tracker(op_flag, det_result_que, trk_result_que, draw_proc_result_que, visualize_bp_que, exit_event, proc_num):

#     center_points_lst = []
#     frame_cnt = 0
#     previous_put_data = {}
    
#     slope = -0.5353805073431241
    
#     glb_inout_map = area1_global_inout_map
#     car_wash_waiting_map = area1_car_wash_waiting_map
#     place0_map = area1_place0_map

#     # area1_slope = -0.5353805073431241
#     # previous_put_data = {}
    
    

#     while True:
#         if exit_event.is_set():
#             break
        
#         dets = det_result_que.get()
#         if type(dets) == type(None):
#             trk_result_que.put(None)
#             visualize_bp_que.put(None)
#             # sys.exit()
#             break
        
#         # if dets == None:
#         #     print('tracker None input break!')
#         #     break
#         dets = filter_out_low_conf(dets, 0.25)
#         eliminate_dup(dets)
#         put_data = {}

#         center_points_lst_orig = center_points_lst.copy()
#         for data in center_points_lst_orig:
#             data_orig = data
#             center_point = data[0]
#             for det in dets:
#                 det = magnify_bbox(det, 0.75)
#                 p_x1, p_y1 = det[0], det[1]
#                 p_x2, p_y2 = det[2], det[3]

#                 if (p_x1 < center_point[0] < p_x2) and (p_y1 < center_point[1] < p_y2):
#                     try:
#                         center_points_lst.remove(data_orig)
#                     except:
#                         pass

#         for det in dets:
#             ct_pt = get_center_pt(det[0:4]) # [x, y]
#             if area1_global_inout_map[int(ct_pt[1]), int(ct_pt[0])] == True: # *
#                 center_points_lst.append([ct_pt, det[0:4], 0])

#         glb_in_cnt = 0 # *
#         car_wash_waiting_cnt = 0 # *
#         place0_cnt = 0 # *

#         pos_data = []
#         for data in center_points_lst_orig:
#             data[2] += 1

#             conf = get_bbox_conf(data[2])
#             if conf <= 0 :
#                 try:
#                     center_points_lst.remove(data)
#                 except:
#                     pass

#             center_point = data[0]
#             if area1_global_inout_map[int(center_point[1]), int(center_point[0])] == True: # *
#                 glb_in_cnt += 1

#                 if area1_car_wash_waiting_map[int(center_point[1]), int(center_point[0])] == True: # *
#                     pos_data.append(round((center_point[1] - slope *(center_point[0])), 2))
#                     car_wash_waiting_cnt += 1
#                 elif area1_place0_map[int(center_point[1]), int(center_point[0])] == True: # *
#                     place0_cnt +=1

#         put_data['pos_data'] = pos_data # *
#         put_data.update({'area':1 , 'global_cnt': glb_in_cnt, 'car_washing_waiting_cnt': car_wash_waiting_cnt, 'place0_cnt': place0_cnt}) # *
#         if previous_put_data != put_data:
#             trk_result_que.put(put_data)

#         previous_put_data = put_data

#         bp_data = []
#         for pts in center_points_lst:
#             bp_data.append(pts[0])
#         # bp_data = center_points_lst
#         visualize_bp_que.put(bp_data)

#         if not draw_proc_result_que.full():
#             draw_put_data = {'cnt_lst': [glb_in_cnt, car_wash_waiting_cnt, place0_cnt], 'center_points_lst': center_points_lst, 'dets': dets }
#             draw_proc_result_que.put(draw_put_data)

#         frame_cnt += 1

#         if det_result_que.qsize() > 10:
#             print('Q size = ' , det_result_que.qsize())
#             if det_result_que.full():
#                 print('Queue is full!!')
#     print('area1 tracker end')