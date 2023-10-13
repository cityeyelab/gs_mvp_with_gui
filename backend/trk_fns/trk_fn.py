import datetime

from ..util_functions import filter_out_low_conf, eliminate_dup, magnify_bbox, trk_fn_get_ct_pt_lst, get_bbox_conf, get_center_pt, cal_IoU_elt_lst
from ..map_vars import area1_global_inout_map, area1_car_wash_waiting_map, area1_place0_map
from .trk_fns_args import TrackingVariables, area1_data_dict, area3_data_dict, area4_data_dict
from .trk_data import TrackingData
from .filter_blacklist import filter_blacklist_fn_callback_lst
# slope: float
# global_inout_map: map
# non_global_maps: list of maps
# maps_names: list of strings

# slope = -0.5353805073431241
# maps_names = ['car_wash_waiting', 'place0']
# non_global_maps = [area1_car_wash_waiting_map, area1_place0_map]


data_dicts_lst = [area1_data_dict, area3_data_dict, area4_data_dict]


def tracker(op_flag, det_result_que, trk_result_que, draw_proc_result_que, visualize_bp_que, collision_que, exit_event, proc_num):

    #data load
    # proc_num_to_area_num = [1, 3, 4]
    data_dict = data_dicts_lst[proc_num]
    tracking_varialbes = TrackingVariables(data_dict)
    get_ct_pt_fn = trk_fn_get_ct_pt_lst[proc_num]
    
    area_number = tracking_varialbes.area_number
    slope = tracking_varialbes.slope
    glb_inout_map = tracking_varialbes.glb_inout_map
    non_global_maps_names = tracking_varialbes.non_global_maps_names
    non_global_maps = tracking_varialbes.non_global_maps
    zone_cnt_vars = [0 for i in range(0, len(non_global_maps_names))]
    
    #initialize
    center_point_lst = []
    frame_cnt = 0
    previous_put_data = {}
    trk_data_lst = []
    tracking_id = 0
    filter_blacklist_fn = filter_blacklist_fn_callback_lst[proc_num]
    
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
        filter_blacklist_fn(dets)
        dets = filter_out_low_conf(dets, 0.25)
        eliminate_dup(dets)
        put_data = {}





    #############################################
        row_num = len(trk_data_lst)
        column_num = len(dets)
        iou_lst = []
        trk_data_to_det_mapping_lst = []
        trk_data_lst_orig = trk_data_lst.copy()
        unmatched_dets_idx = [i for i in range(0, column_num)]
        for i in range(0, row_num):
            trk_data = trk_data_lst_orig[i]
            last_bbox = trk_data.bboxes[-1]
            iou_lst_wrt_trk_data = cal_IoU_elt_lst(last_bbox, dets)
            iou_lst = iou_lst + iou_lst_wrt_trk_data
        
        iou_lst = [iou_lst[i] if iou_lst[i] >= 0.1  else 0 for i in range(0, len(iou_lst))]
        while sum(iou_lst) > 0:
            iou_lst_max_idx = iou_lst.index(max(iou_lst))
            iou_lst_quotient = iou_lst_max_idx//column_num
            iou_lst_remainder = iou_lst_max_idx%column_num
            iou_lst[column_num*iou_lst_quotient:column_num*(iou_lst_quotient+1)] = [0 for _ in range(column_num)]
            for rn in range(0, row_num):
                iou_lst[rn*column_num + iou_lst_remainder] = 0
                
            mapping_tup = (iou_lst_quotient, iou_lst_remainder) #trk -> det : (trk_nb, det_nb)
            unmatched_dets_idx.remove(iou_lst_remainder)
            
            trk_data_to_det_mapping_lst.append(mapping_tup)

        #append data
        for mapping_data in trk_data_to_det_mapping_lst:
            this_cls = trk_data_lst[mapping_data[0]]
            to_be_appended = dets[mapping_data[1]]
            this_cls.bboxes.append(to_be_appended)
            this_cls.center_points_lst.append(get_ct_pt_fn(to_be_appended))
            this_cls.frame_record.append(frame_cnt)
        
        # print('unmatched_dets_idx = ', unmatched_dets_idx)
        # create new data cls
        for new_idx in unmatched_dets_idx:
            det = dets[new_idx]
            center_point = get_ct_pt_fn(det)
            if glb_inout_map[int(center_point[1]), int(center_point[0])] == True:
                new_cls = TrackingData(area_num=area_number, id=tracking_id)
                new_cls.bboxes.append(det)
                new_cls.center_points_lst.append(center_point)
                new_cls.frame_record.append(frame_cnt)
                trk_data_lst.append(new_cls)
                tracking_id+=1
        

        # center_points_lst_orig = center_points_lst.copy()
        # for data in center_points_lst_orig:
        #     data_orig = data
        #     center_point = data[0]
        #     for det in dets:
        #         det = magnify_bbox(det, 0.75)
        #         p_x1, p_y1 = det[0], det[1]
        #         p_x2, p_y2 = det[2], det[3]

        #         if (p_x1 < center_point[0] < p_x2) and (p_y1 < center_point[1] < p_y2):
        #             try:
        #                 center_points_lst.remove(data_orig)
        #             except:
        #                 pass

        # for det in dets:
        #     ct_pt = get_ct_pt_fn(det[0:4]) # [x, y]
        #     if glb_inout_map[int(ct_pt[1]), int(ct_pt[0])] == True: # *
        #         center_points_lst.append([ct_pt, det[0:4], 0])
        # print('class = ' , trk_data_lst)

    #############################################



        glb_in_cnt = 0

        zone_cnt_vars[:] = [0 for _ in range(0, len(non_global_maps_names))]
        trk_data_lst_orig = trk_data_lst.copy()
        pos_data = []
        center_point_lst = []
        # if area_number == 4:
            # print('trk_data_lst_orig : ',trk_data_lst_orig)
        for i, data_cls in enumerate(trk_data_lst_orig):
            # print('diff = ', frame_cnt - data_cls.frame_record[-1])
            # conf = get_bbox_conf(frame_cnt - data_cls.frame_record[-1])
            # print('conf = ' , conf)
            # if conf <= 0 :
            if frame_cnt - data_cls.frame_record[-1] > 30:
                try:
                    # pass
                    data_cls.removed_at = datetime.datetime.now()
                    collision_que.put(data_cls)
                    trk_data_lst.remove(data_cls)
                    # if area_number == 4:
                    #     print('removed!')
                except:
                    pass
            else:
                center_point = data_cls.center_points_lst[-1]
                # center_point_lst.append(data_cls.center_points_lst[-15:])
                # print('center point list = ' , center_point_lst)
                # if area_number == 4:
                #     print('center point = ', center_point)
                if glb_inout_map[int(center_point[1]), int(center_point[0])] == True: # *
                        glb_in_cnt += 1
                        # center_point_lst.append(center_point)
                        center_point_lst.append(data_cls.center_points_lst[-36:])
                        for i in range(0, len(non_global_maps)):
                            map = non_global_maps[i]
                            if map[int(center_point[1]), int(center_point[0])] == True: # *
                                zone_cnt_vars[i] += 1
                                pos_data.append(round((center_point[1] - slope *(center_point[0])), 2))
                                break
        
        # if area_number == 4:
        #     print('cls list = ' , trk_data_lst)
        #     print('ct pt lst = ',center_point_lst)
            
        # pos_data = []
        # for data in center_points_lst_orig:
        #     data[2] += 1

        #     conf = get_bbox_conf(data[2])
        #     if conf <= 0 :
        #         try:
        #             center_point_lst.remove(data)
        #         except:
        #             pass

        #     center_point = data[0]
        #     if glb_inout_map[int(center_point[1]), int(center_point[0])] == True: # *
        #         glb_in_cnt += 1
                
        #         for i in range(0, len(non_global_maps)):
        #             map = non_global_maps[i]
        #             if map[int(center_point[1]), int(center_point[0])] == True: # *
        #                 zone_cnt_vars[i] += 1
        #                 pos_data.append(round((center_point[1] - slope *(center_point[0])), 2))
        #                 break




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

        # bp_data = []
        # for pt in center_point_lst:
        #     bp_data.append(pt)
        bp_data = center_point_lst.copy()
        # bp_data = center_points_lst
        visualize_bp_que.put(bp_data)

        if not draw_proc_result_que.full():
            draw_put_data = {'cnt_lst': [glb_in_cnt] + zone_cnt_vars, 'center_points_lst': center_point_lst, 'dets': dets }
            draw_proc_result_que.put(draw_put_data)

        frame_cnt += 1

        if det_result_que.qsize() > 10:
            print('Q size = ' , det_result_que.qsize())
            if det_result_que.full():
                print('Queue is full!!')
    print('area1 tracker end')


