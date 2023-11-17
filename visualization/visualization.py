import cv2

# from .map_vars import area1_global_inout_map_non_false_idx, area1_car_wash_waiting_map_non_false_idx, area1_place0_map_non_false_idx, area1_global_inout_map_img, area1_car_wash_waiting_map_img, area1_place0_map_img
# from .map_vars import area3_global_inout_map_non_false_idx, area3_car_wash_waiting_map_non_false_idx, area3_global_inout_map_img, area3_car_wash_waiting_map_img
# from .map_vars import area4_global_inout_map_non_false_idx, area4_car_wash_waiting_map_non_false_idx, area4_electric_vehicle_charging_map_non_false_idx, area4_car_interior_washing_map_non_false_idx, area4_global_inout_map_img, area4_car_wash_waiting_map_img, area4_electric_vehicle_charging_map_img, area4_car_interior_washing_map_img
from .util_functions import get_bbox_conf
from .font import font
from .visualization_args import VisualizationArgs

import numpy as np
from PIL import Image, ImageTk

import time



spinner_text_lst = [' |',' /',' -'," \\"]
spinner_period = 3
def get_spinner_text(spinner_cnt, frame):
    if 0 <= spinner_cnt < spinner_period:
        spinner_text = spinner_text_lst[0]
    elif spinner_period <= spinner_cnt < 2*spinner_period:
        spinner_text = spinner_text_lst[1]
    elif 2*spinner_period <= spinner_cnt < 3*spinner_period:
        spinner_text = spinner_text_lst[2]
    elif 3*spinner_period<= spinner_cnt < 4*spinner_period:
        spinner_text = spinner_text_lst[3]
    else:
        spinner_text = spinner_text_lst[3]
        spinner_cnt = 0
    cv2.putText(frame, 'progress indicator: ' + spinner_text, (80, 120), font, 2, (0, 0, 255), 2)
    return spinner_cnt
    


def visualize(op_flag, area_display_value, selected_cam_num, img_q, proc_result_q, area_num_idx, drawing_result_ques, provider_que, exit_event, eco=False, ):
    frame_cnt = 0

    area_num_lst = [1, 3, 4]
    area_num = area_num_lst[area_num_idx]
    displayed_zone_name = 'None'
    eco_mode = eco
    
    visualization_args = VisualizationArgs(area_num)
    

    cnts_lst = visualization_args.cnts_lst
    cnts_lst_str = visualization_args.cnts_lst_str
    display_bool_lst = visualization_args.display_bool_lst
    slope = visualization_args.slope
    non_false_idx_lst = visualization_args.non_false_idx_lst
    map_imgs = visualization_args.map_imgs
    zone_name_strings = visualization_args.zone_name_strings
    
    
    drawing_result_que = drawing_result_ques[area_num_idx]
    
    spinner_cnt = 0
    
    prev_key = 0 # operation key
    
    random_ints = np.random.randint(80, 255, size=(30, 3), dtype=np.uint8)
    colors = [(int(random_ints[i][0]), int(random_ints[i][1]), int(random_ints[i][2])) for i in range(0, len(random_ints))]

    send_cnt = 0
    
    while True:
        if exit_event.is_set():
            break
        
        if not op_flag.is_set():
            while not proc_result_q.empty():
                _ = proc_result_q.get()
            # cv2.destroyAllWindows()
        
        # op_flag.wait()
        
        
        
        if eco_mode:
            _ = img_q.get()
        frame = img_q.get()
        if type(frame) == type(None):
            drawing_result_ques[area_num_idx].put(None)
            # sys.exit()
            break
        
        if op_flag.is_set(): #det result drawing only when op_flag is on
            if eco_mode:
                _ = proc_result_q.get()
            proc_result = proc_result_q.get()
            now_dets = proc_result['dets']
            dets_idx = proc_result['dets_idx']
            center_points_lst = proc_result['center_points_lst']
            for i in range(0, len(cnts_lst)):
                cnts_lst[i] = proc_result['cnt_lst'][i]
            

            for cnts_lst_idx in range(0, len(cnts_lst)):
                cnt_str = cnts_lst_str[cnts_lst_idx]
                cnt_value = cnts_lst[cnts_lst_idx]
                cv2.putText(frame, cnt_str + ': ' + str(cnt_value), (80, 120 + 35*(cnts_lst_idx+1)), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

            

            for i, det in enumerate(now_dets):
                # det = data
                car_idx = dets_idx[i]
                center_points = center_points_lst[i]
                color_idx = car_idx%30
                
                color_selected = colors[color_idx]
                # print('color sel = ' , color_selected)
                p_x1, p_y1 = int(det[0]), int(det[1])
                p_x2, p_y2 = int(det[2]), int(det[3])
                frame = cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), color_selected , 2)
                # frame_for_area_coloring[p_y1:p_y2, p_x1:p_x2] = color_selected
                # frame[p_y1:p_y2, p_x1:p_x2] = color_selected
    
                
                det_conf = round(det[-2], 2)

                area = (p_x2-p_x1)*(p_y2-p_y1)
                
                center_point = center_points[-1]
                refined_draw_ct_pt = center_points[0::4]
                refined_draw_ct_pt.append(center_point)
                inted_pts = np.int32([refined_draw_ct_pt])
                cv2.polylines(frame, inted_pts, False, color_selected, 6, lineType=8)
                cv2.line(frame, (int(center_point[0]), int(center_point[1])), (int(center_point[0]), int(center_point[1])), color_selected , thickness=12, lineType=None, shift=None)
                
                # cv2.putText(frame, f'id: {car_idx}', (int(center_point[0]), int(center_point[1]+20)), font, 1.5, color_selected, 2)
                # if 0 <= int(center_point[1]+33) <= 1070:
                #     cv2.putText(frame, f'det_conf: {det_conf}', (int(center_point[0]), int(center_point[1]+33)), font, 1.5, color_selected, 2)
                # else:
                #     cv2.putText(frame, f'det_conf: {det_conf}', (int(center_point[0]), int(center_point[1]-20)), font, 1.5, color_selected, 2)
                # cv2.putText(frame, f'aera: {area}', (int(center_point[0]), int(center_point[1]-33)), font, 1.5, color_selected, 2)

                # cv2.putText(frame, 'pos:: '+str(round((center_point[1] - slope *(center_point[0])), 2) ), (int(center_point[0]), int(center_point[1]+10)), font, 1, (0, 255, 0), 2)
                # cv2.putText(frame, 'conf: '+str(get_bbox_conf(data[2])), (int(center_point[0]), int(center_point[1]+30)), font, 1, (0, 255, 0), 2)
            
            frame_for_area_coloring = frame.copy()

            for i, det in enumerate(now_dets):
                car_idx = dets_idx[i]
                color_idx = car_idx%30
                color_selected = colors[color_idx]
                p_x1, p_y1 = int(det[0]), int(det[1])
                p_x2, p_y2 = int(det[2]), int(det[3])
                frame_for_area_coloring[p_y1:p_y2, p_x1:p_x2] = color_selected
            
            frame = cv2.addWeighted(frame, 0.72, frame_for_area_coloring, 0.28, 0)

            # frame_to_send_to_provider = frame.copy()
            # provider_que.put(frame_to_send_to_provider)
            if send_cnt%3==0:
                frame_to_send_to_provider = frame.copy()
                provider_que.put(frame_to_send_to_provider)
                if send_cnt > 300:
                    send_cnt = 0
            if provider_que.qsize() > 32:
                print(f'provider_q size at place {area_num} = ' , provider_que.qsize())

            # for pts in center_points_lst:
            #     # for i in range(0, len(pts)-1):
            #     #     former_pt = pts[i]
            #     #     next_pt = pts[i+1]
            #     #     cv2.line(frame, (int(former_pt[0]), int(former_pt[1])), (int(next_pt[0]), int(next_pt[1])), (0,255,255), thickness=4)
            #     # print('pts = ' , pts)
            #     inted_pts = np.int32([pts[0::4]])
            #     # polylines_template = frame.copy()
            #     # cv2.polylines(polylines_template, inted_pts, False, (200, 255, 40), 6, lineType=cv2.LINE_AA)
            #     # cv2.polylines(polylines_template, inted_pts, False, (200, 255, 40), 6, lineType=8)
            #     cv2.polylines(frame, inted_pts, False, (200, 255, 40), 6, lineType=8)
            #     # frame = cv2.addWeighted(frame, 0.6, polylines_template, 0.4, 0)
            #     center_point = pts[-1]
            #     # print('center point = ', center_point)
            #     cv2.line(frame, (int(center_point[0]), int(center_point[1])), (int(center_point[0]), int(center_point[1])), (0,255,0), thickness=12, lineType=None, shift=None)
            #     # cv2.putText(frame, 'pos:: '+str(round((center_point[1] - slope *(center_point[0])), 2) ), (int(center_point[0]), int(center_point[1]+10)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #     cv2.putText(frame, 'pos:: '+str(round((center_point[1] - slope *(center_point[0])), 2) ), (int(center_point[0]), int(center_point[1]+10)), font, 1, (0, 255, 0), 2)
            #     # cv2.putText(frame, 'conf: '+str(get_bbox_conf(data[2])), (int(center_point[0]), int(center_point[1]+30)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #     cv2.putText(frame, 'conf: '+str(get_bbox_conf(data[2])), (int(center_point[0]), int(center_point[1]+30)), font, 1, (0, 255, 0), 2)


        for i in range(0, len(non_false_idx_lst)):
            # non_false_idx = non_false_idx_lst[i]
            need_display = display_bool_lst[i]
            if need_display:
                map_img = map_imgs[i]
                # frame[non_false_idx] = cv2.addWeighted(frame, 0.5, map_img, 0.5, 0)[non_false_idx]
                frame = cv2.addWeighted(frame, 0.6, map_img, 0.4, 0)
                break
            
        
        key = area_display_value.get()
        if  key != prev_key:
            for i in range(0, len(display_bool_lst)):
                displayed_zone_name = 'None'
                display_bool_lst[i] = False
            if key != 0:
                for i in range(0, len(display_bool_lst)):
                    if i == key - 1:
                        displayed_zone_name = str(zone_name_strings[i])
                        display_bool_lst[i] = True
                        break
            prev_key = key
        # cv2.putText(frame, 'Displayed Zone: ' + displayed_zone_name, (80, 120 + 35*(len(cnts_lst))), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Displayed Zone: ' + displayed_zone_name, (1920 - 800, 80), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        

        if selected_cam_num.get() != area_num_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        spinner_cnt = get_spinner_text(spinner_cnt, frame)
        
        # print('drawing result que put')
        # drawing_result_ques[area_num_idx].put(frame)
        drawing_result_que.put(frame)
        # drawing_result_ques[area_num_idx+3].put(frame.copy())
        # cv2.namedWindow('frame: area'+str(area_num), cv2.WINDOW_NORMAL)
        # cv2.imshow('frame: area'+str(area_num), frame)
        
        frame_cnt += 1
        spinner_cnt += 1
        send_cnt += 1

        if img_q.qsize() > 100:
            print(f'img_q size at place {area_num} = ' , img_q.qsize())
            if img_q.full():
                while not img_q.empty():
                    _ = img_q.get()
                    time.sleep(0.005)
                print(f'img_q is full at place {area_num} in visualize fn. Clear img_q')
    print('visualization end')
    cv2.destroyAllWindows()

