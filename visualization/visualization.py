import cv2

from .map_vars import area1_global_inout_map_non_false_idx, area1_car_wash_waiting_map_non_false_idx, area1_place0_map_non_false_idx, area1_global_inout_map_img, area1_car_wash_waiting_map_img, area1_place0_map_img
from .map_vars import area3_global_inout_map_non_false_idx, area3_car_wash_waiting_map_non_false_idx, area3_global_inout_map_img, area3_car_wash_waiting_map_img
from .map_vars import area4_global_inout_map_non_false_idx, area4_car_wash_waiting_map_non_false_idx, area4_electric_vehicle_charging_map_non_false_idx, area4_car_interior_washing_map_non_false_idx, area4_global_inout_map_img, area4_car_wash_waiting_map_img, area4_electric_vehicle_charging_map_img, area4_car_interior_washing_map_img
from .util_functions import get_bbox_conf
from .font import font
from .visualization_args import VisualizationArgs

from PIL import Image, ImageTk

import sys



def visualize(op_flag, area_display_value, img_q, proc_result_q, area_num_idx, drawing_result_ques, exit_event, eco=False, ):
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
    
    # if area_num == 1:
    #     glb_in_cnt = 0
    #     car_wash_waiting_cnt = 0
    #     place0_cnt = 0
    #     cnts_lst = [glb_in_cnt, car_wash_waiting_cnt, place0_cnt]
    #     cnts_lst_str = ['global in', 'car wash wait', 'place0']
    #     area1_view_global_map = False
    #     area1_view_car_wash_waiting_map = False
    #     area1_view_place0_map = False
    #     display_bool_lst = [area1_view_global_map, area1_view_car_wash_waiting_map, area1_view_place0_map]
    #     slope = -0.5353805073431241
    #     non_false_idx_lst = [area1_global_inout_map_non_false_idx, area1_car_wash_waiting_map_non_false_idx, area1_place0_map_non_false_idx]
    #     map_imgs = [area1_global_inout_map_img, area1_car_wash_waiting_map_img, area1_place0_map_img]
    #     zone_name_strings = ['Global In Out', 'Car Wash Waiting', 'Place0']
    # elif area_num == 3:
    #     glb_in_cnt = 0
    #     car_wash_waiting_cnt = 0
    #     cnts_lst = [glb_in_cnt, car_wash_waiting_cnt]
    #     cnts_lst_str = ['global in', 'car wash wait']
    #     area3_view_global_map = False
    #     area3_view_car_wash_waiting_map = False
    #     display_bool_lst = [area3_view_global_map, area3_view_car_wash_waiting_map]
    #     slope = 0.657103825136612
    #     non_false_idx_lst = [area3_global_inout_map_non_false_idx, area3_car_wash_waiting_map_non_false_idx]
    #     map_imgs = [area3_global_inout_map_img, area3_car_wash_waiting_map_img]
    #     zone_name_strings = ['Global In Out', 'Car Wash Waiting']
    # elif area_num == 4:
    #     glb_in_cnt = 0
    #     car_wash_waiting_cnt = 0
    #     electric_vehicle_charging_waiting_cnt = 0
    #     car_interior_washing_waiting_cnt = 0
    #     cnts_lst = [glb_in_cnt, car_wash_waiting_cnt, electric_vehicle_charging_waiting_cnt, car_interior_washing_waiting_cnt] # *
    #     cnts_lst_str = ['global in', 'car wash wait', 'elctric charging', 'car interior wash'] # *
    #     area4_view_global_map = False
    #     area4_view_car_wash_waiting_map = False
    #     area4_view_electric_charging_map = False
    #     area4_view_car_interior_wash_map = False
    #     display_bool_lst = [area4_view_global_map, area4_view_car_wash_waiting_map, area4_view_electric_charging_map, area4_view_car_interior_wash_map] # *
    #     slope = 0.22451456310679613 # *
    #     non_false_idx_lst = [area4_global_inout_map_non_false_idx, area4_car_wash_waiting_map_non_false_idx, area4_electric_vehicle_charging_map_non_false_idx, area4_car_interior_washing_map_non_false_idx] # *
    #     map_imgs = [area4_global_inout_map_img, area4_car_wash_waiting_map_img, area4_electric_vehicle_charging_map_img, area4_car_interior_washing_map_img] # *
    #     zone_name_strings = ['Global In Out', 'Car Wash Waiting', 'Electric Charging Zone', 'Car Interior Washing Zone'] # *

    # available_key_lst = [ord(str(i)) for i in range(0, len(display_bool_lst)+1)]
    prev_key = 0 # operation key
    
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
            sys.exit()
            break
        
        if op_flag.is_set():
            if eco_mode:
                _ = proc_result_q.get()
            proc_result = proc_result_q.get()
            dets = proc_result['dets']
            center_points_lst = proc_result['center_points_lst']
            for i in range(0, len(cnts_lst)):
                cnts_lst[i] = proc_result['cnt_lst'][i]
            
            # for i in range(0, len(non_false_idx_lst)):
            #     # non_false_idx = non_false_idx_lst[i]
            #     map_img = map_imgs[i]
            #     need_display = display_bool_lst[i]
            #     if need_display:
            #         # frame[non_false_idx] = cv2.addWeighted(frame, 0.5, map_img, 0.5, 0)[non_false_idx]
            #         frame = cv2.addWeighted(frame, 0.6, map_img, 0.4, 0)
            #         break
            
            
            # cv2.putText(frame, 'frame cnt: '+str(frame_cnt), (80, 80), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
            for cnts_lst_idx in range(0, len(cnts_lst)):
                cnt_str = cnts_lst_str[cnts_lst_idx]
                cnt_value = cnts_lst[cnts_lst_idx]
                cv2.putText(frame, cnt_str + ': ' + str(cnt_value), (80, 120 + 35*cnts_lst_idx), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

            for data in dets:
                det = data
                p_x1, p_y1 = int(det[0]), int(det[1])
                p_x2, p_y2 = int(det[2]), int(det[3])
                frame = cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (255, 0, 0), 2)
            for data in center_points_lst:
                center_point = data[0]
                cv2.line(frame, (int(center_point[0]), int(center_point[1])), (int(center_point[0]), int(center_point[1])), (0,255,0), thickness=12, lineType=None, shift=None)
                cv2.putText(frame, 'pos:: '+str(round((center_point[1] - slope *(center_point[0])), 2) ), (int(center_point[0]), int(center_point[1]+10)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'conf: '+str(get_bbox_conf(data[2])), (int(center_point[0]), int(center_point[1]+30)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # key = area_display_value.get()
            # if  key != prev_key:
            #     for i in range(0, len(display_bool_lst)):
            #         displayed_zone_name = 'None'
            #         display_bool_lst[i] = False
            #     if key != 0:
            #         for i in range(0, len(display_bool_lst)):
            #             if i == key - 1:
            #                 displayed_zone_name = str(zone_name_strings[i])
            #                 display_bool_lst[i] = True
            #                 break
            #     prev_key = key
        
            # key  = cv2.waitKey(1) & 0xFF
            # if key == ord('q'):
            #     break
            # elif key != 255: # not no input
            #     if key in available_key_lst:
            #         for i in range(0, len(display_bool_lst)):
            #             displayed_zone_name = 'None'
            #             display_bool_lst[i] = False
            #         for i in range(0, len(display_bool_lst)):
            #             if key == ord(str(i+1)):
            #                 displayed_zone_name = str(zone_name_strings[i])
            #                 display_bool_lst[i] = True
            #                 break
            
            # cv2.putText(frame, 'Displayed Zone: ' + displayed_zone_name, (80, 120 + 35*(len(cnts_lst))), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        
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
        
        # print('drawing result que put')
        drawing_result_ques[area_num_idx].put(frame)
        # drawing_result_ques[area_num_idx+3].put(frame.copy())
        # cv2.namedWindow('frame: area'+str(area_num), cv2.WINDOW_NORMAL)
        # cv2.imshow('frame: area'+str(area_num), frame)
        
        frame_cnt += 1

        if img_q.qsize() > 100:
            print(f'img_q size at place {area_num} = ' , img_q.qsize())
            if img_q.full():
                while not img_q.empty():
                    _ = img_q.get() 
                print(f'img_q is full at place {area_num}. Clear img_q')
    print('visualization end')
    cv2.destroyAllWindows()
    
# def cvimg_to_tkimg(frame, width, height):
#     frame = cv2.resize(frame.copy(), (int(width), int(height)))
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(frame)
#     imgtk = ImageTk.PhotoImage(image=img)
#     return imgtk