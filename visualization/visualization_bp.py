import time
import cv2
from ._2d_to_2d_mapping import mapping, load_matrices
from functools import partial
import sys
import numpy as np


blueprint = cv2.imread('visualization/assets/blueprint.png')
height_bp, width_bp  = blueprint.shape[:2]

# car_image = cv2.imread('car_img.png')
# car_image = cv2.imread('backend/car_img_red.png')
car_image = cv2.imread('visualization/assets/car_img_red.png')
car_image = cv2.resize(car_image, (40, 40))
height_car, width_car  = car_image.shape[:2]

def draw_colorbar(img):
    font = cv2.FONT_HERSHEY_PLAIN
    h, w, c = img.shape
    box_w = int(w/7)
    box_h = int(h/30)
    box_start = (int(w/20), int(h/40))
    img = cv2.rectangle(img, box_start, (box_start[0] + box_w, box_start[1]+box_h), (0, 0, 0), 6, 8)
    blank_bar = np.zeros((box_h, box_w), dtype=np.uint8)
    for i in range(0, box_w):
        blank_bar[:,i] = int(254*(i/box_w))
    color_blank_bar =  cv2.applyColorMap(blank_bar, cv2.COLORMAP_JET)
    img[box_start[1]:box_start[1]+box_h, box_start[0]:box_start[0]+box_w] = color_blank_bar
    font_scale = 0.8
    cv2.putText(img, 'low', (box_start[0], box_start[1] + box_h + 20), font, font_scale, (0,0,0), 8, 10)
    cv2.putText(img, 'low', (box_start[0], box_start[1] + box_h + 20), font, font_scale, (255, 255, 255), 2, 10)
    cv2.putText(img, 'high', (box_start[0] + box_w + 10, box_start[1] + box_h + 20), font, font_scale, (0,0,0), 8, 10)
    cv2.putText(img, 'high', (box_start[0] + box_w + 10, box_start[1] + box_h + 20), font, font_scale, (255, 255, 255), 2, 10)




def draw_car_img(img ,ct_pt):
    try:
        img[int(ct_pt[1]):int(ct_pt[1]+height_car), int(ct_pt[0]):int(ct_pt[0]+width_car)] = car_image
    except ValueError:
        print('drawing car img value error. maybe, caused by whole size of img')

area1_matrices = load_matrices('visualization/mapping_matrix/mapping_area1')
area3_matrices = load_matrices('visualization/mapping_matrix/mapping_area3')
area4_matrices = load_matrices('visualization/mapping_matrix/mapping_area4')

mapping_area1 = partial(mapping, loaded_matrices=area1_matrices)
mapping_area3 = partial(mapping, loaded_matrices=area3_matrices)
mapping_area4 = partial(mapping, loaded_matrices=area4_matrices)



font =  cv2.FONT_HERSHEY_PLAIN
spinner_text_lst = [' |',' /',' -'," \\"]
spinner_period = 2
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
    cv2.putText(frame, 'progress indicator: ' + spinner_text, ((width_bp - 400), int(height_bp/10)), font, 2, (0, 0, 255), 2)
    return spinner_cnt
    




def visualize_bp(op_flag, que_area1, que_area3, que_area4, drawing_result_que, exit_event, collision_que, collision_op_flag, stay_time_op_flag, st_que, frame_provider_que_bp):
    # print('bp vis')
    collision_img = blueprint.copy()
    st_img = blueprint.copy()
    collision_img_once_come = False
    st_img_once_come = False
    spinner_cnt = 0
    send_cnt = 0
    while True:
        if exit_event.is_set():
            drawing_result_que.put(None)
            # sys.exit()
            break
    
        
        # if not op_flag.is_set():
        #     # while not drawing_result_que.empty():
        #     #     _ = drawing_result_que.get()
        #     while not que_area1.empty():
        #         _ = que_area1.get()
        #     while not que_area3.empty():
        #         _ = que_area1.get()
        #     while not que_area4.empty():
        #         _ = que_area1.get()
        #     # background_img = blueprint.copy()
        #     # drawing_result_que.put(background_img)
        #     # cv2.destroyAllWindows()
            
        # op_flag.wait()
        background_img = blueprint.copy()
        # print('c0')

        # if op_flag.is_set and (not(que_area1.empty() or que_area3.empty() or que_area4.empty())):
        if op_flag.is_set() and (not que_area1.empty()) and (not que_area3.empty()) and (not que_area4.empty()):
            # print('c1')
            # _ = que_area1.get()
            # _ = que_area3.get()
            # _ = que_area4.get()
            while not que_area1.empty():
                # _ = que_area1.get()
                area1_ct_pts_lst = que_area1.get()
            while not que_area3.empty():
                # _ = que_area3.get()
                area3_ct_pts_lst = que_area3.get()
            while not que_area4.empty():
                # _ = que_area4.get()
                area4_ct_pts_lst = que_area4.get()
            if type(area1_ct_pts_lst) == type(None) or type(area3_ct_pts_lst) == type(None) or type(area4_ct_pts_lst) == type(None):
                break
            # print('c2')
            area1_ct_pts_lst = [area1_ct_pts_lst[i] for i in range(0, len(area1_ct_pts_lst))]
            area3_ct_pts_lst = [area3_ct_pts_lst[i] for i in range(0, len(area3_ct_pts_lst))]
            area4_ct_pts_lst = [area4_ct_pts_lst[i] for i in range(0, len(area4_ct_pts_lst))]
            # area1_ct_pts_lst = [area1_ct_pts_lst[i][-1] for i in range(0, len(area1_ct_pts_lst))]
            # area3_ct_pts_lst = [area3_ct_pts_lst[i][-1] for i in range(0, len(area3_ct_pts_lst))]
            # area4_ct_pts_lst = [area4_ct_pts_lst[i][-1] for i in range(0, len(area4_ct_pts_lst))]
            # print('c3')
            mapped_pts_area1 = mapping_area1(pts_lst=area1_ct_pts_lst)
            mapped_pts_area3 = mapping_area3(pts_lst=area3_ct_pts_lst)
            mapped_pts_area4 = mapping_area4(pts_lst=area4_ct_pts_lst)
            # print('c4')
            for pt in mapped_pts_area1:
                bp_drawing_pt = (pt[0] - width_car/2, pt[1] - height_car/2)
                draw_car_img(background_img, bp_drawing_pt)
                cv2.circle(background_img, pt, 8, (255,0,255), -1)
            for pt in mapped_pts_area3:
                bp_drawing_pt = (pt[0] - width_car/2, pt[1] - height_car/2)
                draw_car_img(background_img, bp_drawing_pt)
                cv2.circle(background_img, pt, 8, (255,0,255), -1)
            for pt in mapped_pts_area4:
                bp_drawing_pt = (pt[0] - width_car/2, pt[1] - height_car/2)
                draw_car_img(background_img, bp_drawing_pt)
                cv2.circle(background_img, pt, 8, (255,0,255), -1)
            # print('c5')
        # elif not op_flag.is_set():
        else:
            # print('v1')
            while not que_area1.empty():
                # print('v2')
                _ = que_area1.get()
                # print('v3')
                time.sleep(0.01)
            while not que_area3.empty():
                _ = que_area3.get()
                time.sleep(0.01)
            while not que_area4.empty():
                _ = que_area4.get()
                time.sleep(0.01)
            # time.sleep(0.1)
            # print('else')

        if not collision_que.empty():
            collision_img = collision_que.get()
            collision_img_once_come = True
            # background_img = cv2.addWeighted(background_img, 0.6, collision_img, 0.4, 0)
            # draw_colorbar(background_img)
        if collision_op_flag.is_set() and collision_img_once_come:
            background_img = cv2.addWeighted(background_img, 0.5, collision_img, 0.5, 0)
            draw_colorbar(background_img)
        
        if not st_que.empty():
            st_img = st_que.get()
            st_img_once_come = True
        if stay_time_op_flag.is_set() and st_img_once_come:
            background_img = cv2.addWeighted(background_img, 0.5, st_img, 0.5, 0)
            draw_colorbar(background_img)

            
        # print('col op : ', collision_op_flag.is_set())
        # print('st op : ', stay_time_op_flag.is_set())
        
        if send_cnt%3 == 0:
            send_img = background_img.copy()
            frame_provider_que_bp.put(send_img)
            if send_cnt > 300 :
                send_cnt = 0
        # print('6')
        spinner_cnt = get_spinner_text(spinner_cnt, background_img)
        drawing_result_que.put(background_img)

        # cv2.namedWindow('blueprint', cv2.WINDOW_NORMAL)
        # cv2.imshow('blueprint', background_img)
        # cv2.waitKey(1)
        
        time.sleep(0.3)
        spinner_cnt += 1
        send_cnt += 1
        
        
    print('visualization bp end')