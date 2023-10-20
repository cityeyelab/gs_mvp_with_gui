from _2d_to_2d_mapping import mapping, load_matrices
from functools import partial
import cv2
import numpy as np
import pickle
import math
import datetime
import time

class TrackingData():
    __slots__ = ['area_num', 'id', 'age', 'bboxes', 'center_points_lst', 'frame_record', 'created_at', 'removed_at']
    
    def __init__(self, area_num, id, bboxes, center_points_lst, frame_record, created_at, removed_at) -> None:
        self.area_num = area_num
        self.id = id
        # self.age = 0
        self.bboxes = bboxes
        self.center_points_lst = center_points_lst
        # self.time_stamp = []
        self.frame_record = frame_record
        self.created_at = created_at
        self.removed_at = removed_at
        
    def __repr__(self) -> str:
        # return f"obj_id:{id(self)}, area_num: {self.area_num}, id: {self.id}, bboxes_orig: {self.bboxes_orig}, frame_rec : {self.frame_record}"
        return f"(obj_id:{id(self)}, area_num: {self.area_num}, id: {self.id}, created_at: {self.created_at}, removed_at: {self.removed_at})"

def cvt_pkl_to_cls(pkl):
    new_cls = TrackingData(pkl[0], pkl[1], pkl[2], pkl[3], pkl[4], pkl[5], pkl[6])
    return new_cls

def intersect(p1, p2, p3, p4):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x,y)


def draw_center_points(frame, center_points, color):
    initial_pt = center_points[0]
    end_pt = center_points[-1]
    inted_pts = np.int32([center_points])
    cv2.polylines(frame, [inted_pts], False, color, 6, lineType=8)
    cv2.circle(frame, (int(initial_pt[0]), int(initial_pt[1])), 4, (0, 0, 255), -1)
    cv2.putText(frame, 'start', (int(initial_pt[0]), int(initial_pt[1])), font, 2, (0, 255, 210), 2)
    cv2.circle(frame, (int(end_pt[0]), int(end_pt[1])), 4, (0, 0, 255), -1)
    cv2.putText(frame, 'end', (int(end_pt[0]), int(end_pt[1])), font, 2, (255, 255, 55), 2)

def smooth_center_points(lst):
    # print('orig lst = ' , lst)
    len_lst = len(lst)
    x = [lst[i][0] for i in range(0, len_lst)]
    y = [lst[i][1] for i in range(0, len_lst)]
    smoothed_x = smooth_along_one_axis(x, 5)
    smoothed_y = smooth_along_one_axis(y, 5)
    result = [(smoothed_x[i], smoothed_y[i]) for i in range(0, len_lst)]
    # print('smoothed result = ' , result)
    return result


def smooth_along_one_axis(lst, window_size):
    len_lst = len(lst)
    results_lst = [lst[0]]
    for i in range(1, len(lst)-1):
        first_idx = max(i - window_size, 1)
        last_idx = min(i + window_size, len_lst-1)
        target_lst = lst[first_idx:last_idx]
        result = sum(target_lst)/(len(target_lst))
        results_lst.append(result)
    results_lst.append(lst[-1])
    return results_lst


def get_distane(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    return ((x2-x1)**2 + (y2-y1)**2)**(1/2)




def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def get_angle(v1, v2):
    return math.acos(abs(dotproduct(v1, v2)) / (length(v1) * length(v2)))    


# start = time.time()

filename="_raw_data"
now = datetime.datetime.now()
today_string = now.strftime('%Y-%m-%d')
file_name = 'data/'+ today_string + filename
filename = 'data/2023-10-19_raw_data'



while True:
    data = []
    # with open(filename, 'rb') as f:
    #     try:
    #         while True:
    #             # data.append(pickle.load(f))
    #             data.append(pickle.load(f))
    #     except EOFError:
    #         pass

    with open(filename, 'rb') as f:
        try:
            while True:
                # data.append(pickle.load(f))
                loaded_data = pickle.load(f)
                cls_cvt = cvt_pkl_to_cls(loaded_data)
                data.append(cls_cvt)
        except EOFError:
            pass

    bg_path = 'backend/data_save/area4_sample.png'
    area4_matrices = load_matrices('visualization/mapping_matrix/mapping_area4')
    area1_matrices = load_matrices('visualization/mapping_matrix/mapping_area1')
    area3_matrices = load_matrices('visualization/mapping_matrix/mapping_area3')
    mapping_area4 = partial(mapping, loaded_matrices=area4_matrices)
    mapping_area1 = partial(mapping, loaded_matrices=area1_matrices)
    mapping_area3 = partial(mapping, loaded_matrices=area3_matrices)

    bp_background = cv2.imread('visualization/assets/blueprint.png')

    empty_template = cv2.imread(bg_path)
    font = cv2.FONT_HERSHEY_PLAIN




    smoothed_center_points_lst = []
    for data_cls in data:
        center_points = data_cls.center_points_lst
        smoothed_center_points = smooth_center_points(center_points.copy())
        smoothed_center_points_lst.append(smoothed_center_points)
        
        
    mapped_pts = []
    for i, smoothed_ct_pt in enumerate(smoothed_center_points_lst):
        area_num = data[i].area_num
        if area_num == 4:
            mapped_pts_area4 = mapping_area4(pts_lst=smoothed_ct_pt)
            mapped_pts.append(mapped_pts_area4)
        if area_num == 1:
            mapped_pts_area1 = mapping_area1(pts_lst=smoothed_ct_pt)
            mapped_pts.append(mapped_pts_area1)
        if area_num == 3:
            mapped_pts_area3 = mapping_area3(pts_lst=smoothed_ct_pt)
            mapped_pts.append(mapped_pts_area3)


    mapped_pt_filtered = []
    for mapped_pt in mapped_pts:
        first_pt = mapped_pt[0]
        end_pt = mapped_pt[-1]
        dist = ((first_pt[0]-end_pt[0])**2 + (first_pt[1]-end_pt[1])**2)**(1/2)
        if dist > 200:
            mapped_pt_filtered.append(mapped_pt)
    mapped_pts = mapped_pt_filtered


    refined_pts_lst = []
    for pts in mapped_pts:
        # print('len pts = ' , len(pts))
        new_lst = []
        contained_nodes_number_lst = []
        index_record = []
        new_lst.append(pts[0])
        last_idx = 0
        is_complete = False
        while True:
            if is_complete:
                break
            last_pt = pts[last_idx]
            for j in range(last_idx+1, len(pts)):
                dist = get_distane(last_pt, pts[j])
                if dist > 10:
                    contained_nodes_number = j - last_idx
                    contained_nodes_number_lst.append(contained_nodes_number)
                    index_record.append(j)
                    last_idx = j
                    new_lst.append(pts[j])
                    if j == len(pts)-1:
                        is_complete = True
                        # contained_nodes_number_lst[-1] += 1 # 좌측 closed, 우측 open interval이라서 보정. -> 중간에 노드갯수 셀때는 각 수치에서 하나씩 빼서 쓰면 되서, (마지막점 빼고) 보정안하고 그냥 써도 될 듯
                    break
                elif j == len(pts)-1:
                    contained_nodes_number = j - last_idx
                    contained_nodes_number_lst.append(contained_nodes_number)
                    index_record.append(j)
                    new_lst.append(pts[j])
                    is_complete = True
                    break
        refined_pts_lst.append(new_lst)


    h, w, c = bp_background.shape
    glb_cvs = np.zeros((h, w), dtype=np.int64)
    results = []
    for i in range(0, len(refined_pts_lst)):
        pts = refined_pts_lst[i]
        
        for j in range(0, len(pts)-1):
            prev_pt = pts[j]
            next_pt = pts[j+1]
            for k in range(i+1, len(refined_pts_lst)):
                other_pts = refined_pts_lst[k]
                for l in range(0, len(other_pts)-1):
                    other_prev_pt = other_pts[l]
                    other_next_pt = other_pts[l+1]
                    result = intersect(prev_pt, next_pt, other_prev_pt, other_next_pt)
                    if type(result) != type(None):
                        v1_x = prev_pt[0] - next_pt[0]
                        v1_y = prev_pt[1] - next_pt[1]
                        v2_x = other_prev_pt[0] - other_next_pt[0]
                        v2_y = other_prev_pt[1] - other_next_pt[1]
                        v1 = (v1_x, v1_y)
                        v2 = (v2_x, v2_y)
                        angle = get_angle(v1, v2)
                        if angle > math.pi/6:
                            results.append((result, prev_pt, next_pt, other_prev_pt, other_next_pt))


    #############################################################




    bp_background = cv2.imread('visualization/assets/blueprint.png')
    h, w, c = bp_background.shape
    # with open("data.pickle","rb") as f:
    #     data = pickle.load(f)
    # print(data)
    data = results

    def strech_vec(v1, v2): #vec = ()
        direction = (v2[0] - v1[0], v2[1] - v1[1])
        former_pt = (v1[0] - direction[0], v1[1] - direction[1])
        futher_pt = (v2[0] + direction[0], v2[1] + direction[1])
        return former_pt, futher_pt


    def draw_colorbar(img):
        font = cv2.FONT_HERSHEY_PLAIN
        # start = time.time()
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
        cv2.putText(img, 'low', (box_start[0], box_start[1] + box_h + 20), font, font_scale, (0,0,0), 6, 8)
        cv2.putText(img, 'low', (box_start[0], box_start[1] + box_h + 20), font, font_scale, (255, 255, 255), 1, 8)
        cv2.putText(img, 'high', (box_start[0] + box_w + 10, box_start[1] + box_h + 20), font, font_scale, (0,0,0), 6, 8)
        cv2.putText(img, 'high', (box_start[0] + box_w + 10, box_start[1] + box_h + 20), font, font_scale, (255, 255, 255), 1, 8)
        # end = time.time()
        # color_bar_time = end - start
        # print('colorbar time = ', color_bar_time)

    # print(len(data))
    glb_cvs = np.zeros((h, w), dtype=np.int64)
    empty_temp = np.full((h, w), False)
    draw_template = np.zeros((h, w), dtype=np.uint8)
    cnt = 0
    for item in data:
        cnt+=1
        
        draw = draw_template.copy()
        tf_template = empty_temp.copy()
        
        inter_pt, prev_pt, next_pt, other_prev_pt, other_next_pt = item
        
        cv2.circle(draw, (int(inter_pt[0]), int(inter_pt[1])), 30, (255, 255, 255), -1)
        # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('draw', draw)
        # cv2.waitKey(0)
        tf_template[draw != 0 ] = True
        
        glb_cvs += tf_template
        
        
        # cv2.circle(draw, (int(inter_pt[0]), int(inter_pt[1])), 30, (255, 255, 255), -1)
        # tf_template[draw != 0 ] = True
        # glb_cvs += tf_template
        
        # print('sum = ' , np.sum(glb_cvs))
        ################################
        # # print(inter_pt)
        # temp_img = bp_background.copy()
        # former_pt, futher_pt = strech_vec(prev_pt, next_pt)
        # other_former_pt, other_futher_pt = strech_vec(other_prev_pt, other_next_pt)
        # cv2.arrowedLine(temp_img, (int(former_pt[0]), int(former_pt[1])), (int(futher_pt[0]), int(futher_pt[1])), (150, 100, 80), 3, 8, tipLength=0.2)
        # cv2.arrowedLine(temp_img, (int(other_former_pt[0]), int(other_former_pt[1])), (int(other_futher_pt[0]), int(other_futher_pt[1])), (150, 80, 120), 3, 8, tipLength=0.3)
        # # cv2.line(temp_img, (int(former_pt[0]), int(former_pt[1])), (int(futher_pt[0]), int(futher_pt[1])), (255, 255, 0), 4, 8)
        # # cv2.line(temp_img, (int(other_former_pt[0]), int(other_former_pt[1])), (int(other_futher_pt[0]), int(other_futher_pt[1])), (255, 255, 0),  4, 8)
        # cv2.line(temp_img, (int(prev_pt[0]), int(prev_pt[1])), (int(next_pt[0]), int(next_pt[1])), (100, 255, 0), 4, 8)
        # cv2.line(temp_img, (int(other_prev_pt[0]), int(other_prev_pt[1])), (int(other_next_pt[0]), int(other_next_pt[1])), (0, 255, 180),  4, 8)
        # cv2.circle(temp_img, (int(inter_pt[0]), int(inter_pt[1])), 4, (0, 0, 255), -1)
        # cv2.imshow('t', temp_img)
        # if cnt <= 10:
        #     wk = 0
        # else:
        #     wk = 10
        # cv2.waitKey(wk)
        ################################


    glb_max_val = np.max(glb_cvs)
    scaled_glb_cvs = 254*(glb_cvs/glb_max_val)
    glb_result = np.uint8(scaled_glb_cvs)

    res_show = cv2.applyColorMap(glb_result, cv2.COLORMAP_JET)
    res_show = cv2.GaussianBlur(res_show,(13,13), 11)
    # cv2.imshow('glb result', glb_result)
    # cv2.imshow('glb result', res_show)
    # cv2.waitKey(0)

    # glb_result = cv2.cvtColor(glb_result, cv2.COLOR_GRAY2BGR)

    # end = time.time()
    # elapsed_time = end - start
    # print('elapsed time = ' , elapsed_time)

    # result = cv2.addWeighted(bp_background.copy(), 0.5, glb_result, 0.5, 0)
    bg_ratio = 0.6
    result = cv2.addWeighted(bp_background.copy(), bg_ratio, res_show, 1-bg_ratio, 0)
    draw_colorbar(result)
    cv2.imshow('result1', result)
    non_zero_idx = glb_result != 0
    result[glb_result == 0] = bp_background[glb_result == 0]
    draw_colorbar(result)
    cv2.imshow('result2', result)
    cv2.waitKey(0)
    time.sleep(60*3)