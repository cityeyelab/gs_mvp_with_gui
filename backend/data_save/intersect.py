import numpy as np
import cv2
import dill
import datetime
import sys
import os
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from  _2d_to_2d_mapping import mapping, load_matrices
from functools import partial
import time

import math

import pickle

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


total_start = time.time()


# filename = 'data/2023-10-15_raw_data'
filename = 'data/2023-10-18_raw_data_7h30m'

data = []
with open(filename, 'rb') as f:
    try:
        while True:
            # data.append(pickle.load(f))
            data.append(dill.load(f, True))
    except EOFError:
        pass

# __slots__ = ['area_num', 'id', 'bboxes', 'center_points_lst', 'frame_record', 'created_at', 'removed_at']

# empty_template = np.zeros((1080, 1920, 3), dtype=np.uint8)
path = 'backend/data_save/area4_sample.png'
area4_matrices = load_matrices('visualization/mapping_matrix/mapping_area4')
area1_matrices = load_matrices('visualization/mapping_matrix/mapping_area1')
area3_matrices = load_matrices('visualization/mapping_matrix/mapping_area3')
mapping_area4 = partial(mapping, loaded_matrices=area4_matrices)
mapping_area1 = partial(mapping, loaded_matrices=area1_matrices)
mapping_area3 = partial(mapping, loaded_matrices=area3_matrices)

bp_background = cv2.imread('visualization/assets/blueprint.png')

empty_template = cv2.imread(path)
font = cv2.FONT_HERSHEY_PLAIN



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


def get_distane(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    return ((x2-x1)**2 + (y2-y1)**2)**(1/2)

refined_pts_lst = []
for pts in mapped_pts:
    print('len pts = ' , len(pts))
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
        # print('last idx = ', last_idx)
        # print('len(pts) = ' , len(pts))
        # print('len pts = ' , len(pts))
        # print('last index + 1 = ', last_idx+1)
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
        
    # new_lst.append(pts[-1])
    refined_pts_lst.append(new_lst)
    print('lst len  = ' , len(new_lst))
    print('contained nodes len = ' , len(contained_nodes_number_lst))
    print('contained nodes number list = ' , contained_nodes_number_lst)
    print('sum contained nodes number list = ' , sum(contained_nodes_number_lst))
    print('idx record = ' , index_record)
    # print('refined')



# for i in range(0, len(refined_pts_lst)):
#     backgound = bp_background.copy()
#     cv2.circle(backgound, (100, 100), 4, (0, 0, 255), -1)
#     cv2.circle(backgound, (100, 120), 4, (0, 0, 255), -1)
#     refined_pts = refined_pts_lst[i]
#     unrefined_pts = mapped_pts[i]
#     draw_center_points(backgound, unrefined_pts, (0, 0, 255))
#     draw_center_points(backgound, refined_pts, (255, 0, 0))
#     # cv2.imshow('result', backgound)
#     # cv2.waitKey(0)





def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def get_angle(v1, v2):
    return math.acos(abs(dotproduct(v1, v2)) / (length(v1) * length(v2)))    

h, w, c = bp_background.shape
glb_cvs = np.zeros((h, w), dtype=np.int64)
print('process starts')
print('len refined pts = ' , len(refined_pts_lst))
start = time.time()
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
                        # results.append((result, prev_pt, next_pt, other_prev_pt, other_next_pt))
                    # print('result :  ' , result)
end = time.time()
print(f'time: {end - start}')


# start = time.time()
# results = []
# for i in range(0, len(mapped_pts)):
#     pts = mapped_pts[i]
    
#     for j in range(0, len(pts)-1):
#         prev_pt = pts[i]
#         next_pt = pts[i+1]
#         for k in range(i+1, len(mapped_pts)):
#             other_pts = mapped_pts[k]
#             for l in range(0, len(other_pts)-1):
#                 other_prev_pt = other_pts[l]
#                 other_next_pt = other_pts[l+1]
#                 result = intersect(prev_pt, next_pt, other_prev_pt, other_next_pt)
#                 if type(result) != type(None):
#                     results.append(result)
#                     # print('result :  ' , result)
# end = time.time()
# print(f'time: {end - start}')



with open("data.pickle","wb") as f:
    pickle.dump(results, f)
    
total_end = time.time()
total_elapsed_time = total_end - total_start

print('total elapsed time = ' , total_elapsed_time)