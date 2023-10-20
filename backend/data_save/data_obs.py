import numpy as np
import cv2
import dill
import datetime
import sys
import os
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from  _2d_to_2d_mapping import mapping, load_matrices
from functools import partial


print(os.getcwd())
sys.path.append('C:/Users/EunKyue Sohn/OneDrive - 시티아이랩/문서/python_scripts/gs_simple_tracker_demo_gui/backend/trk_fns/')
class TrackingData():
    __slots__ = ['area_num', 'id', 'age', 'bboxes', 'center_points_lst', 'frame_record', 'created_at', 'removed_at']
    
    def __init__(self, area_num, id) -> None:
        self.area_num = area_num
        self.id = id
        self.age = 0
        self.bboxes = []
        self.center_points_lst = []
        # self.time_stamp = []
        self.frame_record = []
        self.created_at = datetime.datetime.now()
        self.removed_at = datetime.datetime.now()
        
    def __repr__(self) -> str:
        # return f"obj_id:{id(self)}, area_num: {self.area_num}, id: {self.id}, bboxes_orig: {self.bboxes_orig}, frame_rec : {self.frame_record}"
        return f"(obj_id:{id(self)}, area_num: {self.area_num}, id: {self.id}, created_at: {self.created_at}, removed_at: {self.removed_at})"

filename = 'data/2023-10-15_raw_data'

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
mapping_area4 = partial(mapping, loaded_matrices=area4_matrices)

bp_background = cv2.imread('visualization/assets/blueprint.png')






empty_template = cv2.imread(path)
font = cv2.FONT_HERSHEY_PLAIN

def draw_center_points(frame, center_points, color):
    initial_pt = center_points[0]
    end_pt = center_points[-1]
    inted_pts = np.int32([center_points])
    cv2.polylines(frame, inted_pts, False, color, 6, lineType=8)
    cv2.circle(frame, (int(initial_pt[0]), int(initial_pt[1])), 4, (0, 0, 255), -1)
    cv2.putText(frame, 'start', (int(initial_pt[0]), int(initial_pt[1])), font, 2, (0, 255, 210), 2)
    cv2.circle(frame, (int(end_pt[0]), int(end_pt[1])), 4, (0, 0, 255), -1)
    cv2.putText(frame, 'end', (int(end_pt[0]), int(end_pt[1])), font, 2, (255, 255, 55), 2)
    # for i, pt in enumerate(center_points):
    #     cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (255, 0, 255), -1)
    #     cv2.putText(frame, str(i), (int(pt[0]+5), int(pt[1]+5)), font, 2, (255, 255, 55), 2)

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


h, w, c = bp_background.shape
draw_template = np.zeros((h, w), dtype=np.uint8)
def draw_path(center_points):
    draw_temp = draw_template.copy()
    inted_pts = np.int32([center_points])
    cv2.polylines(draw_temp, [inted_pts], False, (255, 255, 255), 30, lineType=8)
    return draw_temp


# if __name__ == '__main__':


smoothed_center_points_lst = []
for data_cls in data:
    center_points = data_cls.center_points_lst
    smoothed_center_points = smooth_center_points(center_points.copy())
    smoothed_center_points_lst.append(smoothed_center_points)
    
    
mapped_pts = []
for smoothed_ct_pt in smoothed_center_points_lst:
    mapped_pts_area4 = mapping_area4(pts_lst=smoothed_ct_pt)
    mapped_pts.append(mapped_pts_area4)

mapped_pt_filtered = []
for mapped_pt in mapped_pts:
    first_pt = mapped_pt[0]
    end_pt = mapped_pt[-1]
    dist = ((first_pt[0]-end_pt[0])**2 + (first_pt[1]-end_pt[1])**2)**(1/2)
    if dist > 200:
        mapped_pt_filtered.append(mapped_pt)
    

mapped_pts = mapped_pt_filtered


h, w, c = bp_background.shape
bp_paths = []
bp_empty = np.full((h, w), False)
bp_draws = []
for i, mapped_pt in enumerate(mapped_pts):
    # print('mapped pt = ' , mapped_pt)
    bp_temp = bp_empty.copy()
    drawed = draw_path(mapped_pt)
    # drawed = cv2.cvtColor(drawed, cv2.COLOR_BGR2GRAY)
    # cv2.putText(drawed, str(i), (int(50), int(50)), font, 4, (255, 255, 210), 2)
    # cv2.imshow('drawed', drawed)
    # cv2.waitKey(0)
    # record_template[touch_record.any(axis=2) != 0] = True
    bp_temp[drawed != 0 ] = True
    bp_draws.append(drawed)
    bp_paths.append(bp_temp)

h, w, c = bp_background.shape
glb_cvs = np.zeros((h, w), dtype=np.int64)
print('path length = ' , len(bp_paths))
for i, path in enumerate(bp_paths):
    this_draw = bp_draws[i]
    empty_canvas = np.zeros((h, w), dtype=np.int64)
    for j in range(i+1, len(bp_paths)):
        # if i != j:
        if True:
            that_path = bp_paths[j]
            mult_result = np.multiply(path, that_path)
            # empty_canvas += mult_result
            glb_cvs += mult_result
    # max_val = np.max(empty_canvas)
    # if max_val > 0:
    #     scaled_result = 254*(empty_canvas/max_val)
    # else:
    #     scaled_result = empty_canvas
    # scaled_result = np.uint8(scaled_result)
    if i % 10 == 0:
        print('progress check : ', i)
    # cv2.imshow('result', scaled_result)
    # cv2.waitKey(0)
    # if i == 20:
    #     break

print('breaked')
glb_max_val = np.max(glb_cvs)
scaled_glb_cvs = 254*(glb_cvs/glb_max_val)
glb_result = np.uint8(scaled_glb_cvs)
# cv2.imshow('glb result', glb_result)
# cv2.waitKey(0)

glb_result = cv2.cvtColor(glb_result, cv2.COLOR_GRAY2BGR)
# cv2.imshow('glb result', glb_result)
# cv2.waitKey(0)


result_draw = bp_background.copy()

# target_idx = np.where(glb_result != 0)
# # result_draw[target_idx] = glb_result[target_idx]
# result_draw[glb_result.any(axis=2) != 0] = glb_result[glb_result.any(axis=2) != 0] * (0, 0, 1) + result_draw[glb_result.any(axis=2) != 0] * (1, 1, 0)
# cv2.imshow('final result', result_draw)
# cv2.waitKey(0)

add_weighted = cv2.addWeighted(result_draw, 0.5 , glb_result, 0.5, 0)
cv2.imshow('final', add_weighted)
cv2.waitKey(0)

# for path in bp_paths:

# for i in range(0, len(bp_paths)):
#     this_path = bp_paths[i]
#     canvas = this_path.copy()
#     print('canvas = ' , canvas)
#     cv2.imshow('canvas', canvas)
#     cv2.waitKey(0)
#     # for j in range(i+1, len(bp_paths)):
#     #     that_path = bp_paths[j]
#     #     result = cv2.bitwise_and(this_path, that_path)
#     #     cv2.imshow('result', result)
#     #     cv2.waitKey(0)



# cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
# cv2.namedWindow('bp', cv2.WINDOW_NORMAL)

# for data_cls in data:
#     screen = empty_template.copy()
#     center_points = data_cls.center_points_lst
#     draw_center_points(screen, center_points, (200, 255, 40))
#     smoothed_center_points = smooth_center_points(center_points.copy())
#     draw_center_points(screen, smoothed_center_points, (0, 0, 255))
#     # initial_pt = center_points[0]
#     # end_pt = center_points[-1]
#     # inted_pts = np.int32([center_points])
#     # cv2.polylines(screen, inted_pts, False, (200, 255, 40), 6, lineType=8)
#     # cv2.circle(screen, (int(initial_pt[0]), int(initial_pt[1])), 4, (0, 0, 255), -1)
#     # cv2.putText(screen, 'start', (int(initial_pt[0]), int(initial_pt[1])), font, 2, (0, 255, 210), 2)
#     # cv2.circle(screen, (int(end_pt[0]), int(end_pt[1])), 4, (0, 0, 255), -1)
#     # cv2.putText(screen, 'end', (int(end_pt[0]), int(end_pt[1])), font, 2, (255, 255, 55), 2)
#     background = bp_background.copy()
#     mapped_pts_area4 = mapping_area4(pts_lst=smoothed_center_points)
#     draw_center_points(background, mapped_pts_area4, (200, 255, 40))
    
#     cv2.imshow('screen', screen)
#     cv2.imshow('bp', background)
#     cv2.waitKey(0)


