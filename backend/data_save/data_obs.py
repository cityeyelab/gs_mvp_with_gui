import numpy as np
import cv2
import dill

filename = 'data/2023-10-15_raw_data'

data = []
with open(filename, 'rb') as f:
    try:
        while True:
            # data.append(pickle.load(f))
            data.append(dill.load(f))
    except EOFError:
        pass

# __slots__ = ['area_num', 'id', 'bboxes', 'center_points_lst', 'frame_record', 'created_at', 'removed_at']

# empty_template = np.zeros((1080, 1920, 3), dtype=np.uint8)
path = 'backend/data_save/area4_sample.png'

empty_template = cv2.imread(path)
font = cv2.FONT_HERSHEY_PLAIN

def draw_center_points(frame, center_points):
    initial_pt = center_points[0]
    end_pt = center_points[-1]
    inted_pts = np.int32([center_points])
    cv2.polylines(frame, inted_pts, False, (200, 255, 40), 6, lineType=8)
    cv2.circle(frame, (int(initial_pt[0]), int(initial_pt[1])), 4, (0, 0, 255), -1)
    cv2.putText(frame, 'start', (int(initial_pt[0]), int(initial_pt[1])), font, 2, (0, 255, 210), 2)
    cv2.circle(frame, (int(end_pt[0]), int(end_pt[1])), 4, (0, 0, 255), -1)
    cv2.putText(frame, 'end', (int(end_pt[0]), int(end_pt[1])), font, 2, (255, 255, 55), 2)

for data_cls in data:
    screen = empty_template.copy()
    center_points = data_cls.center_points_lst
    draw_center_points(screen, center_points)
    # initial_pt = center_points[0]
    # end_pt = center_points[-1]
    # inted_pts = np.int32([center_points])
    # cv2.polylines(screen, inted_pts, False, (200, 255, 40), 6, lineType=8)
    # cv2.circle(screen, (int(initial_pt[0]), int(initial_pt[1])), 4, (0, 0, 255), -1)
    # cv2.putText(screen, 'start', (int(initial_pt[0]), int(initial_pt[1])), font, 2, (0, 255, 210), 2)
    # cv2.circle(screen, (int(end_pt[0]), int(end_pt[1])), 4, (0, 0, 255), -1)
    # cv2.putText(screen, 'end', (int(end_pt[0]), int(end_pt[1])), font, 2, (255, 255, 55), 2)
    
    cv2.imshow('screen', screen)
    cv2.waitKey(0)


