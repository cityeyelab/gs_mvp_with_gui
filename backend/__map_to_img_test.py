import numpy as np
import cv2


area4_global_inout_map = np.load('area4_global_inout_map.npy')
area4_global_inout_map_img =np.zeros((1080, 1920, 3), np.uint8)
white_img =np.zeros((1080, 1920, 3), np.uint8)

non_false_idx  = np.where(area4_global_inout_map==True)
area4_global_inout_map_img[non_false_idx] = (255,0,0)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', area4_global_inout_map_img)
cv2.waitKey(0)