import numpy as np
import cv2
##########################################################

area1_global_inout_map = np.load('backend/maps/area1_global_inout_map.npy')
area1_inner_map = np.load('backend/maps/area1_inner_map.npy')
area1_car_wash_waiting_map = np.load('backend/maps/area1_car_wash_waiting_map.npy')
area1_place0_map = np.load('backend/maps/area1_place0_map.npy')

# area1_global_inout_map_img =np.zeros((1080, 1920, 3), np.uint8)
area1_global_inout_map_img = 100*np.ones((1080, 1920, 3), np.uint8)
area1_global_inout_map_non_false_idx  = np.where(area1_global_inout_map==True)
area1_global_inout_map_img[area1_global_inout_map_non_false_idx] = (0,255,0)

area1_inner_map_img = 100*np.ones((1080, 1920, 3), np.uint8)
area1_inner_map__non_false_idx  = np.where(area1_inner_map==True)
area1_inner_map_img[area1_inner_map__non_false_idx] = (100,255,200)

area1_global_inout_map_img = cv2.addWeighted(area1_global_inout_map_img, 0.5, area1_inner_map_img, 0.5, 0)

# area1_car_wash_waiting_map_img = np.zeros((1080, 1920, 3), np.uint8)
area1_car_wash_waiting_map_img = 100*np.ones((1080, 1920, 3), np.uint8)
area1_car_wash_waiting_map_non_false_idx  = np.where(area1_car_wash_waiting_map==True)
area1_car_wash_waiting_map_img[area1_car_wash_waiting_map_non_false_idx] = (0,255,0)

# area1_place0_map_img = np.zeros((1080, 1920, 3), np.uint8)
area1_place0_map_img = 100*np.ones((1080, 1920, 3), np.uint8)
area1_place0_map_non_false_idx  = np.where(area1_place0_map==True)
area1_place0_map_img[area1_place0_map_non_false_idx] = (0,255,0)


#######################################################

area3_global_inout_map = np.load('backend/maps/area3_global_inout_map.npy')
area3_inner_map = np.load('backend/maps/area3_inner_map.npy')
area3_car_wash_waiting_map = np.load('backend/maps/area3_car_wash_waiting_map.npy')

# area3_global_inout_map_img =np.zeros((1080, 1920, 3), np.uint8)
area3_global_inout_map_img = 100*np.ones((1080, 1920, 3), np.uint8)
area3_global_inout_map_non_false_idx  = np.where(area3_global_inout_map==True)
area3_global_inout_map_img[area3_global_inout_map_non_false_idx] = (0,255,0)

area3_inner_map_img = 100*np.ones((1080, 1920, 3), np.uint8)
area3_inner_map__non_false_idx  = np.where(area3_inner_map==True)
area3_inner_map_img[area3_inner_map__non_false_idx] = (0,255,0)

area3_global_inout_map_img = cv2.addWeighted(area3_global_inout_map_img, 0.5, area3_inner_map_img, 0.5, 0)

# area3_car_wash_waiting_map_img = np.zeros((1080, 1920, 3), np.uint8)
area3_car_wash_waiting_map_img = 100*np.ones((1080, 1920, 3), np.uint8)
area3_car_wash_waiting_map_non_false_idx  = np.where(area3_car_wash_waiting_map==True)
area3_car_wash_waiting_map_img[area3_car_wash_waiting_map_non_false_idx] = (0,255,0)


##########################################################


area4_global_inout_map = np.load('backend/maps/area4_global_inout_map.npy')
area4_inner_map = np.load('backend/maps/area4_inner_map.npy')
area4_car_wash_waiting_map = np.load('backend/maps/area4_car_wash_waiting_map.npy')
area4_electric_vehicle_charging_map = np.load('backend/maps/area4_electric_vehicle_charging_map.npy')
area4_car_interior_washing_map = np.load('backend/maps/area4_car_interior_washing_map.npy')

# area4_global_inout_map_img =np.zeros((1080, 1920, 3), np.uint8)
area4_global_inout_map_img = 100*np.ones((1080, 1920, 3), np.uint8)
area4_global_inout_map_non_false_idx  = np.where(area4_global_inout_map==True)
area4_global_inout_map_img[area4_global_inout_map_non_false_idx] = (0,255,0)

area4_inner_map_img = 100*np.ones((1080, 1920, 3), np.uint8)
area4_inner_map__non_false_idx  = np.where(area4_inner_map==True)
area4_inner_map_img[area4_inner_map__non_false_idx] = (0,255,0)

area4_global_inout_map_img = cv2.addWeighted(area4_global_inout_map_img, 0.5, area4_inner_map_img, 0.5, 0)

# area4_car_wash_waiting_map_img = np.zeros((1080, 1920, 3), np.uint8)
area4_car_wash_waiting_map_img = 100*np.ones((1080, 1920, 3), np.uint8)
area4_car_wash_waiting_map_non_false_idx  = np.where(area4_car_wash_waiting_map==True)
area4_car_wash_waiting_map_img[area4_car_wash_waiting_map_non_false_idx] = (0,255,0)


# area4_electric_vehicle_charging_map_img = np.zeros((1080, 1920, 3), np.uint8)
area4_electric_vehicle_charging_map_img = 100*np.ones((1080, 1920, 3), np.uint8)
area4_electric_vehicle_charging_map_non_false_idx  = np.where(area4_electric_vehicle_charging_map==True)
area4_electric_vehicle_charging_map_img[area4_electric_vehicle_charging_map_non_false_idx] = (0,255,0)

# area4_car_interior_washing_map_img = np.zeros((1080, 1920, 3), np.uint8)
area4_car_interior_washing_map_img = 100*np.ones((1080, 1920, 3), np.uint8)
area4_car_interior_washing_map_non_false_idx  = np.where(area4_car_interior_washing_map==True)
area4_car_interior_washing_map_img[area4_car_interior_washing_map_non_false_idx] = (0,255,0)

##########################################################