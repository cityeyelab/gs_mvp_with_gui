from .map_vars import area1_global_inout_map_non_false_idx, area1_car_wash_waiting_map_non_false_idx, area1_place0_map_non_false_idx, area1_global_inout_map_img, area1_car_wash_waiting_map_img, area1_place0_map_img
from .map_vars import area3_global_inout_map_non_false_idx, area3_car_wash_waiting_map_non_false_idx, area3_global_inout_map_img, area3_car_wash_waiting_map_img
from .map_vars import area4_global_inout_map_non_false_idx, area4_car_wash_waiting_map_non_false_idx, area4_electric_vehicle_charging_map_non_false_idx, area4_car_interior_washing_map_non_false_idx, area4_global_inout_map_img, area4_car_wash_waiting_map_img, area4_electric_vehicle_charging_map_img, area4_car_interior_washing_map_img


class VisualizationArgs():
    def __init__(self, area_num) -> None:
                
        if area_num == 1:
            self.glb_in_cnt = 0
            self.car_wash_waiting_cnt = 0
            self.place0_cnt = 0
            self.cnts_lst = [self.glb_in_cnt, self.car_wash_waiting_cnt, self.place0_cnt]
            self.cnts_lst_str = ['global in', 'car wash wait', 'place0']
            self.area1_view_global_map = False
            self.area1_view_car_wash_waiting_map = False
            self.area1_view_place0_map = False
            self.display_bool_lst = [self.area1_view_global_map, self.area1_view_car_wash_waiting_map, self.area1_view_place0_map]
            self.slope = -0.5353805073431241
            self.non_false_idx_lst = [area1_global_inout_map_non_false_idx, area1_car_wash_waiting_map_non_false_idx, area1_place0_map_non_false_idx]
            self.map_imgs = [area1_global_inout_map_img, area1_car_wash_waiting_map_img, area1_place0_map_img]
            self.zone_name_strings = ['Global In Out', 'Car Wash Waiting', 'Place0']
        elif area_num == 3:
            self.glb_in_cnt = 0
            self.car_wash_waiting_cnt = 0
            self.cnts_lst = [self.glb_in_cnt, self.car_wash_waiting_cnt]
            self.cnts_lst_str = ['global in', 'car wash wait']
            self.area3_view_global_map = False
            self.area3_view_car_wash_waiting_map = False
            self.display_bool_lst = [self.area3_view_global_map, self.area3_view_car_wash_waiting_map]
            self.slope = 0.657103825136612
            self.non_false_idx_lst = [area3_global_inout_map_non_false_idx, area3_car_wash_waiting_map_non_false_idx]
            self.map_imgs = [area3_global_inout_map_img, area3_car_wash_waiting_map_img]
            self.zone_name_strings = ['Global In Out', 'Car Wash Waiting']
        elif area_num == 4:
           self.glb_in_cnt = 0
           self.car_wash_waiting_cnt = 0
           self.electric_vehicle_charging_waiting_cnt = 0
           self.car_interior_washing_waiting_cnt = 0
           self.cnts_lst = [self.glb_in_cnt, self.car_wash_waiting_cnt, self.electric_vehicle_charging_waiting_cnt, self.car_interior_washing_waiting_cnt] # *
           self.cnts_lst_str = ['global in', 'car wash wait', 'elctric charging', 'car interior wash'] # *
           self.area4_view_global_map = False
           self.area4_view_car_wash_waiting_map = False
           self.area4_view_electric_charging_map = False
           self.area4_view_car_interior_wash_map = False
           self.display_bool_lst = [self.area4_view_global_map, self.area4_view_car_wash_waiting_map, self.area4_view_electric_charging_map, self.area4_view_car_interior_wash_map] # *
           self.slope = 0.22451456310679613 # *
           self.non_false_idx_lst = [area4_global_inout_map_non_false_idx, area4_car_wash_waiting_map_non_false_idx, area4_electric_vehicle_charging_map_non_false_idx, area4_car_interior_washing_map_non_false_idx] # *
           self.map_imgs = [area4_global_inout_map_img, area4_car_wash_waiting_map_img, area4_electric_vehicle_charging_map_img, area4_car_interior_washing_map_img] # *
           self.zone_name_strings = ['Global In Out', 'Car Wash Waiting', 'Electric Charging Zone', 'Car Interior Washing Zone'] # *


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
