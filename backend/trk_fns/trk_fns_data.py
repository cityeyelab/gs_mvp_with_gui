# from dataclasses import dataclass
from ..map_vars import area1_global_inout_map, area1_car_wash_waiting_map, area1_place0_map
from ..map_vars import area3_global_inout_map, area3_car_wash_waiting_map
from ..map_vars import area4_global_inout_map, area4_car_wash_waiting_map, area4_electric_vehicle_charging_map, area4_car_interior_washing_map


# slope = -0.5353805073431241
# glb_inout_map = area1_global_inout_map
# maps_names = ['car_wash_waiting', 'place0']
# non_global_maps = [area1_car_wash_waiting_map, area1_place0_map]
# zone_cnt_vars = [0 for i in range(0, len(maps_names))]


area1_data_dict = {
    'area_number': 1,
    'slope': -0.5353805073431241,
    'glb_inout_map': area1_global_inout_map,
    'non_global_maps_names': ['car_wash_waiting', 'place0'],
    'non_global_maps': [area1_car_wash_waiting_map, area1_place0_map],
}

area3_data_dict = {
    'area_number': 3,
    'slope': -0.5353805073431241,
    'glb_inout_map': area3_global_inout_map,
    'non_global_maps_names': ['car_wash_waiting'],
    'non_global_maps': [area3_car_wash_waiting_map],
}

area4_data_dict = {
    'area_number': 4,
    'slope': 0.22451456310679613,
    'glb_inout_map': area4_global_inout_map,
    'non_global_maps_names': ['car_wash_waiting', 'electric_vehicle_charging', 'car_interior_washing'],
    'non_global_maps': [area4_car_wash_waiting_map, area4_electric_vehicle_charging_map, area4_car_interior_washing_map],
}



class TrackingVariables():
    __slots__ = ['area_number', 'slope', 'glb_inout_map', 'non_global_maps_names', 'non_global_maps']

    def __init__(self, data_dict: dict) -> None:
        self.area_number, self.slope, self.glb_inout_map, self.non_global_maps_names, self.non_global_maps = data_dict['area_number'], data_dict['slope'], data_dict['glb_inout_map'], data_dict['non_global_maps_names'], data_dict['non_global_maps']
        # self.zone_cnt_vars = [0 for i in range(0, len(self.non_global_maps_names))]
        