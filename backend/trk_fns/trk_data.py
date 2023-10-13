from dataclasses import dataclass, field
from typing import List


# @dataclass()
# class CarDetInfo:
#     area: int
#     id: int
#     age: int = 0
#     events: List = field(default_factory=list)

class TrackingData():
    __slots__ = ['area_num', 'id', 'age', 'bboxes_orig', 'center_points_lst', 'time_stamp', 'frame_record']
    
    def __init__(self, area_num, id) -> None:
        self.area_num = area_num
        self.id = id
        self.age = 0
        self.bboxes_orig = []
        self.center_points_lst = []
        self.time_stamp = []
        self.frame_record = []
        
    def __repr__(self) -> str:
        # return f"obj_id:{id(self)}, area_num: {self.area_num}, id: {self.id}, bboxes_orig: {self.bboxes_orig}, frame_rec : {self.frame_record}"
        return f"obj_id:{id(self)}, area_num: {self.area_num}, id: {self.id}"

# inst_car_det = CarDetInfo2(2, 3)
# print(inst_car_det.__dict__)