from dataclasses import dataclass, field
from typing import List
import datetime

# @dataclass()
# class CarDetInfo:
#     area: int
#     id: int
#     age: int = 0
#     events: List = field(default_factory=list)

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
        return f"obj_id:{id(self)}, area_num: {self.area_num}, id: {self.id}, created_at: {self.created_at}, removed_at: {self.removed_at}"

# inst_car_det = CarDetInfo2(2, 3)
# print(inst_car_det.__dict__)