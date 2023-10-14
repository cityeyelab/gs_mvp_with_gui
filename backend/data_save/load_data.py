import pickle
import os
import datetime

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



# if '__name__' == '__main__':
#     data = []
#     filename = 'collision_raw_data'
#     print('cwd = ' , os.getcwd())
#     with open(filename, 'rb') as f:
#         data.append(pickle.load(f))

#     # with open(filename, 'rb') as f:
#     #     try:
#     #         while True:
#     #             data.append(pickle.load(f))
#     #     except EOFError:
#     #         pass

#     print(data)

data = []
filename = 'collision_raw_data'
print('cwd = ' , os.getcwd())

# with open(filename, 'rb') as f:
#     data.append(pickle.load(f))

with open(filename, 'rb') as f:
    try:
        while True:
            data.append(pickle.load(f))
    except EOFError:
        pass

print(data)
print(data[0].bboxes)