import pickle
import time


data = []
# filename = 'data/2023-10-14_raw_data'
# filename = 'data/2023-10-15_raw_data'
# filename = 'data/2023-10-18_raw_data'
filename = 'data/2023-10-19_raw_data'

class TrackingData():
    __slots__ = ['area_num', 'id', 'age', 'bboxes', 'center_points_lst', 'frame_record', 'created_at', 'removed_at']
    
    def __init__(self, area_num, id, bboxes, center_points_lst, frame_record, created_at, removed_at) -> None:
        self.area_num = area_num
        self.id = id
        # self.age = 0
        self.bboxes = bboxes
        self.center_points_lst = center_points_lst
        # self.time_stamp = []
        self.frame_record = frame_record
        self.created_at = created_at
        self.removed_at = removed_at
        
    def __repr__(self) -> str:
        # return f"obj_id:{id(self)}, area_num: {self.area_num}, id: {self.id}, bboxes_orig: {self.bboxes_orig}, frame_rec : {self.frame_record}"
        return f"(obj_id:{id(self)}, area_num: {self.area_num}, id: {self.id}, created_at: {self.created_at}, removed_at: {self.removed_at})"



def cvt_pkl_to_cls(pkl):
    new_cls = TrackingData(pkl[0], pkl[1], pkl[2], pkl[3], pkl[4], pkl[5], pkl[6])
    return new_cls

with open(filename, 'rb') as f:
    try:
        while True:
            # data.append(pickle.load(f))
            loaded_data = pickle.load(f)
            cls_cvt = cvt_pkl_to_cls(loaded_data)
            data.append(cls_cvt)
    except EOFError:
        time.sleep(60)

def create_collision_analysis(op_flags,):
    pass


def analyze_rt(center_points_lst_que):
    pass

def analyze():
    pass