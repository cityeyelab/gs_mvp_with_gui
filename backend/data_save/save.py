import time
import threading
from queue import Queue
import pickle
# import dill
from datetime import datetime
# from ..trk_fns.trk_data import TrackingData

def collect_data(que1, que2, que3):
    save_que = Queue()
    save_thread = threading.Thread(target=save_data, args=(save_que,))
    save_thread.start()
    while True:
        time.sleep(0.01)
        if not que1.empty():
            q1_result = que1.get()
            print('q1 result = ' , q1_result)
            if type(q1_result) == None:
                break
            else:
                save_que.put(q1_result)
        if not que2.empty():
            q2_result = que2.get()
            print('q2 result = ' , q2_result)
            if type(q2_result) == None:
                break
            else:
                save_que.put(q2_result)
        if not que3.empty():
            q3_result = que3.get()
            print('q3 result = ' , q3_result)
            if type(q3_result) == None:
                break
            else:
                save_que.put(q3_result)
        
        time.sleep(0.01)
    print('analysis end')

def save_data(save_que):
    filename="_raw_data"
    while True:
        data = save_que.get()
        if type(data) != type(None):
            data_cvt = cvt_cls_to_pkl(data)
            # data_cvted = convert_data_cls(data)
            now = datetime.now()
            today_string = now.strftime('%Y-%m-%d')
            with open('data/'+today_string+filename, 'ab+') as fp:
                # dill.dump(data, fp)
                # pickle.dump(data, fp)
                pickle.dump(data_cvt, fp)
        time.sleep(0.01)

def cvt_cls_to_pkl(cls):
    result_lst = (cls.area_num, cls.id, cls.bboxes, cls.center_points_lst, cls.frame_record, cls.created_at, cls.removed_at)
    return result_lst

def cvt_cls_to_pkl_lst(cls_lst):
    result_lst = [(cls.area_num, cls.id, cls.bboxes, cls.center_points_lst, cls.frame_record, cls.created_at, cls.removed_at) for cls in cls_lst]
    return result_lst
    # for cls in cls_lst:
        # cls.area_num
        # cls.id
        # cls.bboxes
        # cls.center_points_lst
        # cls.time_stamp
        # cls.frame_record
        # cls.created_at
        # cls.removed_at
        # new_item = (cls.area_num, cls.id, cls.bboxes, cls.center_points_lst, cls.time_stamp, cls.frame_record, cls.created_at, cls.removed_at)


# def cvt_pkl_to_cls(pkl_lst):
#     result_lst = [TrackingData()]
    

# class TrackingDataStore():
#     __slots__ = ['area_num', 'id', 'bboxes', 'center_points_lst', 'frame_record', 'created_at', 'removed_at']
    
#     def __init__(self, area_num, id, bboxes, center_points_lst, frame_record, created_at, removed_at) -> None:
#         self.area_num = area_num
#         self.id = id
#         # self.age = 0
#         self.bboxes = bboxes
#         self.center_points_lst = center_points_lst
#         # self.time_stamp = []
#         self.frame_record = frame_record
#         self.created_at = created_at
#         self.removed_at = removed_at
        
#     def __repr__(self) -> str:
#         # return f"obj_id:{id(self)}, area_num: {self.area_num}, id: {self.id}, bboxes_orig: {self.bboxes_orig}, frame_rec : {self.frame_record}"
#         return f"(obj_id:{id(self)}, area_num: {self.area_num}, id: {self.id}, created_at: {self.created_at}, removed_at: {self.removed_at})"

# def convert_data_cls(tracking_data):
#     new_cls = TrackingDataStore(tracking_data.area_num, tracking_data.id, tracking_data.bboxes, tracking_data.center_points_lst, tracking_data.frame_record,
#                                 tracking_data.created_at, tracking_data.removed_at)
#     return new_cls