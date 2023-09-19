from multiprocessing import Manager, Event, Queue

class SharedVariables():
    def __init__(self) -> None:
        self.manager = Manager()
        whole_operation_flag = Event()
        video_loader_operation_flag = Event()
        yolo_operation_flag = Event()
        tracker_loader_operation_flag = Event()
        det_visualization_operation_flag = Event()
        bp_visualization_operation_flag = Event()
        
        # self.my_bool = self.manager.Value(bool, True)
        # self.my_bool2 = self.manager.Value(bool, True)
        self.args = {'operation_flag': whole_operation_flag, 'video_loader_operation_flag': video_loader_operation_flag,
                     'yolo_operation_flag': yolo_operation_flag, 'tracker_loader_operation_flag': tracker_loader_operation_flag,
                     'det_visualization_operation_flag': det_visualization_operation_flag, 'bp_visualization_operation_flag': bp_visualization_operation_flag
                     }
        
        self.drawing_result_ques = [Queue(200), Queue(200), Queue(200), Queue(200)]

