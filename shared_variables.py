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
        
        area1_display = self.manager.Value(int, 0)
        area2_display = self.manager.Value(int, 0)
        area4_display = self.manager.Value(int, 0)
        area_display_values = [area1_display, area2_display, area4_display]
        
        is_rtsp1_ready = Event()
        is_rtsp2_ready = Event()
        is_rtsp3_ready = Event()
        
        is_yolo_inference1_ready = Event()
        is_yolo_inference2_ready = Event()
        is_yolo_inference3_ready = Event()
        
        # self.my_bool = self.manager.Value(bool, True)
        # self.my_bool2 = self.manager.Value(bool, True)
        self.args = {'operation_flag': whole_operation_flag, 'video_loader_operation_flag': video_loader_operation_flag,
                     'yolo_operation_flag': yolo_operation_flag, 'tracker_loader_operation_flag': tracker_loader_operation_flag,
                     'det_visualization_operation_flag': det_visualization_operation_flag, 'bp_visualization_operation_flag': bp_visualization_operation_flag,
                     'area_display_values': area_display_values,
                     'is_rtsp1_ready': is_rtsp1_ready, 'is_rtsp2_ready': is_rtsp2_ready, 'is_rtsp3_ready': is_rtsp3_ready,
                     'is_yolo_inference1_ready': is_yolo_inference1_ready, 'is_yolo_inference2_ready': is_yolo_inference2_ready, 'is_yolo_inference3_ready': is_yolo_inference3_ready
                     }
        
        self.drawing_result_ques = [Queue(200), Queue(200), Queue(200), Queue(200)]
        self.model_proc_result_ques = [Queue(200), Queue(200), Queue(200),
                                       Queue(200), Queue(200), Queue(200),
                                       Queue(200), Queue(200), Queue(200), ] # imgx3, resultx3, result_bpx3
        
        
        self.exit_event = Event()
        self.process_que = Queue(5)

