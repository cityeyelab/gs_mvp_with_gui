from multiprocessing import Manager, Event, Queue
# import sys

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
        
        selected_cam_num = self.manager.Value(int, 0)
        
        is_rtsp1_ready = Event()
        is_rtsp2_ready = Event()
        is_rtsp3_ready = Event()
        
        is_yolo_inference1_ready = Event()
        is_yolo_inference2_ready = Event()
        is_yolo_inference3_ready = Event()
        
        # self.my_bool = self.manager.Value(bool, True)
        # self.my_bool2 = self.manager.Value(bool, True)
        collision_op_flag = Event()
        collision_rt_op_flag = Event()
        stay_time_op_flag = Event()
        collision_ready_flag = Event()
        collision_rt_ready_flag = Event()
        stay_time_ready_flag = Event()
        
        
        self.args = {'operation_flag': whole_operation_flag, 'video_loader_operation_flag': video_loader_operation_flag,
                     'yolo_operation_flag': yolo_operation_flag, 'tracker_loader_operation_flag': tracker_loader_operation_flag,
                     'det_visualization_operation_flag': det_visualization_operation_flag, 'bp_visualization_operation_flag': bp_visualization_operation_flag,
                     'area_display_values': area_display_values, 'selected_cam_num': selected_cam_num,
                     'is_rtsp1_ready': is_rtsp1_ready, 'is_rtsp2_ready': is_rtsp2_ready, 'is_rtsp3_ready': is_rtsp3_ready,
                     'is_yolo_inference1_ready': is_yolo_inference1_ready, 'is_yolo_inference2_ready': is_yolo_inference2_ready, 'is_yolo_inference3_ready': is_yolo_inference3_ready,
                     'collision_op_flag': collision_op_flag, 'collision_rt_op_flag': collision_rt_op_flag, 'stay_time_op_flag': stay_time_op_flag,
                     'collision_ready_flag': collision_ready_flag, 'collision_rt_ready_flag': collision_rt_ready_flag, 'stay_time_ready_flag': stay_time_ready_flag,
        }
        
        drawing_result_que_a1 = Queue(200)
        drawing_result_que_a3 = Queue(200)
        drawing_result_que_a4 = Queue(200)
        drawing_result_que_bp = Queue(200)
        self.drawing_result_ques = [drawing_result_que_a1, drawing_result_que_a3, drawing_result_que_a4, drawing_result_que_bp]
        
        video_loader_to_vis_a1 = Queue(200)
        video_loader_to_vis_a3 = Queue(200)
        video_loader_to_vis_a4 = Queue(200)
        tracker_to_vis_a1 = Queue(200)
        tracker_to_vis_a3 = Queue(200)
        tracker_to_vis_a4 = Queue(200)
        tracker_to_vis_bp_que_a1 = Queue(200)
        tracker_to_vis_bp_que_a3 = Queue(200)
        tracker_to_vis_bp_que_a4 = Queue(200)
        self.model_proc_result_ques = [video_loader_to_vis_a1, video_loader_to_vis_a3, video_loader_to_vis_a4,
                                       tracker_to_vis_a1, tracker_to_vis_a3, tracker_to_vis_a4, 
                                       tracker_to_vis_bp_que_a1, tracker_to_vis_bp_que_a3, tracker_to_vis_bp_que_a4] # imgx3, resultx3, result_bpx3
        
        self.exit_event = Event()
        self.process_que = Queue(5)
        
        self.collision_analysis_queue = Queue(200)
        self.collision_analysis_rt_queue = Queue(200)
        self.stay_time_queue = Queue(200)
        
        self.bp_heatmap_from_analysis_to_fastapi = Queue(200)
        # print('q100 size = ', sys.getsizeof(Queue(100)))
        # print('q200 size = ', sys.getsizeof(Queue(200)))

