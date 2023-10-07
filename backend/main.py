# from yolov5 import detect


import cv2
from .yolov5.detect_que_orig import inference
import multiprocessing
import time
from multiprocessing import Queue

# import torch.multiprocessing as tcmp

# from .util_functions import *
# from .visualization import visualize
from .trk_fns.area1 import tracker_area1
from .trk_fns.area3 import tracker_area3
from .trk_fns.area4 import tracker_area4
from .trk_fns.trk_fn import tracker
from .control_center import control_center
# from .visualization_bp import visualize_bp

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def create_backend(op_flags, drawing_result_ques, exit_event):
    gs_tracker_instance = gs_tracker(op_flags, drawing_result_ques, exit_event)
    gs_tracker_instance.run()
    print('backend ends')



class gs_tracker():
    def __init__(self, args, model_proc_result_ques, exit_event) -> None:
        print('backend created')
        # self.drawing_result_ques = drawing_result_ques
        self.model_proc_result_ques = model_proc_result_ques
        self.image_que_lst_draw = self.model_proc_result_ques[0:3]
        self.draw_proc_result_que_lst = self.model_proc_result_ques[3:6]
        self.visualize_bp_que_lst = self.model_proc_result_ques[6:9]
        self.exit_event = exit_event
        
        self.image_que_lst_proc = []
        # self.image_que_lst_draw = []
        self.det_result_que_lst = []
        self.trk_result_que_lst = []
        self.video_loader_lst = []
        self.yolo_inference_lst = []
        self.tracking_proc_lst = []
        # self.draw_proc_lst = []
        # self.draw_proc_result_que_lst = []
        # self.visualize_bp_que_lst = []
        
        self.need_visualization = True
        self.visualization_eco_mode = False
        
        self.operation_flag = args['operation_flag']
        self.area_display_values = args['area_display_values']
        
        self.yolo_inference_ready_flag_lst = [args['is_yolo_inference1_ready'], args['is_yolo_inference2_ready'],
                                         args['is_yolo_inference3_ready']]
        
        
        self.rtsp_ready_lst = [args['is_rtsp1_ready'], args['is_rtsp2_ready'], args['is_rtsp3_ready']]

        # yolo_area1_flag = multiprocessing.Event()
        # yolo_area3_flag = multiprocessing.Event()
        # yolo_area4_flag = multiprocessing.Event()
        # self.yolo_inference_flags = [yolo_area1_flag, yolo_area3_flag, yolo_area4_flag]
        # print('flags = ' , flags[0].is_set())
        

        for i in range(0, len(paths)):
            self.image_que_lst_proc.append(Queue(200))
            self.det_result_que_lst.append(Queue(200))
            self.trk_result_que_lst.append(Queue(200))

            self.video_loader_lst.append(multiprocessing.Process(target=video_load, args=(self.rtsp_ready_lst[i], self.operation_flag, self.image_que_lst_proc[i],
                                                                                          self.image_que_lst_draw[i], paths[i], self.need_visualization,
                                                                                          self.exit_event), daemon=False))
            self.yolo_inference_lst.append(multiprocessing.Process(target=yolo_inference, args=(self.yolo_inference_ready_flag_lst[i], self.operation_flag, self.image_que_lst_proc[i],
                                                                                                self.det_result_que_lst[i], self.exit_event), daemon=False))
            # self.tracking_proc_lst.append(multiprocessing.Process(target=lst_of_trk_fns[i], args=(self.operation_flag, self.det_result_que_lst[i], self.trk_result_que_lst[i],
            #                                                                                       self.draw_proc_result_que_lst[i], self.visualize_bp_que_lst[i],
            #                                                                                       self.exit_event, i), daemon=False))
            self.tracking_proc_lst.append(multiprocessing.Process(target=tracker, args=(self.operation_flag, self.det_result_que_lst[i], self.trk_result_que_lst[i],
                                                                                                  self.draw_proc_result_que_lst[i], self.visualize_bp_que_lst[i],
                                                                                                  self.exit_event, i), daemon=False))
        self.post_proc = multiprocessing.Process(target=control_center, args=(self.operation_flag, self.trk_result_que_lst[0], self.trk_result_que_lst[1],
                                                                              self.trk_result_que_lst[2], self.exit_event), daemon=False)

        print('backend init end')

    def run(self):
        
        print('backend run start')
        
        for i in range(0, len(paths)):
            self.video_loader_lst[i].start()
            self.tracking_proc_lst[i].start()

                
        for i in range(0, len(paths)):
            self.yolo_inference_lst[i].start()
            
        self.post_proc.start()

        for i in range(0, len(paths)):
            self.video_loader_lst[i].join()
            self.yolo_inference_lst[i].join()
            self.tracking_proc_lst[i].join()
        self.post_proc.join()

    def close(self): # need feature refactoring using cue sign

        for i in range(0, len(self.yolo_inference_flags)):
            self.yolo_inference_flags[i].clear()

        for i in range(0, len(paths)):
            self.video_loader_lst[i].close()
            self.yolo_inference_lst[i].close()
            self.tracking_proc_lst[i].close()
        self.post_proc.close()


def video_load(rtsp_ready_flag, op_flag, image_que1, image_que2, path, need_visualization, exit_event):
    cap_loader = cv2.VideoCapture(path)
    rtsp_ready_flag.set()
    while True:
        if exit_event.is_set():
            image_que1.put(None)
            image_que2.put(None)
            cap_loader.release()
            break
        _, _ = cap_loader.read()
        ret, frame = cap_loader.read()
        if ret:
            if op_flag.is_set():
                image_que1.put(frame)
            if need_visualization:
                image_que2.put(frame)
        elif not ret:
            image_que1.put(None)
            if need_visualization:
                image_que2.put(None)
            cap_loader.release()
            print('video loader break, put None')
            break
    print('video loader breaked!')





def yolo_inference(ready_flag, op_flag, image_que, result_que, exit_event):
    inference_instance = inference()
    # if flag.is_set():
    y_s = time.time()
    inference_instance.run(ready_flag, op_flag, image_que, result_que, exit_event)
    print('yolo inference breaked!')
    y_e = time.time()
    y_elapsed_time = y_e - y_s
    print('yolo elapsed time = ', y_elapsed_time)

# main6 = area3, main7 = area1, main8 = area4
paths = ['rtsp://admin:self1004@@118.37.223.147:8522/live/main7',
         'rtsp://admin:self1004@@118.37.223.147:8522/live/main6', 
         'rtsp://admin:self1004@@118.37.223.147:8522/live/main8']

lst_of_trk_fns = [tracker_area1, tracker_area3, tracker_area4]


if __name__ == '__main__':
    gs_tracker_instance = gs_tracker()
    gs_tracker_instance.run()
