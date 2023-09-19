# from yolov5 import detect


import cv2
from .yolov5.detect_que_orig import inference
import multiprocessing
import time
from multiprocessing import Queue

# import torch.multiprocessing as tcmp

from .util_functions import *
from .visualization import visualize
from .trk_fns.area1 import tracker_area1
from .trk_fns.area3 import tracker_area3
from .trk_fns.area4 import tracker_area4
from .control_center import control_center
from .visualization_bp import visualize_bp

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def create_backend(op_flag, drawing_result_ques):
    gs_tracker_instance = gs_tracker(op_flag, drawing_result_ques)
    gs_tracker_instance.run()



class gs_tracker():
    def __init__(self, shared_variables, drawing_result_ques) -> None:
        print('backend created')
        self.drawing_result_ques = drawing_result_ques
        self.image_que_lst_proc = []
        self.image_que_lst_draw = []
        self.det_result_que_lst = []
        self.trk_result_que_lst = []
        self.video_loader_lst = []
        self.yolo_inference_lst = []
        self.tracking_proc_lst = []
        self.draw_proc_lst = []
        self.draw_proc_result_que_lst = []
        self.visualize_bp_que_lst = []
        
        self.need_visualization = True
        self.visualization_eco_mode = False
        
        self.operation_flag = shared_variables['operation_flag']

        # yolo_area1_flag = multiprocessing.Event()
        # yolo_area3_flag = multiprocessing.Event()
        # yolo_area4_flag = multiprocessing.Event()
        # self.yolo_inference_flags = [yolo_area1_flag, yolo_area3_flag, yolo_area4_flag]
        # print('flags = ' , flags[0].is_set())
        

        for i in range(0, len(paths)):
            self.image_que_lst_proc.append(Queue(200))
            self.image_que_lst_draw.append(Queue(200))
            self.det_result_que_lst.append(Queue(200))
            self.trk_result_que_lst.append(Queue(200))
            self.draw_proc_result_que_lst.append(Queue(200))
            self.visualize_bp_que_lst.append(Queue(200))

            self.video_loader_lst.append(multiprocessing.Process(target=video_load, args=(self.operation_flag, self.image_que_lst_proc[i], self.image_que_lst_draw[i], paths[i], self.need_visualization), daemon=False))
            # self.yolo_inference_lst.append(multiprocessing.Process(target=yolo_inference, args=(self.yolo_inference_flags[i],self.image_que_lst_proc[i], self.det_result_que_lst[i]), daemon=False))
            self.yolo_inference_lst.append(multiprocessing.Process(target=yolo_inference, args=(self.operation_flag, self.image_que_lst_proc[i], self.det_result_que_lst[i]), daemon=False))
            # self.yolo_inference_lst.append(inference(self.image_que_lst_proc[i], self.det_result_que_lst[i]))
            # self.yolo_inference_lst.append(inference(self.image_que_lst_proc[i], self.det_result_que_lst[i]))
            self.tracking_proc_lst.append(multiprocessing.Process(target=lst_of_trk_fns[i], args=(self.operation_flag, self.det_result_que_lst[i], self.trk_result_que_lst[i], self.draw_proc_result_que_lst[i], self.visualize_bp_que_lst[i],i), daemon=False))
            if self.need_visualization:
                self.draw_proc_lst.append(multiprocessing.Process(target=visualize, args=(self.operation_flag, self.image_que_lst_draw[i], self.draw_proc_result_que_lst[i], i, self.drawing_result_ques[0:3], self.visualization_eco_mode), daemon=False))


        # for i in range(0, len(paths)):
        #     self.yolo_inference_lst[i].start()

        self.post_proc = multiprocessing.Process(target=control_center, args=(self.operation_flag, self.trk_result_que_lst[0], self.trk_result_que_lst[1], self.trk_result_que_lst[2]), daemon=False)
        self.visualize_bp_proc = multiprocessing.Process(target=visualize_bp, args=(self.operation_flag, self.visualize_bp_que_lst[0], self.visualize_bp_que_lst[1], self.visualize_bp_que_lst[2], self.drawing_result_ques[3]), daemon=False)
        
        print('backend init end')

    def run(self):
        
        print('backend run start')
        
        for i in range(0, len(paths)):
            self.video_loader_lst[i].start()
            # self.yolo_inference_lst[i].start()
            self.tracking_proc_lst[i].start()
            if self.need_visualization:
                self.draw_proc_lst[i].start()
                
        for i in range(0, len(paths)):
            self.yolo_inference_lst[i].start()
            
            
        self.post_proc.start()
        self.visualize_bp_proc.start()

        # for i in range(0, len(self.yolo_inference_flags)):
        #     self.yolo_inference_flags[i].set()

        for i in range(0, len(paths)):
            self.video_loader_lst[i].join()
            self.yolo_inference_lst[i].join()
            self.tracking_proc_lst[i].join()
            if self.need_visualization:
                self.draw_proc_lst[i].join()
        self.post_proc.join()
        self.visualize_bp_proc.join()

        # for i in range(0, len(self.yolo_inference_flags)):
        #     self.yolo_inference_flags[i].clear()

    def close(self): # need feature refactoring using cue sign

        for i in range(0, len(self.yolo_inference_flags)):
            self.yolo_inference_flags[i].clear()

        for i in range(0, len(paths)):
            self.video_loader_lst[i].close()
            self.yolo_inference_lst[i].close()
            self.tracking_proc_lst[i].close()
            if self.need_visualization:
                self.draw_proc_lst[i].close()
        self.post_proc.close()
        self.visualize_bp_proc.close()


def video_load(op_flag, image_que1, image_que2, path, need_visualization):
    cap_loader = cv2.VideoCapture(path)
    while True:
        # op_flag.wait()
        _, _ = cap_loader.read()
        ret, frame = cap_loader.read()

        
        if op_flag.is_set() and ret:
            image_que1.put(frame)
            if need_visualization:
                image_que2.put(frame)
        elif not ret:
            image_que1.put(None)
            if need_visualization:
                image_que2.put(None)
            cap_loader.release()
            print('loader break')
            break
    else:
        time.sleep(0.01)
        # print('video loading stopped')
    print('video loader breaked!')





def yolo_inference(op_flag, image_que, result_que):
    inference_instance = inference()
    # if flag.is_set():
    y_s = time.time()
    inference_instance.run(op_flag, image_que, result_que)
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
