# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse

import os
import platform
import sys
from pathlib import Path
import numpy as np

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode




import sys


import datetime
import time
import threading
import pickle
from npy_append_array import NpyAppendArray
from queue import Queue
from copy import deepcopy
# from multiprocessing import Process
# from torch.multiprocessing import Process as TorchMP


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


import os
def make_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

# weights=ROOT / 'sample.pt',  # model path or triton URL
# source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
# data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
# imgsz=(640, 640),  # inference size (height, width)
# conf_thres=0.25,  # confidence threshold
# iou_thres=0.45,  # NMS IOU threshold
# max_det=200,  # maximum detections per image
# device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
# view_img=False,  # show results
# save_txt=False,  # save results to *.txt
# save_conf=False,  # save confidences in --save-txt labels
# save_crop=False,  # save cropped prediction boxes
# nosave=False,  # do not save images/videos
# classes=None,  # filter by class: --class 0, or --class 0 2 3
# agnostic_nms=False,  # class-agnostic NMS
# augment=False,  # augmented inference
# visualize=False,  # visualize features
# update=False,  # update all models
# project=ROOT / 'runs/detect',  # save results to project/name
# name='exp',  # save results to project/name
# exist_ok=False,  # existing project/name ok, do not increment
# line_thickness=3,  # bounding box thickness (pixels)
# hide_labels=False,  # hide labels
# hide_conf=False,  # hide confidences
# half=False,  # use FP16 half-precision inference
# dnn=False,  # use OpenCV DNN for ONNX inference
# vid_stride=1,  # video frame-rate stride

# class inference(Process):
class inference():
# class inference(TorchMP):
    # self.yolo_inference_flags[i],self.image_que_lst_proc[i], self.det_result_que_lst[i]
    @smart_inference_mode()
    def __init__(self, proc_num) -> None:
        # self.img_que = img_que
        # self.det_que = det_que
        # Process.__init__(self)
        # weights=ROOT / 'sample.pt'  # model path or triton URL
        # weights=ROOT / '2022_06_21_cat6_best_S_batch-30_T4.engine'  # model path or triton URL
        # weights=ROOT / '2022_06_21_cat6_best_S_batch-1.onnx'  # model path or triton URL
        # weights=ROOT / 'sample.engine'  # model path or triton URL
        weights=ROOT / 'YOLOv5-s_cat-1_dataver-3_hyp-default_size-640_best.pt'  # model path or triton URL
        
        source=ROOT / 'data/images'  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz=(640, 640)  # inference size (height, width)
        conf_thres=0.25  # confidence threshold
        iou_thres=0.45  # NMS IOU threshold
        max_det=200  # maximum detections per image
        device = '0' if torch.cuda.is_available() else 'cpu' # cuda device, i.e. 0 or 0,1,2,3 or cpu
        # device = 'cuda' if torch.cuda.is_available() else 'cpu' # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False  # show results
        save_txt=False  # save results to *.txt
        save_conf=False  # save confidences in --save-txt labels
        save_crop=False  # save cropped prediction boxes
        nosave=False  # do not save images/videos
        classes=None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False  # class-agnostic NMS
        augment=False  # augmented inference
        visualize=False  # visualize features
        update=False # update all models
        project=ROOT / 'runs/detect'  # save results to project/name
        name='exp'  # save results to project/name
        exist_ok=False  # existing project/name ok, do not increment
        line_thickness=3  # bounding box thickness (pixels)
        hide_labels=False  # hide labels
        hide_conf=False  # hide confidences
        # half=False  # use FP16 half-precision inference
        half=True
        dnn=False  # use OpenCV DNN for ONNX inference
        vid_stride=1  # video frame-rate stride

        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half)
        # print(self.model.__hash__)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
    
        bs = 1  # batch_size
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
        seen, windows, self.dt = 0, [], (Profile(), Profile(), Profile())
        # print('inference init')
        
        proc_to_area = [1,2,4]
        self.area_num = proc_to_area[proc_num]
        
        self.need_save = True

        if self.need_save:
            self.save_img_que = Queue()
            self.save_img_thread = threading.Thread(target=self.save_img, args=(self.save_img_que,))
            
            self.save_dets_que = Queue()
            self.save_dets_thread = threading.Thread(target=self.save_dets, args=(self.save_dets_que,))
            
            self.save_img_thread.start()
            self.save_dets_thread.start()
        

    # self.yolo_inference_flags[i],self.image_que_lst_proc[i], self.det_result_que_lst[i]
    @smart_inference_mode()
    def run(self, ready_flag, op_flag, image_que, result_que, exit_event):
    # def run(self):
        # image_que = self.img_que
        # result_que = self.det_que
        # print('run init')

        # print('model = ' , self.model)
        
        det_cnt = 0
        preds = []
        
       
        # if type(image_q) and image_q.shape == (1080, 1920, 3):
        #     len_img_q = 1
        # else:
        #     len_img_q  = len(image_q)
        
        # len_img_q  = len(image_q)

        
        result_lst = []
        conf_thres=0.25
        iou_thres=0.45
        max_det=200 
        classes=None
        agnostic_nms=False
        print('start making dets in det_que')
        
        # print('ready flag in detect = ', ready_flag.is_set())
        ready_flag.set()
        
        while True:
            if exit_event.is_set(): # maybe useless
                break
            
            
            im0 = image_que.get()
            if self.need_save:
                self.save_img_que.put(im0)
            # im0_orig = deepcopy(im0)
            if type(im0) == type(None):
                result_que.put(None)
                # sys.exit()
                break
                

            # if type(im0) != np.ndarray and im0 == None:
            #     result_que.put(None)
            #     break

            # if type(im0) == np.ndarray:
            im = letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous


            with self.dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

                # Inference
            with self.dt[1]:
                pred = self.model.forward(im, augment=False, visualize=False)

            # NMS
            with self.dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


            for i, det in enumerate(pred): 
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # det[:, 4] = names[int(det[:, 4])]
            
            # pred[0] = pred[0].numpy()

            # print('in detect pred = ' , pred)
            # print('in detect pred[0] = ' , pred[0])
            # det_result = pred[0].numpy().tolist()
            # print('pred[0] = ' , pred[0])
            det_result = pred[0].detach().cpu().numpy().tolist()
            # det_result = 
            # preds.append(det_result)
            # print('before rsult que = ' , det_result)
            
            # det_result = tuple([tuple(det_result[i]) for i in range(0, len(det_result))])
            
            result_que.put(det_result)
            # result_lst.append(det_result)
            # result_que.get()
            # preds.append(pred)
            det_cnt += 1
            # divisor = 1 if len_img_q == 1 else int(len_img_q/10)
            # if det_cnt % divisor == 0:
            #     print('det_cnt = ' + str(det_cnt) + '/' + str(len_img_q))

            # print('det_cnt = ' + str(det_cnt))
            # print('result que = ' , result_que)
            
            if result_que.qsize() > 10:
                print('model output Q size = ' , result_que.qsize())
            if result_que.full():
                print('model output Queue is full!!')
                print('clearing Q')
                while not result_que.empty():
                    _ = result_que.get()

            # im0, det_result 저장
            if self.need_save:
                self.save_dets_que.put(det_result)
            
            # cv2.imshow('im0'+str(self.area_num), im0)
            # cv2.waitKey(3)

                # if det_cnt % 20 == 0:
                #     print('det_cnt = ' + str(det_cnt))


        # with open('result_ya', 'wb') as f:
        #     pickle.dump(result_lst, f)

        # return preds
        
    def save_img(self, save_img_que):
        
        # filename="_img"
        w, h = int(1920 * 0.3), int(1080 * 0.3)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # file_cnt = 0
        frame_cnt = 0
        while True:
            now = datetime.datetime.now()
            today_string = now.strftime('%Y-%m-%d')
            make_dir(f'data/{today_string}/videos/')
            starting_time = now.strftime("%H-%M-%S")
            # file_cnt_string = str(file_cnt).zfill(4)
            # writer=cv2.VideoWriter(f'data/{today_string}/videos/sample_area{self.area_num}_{starting_time}_{file_cnt_string}.avi', fourcc, 1, (w, h), False)
            writer=cv2.VideoWriter(f'data/{today_string}/videos/sample_area{self.area_num}_{starting_time}.avi', fourcc, 1, (w, h), False)
            # writer=SafeVideoWriter(f'data/videos/sample_video_area{self.area_num}_{today_string}_{file_cnt_string}.avi', fourcc, 1, (w, h), False)
            # print('in writer size = ' , (int(1920 * 0.3), int(1080 * 0.3)))
            # frame_cnt = 0
            while True:
                for _ in range(0, 14):
                    _ = save_img_que.get()
                img = save_img_que.get()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img = cv2.resize(img, dsize=(0, 0), fx=0.3, fy=0.3)
                img = cv2.resize(img, dsize=(w, h))
                # print('in while size = ' , img.shape)
                writer.write(img)
                frame_cnt += 1
                if frame_cnt == 60 * 5:
                    print('video write!!')
                    writer.release()
                    frame_cnt = 0
                    break
            # file_cnt += 1
        # with open('data/' + today_string + filename, 'ab+') as f:
        # with NpyAppendArray('data/' + today_string + filename + str(self.area_num) +'.npy', delete_if_exists=False) as npaa:
        #     while True:
        #         img = save_img_que.get()
        #         if type(img) == type(None):
        #             break
        #         # print('save img = ', img)
        #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #         # h, w, c = img.shape
        #         img = cv2.resize(img, dsize=(0, 0), fx=0.3, fy=0.3)
        #         # cv2.imshow('gray'+str(self.area_num), img)
        #         # cv2.waitKey(3)
        #         npaa.append(img)
        #         time.sleep(0.01)
        #         frame_cnt += 1
        #         if frame_cnt == 100:
                    
        #             frame_cnt = 0


# with NpyAppendArray(filename, delete_if_exists=True) as npaa:
#     npaa.append(arr1)
#     npaa.append(arr2)
#     npaa.append(arr2)

    
    def save_dets(self, save_dets_que):
        now = datetime.datetime.now()
        today_string = now.strftime('%Y-%m-%d')
        # filename="_dets"
        make_dir(f'data/{today_string}/')
        with open(f'data/{today_string}/{today_string}_dets', 'ab+') as f:
            while True:
                dets = save_dets_que.get()
                if type(dets) == type(None):
                    break
                # print('dets = ', dets)
                pickle.dump(dets, f)
                time.sleep(0.01)



# def save_data(save_que):
#     filename="_raw_data"
#     while True:
#         data = save_que.get()
#         data_cvt = cvt_cls_to_pkl(data)
#         # data_cvted = convert_data_cls(data)
#         now = datetime.now()
#         today_string = now.strftime('%Y-%m-%d')
#         with open('data/'+today_string+filename, 'ab+') as fp:
#             # dill.dump(data, fp)
#             # pickle.dump(data, fp)
#             pickle.dump(data_cvt, fp)
#         time.sleep(0.01)

###############################################################################

class SafeVideoWriter(cv2.VideoWriter):
    def __enter__(self):
        print('safe writer enter')

    def __exit__(self, type, value, traceback):
        print('context manager works!')
        self.release()


def main(image):
    check_requirements(exclude=('tensorboard', 'thop'))
    # print_args(**vars(opt))
    pred, names = run(image)

    return pred, names


    # if __name__ == '__main__':
    #     opt = parse_opt()
    #     main(opt)
