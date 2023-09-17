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

import pickle

from multiprocessing import Process
from torch.multiprocessing import Process as TorchMP






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
    def __init__(self) -> None:
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

    # self.yolo_inference_flags[i],self.image_que_lst_proc[i], self.det_result_que_lst[i]
    @smart_inference_mode()
    def run(self, op_flag, image_que, result_que):
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

        print('start making dets in det_que')
        result_lst = []
        conf_thres=0.25
        iou_thres=0.45
        max_det=200 
        classes=None
        agnostic_nms=False
        while True:
        # im0 = cv2.imread('intermediate_version/sample_img2.png')
            # if len_img_q == 1:
            #     im0 = image_q
            # else:
            #     im0 = image_q[i]
            # cv2.imshow('im0', im0)
            # cv2.waitKey(0)

            # print('1')

            im0 = image_que.get()
            
            # print('im0 = ' , im0)

            # cv2.namedWindow('im0', cv2.WINDOW_NORMAL)
            # cv2.imshow('im0', im0)
            # cv2.waitKey(1)

            # print('im0 = ', im0)
            # im0 = 
            # fr_num = que_get['frame_num']
            if type(im0) != np.ndarray and im0 == None:
                result_que.put(None)
                break
            # print('im0 = ' , im0)
            # print('image que = ' , image_que)
            if type(im0) == np.ndarray:
                im = letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)  # contiguous

                
                # cv2.namedWindow('im', cv2.WINDOW_NORMAL)
                # cv2.imshow('im', im)
                # cv2.waitKey(1)
                # print('im = ',  im)


                with self.dt[0]:
                    im = torch.from_numpy(im).to(self.model.device)
                    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    # print('im = ' , im)

                    # Inference
                with self.dt[1]:
                    # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    # print('visualize = ' , visualize)
                    
                    # pred = model(im, augment=augment, visualize=False)
                    pred = self.model.forward(im, augment=False, visualize=False)
                    # print('prd = ' , pred)
                    # model(im0=im, augment, vi)

                # print('pred1 = ' , pred)

                

                # print('2')

                # NMS
                with self.dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # print('pred = ' , pred)
                # print('names = ' , names)

                # print('3')

                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

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



                # if det_cnt % 20 == 0:
                #     print('det_cnt = ' + str(det_cnt))


        # with open('result_ya', 'wb') as f:
        #     pickle.dump(result_lst, f)

        # return preds
    


###############################################################################



# @smart_inference_mode()
# def run(image_q):


#     # source = str(source)
#     # save_img = not nosave and not source.endswith('.txt')  # save inference images
#     # is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
#     # is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
#     # webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
#     # screenshot = source.lower().startswith('screen')
#     # if is_url and is_file:
#     #     source = check_file(source)  # download

    
#     # # Directories
#     # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
#     # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    

#     # imgsz *= 2 if len(imgsz) == 1 else 1

#     # weights=ROOT / 'sample.pt'  # model path or triton URL
#     weights=ROOT / '2022_06_21_cat6_best_S_batch-30_T4.engine'  # model path or triton URL
#     source=ROOT / 'data/images'  # file/dir/URL/glob/screen/0(webcam)
#     data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
#     imgsz=(640, 640)  # inference size (height, width)
#     conf_thres=0.25  # confidence threshold
#     iou_thres=0.45  # NMS IOU threshold
#     max_det=200  # maximum detections per image
#     # device= 0 if torch.cuda.is_available() else 'cpu' # cuda device, i.e. 0 or 0,1,2,3 or cpu
#     device= 'cuda' if torch.cuda.is_available() else 'cpu' # cuda device, i.e. 0 or 0,1,2,3 or cpu
#     view_img=False  # show results
#     save_txt=False  # save results to *.txt
#     save_conf=False  # save confidences in --save-txt labels
#     save_crop=False  # save cropped prediction boxes
#     nosave=False  # do not save images/videos
#     classes=None  # filter by class: --class 0, or --class 0 2 3
#     agnostic_nms=False  # class-agnostic NMS
#     augment=False  # augmented inference
#     visualize=False  # visualize features
#     update=False # update all models
#     project=ROOT / 'runs/detect'  # save results to project/name
#     name='exp'  # save results to project/name
#     exist_ok=False  # existing project/name ok, do not increment
#     line_thickness=3  # bounding box thickness (pixels)
#     hide_labels=False  # hide labels
#     hide_conf=False  # hide confidences
#     half=False  # use FP16 half-precision inference
#     dnn=False  # use OpenCV DNN for ONNX inference
#     vid_stride=1  # video frame-rate stride


#     # Load model
#     # print('device = ' , device)
#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
#     # print('model instance initiated!')
#     stride, names, pt = model.stride, model.names, model.pt
#     # print('stride = ' , stride)
#     # print('names00 = ', names)
#     # print('pt = ' , pt)

#     # print('class names = ' , names)

#     imgsz = check_img_size(imgsz, s=stride)  # check image size

#     # # Dataloader
#     bs = 1  # batch_size
#     # if webcam:
#     #     view_img = check_imshow(warn=True)
#     #     dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#     #     bs = len(dataset)
#     # elif screenshot:
#     #     dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
#     # else:
#     #     dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#     #     print('dataset = ' , dataset)


#     # vid_path, vid_writer = [None] * bs, [None] * bs


#     # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#     # print('dataset = ' , dataset)

#     # Run inference
#     # print('before model warmup')
#     model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
#     seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

#     det_cnt = 0
#     preds = []

#     # print('path = ' , path)
#     # print('current working directory = ' , os.getcwd())
#     # im0 = cv2.imread(path)  # BGR

#     len_img_q  = len(image_q)

#     for i in range(0, len_img_q):
#     # im0 = cv2.imread('intermediate_version/sample_img2.png')
#         im0 = image_q[i]
#         # cv2.imshow('im0', im0)
#         # cv2.waitKey(0)

#         im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]  # padded resize
#         im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#         im = np.ascontiguousarray(im)  # contiguous


#         with dt[0]:
#             im = torch.from_numpy(im).to(model.device)
#             im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
#             im /= 255  # 0 - 255 to 0.0 - 1.0
#             if len(im.shape) == 3:
#                 im = im[None]  # expand for batch dim
#             # print('im = ' , im)

#             # Inference
#         with dt[1]:
#             # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
#             # print('visualize = ' , visualize)
            
#             # pred = model(im, augment=augment, visualize=False)
#             pred = model.forward(im, augment=augment, visualize=visualize)
#             # model(im0=im, augment, vi)

#         # print('pred1 = ' , pred)

#         # NMS
#         with dt[2]:
#             pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

#         # print('pred = ' , pred)
#         # print('names = ' , names)


#         # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

#         for i, det in enumerate(pred): 
#             det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
#             # det[:, 4] = names[int(det[:, 4])]
        
#         print('in detect pred = ', pred.numpy())
#         preds.append(pred)
#         det_cnt += 1

#         if det_cnt % int(len_img_q/10) == 0:
#             print('det_cnt ah-ny!! = ' + str(det_cnt) + '/' + str(len_img_q))

#     return preds, names
    



    # for path, im, im0s, vid_cap, s in dataset:
        
    #     print('path = ' , path)

    #     with dt[0]:
    #         im = torch.from_numpy(im).to(model.device)
    #         im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    #         im /= 255  # 0 - 255 to 0.0 - 1.0
    #         if len(im.shape) == 3:
    #             im = im[None]  # expand for batch dim

    #     # Inference
    #     with dt[1]:
    #         # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
    #         # print('visualize = ' , visualize)
    #         pred = model(im, augment=augment, visualize=False)

    #     # print('pred1 = ' , pred)

    #     # NMS
    #     with dt[2]:
    #         pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    #     print('pred = ' , pred)
    #     print('names = ' , names)

    #     # pred = 위치정보4개 xy xy or xy wh , conf, class number 이렇게 6차원 아웃풋
    #     # names = {0: 'car', 1: 'bus-s', 2: 'bus-m', 3: 'truck-s', 4: 'truck-m', 5: 'truck-x'}

    #     # Second-stage classifier (optional)
    #     # import utils
    #     # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)


#################################


    #     # Process predictions
    #     for i, det in enumerate(pred):  # per image
    #         seen += 1
    #         # if webcam:  # batch_size >= 1
    #         #     p, im0, frame = path[i], im0s[i].copy(), dataset.count
    #         #     s += f'{i}: '
    #         # else:
    #         #     p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

    #         # p = Path(p)  # to Path
    #         # save_path = str(save_dir / p.name)  # im.jpg
    #         # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
    #         s += '%gx%g ' % im.shape[2:]  # print string
    #         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #         imc = im0.copy() if save_crop else im0  # for save_crop
    #         annotator = Annotator(im0, line_width=line_thickness, example=str(names))
    #         if len(det):
    #             # Rescale boxes from img_size to im0 size
    #             det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

    #             # Print results
    #             for c in det[:, 5].unique():
    #                 n = (det[:, 5] == c).sum()  # detections per class
    #                 s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

    #             # Write results
    #             for *xyxy, conf, cls in reversed(det):
    #                 if save_txt:  # Write to file
    #                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
    #                     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
    #                     # with open(f'{txt_path}.txt', 'a') as f:
    #                     #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

    #                 # if save_img or save_crop or view_img:  # Add bbox to image
    #                 #     c = int(cls)  # integer class
    #                 #     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
    #                 #     annotator.box_label(xyxy, label, color=colors(c, True))
    #                 # if save_crop:
    #                 #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

    #                 print('each class', cls)

    #         # Stream results
    #         im0 = annotator.result()
    #         if view_img:
    #             if platform.system() == 'Linux' and p not in windows:
    #                 windows.append(p)
    #                 cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    #                 cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
    #             cv2.imshow(str(p), im0)
    #             cv2.waitKey(1)  # 1 millisecond

    #         # Save results (image with detections)
    #         # if save_img:
    #         #     if dataset.mode == 'image':
    #         #         cv2.imwrite(save_path, im0)
    #         #     else:  # 'video' or 'stream'
    #         #         if vid_path[i] != save_path:  # new video
    #         #             vid_path[i] = save_path
    #         #             if isinstance(vid_writer[i], cv2.VideoWriter):
    #         #                 vid_writer[i].release()  # release previous video writer
    #         #             if vid_cap:  # video
    #         #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #         #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #         #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #         #             else:  # stream
    #         #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
    #         #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
    #         #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #         #         vid_writer[i].write(im0)

    #     # Print time (inference-only)
    #     LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # # Print results
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # if update:
    #     strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)



#####################################################


# # --weights sample.pt --source C:/test_images/sample_img.png
# model_file_name = 'sample.pt'
# # model_file_name = 'yolov5s.pt'
# target_file_name = 'C:/test_images/sample_img1.png'

# def parse_opt():
#     parser = argparse.ArgumentParser()

#     # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
#     # parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
#     print('ROOT = ' , ROOT)
#     print('type of ROOT = ' , type(ROOT))
#     parser.add_argument('--weights', nargs='+', type=str, default=ROOT / model_file_name, help='model path or triton URL')
#     parser.add_argument('--source', type=str, default=ROOT / target_file_name, help='file/dir/URL/glob/screen/0(webcam)')
#     parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
#     parser.add_argument('--max-det', type=int, default=200, help='maximum detections per image')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='show results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--visualize', action='store_true', help='visualize features')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
#     parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
#     parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
#     print_args(vars(opt))
#     return opt


def main(image):
    check_requirements(exclude=('tensorboard', 'thop'))
    # print_args(**vars(opt))
    pred, names = run(image)

    return pred, names


    # if __name__ == '__main__':
    #     opt = parse_opt()
    #     main(opt)
