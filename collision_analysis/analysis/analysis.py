from ._2d_to_2d_mapping import mapping, load_matrices
from functools import partial
import cv2
import numpy as np
import pickle
import math
# import datetime
import time
from os.path import isfile
from datetime import datetime, timedelta



import asyncio
import websockets

import base64

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

def check_intersect(p1, p2, p3, p4):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x,y)


def draw_center_points(frame, center_points, color):
    initial_pt = center_points[0]
    end_pt = center_points[-1]
    inted_pts = np.int32([center_points])
    cv2.polylines(frame, [inted_pts], False, color, 6, lineType=8)
    cv2.circle(frame, (int(initial_pt[0]), int(initial_pt[1])), 4, (0, 0, 255), -1)
    cv2.putText(frame, 'start', (int(initial_pt[0]), int(initial_pt[1])), font, 2, (0, 255, 210), 2)
    cv2.circle(frame, (int(end_pt[0]), int(end_pt[1])), 4, (0, 0, 255), -1)
    cv2.putText(frame, 'end', (int(end_pt[0]), int(end_pt[1])), font, 2, (255, 255, 55), 2)

def smooth_center_points(lst):
    # print('orig lst = ' , lst)
    len_lst = len(lst)
    x = [lst[i][0] for i in range(0, len_lst)]
    y = [lst[i][1] for i in range(0, len_lst)]
    smoothed_x = smooth_along_one_axis(x, 5)
    smoothed_y = smooth_along_one_axis(y, 5)
    result = [(smoothed_x[i], smoothed_y[i]) for i in range(0, len_lst)]
    # print('smoothed result = ' , result)
    return result


def smooth_along_one_axis(lst, window_size):
    len_lst = len(lst)
    results_lst = [lst[0]]
    for i in range(1, len(lst)-1):
        first_idx = max(i - window_size, 1)
        last_idx = min(i + window_size, len_lst-1)
        target_lst = lst[first_idx:last_idx]
        result = sum(target_lst)/(len(target_lst))
        results_lst.append(result)
    results_lst.append(lst[-1])
    return results_lst


def get_distane(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    return ((x2-x1)**2 + (y2-y1)**2)**(1/2)




def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def get_angle(v1, v2):
    return math.acos(abs(dotproduct(v1, v2)) / (length(v1) * length(v2)))

def strech_vec(v1, v2): #vec = ()
        direction = (v2[0] - v1[0], v2[1] - v1[1])
        former_pt = (v1[0] - direction[0], v1[1] - direction[1])
        futher_pt = (v2[0] + direction[0], v2[1] + direction[1])
        return former_pt, futher_pt


def draw_colorbar(img):
    font = cv2.FONT_HERSHEY_PLAIN
    h, w, c = img.shape
    box_w = int(w/7)
    box_h = int(h/30)
    box_start = (int(w/20), int(h/40))
    img = cv2.rectangle(img, box_start, (box_start[0] + box_w, box_start[1]+box_h), (0, 0, 0), 6, 8)
    blank_bar = np.zeros((box_h, box_w), dtype=np.uint8)
    for i in range(0, box_w):
        blank_bar[:,i] = int(254*(i/box_w))
    color_blank_bar =  cv2.applyColorMap(blank_bar, cv2.COLORMAP_JET)
    img[box_start[1]:box_start[1]+box_h, box_start[0]:box_start[0]+box_w] = color_blank_bar
    font_scale = 0.8
    cv2.putText(img, 'low', (box_start[0], box_start[1] + box_h + 20), font, font_scale, (0,0,0), 6, 8)
    cv2.putText(img, 'low', (box_start[0], box_start[1] + box_h + 20), font, font_scale, (255, 255, 255), 1, 8)
    cv2.putText(img, 'high', (box_start[0] + box_w + 10, box_start[1] + box_h + 20), font, font_scale, (0,0,0), 6, 8)
    cv2.putText(img, 'high', (box_start[0] + box_w + 10, box_start[1] + box_h + 20), font, font_scale, (255, 255, 255), 1, 8)


# start = time.time()



bg_path = 'backend/data_save/area4_sample.png'
area4_matrices = load_matrices('visualization/mapping_matrix/mapping_area4')
area1_matrices = load_matrices('visualization/mapping_matrix/mapping_area1')
area3_matrices = load_matrices('visualization/mapping_matrix/mapping_area3')
mapping_area4 = partial(mapping, loaded_matrices=area4_matrices)
mapping_area1 = partial(mapping, loaded_matrices=area1_matrices)
mapping_area3 = partial(mapping, loaded_matrices=area3_matrices)

bp_background = cv2.imread('visualization/assets/blueprint.png')



empty_template = cv2.imread(bg_path)
font = cv2.FONT_HERSHEY_PLAIN

# bp_background = cv2.imread('visualization/assets/blueprint.png')
h, w, c = bp_background.shape



def smooth_center_points_lst_from_cls_lst(cls_lst):
    smoothed_center_points_lst = []
    for data_cls in cls_lst:
        center_points = data_cls.center_points_lst
        smoothed_center_points = smooth_center_points(center_points.copy())
        smoothed_center_points_lst.append(smoothed_center_points)
    return smoothed_center_points_lst


def map_points(points_lst, cls_lst):
    mapped_pts = []
    for i, pt in enumerate(points_lst):
        area_num = cls_lst[i].area_num
        if area_num == 4:
            mapped_pts_area4 = mapping_area4(pts_lst=pt)
            mapped_pts.append(mapped_pts_area4)
        if area_num == 1:
            mapped_pts_area1 = mapping_area1(pts_lst=pt)
            mapped_pts.append(mapped_pts_area1)
        if area_num == 3:
            mapped_pts_area3 = mapping_area3(pts_lst=pt)
            mapped_pts.append(mapped_pts_area3)
    return mapped_pts

def filter_out_no_movement(points_lst, dist_thr):
    filtered_points_lst = []
    for points in points_lst:
        first_pt = points[0]
        end_pt = points[-1]
        dist = ((first_pt[0]-end_pt[0])**2 + (first_pt[1]-end_pt[1])**2)**(1/2)
        if dist > dist_thr:
            filtered_points_lst.append(points)
    return filtered_points_lst

def refine_points_lst_by_moving_distance(points_lst, dist_thr = 10):
    refined_pts_lst = []
    contained_nodes_numbers_lst = []
    for pts in points_lst:
        # print('len pts = ' , len(pts))
        new_lst = []
        contained_nodes_numbers = []
        index_record = []
        new_lst.append(pts[0])
        last_idx = 0
        is_complete = False
        while True:
            if is_complete:
                break
            last_pt = pts[last_idx]
            for j in range(last_idx+1, len(pts)):
                dist = get_distane(last_pt, pts[j])
                if dist > dist_thr:
                    contained_nodes_number = j - last_idx
                    contained_nodes_numbers.append(contained_nodes_number)
                    index_record.append(j)
                    last_idx = j
                    new_lst.append(pts[j])
                    if j == len(pts)-1:
                        is_complete = True
                        # contained_nodes_number_lst[-1] += 1 # 좌측 closed, 우측 open interval이라서 보정. -> 중간에 노드갯수 셀때는 각 수치에서 하나씩 빼서 쓰면 되서, (마지막점 빼고) 보정안하고 그냥 써도 될 듯
                    break
                elif j == len(pts)-1:
                    contained_nodes_number = j - last_idx
                    contained_nodes_numbers.append(contained_nodes_number)
                    index_record.append(j)
                    new_lst.append(pts[j])
                    is_complete = True
                    break
        refined_pts_lst.append(new_lst)
        contained_nodes_numbers_lst.append(contained_nodes_numbers)
    return refined_pts_lst, contained_nodes_numbers_lst


def detect_collisions(points_lst, angle_thr = math.pi/6):
    results = []
    for i in range(0, len(points_lst)):
        pts = points_lst[i]
        
        for j in range(0, len(pts)-1):
            prev_pt = pts[j]
            next_pt = pts[j+1]
            for k in range(i+1, len(points_lst)):
                other_pts = points_lst[k]
                for l in range(0, len(other_pts)-1):
                    other_prev_pt = other_pts[l]
                    other_next_pt = other_pts[l+1]
                    result = check_intersect(prev_pt, next_pt, other_prev_pt, other_next_pt)
                    if type(result) != type(None):
                        v1_x = prev_pt[0] - next_pt[0]
                        v1_y = prev_pt[1] - next_pt[1]
                        v2_x = other_prev_pt[0] - other_next_pt[0]
                        v2_y = other_prev_pt[1] - other_next_pt[1]
                        v1 = (v1_x, v1_y)
                        v2 = (v2_x, v2_y)
                        angle = get_angle(v1, v2)
                        if angle > angle_thr:
                            results.append((result, prev_pt, next_pt, other_prev_pt, other_next_pt))
    return results


def detect_collisons_partly(old_points_lst, new_points_lst, angle_thr=math.pi/6):
    results = []
    for i in range(0, len(new_points_lst)):
        pts = new_points_lst[i]
        
        for j in range(0, len(pts)-1):
            prev_pt = pts[j]
            next_pt = pts[j+1]
            for k in range(i+1, len(new_points_lst)):
                other_pts = new_points_lst[k]
                for l in range(0, len(other_pts)-1):
                    other_prev_pt = other_pts[l]
                    other_next_pt = other_pts[l+1]
                    result = check_intersect(prev_pt, next_pt, other_prev_pt, other_next_pt)
                    if type(result) != type(None):
                        v1_x = prev_pt[0] - next_pt[0]
                        v1_y = prev_pt[1] - next_pt[1]
                        v2_x = other_prev_pt[0] - other_next_pt[0]
                        v2_y = other_prev_pt[1] - other_next_pt[1]
                        v1 = (v1_x, v1_y)
                        v2 = (v2_x, v2_y)
                        angle = get_angle(v1, v2)
                        if angle > angle_thr:
                            results.append((result, prev_pt, next_pt, other_prev_pt, other_next_pt))
                            
            for k in range(0, len(old_points_lst)):
                other_pts = old_points_lst[k]
                for l in range(0, len(other_pts)-1):
                    other_prev_pt = other_pts[l]
                    other_next_pt = other_pts[l+1]
                    result = check_intersect(prev_pt, next_pt, other_prev_pt, other_next_pt)
                    if type(result) != type(None):
                        v1_x = prev_pt[0] - next_pt[0]
                        v1_y = prev_pt[1] - next_pt[1]
                        v2_x = other_prev_pt[0] - other_next_pt[0]
                        v2_y = other_prev_pt[1] - other_next_pt[1]
                        v1 = (v1_x, v1_y)
                        v2 = (v2_x, v2_y)
                        angle = get_angle(v1, v2)
                        if angle > angle_thr:
                            results.append((result, prev_pt, next_pt, other_prev_pt, other_next_pt))
            
    return results


def draw_stay_time(cvs, center_points_lst, contained_points_lst):
    # print('contained pt lst = ' , contained_points_lst)
    # print(len(center_points_lst))
    # print(len(contained_points_lst))
    for i, pts in enumerate(center_points_lst):
        # print('check st : ', i)
        add_values = contained_points_lst[i]
        # print('add values = ', add_values)
        for j in range(0, len(pts)-1):
            # print('st j = ', j)
            add_value = add_values[j]
            # print('add value = ', add_value)
            # print('add value[0] = ' , add_value[0])
            # print('1')
            tf_template = np.full((h, w), False)
            # print('2')
            draw_template = np.zeros((h, w), dtype=np.uint8)
            # print('3')
            prev_pt = pts[j]
            next_pt = pts[j+1]
            # print('4')
            ct_pt = ((prev_pt[0]+next_pt[0])/2, (prev_pt[1]+next_pt[1])/2)
            # print('5')
            # cv2.circle(draw_template, (int(ct_pt[0]), int(ct_pt[1])), 30, (255, 255, 255), -1)
            cv2.circle(draw_template, (int(ct_pt[0]), int(ct_pt[1])), 30,  255, -1)
            # print('6')
            tf_template[draw_template != 0 ] = True
            # print('7')
            # for _ in range(0, add_value):
            #     cvs += tf_template
            cvs += add_value*tf_template
            # print('8')



import os
def make_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


base_data_path = None
base_data = []
if type(base_data_path) != type(None):
    with open(base_data_path, 'rb') as f:
        try:
            while True:
                # data.append(pickle.load(f))
                loaded_data = pickle.load(f)
                cls_cvt = cvt_pkl_to_cls(loaded_data)
                base_data.append(cls_cvt)
        except EOFError:
            print('EOFError')


async def connect():

    # 웹 소켓에 접속을 합니다.
    # async with websockets.connect("ws://14.36.1.6:8000/ws/bp_heatmap") as websocket:
    async with websockets.connect("ws://localhost:8001/ws/bp_heatmap") as websocket:
        # websocket.close_timeout()
        # websockets.

        # str = input('Python 웹소켓으로 전송할 내용을 입력하세요[종료하려면 quit 입력!]: ')     # 사용자의 입력을 변수에 저장
        # #print(str)  # 변수의 내용을 출력

        # #입력받은 값을 파일로 저장
        # f = open('./chat/chatlog2.txt', 'w')
        # f.write(str)
        cnt = 0
        # rcv = await websocket.recv()
        # print('client rcv = ' , rcv)
        while True:
            cnt += 1
            try:
                print('analysis connect while goes on')
                # print('websocket open = ', websocket.ensure_open)
                # quit가 들어오기 전까지 계속 입력받은 문자열을 전송하고 에코를 수신한다.
                data = 'hihihi ' + str(cnt)
                await websocket.send(data)
                # rcv = await websocket.recv()
                # print('sent data = ', data)

                # 웹 소켓 서버로 부터 메시지가 오면 콘솔에 출력합니다.
                # data = await websocket.recv();
                # print(data);
                # f.write(data)

                # str = input('Python 웹소켓으로 전송할 내용을 입력하세요[종료하려면 quit 입력!]: ')  # 사용자의 입력을 변수에 저장
                # print(str)  # 변수의 내용을 출력
            except Exception as e:
                print('error : ' , e)
            
            await asyncio.sleep(1.0)

        # f.close()




async def analyze(collision_op_flag, stay_time_op_flag, collision_ready_flag, stay_time_ready_flag, collision_que, st_que, ):
    
    # print('analysis time sleep')
    # time.sleep(5)
    # asyncio.get_event_loop().run_until_complete(connect())
    # # asyncio.get_event_loop().run_forever()
    # # connect()
    # print('ws bp heatmap connected')
    
    time_interval = 10
    # filename="_raw_data"
    
    # filename = 'data/2023-10-19_raw_data'

    
    glb_cvs = np.zeros((h, w), dtype=np.int64)
    stay_time_cvs = np.zeros((h, w), dtype=np.int64)
    # prev_cvs = np.zeros((h, w), dtype=np.int64)
    
    # time.sleep(10)
    # collision_ready_flag.set()
    # stay_time_ready_flag.set()
    now = datetime.now()
    today_string = now.strftime('%Y-%m-%d')
    filename = f'data/{today_string}/{today_string}_raw_data'
    while True:
        # now = datetime.now()
        # today_string = now.strftime('%Y-%m-%d')
        # filename = 'data/'+ today_string + '_raw_data'
        try:
            if isfile(filename):
                print('file exist in collision')
                break
            else:
                print('file checking in collision analysis...1')
                # if base_data:
                #     new_data = base_data
                #     break
                time.sleep(2)
            # with open(filename, 'rb') as f:
            #     break
        except:
            print('file checking in collision analysis...2')
            time.sleep(3)
            # if base_data:
            #     new_data = base_data
            #     break
    # if 읽어들인 파일의 data len > 0 and 저장된 캐쉬들과 len 일치:
    #     저장된 global cvs, stay time cvs 읽어서 초기화된 glb_cvs, stay_time_cvs에 더함.
    #     얘네들 이미지 만들어서 vis bp에 전송함
    #     refined_pts_lst를 prev_data에 넣음
    #     저장할것 -> glb_cvs, stay_time_cvs, refined_pts_lst
    
    pre_loaded_data_available = False
    try:
        print('try preload collision data')
        pre_loaded_data_lst = []
        with open(filename, 'rb') as f:
            while True:
                try:
                    while True: # 이 프로세스를 한번으로 줄이면 좋음
                        pre_loaded_data = pickle.load(f) # loads?
                        cls_cvt = cvt_pkl_to_cls(pre_loaded_data)
                        pre_loaded_data_lst.append(cls_cvt)
                except EOFError:
                    print('EOFError 1')
                    break
                except:
                    print('preload raw data file failure')
                    break
    
    # with open(f'data/{today_string}/{today_string}_refined_pts_lst_cached', 'wb') as f1:
    #     pickle.dump(refined_pts_lst, f1)
    # np.save(f'data/{today_string}/{today_string}_glb_cvs_cached', glb_cvs)
    # np.save(f'data/{today_string}/{today_string}_st_cvs_cached', stay_time_cvs)
    # with open(f'data/{today_string}/{today_string}_meta', 'wb') as f2:
    
    
        print('preload check1')
        with open(f'data/{today_string}/{today_string}_refined_pts_lst_cached', 'rb') as f:
            refined_pts_lst_loaded = pickle.load(f)
        print('preload check2')
        with open(f'data/{today_string}/{today_string}_meta', 'rb') as f:
            meta_dict_loaded = pickle.load(f)
            len_refined_pts_lst = meta_dict_loaded['len_refined_pts_lst']
            glb_max_val_cached = meta_dict_loaded['glb_max_val']
            st_max_val_cached = meta_dict_loaded['st_max_val']
            # meta_dict = {'len_refined_pts_lst': len(prev_data), 'glb_max_val':glb_max_val, 'st_max_val': st_max_val, }
        print('preload check3')
        
        pre_loaded_glb_cvs = np.load(f'data/{today_string}/{today_string}_glb_cvs_cached.npy')
        pre_loaded_st_cvs = np.load(f'data/{today_string}/{today_string}_st_cvs_cached.npy')
        print('preload check4')
        print('meta pts len = ' , len_refined_pts_lst)
        print('real pts len = ' , len(refined_pts_lst_loaded))
        print('valid pre load data = ' , len_refined_pts_lst == len(refined_pts_lst_loaded))
        if len_refined_pts_lst == len(refined_pts_lst_loaded):
        
            # glb_max_val = np.max(pre_loaded_glb_cvs)
            if glb_max_val_cached != 0:
                scaled_glb_cvs = 254*(pre_loaded_glb_cvs/glb_max_val_cached)
                glb_result = np.uint8(scaled_glb_cvs)
                res_show = cv2.applyColorMap(glb_result, cv2.COLORMAP_JET)
                res_show = cv2.blur(res_show, (7, 7))
                collision_que.put(res_show)
                
                # async with websockets.connect("ws://localhost:8001/ws/bp_collision_heatmap_innerpass") as websocket:
                async with websockets.connect("ws://127.0.0.1:8001/ws/bp_collision_heatmap_innerpass") as websocket:
                    try:
                        retval, buffer = cv2.imencode('.jpg', res_show, [cv2.IMWRITE_JPEG_QUALITY,80])
                        # print('buffer : ' , buffer)
                        data = base64.b64encode(buffer)
                        # print('sent data : ' , data)
                        # data = data.decode('utf-8')
                        await websocket.send(data)
                        await websocket.close()
                    except Exception as e:
                        print(f'something went wrong in ws send, error : {e}')
                        await websocket.close()
                
                # bp_heatmap_que.put(res_show)
                
                collision_ready_flag.set()
            else:
                print('not valid result in pre loaded glb cvs calc')
            
            print('preload check5')
            # st_max_val = np.max(pre_loaded_st_cvs)
            if st_max_val_cached != 0:
                scaled_st_cvs = 254*(pre_loaded_st_cvs/st_max_val_cached)
                st_result = np.uint8(scaled_st_cvs)
                st_res = cv2.applyColorMap(st_result, cv2.COLORMAP_JET)
                # st_res = cv2.GaussianBlur(st_res, (13,13), 11)
                st_res = cv2.blur(st_res, (7, 7))
                st_que.put(st_res)
                stay_time_ready_flag.set()
                
                
                # async with websockets.connect("ws://localhost:8001/ws/bp_st_heatmap_innerpass") as websocket:
                async with websockets.connect("ws://127.0.0.1:8001/ws/bp_st_heatmap_innerpass") as websocket:
                    try:
                        retval, buffer = cv2.imencode('.jpg', st_res, [cv2.IMWRITE_JPEG_QUALITY,80])
                        # print('buffer : ' , buffer)
                        data = base64.b64encode(buffer)
                        # print('sent data : ' , data)
                        # data = data.decode('utf-8')
                        await websocket.send(data)
                        await websocket.close()
                    except Exception as e:
                        print(f'something went wrong in ws send, error : {e}')
                        await websocket.close()
            else:
                print('not valid result in pre loaded stay time calc')
            
            glb_cvs += pre_loaded_glb_cvs
            stay_time_cvs += pre_loaded_st_cvs
            
            refined_pts_lst = refined_pts_lst_loaded
            
            pre_loaded_data_available = True
        print('preload check6')
    except:
        print('fail: preload collision data')
    
    # 중간에 실패시 모두 롤백하는 기능 필요
    
    
    
    # data lenght validation feature needed in the future
    
    
    
    pre_loaded_once = False
    pass_load_raw_data_once = False
    while True:
        now = datetime.now()
        today_string = now.strftime('%Y-%m-%d')
        make_dir(f'data/{today_string}/')
        filename = f'data/{today_string}/{today_string}_raw_data'
        # len_data = 0
        if  (not pre_loaded_once) and pre_loaded_data_available:
            prev_data = refined_pts_lst
            pre_loaded_once = True
        else:
            prev_data = [] #ct pts lst
        
        with open(filename, 'rb') as f:
            while True:
                # print('inner while starts')
                # with open(filename, 'rb') as f:
                #     try:
                #         while True:
                #             # data.append(pickle.load(f))
                #             data.append(pickle.load(f))
                #     except EOFError:
                #         pass
                
                new_data = [] # ct pts lst
                
                # print('before try')
                try:
                    while True:
                        # data.append(pickle.load(f))
                        loaded_data = pickle.load(f)
                        cls_cvt = cvt_pkl_to_cls(loaded_data)
                        new_data.append(cls_cvt)
                except EOFError:
                    print('EOFError 2')
                
                if not pass_load_raw_data_once:
                    new_data = []
                    pass_load_raw_data_once = True
                
                if not new_data:
                    print('data not appended')
                    now_time = datetime.now()
                    if (now_time + timedelta(seconds=15)).day == (datetime.now()).day + 1: # 잠시 딜레이 될 때 비록 새벽이지만 Que가 약간 쌓일 위험성이 있음.
                        time.sleep(16)
                        # print('mid level break')
                        break
                    time.sleep(time_interval)
                    
                
                elif new_data:
                    # len_data += len(new_data)
                    print('new data added')
                    print('len new data = ' , len(new_data))
                    start = time.time()
                    cls_lst = new_data
    
                    smoothed_center_points_lst = smooth_center_points_lst_from_cls_lst(cls_lst)

                    mapped_pts = map_points(smoothed_center_points_lst, cls_lst)

                    filtered_points_lst = filter_out_no_movement(mapped_pts, 200)


                    refined_pts_lst, contained_pts_lst = refine_points_lst_by_moving_distance(filtered_points_lst)

                    # collison_events_lst = detect_collisions(refined_pts_lst)
                    collison_events_lst = detect_collisons_partly(prev_data, refined_pts_lst)
                    
                    # print('len collision events lst = ', len(collison_events_lst))
                    
                    

                    #############################################################
                    print('start draw stay time')
                    draw_stay_time(stay_time_cvs, refined_pts_lst, contained_pts_lst)


                    
                    # with open("data.pickle","rb") as f:
                    #     data = pickle.load(f)
                    # print(data)
                    data = collison_events_lst

                

                    # print(len(data))
                    # glb_cvs = np.zeros((h, w), dtype=np.int64)
                    empty_temp = np.full((h, w), False)
                    draw_template = np.zeros((h, w), dtype=np.uint8)
                    cnt = 0
                    for item in data:
                        cnt+=1
                        
                        draw = draw_template.copy()
                        tf_template = empty_temp.copy()
                        
                        inter_pt, prev_pt, next_pt, other_prev_pt, other_next_pt = item
                        
                        cv2.circle(draw, (int(inter_pt[0]), int(inter_pt[1])), 30, (255, 255, 255), -1)
                        # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
                        # cv2.imshow('draw', draw)
                        # cv2.waitKey(0)
                        tf_template[draw != 0 ] = True
                        
                        glb_cvs += tf_template
                        
                        
                        # cv2.circle(draw, (int(inter_pt[0]), int(inter_pt[1])), 30, (255, 255, 255), -1)
                        # tf_template[draw != 0 ] = True
                        # glb_cvs += tf_template
                        
                        # print('sum = ' , np.sum(glb_cvs))
                        ################################
                        # # print(inter_pt)
                        # temp_img = bp_background.copy()
                        # former_pt, futher_pt = strech_vec(prev_pt, next_pt)
                        # other_former_pt, other_futher_pt = strech_vec(other_prev_pt, other_next_pt)
                        # cv2.arrowedLine(temp_img, (int(former_pt[0]), int(former_pt[1])), (int(futher_pt[0]), int(futher_pt[1])), (150, 100, 80), 3, 8, tipLength=0.2)
                        # cv2.arrowedLine(temp_img, (int(other_former_pt[0]), int(other_former_pt[1])), (int(other_futher_pt[0]), int(other_futher_pt[1])), (150, 80, 120), 3, 8, tipLength=0.3)
                        # # cv2.line(temp_img, (int(former_pt[0]), int(former_pt[1])), (int(futher_pt[0]), int(futher_pt[1])), (255, 255, 0), 4, 8)
                        # # cv2.line(temp_img, (int(other_former_pt[0]), int(other_former_pt[1])), (int(other_futher_pt[0]), int(other_futher_pt[1])), (255, 255, 0),  4, 8)
                        # cv2.line(temp_img, (int(prev_pt[0]), int(prev_pt[1])), (int(next_pt[0]), int(next_pt[1])), (100, 255, 0), 4, 8)
                        # cv2.line(temp_img, (int(other_prev_pt[0]), int(other_prev_pt[1])), (int(other_next_pt[0]), int(other_next_pt[1])), (0, 255, 180),  4, 8)
                        # cv2.circle(temp_img, (int(inter_pt[0]), int(inter_pt[1])), 4, (0, 0, 255), -1)
                        # cv2.imshow('t', temp_img)
                        # if cnt <= 10:
                        #     wk = 0
                        # else:
                        #     wk = 10
                        # cv2.waitKey(wk)
                        ################################


                    glb_max_val = np.max(glb_cvs)
                    if glb_max_val != 0:
                        scaled_glb_cvs = 254*(glb_cvs/glb_max_val)
                        glb_result = np.uint8(scaled_glb_cvs)

                        res_show = cv2.applyColorMap(glb_result, cv2.COLORMAP_JET)
                        # res_show = cv2.GaussianBlur(res_show,(13,13), 11)
                        res_show = cv2.blur(res_show, (7, 7))
                        # cv2.imshow('glb result', glb_result)
                        # cv2.imshow('glb result', res_show)
                        # cv2.waitKey(0)

                        # glb_result = cv2.cvtColor(glb_result, cv2.COLOR_GRAY2BGR)

                        # end = time.time()
                        # elapsed_time = end - start
                        # print('elapsed time = ' , elapsed_time)

                        # result = cv2.addWeighted(bp_background.copy(), 0.5, glb_result, 0.5, 0)
                        # bg_ratio = 0.6
                        # result = cv2.addWeighted(bp_background.copy(), bg_ratio, res_show, 1-bg_ratio, 0)
                        # draw_colorbar(result)
                        # cv2.imshow('result1', result)
                        # non_zero_idx = glb_result != 0
                        
                        # result[glb_result == 0] = bp_background[glb_result == 0]
                        # draw_colorbar(result)
                        
                        # cv2.imshow('result2', result)
                        # cv2.waitKey(600)
                        collision_que.put(res_show)
                        collision_ready_flag.set()
                        
                        # async with websockets.connect("ws://localhost:8001/ws/bp_collision_heatmap_innerpass") as websocket:
                        async with websockets.connect("ws://127.0.0.1:8001/ws/bp_collision_heatmap_innerpass") as websocket:
                            try:
                                retval, buffer = cv2.imencode('.jpg', res_show, [cv2.IMWRITE_JPEG_QUALITY,80])
                                data = base64.b64encode(buffer)
                                await websocket.send(data)
                                await websocket.close()
                            except Exception as e:
                                print(f'something went wrong in ws send, error : {e}')
                                await websocket.close()
                        # bp_heatmap_que.put(res_show)
                        
                    else:
                        print('valid result not created, in collision analysis')

                    prev_data = prev_data + refined_pts_lst
                    # print('prev data = ', prev_data)
                    # prev_cvs = glb_cvs.copy()
                    # print('glb cvs total sum',np.sum(glb_cvs))
                    
                    
                    ###############################
                    # stay_time_cvs
                    st_max_val = np.max(stay_time_cvs)
                    if st_max_val != 0:
                        scaled_st_cvs = 254*(stay_time_cvs/st_max_val)
                        st_result = np.uint8(scaled_st_cvs)
                        st_res = cv2.applyColorMap(st_result, cv2.COLORMAP_JET)
                        # st_res = cv2.GaussianBlur(st_res, (13,13), 11)
                        st_res = cv2.blur(st_res, (7, 7))
                        st_que.put(st_res)
                        stay_time_ready_flag.set()
                        # bg_ratio = 0.5
                        # st_result = cv2.addWeighted(bp_background.copy(), bg_ratio, st_res, 1-bg_ratio, 0)
                        # cv2.imshow('st', st_result)
                        # cv2.waitKey(300)
                        # async with websockets.connect("ws://localhost:8001/ws/bp_st_heatmap_innerpass") as websocket:
                        async with websockets.connect("ws://127.0.0.1:8001/ws/bp_st_heatmap_innerpass") as websocket:
                            try:
                                retval, buffer = cv2.imencode('.jpg', st_res, [cv2.IMWRITE_JPEG_QUALITY,80])
                                # print('buffer : ' , buffer)
                                data = base64.b64encode(buffer)
                                # print('sent data : ' , data)
                                # data = data.decode('utf-8')
                                await websocket.send(data)
                                await websocket.close()
                            except Exception as e:
                                print(f'something went wrong in ws send, error : {e}')
                                await websocket.close()
                        
                    else:
                        print('valid result not created, in stay time')


                    # pre_loaded_glb_cvs = np.load('glb_cvs_cached.npy')
                    # pre_loaded_st_cvs = np.load('st_cvs_cached.npy')

                    with open(f'data/{today_string}/{today_string}_refined_pts_lst_cached', 'wb') as f1:
                        pickle.dump(prev_data, f1)
                    np.save(f'data/{today_string}/{today_string}_glb_cvs_cached', glb_cvs)
                    np.save(f'data/{today_string}/{today_string}_st_cvs_cached', stay_time_cvs)
                    with open(f'data/{today_string}/{today_string}_meta', 'wb') as f2:
                        meta_dict = {'len_refined_pts_lst': len(prev_data), 'glb_max_val':glb_max_val, 'st_max_val': st_max_val, }
                        pickle.dump(meta_dict, f2)



                    print('process end')
                    end = time.time()
                    print('collision analysis elapsed time = ' , end - start)
                    now_time = datetime.now()
                    if (now_time + timedelta(seconds=15)).day == (datetime.now()).day + 1: # 잠시 딜레이 될 때 비록 새벽이지만 Que가 약간 쌓일 위험성이 있음.
                        time.sleep(16)
                        # print('mid level break')
                        break
                    time.sleep(time_interval)
                    print('after time sleep')
                    # if not collision_op_flag.is_set():
                    #     print('waiting.. collision op flag is not set.')
                    # collision_op_flag.wait()

def run_analyze(collision_op_flag, stay_time_op_flag, collision_ready_flag, stay_time_ready_flag, collision_que, st_que, ):
    asyncio.get_event_loop().run_until_complete(analyze(collision_op_flag, stay_time_op_flag, collision_ready_flag, stay_time_ready_flag, collision_que, st_que, ))
    asyncio.get_event_loop().run_forever()