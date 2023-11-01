# import joblib
# loaded_model = joblib.load('xgboost_model.pkl')

def area1_filter_blacklist(dets):
    area1_del_list = []
    for i, det in enumerate(dets[:]):
        if area1_condition_check(det):
            area1_del_list.append(i)

    for index in reversed(area1_del_list):
        # print(f"remove area1 : {dets[index]}")
        del dets[index]

def area1_condition_check(det):
    p_x1, p_y1, p_x2, p_y2 = map(int, det[:4])
    
    w = p_x2 - p_x1
    h = p_y2 - p_y1

    area_of_bbox = w * h
    if (500 <= p_x1 <= 520) and (1010 <= p_y1 <= 1025):
        return True
    elif (10 <= p_x1 <= 30) and (440 <= p_y1 <= 765) and (850 <= p_x2) and (1070 <= p_y2):
        return True
    elif area_of_bbox >= 300000:
        return True
    
def area3_filter_blacklist(dets):
    area3_del_list = []
    for i, det in enumerate(dets[:]):
        if area3_condition_check(det):
            area3_del_list.append(i)

    for index in reversed(area3_del_list):
        # print(f"remove area3 : {dets[index]}")
        del dets[index]

def area3_condition_check(det):
    p_x1, p_y1 = map(int, det[:2])
    if (1830 <= p_x1 <= 1899):
        return True
    elif (p_x1 <= 20) and (p_y1 <= 500):
        return True
    
def area4_filter_blacklist(dets):
    area4_del_list = []
    for i, det in enumerate(dets[:]):
        if area4_condition_check(det):
            area4_del_list.append(i)

    for index in reversed(area4_del_list):
        print(f"remove area4 : {dets[index]}")
        del dets[index]

def area4_condition_check(det):
    p_x1, p_y1 = map(int, det[:2])
    if (p_x1 <= 25) and (840 <= p_y1 <= 899):
        return True
    elif (p_x1 < 120 and p_y1 < 200):
        return True
    elif (1540 <= p_x1 <= 1570) and (60 <= p_y1 <= 80):
        return True

filter_blacklist_fn_callback_lst = [area1_filter_blacklist, area3_filter_blacklist, area4_filter_blacklist]