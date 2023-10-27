# import joblib
# loaded_model = joblib.load('xgboost_model.pkl')
def area1_filter_blacklist(dets):
    area1_del_list = []
    for i, det in enumerate(dets[:]):
        p_x1, p_y1 = int(det[0]), int(det[1])
        p_x2, p_y2 = int(det[2]), int(det[3])
       
        w = p_x2 - p_x1
        h = p_y2 - p_y1

        area_of_bbox = w * h

        condition1 = ((500 <= p_x1 <= 520) and (1010 <= p_y1 <= 1025))
        condition2 = ((10 <= p_x1 <= 30) and (440 <= p_y1 <= 765) and (850 <= p_x2) and (1070 <= p_y2))
        condition3 = (area_of_bbox >= 300000)

        if condition1 or condition2 or condition3:
            area1_del_list.append(i)

    for index in reversed(area1_del_list):
        print(f"remove area1 : {dets[index]}")
        del dets[index]

def area3_filter_blacklist(dets):
    area3_del_list = []
    for i, det in enumerate(dets[:]):
        p_x1, p_y1 = int(det[0]), int(det[1])
        p_x2, p_y2 = int(det[2]), int(det[3])
       
        # w = p_x2 - p_x1
        # h = p_y2 - p_y1

        condition1 = (1830 <= p_x1 <= 1899)
        condition2 = (p_x1 <= 20) and (p_y1 <= 500)

        if condition1 or condition2:
            area3_del_list.append(i)

    for index in reversed(area3_del_list):
        print(f"remove area3 : {dets[index]}")
        del dets[index]

def area4_filter_blacklist(dets):
    area4_del_list = []
    for i, det in enumerate(dets[:]):
        p_x1, p_y1 = int(det[0]), int(det[1])
        p_x2, p_y2 = int(det[2]), int(det[3])

        condition1 = (p_x1 <= 10) and (755 <= p_y1 <= 899) and (280 <= p_x2 <= 289) and (1070 <= p_y2 <= 1080)
        condition2 = (p_x1 < 150 and p_y1 < 300)
        condition3 = (1540 <= p_x1 <= 1570) and (60 <= p_y1 <= 80)
        
        if condition1 or condition2 or condition3 :
            area4_del_list.append(i)

    for index in reversed(area4_del_list):
        print(f"remove area4 : {dets[index]}")
        del dets[index]

filter_blacklist_fn_callback_lst = [area1_filter_blacklist, area3_filter_blacklist, area4_filter_blacklist]