# import joblib
# loaded_model = joblib.load('xgboost_model.pkl')
# condition를 list, tuple로 
def area1_filter_blacklist(dets):
    # pass
    for det in dets[:]:
        p_x1, p_y1 = int(det[0]), int(det[1])
        p_x2, p_y2 = int(det[2]), int(det[3])
       
        w = p_x2 - p_x1
        h = p_y2 - p_y1
 
        ratio = round(w / h, 4)
        ratio_difference = abs(6.0 - ratio)
   
        area_of_bbox = w * h
        if ((p_x1 >=  500 and p_x1 <= 520) and (p_y1 <= 1025 and p_y1 >= 1010)) or (ratio_difference < 2):
            dets.remove(det)
            print(f"remove area1 : {det}")
        elif (p_x1 >= 1230 and p_y1 >= 190):
            dets.remove(det)
            print(f"remove area1 : {det}")            
        elif ((p_x1 >=  10 and p_x1 <= 30) and (p_y1 <= 765 and p_y1 >= 740)):
            dets.remove(det)
            print(f"remove area1 : {det}")
        elif area_of_bbox < 5000:
            dets.remove(det)
            print(f"remove area1 : {det}")
        # X_test = 0
        # result = loaded_model.predict(X_test)

def area3_filter_blacklist(dets):
    # pass
    for det in dets[:]:
        p_x1, p_y1 = int(det[0]), int(det[1])
        p_x2, p_y2 = int(det[2]), int(det[3])
       
        w = p_x2 - p_x1
        h = p_y2 - p_y1
 
        ratio = round(w / h, 4)
        ratio_difference = abs(0.2 - ratio)
   
        area_of_bbox = w * h
        if p_x1 > 1830 and p_x1 < 1899 and ratio_difference < 0.2:
            dets.remove(det)
            print(f"remove area3 : {det}")
        elif (p_x1 >= 0  and p_x1 <= 20) and (p_y1 <= 340 and p_y1 >= 0):
            dets.remove(det)
            print(f"remove area3 : {det}")
        # X_test = 0
        # result = loaded_model.predict(X_test)
def area4_filter_blacklist(dets):
    # pass
    for det in dets[:]:
        p_x1, p_y1 = int(det[0]), int(det[1])
        p_x2, p_y2 = int(det[2]), int(det[3])
       
        w = p_x2 - p_x1
        h = p_y2 - p_y1
 
        ratio = round(w / h, 4)
        ratio_difference = abs(6.0 - ratio)
   
        area_of_bbox = w * h
        if (p_x1 >=  0 and p_x1 <= 16) and (p_y1 <= 899 and p_y1 >= 840):
            dets.remove(det)
            print(f"remove area4 : {det}")
        elif (p_x1 >=  0 and p_x1 <= 10) and (p_y1 <= 810 and p_y1 >= 755) and (p_x2 >= 280 and p_x2 <= 289) and (p_y2 >= 1070 and p_y2 <= 1080):
            dets.remove(det)
            print(f"remove area4 : {det}")
        elif (p_x1 >=  1540 and p_x1 <= 1570) and (p_y1 <= 80 and p_y1 >= 60):
            dets.remove(det)
            print(f"remove area4 : {det}")
        # X_test = 0
        # result = loaded_model.predict(X_test)


filter_blacklist_fn_callback_lst = [area1_filter_blacklist, area3_filter_blacklist, area4_filter_blacklist]