# import joblib
# loaded_model = joblib.load('xgboost_model.pkl')

def area1_filter_blacklist(dets):
    pass
    # for det in dets:
    #     p_x1, p_y1 = int(det[0]), int(det[1])
    #     p_x2, p_y2 = int(det[2]), int(det[3])
       
    #     w = p_x2 - p_x1
    #     h = p_y2 - p_y1
 
    #     ratio = round(w / h, 4)
    #     ratio_difference = abs(6.0 - ratio)
   
    #     area_of_bbox = w * h
    #     X_test = 0
    #     result = loaded_model.predict(X_test)

def area3_filter_blacklist(dets):
    pass
    # for det in dets:
    #     p_x1, p_y1 = int(det[0]), int(det[1])
    #     p_x2, p_y2 = int(det[2]), int(det[3])
       
    #     w = p_x2 - p_x1
    #     h = p_y2 - p_y1
 
    #     ratio = round(w / h, 4)
    #     ratio_difference = abs(6.0 - ratio)
   
    #     area_of_bbox = w * h
    #     X_test = 0
    #     result = loaded_model.predict(X_test)
def area4_filter_blacklist(dets):
    pass
    # for det in dets:
    #     p_x1, p_y1 = int(det[0]), int(det[1])
    #     p_x2, p_y2 = int(det[2]), int(det[3])
       
    #     w = p_x2 - p_x1
    #     h = p_y2 - p_y1
 
    #     ratio = round(w / h, 4)
    #     ratio_difference = abs(6.0 - ratio)
   
    #     area_of_bbox = w * h
    #     X_test = 0
    #     result = loaded_model.predict(X_test)


filter_blacklist_fn_callback_lst = [area1_filter_blacklist, area3_filter_blacklist, area4_filter_blacklist]