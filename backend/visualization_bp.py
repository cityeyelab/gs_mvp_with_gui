import cv2
from ._2d_to_2d_mapping import mapping

blueprint = cv2.imread('backend/blueprint.png')
height_bp, width_bp  = blueprint.shape[:2]

# car_image = cv2.imread('car_img.png')
car_image = cv2.imread('backend/car_img_red.png')
car_image = cv2.resize(car_image, (40, 40))
height_car, width_car  = car_image.shape[:2]





def draw_car_img(img ,ct_pt):
    try:
        img[int(ct_pt[1]):int(ct_pt[1]+height_car), int(ct_pt[0]):int(ct_pt[0]+width_car)] = car_image
    except ValueError:
        print('drawing car img value error. maybe, caused by whole size of img')

def visualize_bp(op_flag, que_area1, que_area3, que_area4, drawing_result_que):
    # print('bp vis')
    while True:
        # if not op_flag.is_set():
        #     cv2.destroyAllWindows()
        op_flag.wait()
        background_img = blueprint.copy()

        area1_ct_pts_lst = que_area1.get()
        area3_ct_pts_lst = que_area3.get()
        area4_ct_pts_lst = que_area4.get()

        mapped_pts_arear1 = mapping('backend/mapping_matrix/mapping_area1', area1_ct_pts_lst)
        mapped_pts_arear3 = mapping('backend/mapping_matrix/mapping_area3', area3_ct_pts_lst)
        mapped_pts_arear4 = mapping('backend/mapping_matrix/mapping_area4', area4_ct_pts_lst)

        # print('mapped pts area1 =  ', mapped_pts_arear1)
        for pt in mapped_pts_arear1:
            bp_drawing_pt = (pt[0] - width_car/2, pt[1] - height_car/2)
            draw_car_img(background_img, bp_drawing_pt)
            cv2.circle(background_img, pt, 8, (255,0,255), -1)
        for pt in mapped_pts_arear3:
            bp_drawing_pt = (pt[0] - width_car/2, pt[1] - height_car/2)
            draw_car_img(background_img, bp_drawing_pt)
            cv2.circle(background_img, pt, 8, (255,0,255), -1)
        for pt in mapped_pts_arear4:
            bp_drawing_pt = (pt[0] - width_car/2, pt[1] - height_car/2)
            draw_car_img(background_img, bp_drawing_pt)
            cv2.circle(background_img, pt, 8, (255,0,255), -1)


        drawing_result_que.put(background_img)

        # cv2.namedWindow('blueprint', cv2.WINDOW_NORMAL)
        # cv2.imshow('blueprint', background_img)
        # cv2.waitKey(1)