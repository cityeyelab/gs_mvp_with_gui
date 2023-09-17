import cv2
import argparse
import numpy as np
import datetime

# map_file_name = './maps/area4_electric_vehicle_charging_map.npy'
map_file_name = './maps/area4_car_interior_washing_map.npy'
check_target = np.load(map_file_name)
target_map_img = np.zeros((1080, 1920, 3), np.uint8)
target_non_false_idx  = np.where(check_target==True)
target_map_img[target_non_false_idx] = (0,255,0)

ref_point = []
rectangle_lst = []


blank_image = np.zeros((1080, 1920, 3), np.uint8)
blank_image_orig = np.zeros((1080, 1920, 3), np.uint8)

touch_record = np.zeros((1080, 1920, 3), np.uint8)

hover_img = np.zeros((1080, 1920, 3), np.uint8)
hover_img_orig = np.zeros((1080, 1920, 3), np.uint8)


is_clicked = False
first_point = (0, 0)

point_size = 40

def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    # global ref_point, crop
    global is_clicked
    # global blank_image
    # global first_point
    global touch_record
    global point_size
    global hover_img
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed

    # print('event = ' , event)
    # print('x, y = ' , x, ' ', y)
    hover_img = hover_img_orig.copy()
    cv2.line(hover_img, (x,y), (x,y), (0, 255, 0), point_size)

    if not is_clicked:
        first_point = (0, 0)

    if event == cv2.EVENT_LBUTTONDOWN:
        # ref_point = [(x, y)]
        # first_point = (x, y)
        is_clicked = True
        # print('event = ' , event)
        # print('x, y = ' , x, ' ', y)

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        # ref_point.append((x, y))
        # blank_image = blank_image_orig.copy()
        # cv2.rectangle(blank_image, ref_point[0], ref_point[1], (0, 255, 0), 5)
        # rectangle_lst.append(ref_point)
        is_clicked = False
        # cv2.rectangle(rectangle_record, ref_point[0], ref_point[1], (0, 255, 0), 5)
        # print('rectangle list = ' , rectangle_lst)
        # print('event = ' , event)
        # print('x, y = ' , x, ' ', y)
    
    if is_clicked:
        # blank_image = blank_image_orig.copy()
        cv2.line(touch_record, (x,y), (x,y), (0, 255, 0), point_size)
        # draw a rectangle around the region of interest
        # cv2.rectangle(blank_image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        # cv2.imshow("frame", blank_image)


# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
# image = cv2.imread(args["image"])
path = 'rtsp://admin:self1004@@118.37.223.147:8522/live/main8' # area4
# path = 'rtsp://admin:self1004@@118.37.223.147:8522/live/main7' # area1 
# path = 'rtsp://admin:self1004@@118.37.223.147:8522/live/main6' # area3
cap_loader = cv2.VideoCapture(path)


result = np.zeros((1080, 1920, 3), np.uint8)





def make_map(map=touch_record):
    # record_template = np.zeros((1080, 1920), dtype=bool)
    record_template = np.full((1080, 1920), False)
    record_template[touch_record.any(axis=2) != 0] = True
    now_time = datetime.datetime.now()
    now_string = now_time.strftime("%Y-%m-%d_%H%M%S")
    np.save(now_string+'_area_template', record_template)

while True:
    
    # print('data load')
    # _, _ = cap_loader.read()
    # _, _ = cap_loader.read()
    ret, frame = cap_loader.read()

    # video_frame_cnt += 1
    # if video_frame_cnt == 60:
    #     image_que.put(None)
    #     cap_loader.release()
    #     print('loader break')
    #     break


    if not ret:
        break
    if ret:
        # clone = frame.copy()

        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("frame", shape_selection)


        frame[target_non_false_idx] = cv2.addWeighted(frame, 0.5, target_map_img, 0.5, 0)[target_non_false_idx] 


        # touch_rec_idx = np.where(touch_record!=0)
        # frame[touch_rec_idx] = touch_record[touch_rec_idx]

        # hover_img_idx = np.where(hover_img!=0)
        # frame[hover_img_idx] = hover_img[hover_img_idx]


        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # press 'r' to reset the window
        # if key == ord("r"):
        #     frame = clone.copy()

        # if the 'c' key is pressed, break from the loop
        if key == ord('o'):
            if point_size <= 40:
                point_size = 40
            else:
                point_size -= 20
        elif key == ord('p'):
            if point_size >= 300:
                point_size = 300
            else:
                point_size += 20
        elif key == ord("c"):
            touch_record = np.zeros((1080, 1920, 3), np.uint8)
        elif key == ord("s"):
            make_map()
            print('saved!')
        elif key == ord("q"):
            break






cv2.destroyAllWindows() 