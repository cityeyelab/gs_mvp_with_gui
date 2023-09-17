import cv2
import argparse
import numpy as np
# now let's initialize the list of reference point

ref_point = []
rectangle_lst = []


blank_image = np.zeros((1080, 1920, 3), np.uint8)
blank_image_orig = np.zeros((1080, 1920, 3), np.uint8)
rectangle_record = np.zeros((1080, 1920, 3), np.uint8)

is_clicked = False
first_point = (0, 0)

def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, crop
    global is_clicked
    global blank_image
    global first_point
    global rectangle_record
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed

    # print('event = ' , event)
    # print('x, y = ' , x, ' ', y)

    if not is_clicked:
        first_point = (0, 0)

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        first_point = (x, y)
        is_clicked = True
        # print('event = ' , event)
        # print('x, y = ' , x, ' ', y)

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))
        blank_image = blank_image_orig.copy()
        cv2.rectangle(blank_image, ref_point[0], ref_point[1], (0, 255, 0), 5)
        rectangle_lst.append(ref_point)
        is_clicked = False
        cv2.rectangle(rectangle_record, ref_point[0], ref_point[1], (0, 255, 0), 5)
        # print('rectangle list = ' , rectangle_lst)
        # print('event = ' , event)
        # print('x, y = ' , x, ' ', y)
    
    if is_clicked:
        blank_image = blank_image_orig.copy()
        cv2.rectangle(blank_image, first_point, (x,y), (0, 255, 0), 2)
        # draw a rectangle around the region of interest
        # cv2.rectangle(blank_image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        # cv2.imshow("frame", blank_image)


# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
# image = cv2.imread(args["image"])
path = 'rtsp://admin:self1004@@118.37.223.147:8522/live/main8'
cap_loader = cv2.VideoCapture(path)


result = np.zeros((1080, 1920, 3), np.uint8)

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

        blank_idx = np.where(blank_image!=0)
        rectangle_rec_idx = np.where(rectangle_record!=0)
        # blank_idx_complementary = np.where(blank_image == 0)
        # result = cv2.addWeighted(blank_image, 0.5, frame, 0.5, 0.0)

        frame[blank_idx] = blank_image[blank_idx]
        frame[rectangle_rec_idx] = rectangle_record[rectangle_rec_idx]
        # result[blank_idx_complementary] = frame[blank_idx_complementary]

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # press 'r' to reset the window
        # if key == ord("r"):
        #     frame = clone.copy()

        # if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            break



# # keep looping until the 'q' key is pressed
# while True:
#     # display the image and wait for a keypress
#     cv2.imshow("image", image)
#     key = cv2.waitKey(1) & 0xFF

#     # press 'r' to reset the window
#     if key == ord("r"):
#         image = clone.copy()

#     # if the 'c' key is pressed, break from the loop
#     elif key == ord("c"):
#         break

# close all open windows
cv2.destroyAllWindows() 