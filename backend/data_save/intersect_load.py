import pickle
import numpy as np
import cv2
import time



bp_background = cv2.imread('visualization/assets/blueprint.png')
h, w, c = bp_background.shape
with open("data.pickle","rb") as f:
    data = pickle.load(f)
# print(data)

def strech_vec(v1, v2): #vec = ()
    direction = (v2[0] - v1[0], v2[1] - v1[1])
    former_pt = (v1[0] - direction[0], v1[1] - direction[1])
    futher_pt = (v2[0] + direction[0], v2[1] + direction[1])
    return former_pt, futher_pt


def draw_colorbar(img):
    font = cv2.FONT_HERSHEY_PLAIN
    # start = time.time()
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
    # end = time.time()
    # color_bar_time = end - start
    # print('colorbar time = ', color_bar_time)

print(len(data))
glb_cvs = np.zeros((h, w), dtype=np.int64)
empty_temp = np.full((h, w), False)
draw_template = np.zeros((h, w), dtype=np.uint8)
cnt = 0
start = time.time()
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
end = time.time()
elapsed_time = end - start
print('proc time = ' , elapsed_time)

glb_max_val = np.max(glb_cvs)
scaled_glb_cvs = 254*(glb_cvs/glb_max_val)
glb_result = np.uint8(scaled_glb_cvs)

# glb_result = cv2.GaussianBlur(glb_result,(13,13), 11)

res_show = cv2.applyColorMap(glb_result, cv2.COLORMAP_JET)
# res_show = cv2.GaussianBlur(res_show,(13,13), 11)
# cv2.imshow('glb result', glb_result)
# cv2.imshow('glb result', res_show)
# cv2.waitKey(0)

# glb_result = cv2.cvtColor(glb_result, cv2.COLOR_GRAY2BGR)


# result = cv2.addWeighted(bp_background.copy(), 0.5, glb_result, 0.5, 0)
bg_ratio = 0.6
result = cv2.addWeighted(bp_background.copy(), bg_ratio, res_show, 1-bg_ratio, 0)
draw_colorbar(result)
cv2.imshow('result1', result)
non_zero_idx = glb_result != 0
result[glb_result == 0] = bp_background[glb_result == 0]
draw_colorbar(result)
cv2.imshow('result2', result)
cv2.waitKey(0)