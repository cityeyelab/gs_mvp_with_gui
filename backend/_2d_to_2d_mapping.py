import pickle
import numpy as np


def load_matrices(path):
    # print('check')
    with open(path, "rb") as f:
        result = pickle.load(f)
    # print('pickled result = ' , result)
    return result

def cvt_loaded_data(dic):
    M_f = dic['cam_matrix']
    M_inv = dic['cam_matrix_inv']
    M_bp = dic['blueprint_matrix']
    M_r = dic['rotation_matrix']
    return M_f, M_inv, M_bp, M_r



def point_transform(source_point, matrix):
    result = np.squeeze(matrix @ np.expand_dims(np.append(source_point, 1), axis=1))
    return result

def normalize_pt(point):
    return point[:2]/point[2]


def transform_points(pts_lst, M_inv, M_r, M_bp):
    result_lst = []
    for pt in pts_lst:
        point1 = point_transform(pt, M_inv)
        point2 = normalize_pt(point1)
        point3 = point_transform(point2, M_r)
        point4 = point_transform(point3, M_bp)
        point5 = normalize_pt(point4)
        result = (int(point5[0]),int(point5[1]))
        result_lst.append(result)
    return result_lst

def mapping(data_path, pts_lst):
    # print('pts lst in mapping = ' , pts_lst)
    loaded_data_dict = load_matrices(data_path)
    cvted_data = cvt_loaded_data(loaded_data_dict)
    M_f, M_inv, M_bp, M_r = cvted_data
    result_pts_lst = transform_points(pts_lst, M_inv, M_r, M_bp)

    return result_pts_lst

