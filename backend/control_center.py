
import time
import numpy as np
from .firebase import firebase_upload
import sys

def control_center(op_flag, que1, que2, que3, exit_event):
    # print('cc check!')
    area1_global_cnt = 0
    area1_car_wash_waiting_cnt = 0
    area1_place0_cnt = 0
    area1_pos = []

    area3_global_cnt = 0
    area3_car_wash_waiting_cnt = 0
    area3_pos = []

    area4_global_cnt = 0
    area4_car_wash_waiting_cnt = 0
    area4_electric_vehicle_charging_cnt = 0
    area4_car_interior_wash_cnt = 0
    area4_pos = []

    congestion = 0
    number_of_waiting_cars = 0
    waiting_time = 0
    electric_charging_waiting_cnt = 0
    car_interior_wash_cnt = 0

    updated = True

    data = {'congestion':congestion, 'number_of_waiting_cars': number_of_waiting_cars, 'waiting_time':waiting_time,
            'electric_charging_waiting_cnt':electric_charging_waiting_cnt, 'car_interior_wash_cnt':car_interior_wash_cnt}
    prev_data = {'congestion':congestion, 'number_of_waiting_cars': number_of_waiting_cars, 'waiting_time':waiting_time,
            'electric_charging_waiting_cnt':electric_charging_waiting_cnt, 'car_interior_wash_cnt':car_interior_wash_cnt}

    while True:
        if exit_event.is_set():
            break
        
        time.sleep(0.02)
        # print('cc check2')
        if not que1.empty(): # area1
            qdata1 = que1.get()
            if type(qdata1) == type(None):
                sys.exit()
                break
            # qdata1 =  {'pos_data': [], 'area': 1, 'global_cnt': 0, 'car_washing_waiting_cnt': 0, 'place0_cnt': 0}
            area1_global_cnt = qdata1['global_cnt']
            area1_car_wash_waiting_cnt = qdata1['car_wash_waiting']
            area1_place0_cnt = qdata1['place0']
            area1_pos = qdata1['pos_data']
            # print('qdata1 = ', qdata1)
            updated = True

        if not que2.empty(): # area3
            qdata2 = que2.get()
            if type(qdata2) == type(None):
                sys.exit()
                break
            # qdata2 =  {'pos_data': [], 'area': 3, 'global_cnt': 0, 'car_washing_waiting_cnt': 0}
            area3_global_cnt = qdata2['global_cnt']
            area3_car_wash_waiting_cnt = qdata2['car_wash_waiting']
            area3_pos = qdata2['pos_data']
            # print('qdata2 = ' , qdata2)
            updated = True

        if not que3.empty(): #area4
            qdata3 = que3.get()
            if type(qdata3) == type(None):
                sys.exit()
                break
            # qdata3 =  {'pos_data': [], 'area': 4, 'global_cnt': 3, 'car_washing_waiting_cnt': 0,
            #               'car_interior_washing_waiting_cnt': 2, 'electric_vehicle_charging_waiting_cnt': 0}
            area4_global_cnt = qdata3['global_cnt']
            area4_car_wash_waiting_cnt = qdata3['car_wash_waiting']
            area4_car_interior_wash_cnt = qdata3['car_interior_washing']
            area4_electric_vehicle_charging_cnt = qdata3['electric_vehicle_charging']
            area4_pos = qdata3['pos_data']
            # print('qdata3 = ' , qdata3)
            updated = True
        

        if updated:
            total_cnt = area1_global_cnt + area3_global_cnt + area4_global_cnt
            car_wash_cnt = area1_car_wash_waiting_cnt + area3_car_wash_waiting_cnt + area4_car_wash_waiting_cnt
            electric_charging_waiting_cnt = area4_electric_vehicle_charging_cnt
            car_interior_wash_cnt = area4_car_interior_wash_cnt
            place0_cnt = area1_place0_cnt

            congestion = calc_congetsion(total_cnt=total_cnt, car_wash_cnt=car_wash_cnt, electric_charging_waiting_cnt=electric_charging_waiting_cnt, car_interior_wash_cnt=car_interior_wash_cnt)
            number_of_waiting_cars = car_wash_cnt
            waiting_time = 240 * number_of_waiting_cars

            # data = {'congestion':congestion, 'number_of_waiting_cars': number_of_waiting_cars, 'waiting_time':waiting_time}
            data['congestion'] = congestion
            data['number_of_waiting_cars'] = number_of_waiting_cars
            data['waiting_time'] = waiting_time
            data['electric_charging_waiting_cnt'] = electric_charging_waiting_cnt
            data['car_interior_wash_cnt'] = car_interior_wash_cnt

            if data != prev_data:
                firebase_upload(data=data)
                # print('data updated : ', data)
                prev_data = data.copy()


        updated = False
    
    print('control center end')
        
    
def custom_sigmoid(x):
    a = 0.6
    b = 5
    return 10/(1 + np.exp(-1*a*(x - b)))

def calc_congetsion(total_cnt, car_wash_cnt, electric_charging_waiting_cnt, car_interior_wash_cnt):
    weighted_sum = total_cnt + 0.8*car_wash_cnt - 0.6*electric_charging_waiting_cnt - 0.4*car_interior_wash_cnt
    result = custom_sigmoid(weighted_sum)
    result = round(result, 2)
    return result
