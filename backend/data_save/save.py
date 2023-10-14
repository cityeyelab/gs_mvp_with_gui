import time
import threading
from queue import Queue
import pickle
from datetime import datetime


def collect_data(que1, que2, que3):
    save_que = Queue()
    save_thread = threading.Thread(target=save_data, args=(save_que,))
    save_thread.start()
    while True:
        time.sleep(0.01)
        if not que1.empty():
            q1_result = que1.get()
            print('q1 result = ' , q1_result)
            if type(q1_result) == None:
                break
            save_que.put(q1_result)
        if not que2.empty():
            q2_result = que2.get()
            print('q2 result = ' , q2_result)
            if type(q2_result) == None:
                break
            save_que.put(q2_result)
        if not que3.empty():
            q3_result = que3.get()
            print('q3 result = ' , q3_result)
            if type(q3_result) == None:
                break
            save_que.put(q3_result)
        
        time.sleep(0.01)
    print('analysis end')

def save_data(save_que):
    filename="_raw_data"
    while True:
        data = save_que.get()
        now = datetime.now()
        today_string = now.strftime('%Y-%m-%d')
        with open('data/'+today_string+filename, 'ab+') as fp:
            pickle.dump(data, fp)
        time.sleep(0.01)