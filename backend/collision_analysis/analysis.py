import time

def analyze(que1, que2, que3):
    while True:
        time.sleep(0.01)
        if not que1.empty():
            q1_result = que1.get()
            print('q1 result = ' , q1_result)
        if not que2.empty():
            q2_result = que2.get()
            print('q2 result = ' , q2_result)
        if not que3.empty():
            q3_result = que3.get()
            print('q3 result = ' , q3_result)
        time.sleep(0.01)
        
        