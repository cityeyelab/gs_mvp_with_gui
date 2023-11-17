from backend.main import create_backend
from frontend.main import create_frontend
from visualization.main import create_visualization
from shared_variables import SharedVariables
from network_backend.main import create_network_backend
# from network_backend.main import app

from multiprocessing import Process
from collision_analysis.main import create_collision_analysis
import time
# from datetime import datetime, timedelta
# # import sys
# import gc
import os
import subprocess

def run_nginx():
    os.system('cd network_backend')
    os.chdir('network_backend')
    # os.system('dir')
    os.chdir('nginx')
    # os.system('cd nginx')
    # os.system('dir')
    os.system('nginx -s stop')
    os.system('nginx -t')
    print('nginx starts')
    subprocess.run(["nginx"])
    # os.system('nginx')
    # os.system('ipconfig')

def stop_nginx():
    os.system('cd network_backend')
    os.chdir('network_backend')
    # os.system('dir')
    os.chdir('nginx')
    # os.system('cd nginx')
    # os.system('dir')
    os.system('nginx -s stop')
    # os.system('nginx -t')
    # subprocess.run(["nginx"])
    print('nginx stopped')

def main_run():
    # whole_proc_test_cnt = 0
    # 
    # while True:
        # whole_proc_test_cnt += 1
        # print('whole cnt = ' , whole_proc_test_cnt)
    p0 = Process(target=run_nginx)
    p0.start()
        
    
    shared_variables = SharedVariables()
    

    p1 = Process(target = create_frontend, args= (shared_variables.args, shared_variables.drawing_result_ques, shared_variables.exit_event), daemon=False)
    p2 = Process(target = create_backend, args= (shared_variables.args, shared_variables.model_proc_result_ques, shared_variables.exit_event, ), daemon=False)
    p3 = Process(target = create_visualization, args=(shared_variables.args, shared_variables.model_proc_result_ques, shared_variables.drawing_result_ques, shared_variables.exit_event,
                                                    shared_variables.collision_analysis_queue, shared_variables.collision_analysis_rt_queue, shared_variables.stay_time_queue))
    # p4 = Process(target = check_exit, args=(shared_variables.exit_event, p2, p3))
    p4 = Process(target= create_collision_analysis, args=(shared_variables.args, shared_variables.collision_analysis_queue, shared_variables.collision_analysis_rt_queue, shared_variables.stay_time_queue,))
    p5 = Process(target=create_network_backend, )
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()

    # test_cnt = 0
    # while True:
        # test_cnt += 1
        # print('test_cnt = ', test_cnt)
        
        # # now_time = datetime.now()
        # # if (now_time + timedelta(seconds=30)).day == (datetime.now()).day + 1:
        # if test_cnt == 30:
        #     shared_variables.args['operation_flag'].clear()
        #     time.sleep(5)
        #     p1.terminate()
        #     p2.terminate()
        #     p3.terminate()
        #     p4.terminate()
        #     del p1
        #     del p2
        #     del p3
        #     del p4
        #     time.sleep(30)
        #     gc.collect()
        #     break

    while True:
        if shared_variables.exit_event.is_set():
            # print('sys exit!!')
            stop_nginx()
            time.sleep(3)
            p1.terminate()
            p2.terminate()
            p3.terminate()
            p4.terminate()
            # sys.exit()
            break
        else:
            time.sleep(2)
            # print('p1 size', sys.getsizeof(p1))
            # print('p2 size', sys.getsizeof(p2))
            # print('p3 size', sys.getsizeof(p3))
            # print('shared varialbes size = ' , sys.getsizeof(shared_variables.args))
    # time.sleep(10)
    
if __name__ == '__main__':
    main_run()

# 사무실 외부 ip 10.100.71.25
# 내부 ip 192.168.1.10 / 상위 192.168.100.26:8001

# 라우터1 
# 접속-. 192.168.100.1
# 외부주소 10.100.71.25
# 할당 범위 192.168.100.2 -  192.168.100.254
# 포트포워딩 시도 : 접속주소 ->  192.168.100.26 로 나옴

# 라우터2
# 외부주소 192.168.100.26
# 할당 범위 192.168.1.2 -  192.168.1.254

# 사무실 컴퓨터
# 192.168.1.10

# 새로연결한 와이파이 외부 ip 121.169.212.87