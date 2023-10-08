from backend.main import gs_tracker
from backend.main import create_backend
from frontend.main import create_frontend
from visualization.main import create_visualization
from shared_variables import SharedVariables
from multiprocessing import Process
import time
# import multiprocessing
# import os
# from functools import partial

# import sys

# from cProfile import Profile
# from pstats import Stats
# import pstats
# import cProfile
# import re
# import datetime

        
# def exit_app(p2, p3):
#     p2.terminate()
#     p3.terminate()
    
# def check_exit(exit_event, exit_callback):
#     # print('check exit start')
#     while True:
#         if exit_event.is_set():
#             exit_callback()
#             break
#         else:
#             time.sleep(1)

# def check_exit(exit_event, p2, p3):
#     print('check exit start')
#     while True:
#         if exit_event.is_set():
#             exit_app(p2, p3)
#             break
#         else:
#             time.sleep(1)

# partial(exit_whole_app, (p1=p1, p2=))

# def check_exit(exit_event):
#     # print('check exit start')
#     while True:
#         if exit_event.is_set():
#             print('sys exit!!')
#             sys.exit()
#             break
#         else:
#             print('just exit check')
#             time.sleep(1)

def main_run():
    
    shared_variables = SharedVariables()
    

    
    p1 = Process(target = create_frontend, args= (shared_variables.args, shared_variables.drawing_result_ques, shared_variables.exit_event), daemon=False)
    p2 = Process(target = create_backend, args= (shared_variables.args, shared_variables.model_proc_result_ques, shared_variables.exit_event), daemon=False)
    p3 = Process(target = create_visualization, args=(shared_variables.args, shared_variables.model_proc_result_ques, shared_variables.drawing_result_ques, shared_variables.exit_event))
    # p4 = Process(target = check_exit, args=(shared_variables.exit_event, p2, p3))
    
    
    p1.start()
    p2.start()
    p3.start()

    while True:
        if shared_variables.exit_event.is_set():
            # print('sys exit!!')
            p1.terminate()
            p2.terminate()
            p3.terminate()
            # sys.exit()
            break
        else:
            time.sleep(1)


if __name__ == '__main__':
    main_run()







#######################################################################
# if __name__ == '__main__':
    
#     # profiler = Profile()
#     # profiler.runcall(main_run)
    
#     cProfile.run('main_run()', 'profile_results', sort='tottime')
    
#     now_time = datetime.datetime.now()
#     now_string = now_time.strftime('%Y-%m-%d_%H-%M-%S')
#     with open(f'profile_results_{now_string}.txt', 'w') as file:
#         profile = pstats.Stats('profile_results', stream=file)
#         profile.print_stats()
#         file.close()
    
    
    
    
#     # shared_variables = SharedVariables()
    

    
#     # p1 = Process(target = create_frontend, args= (shared_variables.args, shared_variables.drawing_result_ques, shared_variables.exit_event), daemon=False)
#     # p2 = Process(target = create_backend, args= (shared_variables.args, shared_variables.model_proc_result_ques, shared_variables.exit_event), daemon=False)
#     # p3 = Process(target = create_visualization, args=(shared_variables.args, shared_variables.model_proc_result_ques, shared_variables.drawing_result_ques, shared_variables.exit_event))
#     # # p4 = Process(target = check_exit, args=(shared_variables.exit_event,), daemon=False)
    
    
    
#     # p1.start()
#     # p2.start()
#     # p3.start()
    
    
#     # # exit_app_callback = partial(exit_app, p1, p2)
#     # # p4 = Process(target = check_exit, args=(shared_variables.exit_event, exit_app_callback))
#     # # p4.start()

#     # # while True:
#     # #     print('check')
#     # #     time.sleep(1)
#     # #     if shared_variables.exit_event.is_set():
#     # #         print('sys exit!')
#     # #         sys.exit()

  
    
#     # # p1.join()
#     # # p2.join()
#     # # p3.join()
#     # # p4.join()
#     # while True:
#     #     if shared_variables.exit_event.is_set():
#     #         print('sys exit!!')
#     #         sys.exit()
#     #         break
#     #     else:
#     #         time.sleep(1)
    
    
#     # # profiler = Profile()
#     # # profiler.runcall(main_run)

#     # # stats = Stats(profiler)
#     # # stats.strip_dirs()
#     # # stats.sort_stats('cumulative')
#     # # stats.print_stats()
        
#     # # p4.join()
#     # # print('all proc end')



# #ends
# #yolo tracker3 tracker4 tracker1 yolox3 controlcenter frontend visualization_bp visualizationX3 loader