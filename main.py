from backend.main import create_backend
from frontend.main import create_frontend
from visualization.main import create_visualization
from shared_variables import SharedVariables
from multiprocessing import Process
from collision_analysis.main import create_collision_analysis
import time

# import sys

def main_run():
    
    shared_variables = SharedVariables()
    

    p1 = Process(target = create_frontend, args= (shared_variables.args, shared_variables.drawing_result_ques, shared_variables.exit_event), daemon=False)
    p2 = Process(target = create_backend, args= (shared_variables.args, shared_variables.model_proc_result_ques, shared_variables.exit_event), daemon=False)
    p3 = Process(target = create_visualization, args=(shared_variables.args, shared_variables.model_proc_result_ques, shared_variables.drawing_result_ques, shared_variables.exit_event,
                                                    shared_variables.collision_analysis_queue, shared_variables.collision_analysis_rt_queue))
    # p4 = Process(target = check_exit, args=(shared_variables.exit_event, p2, p3))
    p4 = Process(target= create_collision_analysis, args=(shared_variables.collision_analysis_queue, shared_variables.collision_analysis_rt_queue))
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()

    while True:
        if shared_variables.exit_event.is_set():
            # print('sys exit!!')
            time.sleep(2)
            p1.terminate()
            p2.terminate()
            p3.terminate()
            # sys.exit()
            break
        else:
            time.sleep(1)
            # print('p1 size', sys.getsizeof(p1))
            # print('p2 size', sys.getsizeof(p2))
            # print('p3 size', sys.getsizeof(p3))
            # print('shared varialbes size = ' , sys.getsizeof(shared_variables.args))


if __name__ == '__main__':
    main_run()