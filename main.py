from backend.main import gs_tracker
from backend.main import create_backend
from frontend.main import create_frontend
from shared_variables import SharedVariables
from multiprocessing import Process
import multiprocessing



if __name__ == '__main__':
    # gs_tracker_instance = gs_tracker()
    
    # print('check')
    
    # operation_flag = multiprocessing.Event()
    
    shared_variables = SharedVariables()
    
    
    # video_loader_op_flag.
    # operation_flag.set()
    p2 = Process(target = create_backend, args= (shared_variables.args,), daemon=False)
    
    p1 = Process(target = create_frontend, args= (shared_variables.args,), daemon=False)
    
    # gs_tracker_instance.video_loader_op_flag.set()
    # operation_flag.set()
    p2.start()
    p1.start()
    # operation_flag.clear()
    
    
    p1.join()
    p2.join()
    
    
    
    