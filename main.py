from backend.main import gs_tracker
from backend.main import create_backend
from frontend.main import create_frontend
from shared_variables import SharedVariables
from multiprocessing import Process
import multiprocessing



if __name__ == '__main__':
    # gs_tracker_instance = gs_tracker()
    
    # print('check')
    
    video_loader_op_flag = multiprocessing.Event()
    
    
    # video_loader_op_flag.
    video_loader_op_flag.set()
    p2 = Process(target = create_backend, args= (video_loader_op_flag,), daemon=False)
    
    p1 = Process(target = create_frontend, args= (video_loader_op_flag,), daemon=False)
    
    # gs_tracker_instance.video_loader_op_flag.set()
    video_loader_op_flag.set()
    p2.start()
    p1.start()
    video_loader_op_flag.clear()
    
    
    p1.join()
    p2.join()
    
    
    
    