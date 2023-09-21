from backend.main import gs_tracker
from backend.main import create_backend
from frontend.main import create_frontend
from visualization.main import create_visualization
from shared_variables import SharedVariables
from multiprocessing import Process
# import multiprocessing



if __name__ == '__main__':
    
    shared_variables = SharedVariables()
    
    # drawing_result_ques = [Queue(200), Queue(200), Queue(200), Queue(200), Queue(200), Queue(200), Queue(200)] # visualize, visualize, visualize, eachview<->mainview x 3, visualize_bp, 
    # drawing_result_ques = [Queue(200), Queue(200), Queue(200), Queue(200)] # visualize, visualize, visualize, visualize_bp, 
    # shared_variables.model_proc_result_ques -> model to drawing
    # 
    p1 = Process(target = create_frontend, args= (shared_variables.args, shared_variables.drawing_result_ques), daemon=False)
    p2 = Process(target = create_backend, args= (shared_variables.args, shared_variables.model_proc_result_ques), daemon=False)
    p3 = Process(target = create_visualization, args=(shared_variables.args, shared_variables.model_proc_result_ques, shared_variables.drawing_result_ques))
    
    p1.start()
    p2.start()
    p3.start()
    
    p1.join()
    p2.join()
    p3.join()