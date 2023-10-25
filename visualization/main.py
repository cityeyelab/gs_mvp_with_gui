from .visualization import visualize
from .visualization_bp import visualize_bp
from multiprocessing import Process

def create_visualization(args, model_proc_result_ques, drawing_result_ques, exit_event, collision_que, collision_rt_que, st_que):
    main_visualizer = VisualizationMain(args, model_proc_result_ques,  drawing_result_ques, exit_event, collision_que, collision_rt_que, st_que)
    main_visualizer.run()
    print('visualizer ends')
    
    
    
class VisualizationMain():
    def __init__(self, args, model_proc_result_ques,  drawing_result_ques, exit_event, collision_que, collision_rt_que, st_que) -> None:
        self.draw_proc_lst = []
        self.args = args
        self.model_proc_result_ques = model_proc_result_ques
        self.drawing_result_ques = drawing_result_ques
        self.exit_event = exit_event
        # shared_variables.args, shared_variables.model_proc_result_ques, shared_variables.drawing_result_ques
        
        self.operation_flag = self.args['operation_flag']
        self.area_display_values = args['area_display_values']
        self.selected_cam_num = args['selected_cam_num']
        self.collision_op_flag = args['collision_op_flag']
        self.stay_time_op_flag = args['stay_time_op_flag']
        # self.collision_ready_flag = args['collision_ready_flag']
        # self.stay_time_ready_flag = args['stay_time_ready_flag']
        
        self.image_que_lst_draw = self.model_proc_result_ques[0:3]
        self.draw_proc_result_que_lst = self.model_proc_result_ques[3:6]
        self.visualize_bp_que_lst = self.model_proc_result_ques[6:9]
        
        self.exit_event = exit_event
        
        self.collision_que = collision_que
        self.collision_rt_que = collision_rt_que
        self.st_que = st_que
        
        # self.visualization_eco_mode = False
        
        
        for i in range(0, 3):
            self.draw_proc_lst.append(Process(target=visualize, args=(self.operation_flag, self.area_display_values[i], self.selected_cam_num,
                                                                self.image_que_lst_draw[i], self.draw_proc_result_que_lst[i], i,
                                                                self.drawing_result_ques[0:3], self.exit_event),
                                              daemon=False))
        self.visualize_bp_proc = Process(target=visualize_bp, args=(self.operation_flag, self.visualize_bp_que_lst[0], self.visualize_bp_que_lst[1],
                                                                    self.visualize_bp_que_lst[2], self.drawing_result_ques[3], self.exit_event, collision_que, self.collision_op_flag, self.stay_time_op_flag, self.st_que), daemon=False)
    
    def run(self):
        for i in range(0, 3):
            self.draw_proc_lst[i].start()
        self.visualize_bp_proc.start()
        
        for i in range(0, 3):
            self.draw_proc_lst[i].join()
        self.visualize_bp_proc.join()