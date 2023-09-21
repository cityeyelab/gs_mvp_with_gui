import tkinter
import tkinter.messagebox
import customtkinter
import cv2
import numpy as np
import PIL
from PIL import Image, ImageTk
import time
import threading
from multiprocessing import Process
from .widgets.rtsp_cards import RtspFrame
from .widgets.zone_radio_button import ZoneRadioButton

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

def create_frontend(op_flags, drawing_result_ques):
    app_instance = App(op_flags, drawing_result_ques)
    app_instance.mainloop()
    # app_process.join()


class App(customtkinter.CTk):
    def __init__(self, shared_variables, drawing_result_ques):
        print('App created')
        super().__init__()
        self.drawing_result_ques = drawing_result_ques
        self.operation_flag = shared_variables['operation_flag']
        self.area_display_values = shared_variables['area_display_values']
        self.yolo_inference_ready_flag_lst = [shared_variables['is_yolo_inference1_ready'], shared_variables['is_yolo_inference2_ready'],
                                         shared_variables['is_yolo_inference3_ready']]
        self.rtsp_ready_lst = [shared_variables['is_rtsp1_ready'], shared_variables['is_rtsp2_ready'], shared_variables['is_rtsp3_ready']]
        
        self.title("GS demonstration MVP GUI")
        self.geometry(f"{1400}x{800}")

        self.grid_rowconfigure(5, weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, corner_radius=8, fg_color='white', border_width=4, border_color='grey')
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        # logo_image_ = cv2.imread('assets/main_logo.svg')
        # logo_image = Image.open('.assets/main_logo.svg')
        logo_image = Image.open('frontend/assets/logo.png')
        imgtk = customtkinter.CTkImage(logo_image, size= (220, 50))
        # img = Image.fromarray()
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=20)
        self.logo_label.configure(image=imgtk)
        
        # self.logo_text = customtkinter.CTkLabel(self.sidebar_frame, text="CityEyeLab", font=customtkinter.CTkFont(size=20, weight="bold"))
        # self.logo_text.grid(row=1, column=0, padx=20, pady=(20, 10))
        
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text='button1', command=None)
        self.sidebar_button_1.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text='button2', command=None)
        self.sidebar_button_2.grid(row=3, column=0, padx=20, pady=10)
        
        self.wait_op_button_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=4, fg_color='gray80', border_width=2, border_color='grey')
        self.wait_op_button_frame.grid(row=4, column=0, padx=8, pady=8, sticky="nsew")
        self.wait_op_button_lbl = customtkinter.CTkLabel(self.wait_op_button_frame, text="Wait until the model is ready...")
        self.wait_op_button_lbl.grid(row=0, column=0, padx=4, pady=4, )
        self.wait_op_button_bar = customtkinter.CTkProgressBar(self.wait_op_button_frame)
        self.wait_op_button_bar.grid(row=1, column=0, padx=4, pady=4, )

        self.wait_op_button_frame_task = self.wait_op_button_frame.after(200, self.load_op_button)
        self.op_button = customtkinter.CTkButton(self.sidebar_frame, text='run / pause', command=self.toggle_op_flag)
        # self.op_button.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")
        
        # self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text='button3', command=None)
        # self.sidebar_button_3.grid(row=4, column=0, padx=20, pady=10)
        
        
        self.radiobutton_frame = ZoneRadioButton(self, 1000, 160, area_display_values=self.area_display_values)
        # self.radiobutton_frame = customtkinter.CTkFrame(self, border_color='black', border_width=2)
        self.radiobutton_frame.grid(row=4, column=1, columnspan=3, padx=12, pady=12, sticky="nw")
        self.radio_button_callback = self.radiobutton_frame.set_selected_cam_num
        
        
        # rtsp frame
        self.wait_rtsp_frame = customtkinter.CTkFrame(self, width=400*(16/9), height=400)
        self.wait_rtsp_frame.grid(row=0, column=1, rowspan=3, columnspan=3, padx=10, pady=10, sticky="nsew")
        self.wait_rtsp_frame_lbl = customtkinter.CTkLabel(self.wait_rtsp_frame, text="Wait until rtsp protocol is ready...", width=385*(16/9), height=385)
        # self.wait_rtsp_frame_lbl = customtkinter.CTkLabel(self.wait_rtsp_frame, text="Wait until rtsp protocol is ready...",)
        self.wait_rtsp_frame_lbl.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.wait_rtsp_frame_bar = customtkinter.CTkProgressBar(self.wait_rtsp_frame)
        self.wait_rtsp_frame_bar.grid(row=1, column=0, padx=10, pady=10,)
        self.wait_rtsp_frame_task = self.wait_rtsp_frame.after(200, self.load_rtsp_frame)
        self.rtsp_frame = RtspFrame(self, self.drawing_result_ques, self.radio_button_callback)
        # self.rtsp_frame.grid(row=0, column=1, rowspan=3, columnspan=3, padx=10, pady=10)
        
        # sized box
        self.empty_block = customtkinter.CTkFrame(self, corner_radius=0, fg_color='transparent')
        self.empty_block.grid(row=5, column=0, rowspan=3, sticky="nsew")
        
        #bottom ui setting bar
        self.bottom_ui_setting_bar = customtkinter.CTkFrame(self, corner_radius=8, fg_color='light gray')
        self.bottom_ui_setting_bar.grid(row=6, column=0, rowspan=3, sticky="nsew")
        self.appearance_mode_label = customtkinter.CTkLabel(self.bottom_ui_setting_bar, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=0, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.bottom_ui_setting_bar, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=1, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.bottom_ui_setting_bar, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=2, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.bottom_ui_setting_bar, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=3, column=0, padx=20, pady=(10, 20))
        
        
    
        # self.test_frame = customtkinter.CTkFrame(self, corner_radius=0, bg_color='black', fg_color='yellow')
        # self.test_frame.grid(row=1, column=1, padx=10, pady=10)
        
        # self.test_frame2 = customtkinter.CTkFrame(self, corner_radius=0, bg_color='black', fg_color='yellow')
        # self.test_frame2.grid(row=2, column=1, padx=10, pady=10)
        
        # self.test_frame3 = customtkinter.CTkFrame(self, corner_radius=0, bg_color='black', fg_color='yellow')
        # self.test_frame3.grid(row=3, column=1, padx=10, pady=10)
        
        # self.rtsp_card_1.make_selected()
        # self.rtsp_main_card.selected_cam_num = 1
        
        
        self.scaling_optionemenu.set("100%")
        self.wait_op_button_bar.configure(mode="indeterminnate")
        self.wait_op_button_bar.start()
        self.wait_rtsp_frame_bar.configure(mode="indeterminnate")
        self.wait_rtsp_frame_bar.start()
        
        print('frontend init end')
        

    def load_op_button(self):
        flag1 = self.yolo_inference_ready_flag_lst[0].is_set()
        flag2 = self.yolo_inference_ready_flag_lst[1].is_set()
        flag3 = self.yolo_inference_ready_flag_lst[2].is_set()
        
        if flag1 and flag2 and flag3:
            self.wait_op_button_frame.grid_forget()
            self.op_button.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")
            self.wait_op_button_bar.after_cancel(self.wait_op_button_frame_task)
            self.wait_op_button_frame.destroy()
        else:
            self.wait_op_button_frame_task = self.wait_op_button_frame.after(200, self.load_op_button)
        
        
    def load_rtsp_frame(self):
        flag1 = self.rtsp_ready_lst[0].is_set()
        flag2 = self.rtsp_ready_lst[1].is_set()
        flag3 = self.rtsp_ready_lst[2].is_set()
        
        if flag1 and flag2 and flag3:
            self.wait_rtsp_frame.grid_forget()
            self.rtsp_frame.grid(row=0, column=1, rowspan=3, columnspan=3, padx=10, pady=10)
            self.wait_rtsp_frame.after_cancel(self.wait_rtsp_frame_task)
            self.wait_rtsp_frame.destroy()
        else:
            self.wait_rtsp_frame_task = self.wait_rtsp_frame.after(200, self.load_rtsp_frame)
    
    # def radio_button_command(self):
    #     radio_var1 = self.radio_var_area1.get()
    #     print('radio var 1 = ' , radio_var1)
    #     self.area_display_values[0].set(radio_var1)
    #     print('self.area_display_values[0] = ', self.area_display_values[0])
    #     # area1_zone_display_value.set()
    #     # self.area_display_values = radio_var1
        
    
    def toggle_op_flag(self):
        if self.operation_flag.is_set():
            self.operation_flag.clear()
            # cv2.destroyAllWindows()
        else:
            self.operation_flag.set()
        
    def whole_clicked(self, event):
        print('self clicked')
    
    # def select_rtsp_card(self, event, card_num):
    #     # print('select event = ', args)
    #     print('frame clicked! card num = ', card_num)
    #     self.rtsp_card_1.make_unselected()
    #     self.rtsp_card_2.make_unselected()
    #     self.rtsp_card_3.make_unselected()
        
    #     # print('card num = ' , card_num)
    #     if card_num == 1:
    #         self.rtsp_card_1.make_selected()
    #         self.rtsp_main_card.selected_cam_num = 1
    #         self.radiobutton_frame.set_selected_cam_num(1)
    #     elif card_num == 2:
    #         self.rtsp_card_2.make_selected()
    #         self.rtsp_main_card.selected_cam_num = 2
    #         self.radiobutton_frame.set_selected_cam_num(2)
    #     elif card_num == 3:
    #         self.rtsp_card_3.make_selected()
    #         self.rtsp_main_card.selected_cam_num = 3
    #         self.radiobutton_frame.set_selected_cam_num(3)
    
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)
    
    
    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    
    def run(self):
        self.mainloop()
        
    # def rtsp_frame(self, path):
    #     frame = customtkinter.CTkFrame(self, width=180, height=180)
    #     lbl = customtkinter.CTkLabel(frame, text='', width=150, height=150)
    #     lbl.grid(row=1, column=0, columnspan=4, padx=(10, 10), pady=(10, 10), sticky="nsew")
    #     return frame


    
        




if __name__ == "__main__":
    pass
    # app = App()
    # app.mainloop()