import tkinter
import tkinter.messagebox
import customtkinter
from customtkinter import DISABLED, NORMAL
import cv2
import numpy as np
import PIL
from PIL import Image, ImageTk
import time
import threading
from multiprocessing import Process
import os

from .widgets.rtsp_cards import RtspFrame
from .widgets.zone_radio_button import ZoneRadioButton

import sys

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

def create_frontend(args, drawing_result_ques, exit_event):
    app_instance = App(args, drawing_result_ques)
    app_instance.mainloop()
    exit_event.set()
    print('exit frontend')
    del app_instance
    sys.exit()



class App(customtkinter.CTk):
    def __init__(self, args, drawing_result_ques):
        print('App created')
        super().__init__()
        self.drawing_result_ques = drawing_result_ques
        self.operation_flag = args['operation_flag']
        self.area_display_values = args['area_display_values']
        self.selected_cam_num = args['selected_cam_num']
        self.yolo_inference_ready_flag_lst = [args['is_yolo_inference1_ready'], args['is_yolo_inference2_ready'],
                                         args['is_yolo_inference3_ready']]
        self.rtsp_ready_lst = [args['is_rtsp1_ready'], args['is_rtsp2_ready'], args['is_rtsp3_ready']]
        self.collision_op_flag, self.collision_rt_op_flag, self.stay_time_op_flag = args['collision_op_flag'], args['collision_rt_op_flag'], args['stay_time_op_flag']
        self.collision_ready_flag, self.collision_rt_ready_flag, self.stay_time_ready_flag = args['collision_ready_flag'], args['collision_rt_ready_flag'], args['stay_time_ready_flag']
        
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

        # self.wait_op_button_frame_task = self.wait_op_button_frame.after(200, self.load_op_button)
        self.load_op_button()
        self.op_button = customtkinter.CTkButton(self.sidebar_frame, text='run / pause', command=self.toggle_op_flag)
        # self.op_button.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")
        
        # self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text='button3', command=None)
        # self.sidebar_button_3.grid(row=4, column=0, padx=20, pady=10)
        
        # self.status_text = '-'
        self.status_label = customtkinter.CTkLabel(self.sidebar_frame, text='Status: -')
        self.status_label.grid(row=5, column=0, padx=20, pady=20)
        
        ####
        # self.wait_col_cb_lbl = customtkinter.CTkLabel(self.sidebar_frame, text="x")
        # self.wait_col_cb_lbl.grid(row=6, column=0, pady=10, padx=10, sticky="nsew")
        # # self.wait_col_cb_task = self.wait_col_cb_lbl.after(200, self.load_col_cb)
        # self.load_col_cb()
        # # self.checkbox1.grid(row=6, column=0, pady=10, padx=10, sticky="nsew")
        # self.collision_cb = customtkinter.CTkCheckBox(master=self.sidebar_frame, text='Collision anlysis', command = self.toggle_collision_op)
        # # self.collision_checkbox.grid(row=6, column=0, pady=10, padx=10, sticky="nsew")
        
        
        
        # self.collision_rt_cb = customtkinter.CTkCheckBox(master=self.sidebar_frame, text='Collision anlysis RT', command = None)
        # self.collision_rt_cb.grid(row=7, column=0, pady=10, padx=10, sticky="nsew")
        
        # self.wait_st_cb_lbl = customtkinter.CTkLabel(self.sidebar_frame, text="x")
        # self.wait_st_cb_lbl.grid(row=8, column=0, pady=10, padx=10, sticky="nsew")
        # self.load_st_cb()
        # self.stay_time_cb = customtkinter.CTkCheckBox(master=self.sidebar_frame, text='Stay time analysis', command = self.toggle_stay_time_op)
        # # self.stay_time_cb.grid(row=7, column=0, pady=10, padx=10, sticky="nsew")
        
        
        self.radio_var = tkinter.IntVar()
        self.rb_none = customtkinter.CTkRadioButton(master = self.sidebar_frame, text='None', variable=self.radio_var, value=0,
                                                          command=self.radio_button_command, width=180, height=30)
        self.rb_none.grid(row=6, column=0, pady=8, padx=8, sticky="nsew")
        
        
        self.rb_col = customtkinter.CTkRadioButton(master = self.sidebar_frame, text='Collision Analysis: Loading..', variable=self.radio_var, value=1,
                                                          command=self.radio_button_command, width=180, height=30)
        self.rb_col.grid(row=7, column=0, pady=8, padx=8, sticky="nsew")
        self.rb_col.after(200, self.rb_col_load)
        self.rb_col.configure(state=DISABLED)
        # self.radio_button2.grid(row=1, column=1, pady=4, padx=4, sticky="n")
        
        self.rb_col_rt = customtkinter.CTkRadioButton(master = self.sidebar_frame, text='Collision RT Analysis', variable=self.radio_var, value=2,
                                                          command=self.radio_button_command, width=180, height=30)
        self.rb_col_rt.grid(row=8, column=0, pady=8, padx=8, sticky="nsew")
        self.rb_col_rt.after(200, self.rb_col_rt_load)
        self.rb_col_rt.configure(state=DISABLED)
        self.rb_st = customtkinter.CTkRadioButton(master = self.sidebar_frame, text='Stay Time Analysis: Loading..', variable=self.radio_var, value=3,
                                                          command=self.radio_button_command, width=180, height=30)
        self.rb_st.grid(row=9, column=0, pady=8, padx=8, sticky="nsew")
        self.rb_st.after(200, self.rb_st_load)
        self.rb_st.configure(state=DISABLED)
        
        # def radio_button_command(self):
        #     # pass
        #     radio_var = self.radio_var.get()
            
        #     self.area_display_values[0].set(0)
        #     self.area_display_values[1].set(0)
        #     self.area_display_values[2].set(0)
            
        #     if self.selected_cam_num == 1:
        #         self.area_display_values[0].set(radio_var)
        #     elif self.selected_cam_num == 2:
        #         self.area_display_values[1].set(radio_var)
        #     elif self.selected_cam_num == 3:
        #         self.area_display_values[2].set(radio_var)
        
        
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
        # self.wait_rtsp_frame_task = self.wait_rtsp_frame.after(200, self.load_rtsp_frame)
        self.load_rtsp_frame()
        self.rtsp_frame = RtspFrame(self, self.drawing_result_ques, self.radio_button_callback, self.selected_cam_num)
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
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.bottom_ui_setting_bar, values=["60%", "80%", "100%", "120%"],
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
        
        # default init
        self.scaling_optionemenu.set("100%")
        self.wait_op_button_bar.configure(mode="indeterminnate")
        self.wait_op_button_bar.start()
        self.wait_rtsp_frame_bar.configure(mode="indeterminnate")
        self.wait_rtsp_frame_bar.start()
        self.radio_var.set(0)
        # self.checkbox1.configure(state=DISABLED)
        # self.checkbox2.configure(state=DISABLED)
        # self.checkbox3.configure(state=DISABLED)
        
        print('frontend init end')
    
    def radio_button_command(self):
        var = self.radio_var.get()
        self.collision_op_flag.clear()
        self.collision_rt_op_flag.clear()
        self.stay_time_op_flag.clear()
        if var == 0:
            pass
        elif var == 1:
            self.collision_op_flag.set()
        elif var == 2:
            self.collision_rt_op_flag.set()
        elif var == 3:
            self.stay_time_op_flag.set()
            
    
    def rb_col_load(self):
        flag = self.collision_ready_flag.is_set()
        # print('rb col flag = ' , flag)
        if flag:
            self.rb_col.configure(state=NORMAL, text='Collision Analysis')
            # self.wait_col_cb_lbl.after_cancel(self.wait_col_cb_task)
        else:
            self.wait_rb_col_task = self.rb_col.after(500, self.rb_col_load)

    def rb_col_rt_load(self):
        flag = self.collision_rt_ready_flag.is_set()
        if flag:
            self.rb_col_rt.configure(state=NORMAL, text='Collision RT Analysis')
        else:
            self.wait_rb_col_rt_task = self.rb_col_rt.after(500, self.rb_col_rt_load)
    
    def rb_st_load(self):
        flag = self.stay_time_ready_flag.is_set()
        if flag:
            self.rb_st.configure(state=NORMAL, text='Stay Time Analysis')
        else:
            self.wait_rb_st_task = self.rb_st.after(500, self.rb_st_load)
        
    # def toggle_collision_op(self):
    #     if self.collision_op_flag.is_set():
    #         self.collision_op_flag.clear()
    #     else:
    #         self.collision_op_flag.set()
    
    # def toggle_stay_time_op(self):
    #     if self.stay_time_op_flag.is_set():
    #         self.stay_time_op_flag.clear()
    #     else:
    #         self.stay_time_op_flag.set()
    
    
    # def load_col_cb(self):
    #     flag = self.collision_ready_flag.is_set()
    #     if flag:
    #         self.wait_col_cb_lbl.grid_forget()
    #         self.collision_cb.grid(row=6, column=0, pady=10, padx=10, sticky="nsew")
    #         self.wait_col_cb_lbl.after_cancel(self.wait_col_cb_task)
    #         self.wait_col_cb_lbl.destroy()
    #     else:
    #         self.wait_col_cb_task = self.wait_col_cb_lbl.after(200, self.load_col_cb)
    
    # def load_st_cb(self):
    #     flag = self.stay_time_ready_flag.is_set()
    #     if flag:
    #         self.wait_st_cb_lbl.grid_forget()
    #         self.stay_time_cb.grid(row=8, column=0, pady=10, padx=10, sticky="nsew")
    #         self.wait_st_cb_lbl.after_cancel(self.wait_st_cb_task)
    #         self.wait_st_cb_lbl.destroy()
    #     else:
    #         self.wait_st_cb_task = self.wait_st_cb_lbl.after(200, self.load_st_cb)
    
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
            # self.status_text = '-'
            self.status_label.configure(text='Status: -')
            # cv2.destroyAllWindows()
            # self.checkbox1.configure(state=DISABLED)
            # self.checkbox2.configure(state=DISABLED)
            # self.checkbox3.configure(state=DISABLED)
            # self.checkbox1.deselect()
            # self.checkbox2.deselect()
            # self.checkbox3.deselect()
        else:
            self.operation_flag.set()
            self.status_label.configure(text='Status: Running..')
            # self.checkbox1.configure(state=NORMAL)
            # self.checkbox2.configure(state=NORMAL)
            # self.checkbox3.configure(state=NORMAL)
            
        
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


        
        
    # def rtsp_frame(self, path):
    #     frame = customtkinter.CTkFrame(self, width=180, height=180)
    #     lbl = customtkinter.CTkLabel(frame, text='', width=150, height=150)
    #     lbl.grid(row=1, column=0, columnspan=4, padx=(10, 10), pady=(10, 10), sticky="nsew")
    #     return frame


    
        




if __name__ == "__main__":
    pass
    # app = App()
    # app.mainloop()