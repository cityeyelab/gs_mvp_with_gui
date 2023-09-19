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
from .widgets.rtsp_cards import RtspCard, RtspMainView, BpView

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

def create_frontend(video_op_flag, drawing_result_ques):
    app_instance = App(video_op_flag, drawing_result_ques)
    app_instance.mainloop()
    # app_process.join()


class App(customtkinter.CTk):
    def __init__(self, shared_variables, drawing_result_ques):
        print('App created')
        super().__init__()
        self.drawing_result_ques = drawing_result_ques
        self.operation_flag = shared_variables['operation_flag']

        self.title("GS demonstration MVP GUI")
        self.geometry(f"{1400}x{800}")

        self.grid_rowconfigure(4, weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, corner_radius=8, fg_color='white', border_width=4, border_color='grey')
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
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
        
        self.op_button = customtkinter.CTkButton(self.sidebar_frame, text='run/pause', command=self.toggle_video_op_flag)
        self.op_button.grid(row=5, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        # self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text='button3', command=None)
        # self.sidebar_button_3.grid(row=4, column=0, padx=20, pady=10)
        
        
        
        # rtsp frame
        self.rtsp_frame = customtkinter.CTkFrame(self, height=100, corner_radius=0)
        self.rtsp_frame.grid(row=0, column=1, rowspan=3, columnspan=3, padx=10, pady=10)
        
        self.rtsp_card_1 = RtspCard(self.rtsp_frame, 'rtsp://admin:self1004@@118.37.223.147:8522/live/main7', 85, self.drawing_result_ques[0])
        self.rtsp_card_1.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        
        self.rtsp_card_2 = RtspCard(self.rtsp_frame, 'rtsp://admin:self1004@@118.37.223.147:8522/live/main6', 85, self.drawing_result_ques[1])
        self.rtsp_card_2.grid(row=0, column=1, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        
        self.rtsp_card_3 = RtspCard(self.rtsp_frame, 'rtsp://admin:self1004@@118.37.223.147:8522/live/main8', 85, self.drawing_result_ques[2])
        self.rtsp_card_3.grid(row=0, column=2, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        # self.rtsp_card_1.tkraise()
        # print('bind tags = ' , self.bindtags())
        self.rtsp_card_1.lbl.bind("<Button-1>", lambda e: self.select_rtsp_card(event=e, card_num=1))
        self.rtsp_card_2.lbl.bind("<Button-1>", lambda e: self.select_rtsp_card(event=e, card_num=2))
        self.rtsp_card_3.lbl.bind("<Button-1>", lambda e: self.select_rtsp_card(event=e, card_num=3))
        
        self.rtsp_main_card = RtspMainView(self.rtsp_frame, 300, self.drawing_result_ques[0:3])
        self.rtsp_main_card.grid(row=1, column=0, rowspan=2, columnspan=3, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        self.bp_view = BpView(self.rtsp_frame, 300, self.drawing_result_ques[3])
        self.bp_view.grid(row=0, column=3, rowspan=2, columnspan=2, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        
        
        
        
        # self.frame_click_test = customtkinter.CTkFrame(self.rtsp_frame, corner_radius=0, fg_color='transparent')
        # self.frame_click_test.grid(row=3, column=0, rowspan=2, columnspan=3, padx=(0, 0), pady=(0, 0), sticky="nsew")
        # self.click_test_lbl = customtkinter.CTkLabel(self.frame_click_test, text='test')
        # self.click_test_lbl.pack()
        # self.frame_click_test._canvas.bind("<Button-1>", self.select_rtsp_card)
        # self.frame_click_test.canva
        
        
        
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
        
        self.rtsp_card_1.make_selected()
        self.rtsp_main_card.selected_cam_num = 1
        self.scaling_optionemenu.set("100%")
        
        print('frontend init end')
        
    def toggle_video_op_flag(self):
        if self.operation_flag.is_set():
            self.operation_flag.clear()
            cv2.destroyAllWindows()
        else:
            self.operation_flag.set()
        
    def whole_clicked(self, event):
        print('self clicked')
    
    def select_rtsp_card(self, event, card_num):
        # print('select event = ', args)
        print('frame clicked! card num = ', card_num)
        self.rtsp_card_1.make_unselected()
        self.rtsp_card_2.make_unselected()
        self.rtsp_card_3.make_unselected()
        
        if card_num == 1:
            self.rtsp_card_1.make_selected()
            self.rtsp_main_card.selected_cam_num = 1
        elif card_num == 2:
            self.rtsp_card_2.make_selected()
            self.rtsp_main_card.selected_cam_num = 2
        elif card_num == 3:
            self.rtsp_card_3.make_selected()
            self.rtsp_main_card.selected_cam_num = 3
    
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