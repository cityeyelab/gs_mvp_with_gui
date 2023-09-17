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

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


# class RtspCard(customtkinter.CTkFrame):
class RtspCard(customtkinter.CTkFrame):
    def __init__(self, parent, path, height) -> None:
        super().__init__(parent)
        self.is_selected = False
        self.card_height = height
        self.card_width = int(self.card_height * (16/9))
        self.lbl = customtkinter.CTkLabel(self, text='', width=self.card_width, height=self.card_height, fg_color='light grey', )
        # lbl.grid(row=0, column=0, columnspan=4, padx=(10, 10), pady=(10, 10), sticky="nsew")
        self.lbl.pack(anchor='center', fill='both', expand=True)
        # self.lbl.bind("<Button-1>", self.make_selected)
        
        self.lbl.bind("<Enter>", self.on_hover)
        self.lbl.bind("<Leave>", self.off_hover)
        
        # self.lbl.grid(row=0, column=0, padx=0, pady=0)
        # self.video_button = customtkinter.CTkButton(self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE")
        #                                             ,text='video button', command=self.run_video)
        # self.video_button.pack()
        self.path = path
        self.cap_loader = cv2.VideoCapture(path)
        _, self.current_frame = self.cap_loader.read()
        
        self.run_video_thread = threading.Thread(target=self.run_video, args=(), daemon=False)
        self.run_video_thread.start()
        
        # self.update_frame_thread = threading.Thread(target=self.update_frame, args = () , daemon=False)
        # self.update_frame_thread.start()
        self.update_frame()
        # print('init end')
        
    
    def on_hover(self, event):
        print('event = ', event)
        if not self.is_selected:
            self.lbl.configure(fg_color = 'light sky blue')
    
    def off_hover(self, event):
        print('event = ', event)
        if not self.is_selected:
            self.lbl.configure(fg_color = 'light grey')
        
    def make_selected(self):
        self.is_selected = True
        self.lbl.configure(fg_color = 'blue')
            
    def make_unselected(self):
        self.is_selected = False
        self.lbl.configure(fg_color = 'light grey')
    
    def run_video(self):
        # print('run video check2')
        while True:
            ret, self.current_frame = self.cap_loader.read()
            _ , _ = self.cap_loader.read()
            if not ret:
                self.cap_loader.release()
            # print('frame loaded')
    
    def update_frame(self):
        # print('update frame')
        frame = self.current_frame
        if not self.is_selected:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # h, s, v = cv2.split(hsv)
            
            # lim = 30
            # v[v < lim] = 0 
            # v[v >= lim] -= lim

            # final_hsv = cv2.merge((h, s, v))
            # frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            # frame = frame - 10
            # frame = np.clip(frame, 0, 255)
            
        img = Image.fromarray(frame)
        imgtk = customtkinter.CTkImage(img, size= (int(0.95 * self.card_width), int(0.95 * self.card_height)))
        self.lbl.imgtk = ImageTk
        self.lbl.configure(image=imgtk)
        # self.lbl.update()
        self.after(256, self.update_frame)




class RtspMainView(customtkinter.CTkFrame):
    def __init__(self, parent, height) -> None:
        super().__init__(parent)

        self.card_height = height
        self.card_width = int(self.card_height * (16/9))
        self.lbl = customtkinter.CTkLabel(self, text='', width=self.card_width, height=self.card_height, fg_color='light grey', )
        # lbl.grid(row=0, column=0, columnspan=4, padx=(10, 10), pady=(10, 10), sticky="nsew")
        self.lbl.pack(anchor='center', fill='both', expand=True)
        self.paths = ['rtsp://admin:self1004@@118.37.223.147:8522/live/main7', 'rtsp://admin:self1004@@118.37.223.147:8522/live/main6',
                 'rtsp://admin:self1004@@118.37.223.147:8522/live/main8']
        
        self.cap_loader1 = cv2.VideoCapture(self.paths[0])
        _, self.current_frame = self.cap_loader1.read()
        self.cap_loader2 = cv2.VideoCapture(self.paths[1])
        self.cap_loader3 = cv2.VideoCapture(self.paths[2])
        
        self.selected_cam_num = 1
        
        self.run_video_thread = threading.Thread(target=self.run_video, args=(), daemon=False)
        self.run_video_thread.start()
        
        self.update_frame()

    def run_video(self):
        # print('run video check2')
        while True:
            
            if self.selected_cam_num == 1:
                _, _ = self.cap_loader1.read()
                ret, self.current_frame = self.cap_loader1.read()
                
                _, _ = self.cap_loader2.read()
                _, _ = self.cap_loader2.read()
                _, _ = self.cap_loader3.read()
                _, _ = self.cap_loader3.read()
                if not ret:
                    self.cap_loader1.release()
            elif self.selected_cam_num == 2:
                _, _ = self.cap_loader2.read()
                ret, self.current_frame = self.cap_loader2.read()
                
                _, _ = self.cap_loader1.read()
                _, _ = self.cap_loader1.read()
                _, _ = self.cap_loader3.read()
                _, _ = self.cap_loader3.read()
                if not ret:
                    self.cap_loader2.release()
            elif self.selected_cam_num == 3:
                _, _ = self.cap_loader3.read()
                ret, self.current_frame = self.cap_loader3.read()
                
                _, _ = self.cap_loader2.read()
                _, _ = self.cap_loader2.read()
                _, _ = self.cap_loader1.read()
                _, _ = self.cap_loader1.read()
                if not ret:
                    self.cap_loader3.release()

    
    def update_frame(self):
        # print('update frame')
        frame = self.current_frame
        img = Image.fromarray(frame)
        imgtk = customtkinter.CTkImage(img, size= (int(0.95 * self.card_width), int(0.95 * self.card_height)))
        self.lbl.imgtk = ImageTk
        self.lbl.configure(image=imgtk)
        # self.lbl.update()
        self.after(64, self.update_frame)

def create_frontend(video_op_flag):
    app_instance = App(video_op_flag)
    app_instance.mainloop()
    # app_process.join()

# class CreateApp():
#     def __init__(self) -> None:
#         pass
        
#     def run(self):
#         self.app_instance = App()
#         self.app_process = Process(target=self.app_instance.mainloop(), daemon=False)
#         self.app_process.start()

class App(customtkinter.CTk):
    def __init__(self, shared_variables):
        print('App created')
        # Process.__init__(self)
        super().__init__()
        
        self.operation_flag = shared_variables['operation_flag']
        
    
        
        self.title("GS demonstration MVP GUI")
        self.geometry(f"{1200}x{800}")

        self.grid_rowconfigure(4, weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="CityEyeLab", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text='button1', command=None)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text='button1', command=None)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text='button1', command=None)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        
        
        
        # rtsp frame
        self.rtsp_frame = customtkinter.CTkFrame(self, height=100, corner_radius=0)
        self.rtsp_frame.grid(row=0, column=1, rowspan=3, columnspan=3, padx=10, pady=10)
        
        self.rtsp_card_1 = RtspCard(self.rtsp_frame, 'rtsp://admin:self1004@@118.37.223.147:8522/live/main7', 70)
        self.rtsp_card_1.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        
        self.rtsp_card_2 = RtspCard(self.rtsp_frame, 'rtsp://admin:self1004@@118.37.223.147:8522/live/main6', 70)
        self.rtsp_card_2.grid(row=0, column=1, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        
        self.rtsp_card_3 = RtspCard(self.rtsp_frame, 'rtsp://admin:self1004@@118.37.223.147:8522/live/main8', 70)
        self.rtsp_card_3.grid(row=0, column=2, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        # self.rtsp_card_1.tkraise()
        # print('bind tags = ' , self.bindtags())
        self.rtsp_card_1.lbl.bind("<Button-1>", lambda e: self.select_rtsp_card(event=e, card_num=1))
        self.rtsp_card_2.lbl.bind("<Button-1>", lambda e: self.select_rtsp_card(event=e, card_num=2))
        self.rtsp_card_3.lbl.bind("<Button-1>", lambda e: self.select_rtsp_card(event=e, card_num=3))
        
        # self.bind("<Button-1>", self.whole_clicked)
        # self.rtsp_frame.bind_all("<Button-1>", self.whole_clicked)
        # self.rtsp_card_1.bind("<Button-1>", lambda event : self.select_rtsp_card(event=event, card_num=1))
        # self.rtsp_card_2.bind("<Button-1>", lambda event : self.select_rtsp_card(event=event, card_num=2))
        # self.rtsp_card_3.bind("<Button-1>", lambda event : self.select_rtsp_card(event=event, card_num=3))
        
        self.rtsp_main_card = RtspMainView(self.rtsp_frame, 200)
        self.rtsp_main_card.grid(row=1, column=0, rowspan=2, columnspan=3, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        self.op_button = customtkinter.CTkButton(self, text='run', command=self.toggle_video_op_flag)
        self.op_button.grid(row=3, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
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
        
        
    # def mainloop(self):
        # self.rtsp_card_1.run_video()
        # self.rtsp_card_2.run_video()
        # self.rtsp_card_3.run_video()
        # super().mainloop()
        # main_loop_thread = threading.Thread(target=super().mainloop)
        # run_video_thread = threading.Thread(target=self.rtsp_card_1.run_video)
        # main_loop_thread.start()
        # main_loop_thread.join()
        # run_video_thread.start()
        # run_video_thread.join()
    
    def run(self):
        self.mainloop()
        
        # self.rtsp_card_1.run_video()
    
    # def rtsp_frame(self, path):
    #     frame = customtkinter.CTkFrame(self, width=180, height=180)
    #     lbl = customtkinter.CTkLabel(frame, text='', width=150, height=150)
    #     lbl.grid(row=1, column=0, columnspan=4, padx=(10, 10), pady=(10, 10), sticky="nsew")
    #     return frame


    
        




if __name__ == "__main__":
    app = App()
    app.mainloop()