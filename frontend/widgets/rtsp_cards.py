from typing import Optional, Tuple, Union
import customtkinter
import cv2
import threading
from multiprocessing import Process
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
import time


def cvimg_to_tkimg(frame, width, height):
    frame = cv2.resize(frame.copy(), (int(width), int(height)))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    return imgtk

class RtspFrame(customtkinter.CTkFrame):
    # def __init__(self, master: any, width: int = 200, height: int = 200, corner_radius: int | str | None = None, border_width: int | str | None = None, bg_color: str | Tuple[str, str] = "transparent", fg_color: str | Tuple[str, str] | None = None, border_color: str | Tuple[str, str] | None = None, background_corner_colors: Tuple[str | Tuple[str, str]] | None = None, overwrite_preferred_drawing_method: str | None = None, **kwargs):
    #     super().__init__(master, width, height, corner_radius, border_width, bg_color, fg_color, border_color, background_corner_colors, overwrite_preferred_drawing_method, **kwargs)
    def __init__(self, parent, drawing_result_que, radiobutton_callback, selected_cam_num):
        super().__init__(master=parent)
        self.radiobutton_callback = radiobutton_callback
        self.drawing_result_que = drawing_result_que
        self.selected_cam_num = selected_cam_num
        # self.rtsp_card_1 = RtspCard(self, 85, self.drawing_result_ques[3])
        
        self.grid_columnconfigure((0,1,2,3,4), weight=1)
        self.grid_rowconfigure((0,1), weight=1)
        
        
        self.card_height = 85
        self.card_width = int(self.card_height * (16/9))
        
        self.main_view_height = 300
        self.main_view_width = int(self.main_view_height * (16/9))
        
        
        self.rtsp_card_1 = RtspCard(self, self.card_height)
        self.rtsp_card_1.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")


        # self.rtsp_card_2 = RtspCard(self, 85, self.drawing_result_ques[4]) 
        self.rtsp_card_2 = RtspCard(self, self.card_height)
        self.rtsp_card_2.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")


        # self.rtsp_card_3 = RtspCard(self, 85, self.drawing_result_ques[5])
        self.rtsp_card_3 = RtspCard(self, self.card_height)
        self.rtsp_card_3.grid(row=0, column=2, padx=4, pady=4, sticky="nsew")
        
        

        self.rtsp_card_1.lbl.bind("<Button-1>", lambda e: self.select_rtsp_card(event=e, card_num=1))
        self.rtsp_card_2.lbl.bind("<Button-1>", lambda e: self.select_rtsp_card(event=e, card_num=2))
        self.rtsp_card_3.lbl.bind("<Button-1>", lambda e: self.select_rtsp_card(event=e, card_num=3))

        self.rtsp_main_card = RtspMainView(self, 300)
        self.rtsp_main_card.grid(row=1, column=0, rowspan=2, columnspan=3, padx=4, pady=4, sticky="nsew")

        self.bp_view = BpView(self, 300)
        # self.bp_view.grid(row=0, column=3, rowspan=2, columnspan=2, padx=(0, 0), pady=(0, 0), sticky="nsew")
        self.bp_view.grid(row=0, column=3, rowspan=2, columnspan=2, padx=4, pady=4)
        
        self.current_frame_1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.current_frame_2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.current_frame_3 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.current_frame_bp = cv2.imread('frontend/assets/blueprint.png')
        
        self.rtsp_card_1.make_selected()
        self.rtsp_main_card.selected_cam_num = 1
        self.selected_cam_num.set(0)
        
        self.run_video_thread = threading.Thread(target=self.run_video, args=(), daemon=False)
        self.run_video_thread.start()
        # self.run_video_proc = Process(target=self.run_video, args=(), daemon=True)
        # self.run_video_proc.start()
        
        self.update_frame_task = self.update_frame()
        
        
    def select_rtsp_card(self, event, card_num):
        # print('frame clicked! card num = ', card_num)
        self.rtsp_card_1.make_unselected()
        self.rtsp_card_2.make_unselected()
        self.rtsp_card_3.make_unselected()
        
        if card_num == 1:
            self.rtsp_card_1.make_selected()
            self.rtsp_main_card.selected_cam_num = 1
            self.selected_cam_num.set(0)
            self.radiobutton_callback(1)
        elif card_num == 2:
            self.rtsp_card_2.make_selected()
            self.rtsp_main_card.selected_cam_num = 2
            self.selected_cam_num.set(1)
            self.radiobutton_callback(2)
        elif card_num == 3:
            self.rtsp_card_3.make_selected()
            self.rtsp_main_card.selected_cam_num = 3
            self.selected_cam_num.set(2)
            self.radiobutton_callback(3)
            
    def run_video(self):
        # print('run video')
        while True:
            if (not self.drawing_result_que[0].empty()) and (not self.drawing_result_que[1].empty()) and (not self.drawing_result_que[2].empty()):
                self.current_frame_1 = self.drawing_result_que[0].get()
                self.current_frame_2 = self.drawing_result_que[1].get()
                self.current_frame_3 = self.drawing_result_que[2].get()
            else:
                time.sleep(0.01)
            if not self.drawing_result_que[3].empty():
                self.current_frame_bp = self.drawing_result_que[3].get()
            else:
                time.sleep(0.01)
            # if type(self.current_frame_1) == type(None) or type(self.current_frame_2) == type(None) or type(self.current_frame_3) == type(None) or type(self.current_frame_bp) == type(None):
            #     print('run video None type')
            #     self.after_cancel(self.update_frame_task)
            #     break
        # print('run video in gui end')
        
    def update_frame(self):
        # print('update frame gui rtsp')
        frame1 = self.current_frame_1
        frame2 = self.current_frame_2
        frame3 = self.current_frame_3
        frame_bp = self.current_frame_bp
        
        if self.rtsp_main_card.selected_cam_num == 1:
            # frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            # frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
            imgtk = cvimg_to_tkimg(frame1, int(1.5 * self.main_view_width),  int(1.5 * self.main_view_height))
            self.rtsp_main_card.lbl.configure(image=imgtk)
        elif self.rtsp_main_card.selected_cam_num == 2:
            # frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            # frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
            imgtk = cvimg_to_tkimg(frame2, int(1.5 * self.main_view_width),  int(1.5 * self.main_view_height))
            self.rtsp_main_card.lbl.configure(image=imgtk)
        elif self.rtsp_main_card.selected_cam_num == 3:
            # frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            # frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            imgtk = cvimg_to_tkimg(frame3, int(1.5 * self.main_view_width),  int(1.5 * self.main_view_height))
            self.rtsp_main_card.lbl.configure(image=imgtk)
        
        imgtk1 = cvimg_to_tkimg(frame1, int(1.5 * self.card_width),  int(1.5 * self.card_height))
        self.rtsp_card_1.lbl.configure(image=imgtk1)

        imgtk2 = cvimg_to_tkimg(frame2, int(1.5 * self.card_width),  int(1.5 * self.card_height))
        self.rtsp_card_2.lbl.configure(image=imgtk2)

        imgtk3 = cvimg_to_tkimg(frame3, int(1.5 * self.card_width),  int(1.5 * self.card_height))
        self.rtsp_card_3.lbl.configure(image=imgtk3)

        imgtk_bp = cvimg_to_tkimg(frame_bp, int(1.5 * self.main_view_width),  int(1.7 * self.main_view_height))
        self.bp_view.lbl.configure(image=imgtk_bp)

        self.update_frame_task = self.after(80, self.update_frame)
        

class RtspCard(customtkinter.CTkFrame):
    def __init__(self, parent, height) -> None:
        super().__init__(parent)
        self.is_selected = False
        self.card_height = height
        self.card_width = int(self.card_height * (16/9))
        self.lbl = customtkinter.CTkLabel(self, text='', width=self.card_width, height=self.card_height, fg_color='light grey', )
        self.lbl.pack(anchor='center', fill='both', expand=True)
        
        self.lbl.bind("<Enter>", self.on_hover)
        self.lbl.bind("<Leave>", self.off_hover)

    def on_hover(self, event):
        # print('event = ', event)
        if not self.is_selected:
            self.lbl.configure(fg_color = 'light sky blue')

    def off_hover(self, event):
        # print('event = ', event)
        if not self.is_selected:
            self.lbl.configure(fg_color = 'light grey')

    def make_selected(self):
        self.is_selected = True
        self.lbl.configure(fg_color = 'blue')
            
    def make_unselected(self):
        self.is_selected = False
        self.lbl.configure(fg_color = 'light grey')


class RtspMainView(customtkinter.CTkFrame):
    def __init__(self, parent, height) -> None:
        super().__init__(parent)
        self.card_height = height
        self.card_width = int(self.card_height * (16/9))
        self.lbl = customtkinter.CTkLabel(self, text='', width=self.card_width, height=self.card_height, fg_color='light grey', )
        self.lbl.pack(anchor='center', fill='both', expand=True)

        self.selected_cam_num = 1


class BpView(customtkinter.CTkFrame):
    def __init__(self, parent, height) -> None:
        super().__init__(parent)
        self.card_height = height
        self.card_width = int(self.card_height * (16/9))
        self.lbl = customtkinter.CTkLabel(self, text='', width=self.card_width, height=self.card_height, fg_color='light grey', )
        self.lbl.pack(anchor='center', fill='both', expand=True)

