import customtkinter
import cv2
import threading
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk

class RtspCard(customtkinter.CTkFrame):
    def __init__(self, parent, path, height, drawing_result_que) -> None:
        super().__init__(parent)
        self.drawing_result_que = drawing_result_que
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
        # self.path = path
        # self.cap_loader = cv2.VideoCapture(path)
        # _, self.current_frame = self.cap_loader.read()
        self.current_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        self.run_video_thread = threading.Thread(target=self.run_video, args=(), daemon=False)
        self.run_video_thread.start()
        
        # self.update_frame_thread = threading.Thread(target=self.update_frame, args = () , daemon=False)
        # self.update_frame_thread.start()
        self.update_frame()
        # print('init end')
        
    
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
    
    def run_video(self):
        # print('run video check2')
        while True:
            self.current_frame = self.drawing_result_que.get()
            # ret, self.current_frame = self.cap_loader.read()
            # _ , _ = self.cap_loader.read()
            # if not ret:
            #     self.cap_loader.release()
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
        
        frame = cv2.resize(frame, (int(1.5 * self.card_width), int(1.5 * self.card_height)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        
        img = Image.fromarray(frame)
        # imgtk = customtkinter.CTkImage(img, size= (int(0.95 * self.card_width), int(0.95 * self.card_height)))
        imgtk = ImageTk.PhotoImage(image=img)
        # self.lbl.imgtk = imgtk
        self.lbl.configure(image=imgtk)
        # self.lbl.update()
        # self.after(256, self.update_frame)
        self.lbl.after(256, self.update_frame)




class RtspMainView(customtkinter.CTkFrame):
    def __init__(self, parent, height, drawing_result_ques) -> None:
        super().__init__(parent)
        self.drawing_result_ques = drawing_result_ques
        self.card_height = height
        self.card_width = int(self.card_height * (16/9))
        self.lbl = customtkinter.CTkLabel(self, text='', width=self.card_width, height=self.card_height, fg_color='light grey', )
        # lbl.grid(row=0, column=0, columnspan=4, padx=(10, 10), pady=(10, 10), sticky="nsew")
        self.lbl.pack(anchor='center', fill='both', expand=True)
        self.paths = ['rtsp://admin:self1004@@118.37.223.147:8522/live/main7', 'rtsp://admin:self1004@@118.37.223.147:8522/live/main6',
                 'rtsp://admin:self1004@@118.37.223.147:8522/live/main8']
        
        # self.cap_loader1 = cv2.VideoCapture(self.paths[0])
        # _, self.current_frame = self.cap_loader1.read()
        # self.cap_loader2 = cv2.VideoCapture(self.paths[1])
        # self.cap_loader3 = cv2.VideoCapture(self.paths[2])
        
        self.current_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        self.selected_cam_num = 1
        
        self.run_video_thread = threading.Thread(target=self.run_video, args=(), daemon=False)
        self.run_video_thread.start()
        
        self.update_frame()

    def run_video(self):
        # print('run video check2')
        while True:
            frame1 = self.drawing_result_ques[0].get()
            frame2 = self.drawing_result_ques[1].get()
            frame3 = self.drawing_result_ques[2].get()
            
            if self.selected_cam_num == 1:
                self.current_frame = frame1
            if self.selected_cam_num == 2:
                self.current_frame = frame2
            if self.selected_cam_num == 3:
                self.current_frame = frame3
                
            
            # if self.selected_cam_num == 1:
            #     _, _ = self.cap_loader1.read()
            #     ret, self.current_frame = self.cap_loader1.read()
                
            #     _, _ = self.cap_loader2.read()
            #     _, _ = self.cap_loader2.read()
            #     _, _ = self.cap_loader3.read()
            #     _, _ = self.cap_loader3.read()
            #     if not ret:
            #         self.cap_loader1.release()
            # elif self.selected_cam_num == 2:
            #     _, _ = self.cap_loader2.read()
            #     ret, self.current_frame = self.cap_loader2.read()
                
            #     _, _ = self.cap_loader1.read()
            #     _, _ = self.cap_loader1.read()
            #     _, _ = self.cap_loader3.read()
            #     _, _ = self.cap_loader3.read()
            #     if not ret:
            #         self.cap_loader2.release()
            # elif self.selected_cam_num == 3:
            #     _, _ = self.cap_loader3.read()
            #     ret, self.current_frame = self.cap_loader3.read()
                
            #     _, _ = self.cap_loader2.read()
            #     _, _ = self.cap_loader2.read()
            #     _, _ = self.cap_loader1.read()
            #     _, _ = self.cap_loader1.read()
            #     if not ret:
            #         self.cap_loader3.release()

    
    def update_frame(self):
        # print('update frame')
        frame = self.current_frame
        frame = cv2.resize(frame, (int(1.5 * self.card_width), int(1.5 * self.card_height)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(frame)
        # imgtk = customtkinter.CTkImage(img, size= (int(0.95 * self.card_width), int(0.95 * self.card_height)))
        imgtk = ImageTk.PhotoImage(image=img)
        # self.lbl.imgtk = ImageTk
        self.lbl.configure(image=imgtk)
        # self.lbl.update()
        # self.after(10, self.update_frame)
        self.lbl.after(10, self.update_frame)
        



class BpView(customtkinter.CTkFrame):
    def __init__(self, parent, height, drawing_result_que) -> None:
        super().__init__(parent)
        self.drawing_result_ques = drawing_result_que
        self.card_height = height
        self.card_width = int(self.card_height * (16/9))
        self.lbl = customtkinter.CTkLabel(self, text='', width=self.card_width, height=self.card_height, fg_color='light grey', )
        # lbl.grid(row=0, column=0, columnspan=4, padx=(10, 10), pady=(10, 10), sticky="nsew")
        self.lbl.pack(anchor='center', fill='both', expand=True)
        # self.paths = ['rtsp://admin:self1004@@118.37.223.147:8522/live/main7', 'rtsp://admin:self1004@@118.37.223.147:8522/live/main6',
        #          'rtsp://admin:self1004@@118.37.223.147:8522/live/main8']
        
        
        self.current_frame = cv2.imread('frontend/assets/blueprint.png')
        # self.cap_loader1 = cv2.VideoCapture(self.paths[0])
        # _, self.current_frame = self.cap_loader1.read()
        # self.cap_loader2 = cv2.VideoCapture(self.paths[1])
        # self.cap_loader3 = cv2.VideoCapture(self.paths[2])
        
        # self.selected_cam_num = 1
        
        self.run_video_thread = threading.Thread(target=self.run_video, args=(), daemon=False)
        self.run_video_thread.start()
        
        self.update_frame()

    def run_video(self):
        # print('run video check2')
        while True:
            frame = self.drawing_result_ques.get()
            self.current_frame = frame

                
            
    
    def update_frame(self):
        # print('update frame')
        frame = self.current_frame
        frame = cv2.resize(frame, (int(1.5 * self.card_width), int(1.5 * self.card_height)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(frame)
        # imgtk = customtkinter.CTkImage(img, size= (int(0.95 * self.card_width), int(0.95 * self.card_height)))
        imgtk = ImageTk.PhotoImage(image=img)
        # self.lbl.imgtk = ImageTk
        self.lbl.configure(image=imgtk)
        # self.lbl.update()
        # self.after(64, self.update_frame)
        self.lbl.after(64, self.update_frame)
        