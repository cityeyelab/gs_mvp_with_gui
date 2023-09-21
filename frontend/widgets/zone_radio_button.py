from typing import Optional, Tuple, Union
import customtkinter
import tkinter

# self.radiobutton_frame = customtkinter.CTkFrame(self, border_color='black', border_width=2)
#         self.radiobutton_frame.grid(row=5, column=2, padx=12, pady=12, sticky="nsew")
#         self.label_radio_group = customtkinter.CTkLabel(master=self.radiobutton_frame, text="Area1 Zone Display", fg_color='grey', corner_radius=8, )
#         self.label_radio_group.grid(row=0, column=1, padx=10, pady=10, )
#         self.radio_var_area1 = tkinter.IntVar(value=0)
#         self.radio_button1 = customtkinter.CTkRadioButton(master = self.radiobutton_frame, text='None', variable=self.radio_var_area1, value=0,
#                                                           command=self.radio_button_command)
#         self.radio_button1.grid(row=1, column=0, pady=8, padx=8)
#         self.radio_button2 = customtkinter.CTkRadioButton(master = self.radiobutton_frame, text='1', variable=self.radio_var_area1, value=1,
#                                                           command=self.radio_button_command)
#         self.radio_button2.grid(row=1, column=1, pady=8, padx=8)
#         # self.radio_button2.grid(row=1, column=1, pady=4, padx=4, sticky="n")
#         self.radio_button3 = customtkinter.CTkRadioButton(master = self.radiobutton_frame, text='2', variable=self.radio_var_area1, value=2,
#                                                           command=self.radio_button_command)
#         self.radio_button3.grid(row=1, column=2, pady=8, padx=8)
#         self.radio_button3 = customtkinter.CTkRadioButton(master = self.radiobutton_frame, text='3', variable=self.radio_var_area1, value=3,
#                                                           command=self.radio_button_command)
#         self.radio_button3.grid(row=1, column=3, pady=8, padx=8)


class ZoneRadioButton(customtkinter.CTkFrame):
    # def __init__(self, master: any, width: int = 200, height: int = 200, corner_radius: int | str | None = None, border_width: int | str | None = None, bg_color: str | Tuple[str, str] = "transparent", fg_color: str | Tuple[str, str] | None = None, border_color: str | Tuple[str, str] | None = None, background_corner_colors: Tuple[str | Tuple[str, str]] | None = None, overwrite_preferred_drawing_method: str | None = None, **kwargs):
    #     super().__init__(master, width, height, corner_radius, border_width, bg_color, fg_color, border_color, background_corner_colors, overwrite_preferred_drawing_method, **kwargs)
    def __init__(self, parent, width, height, area_display_values):
        super().__init__(master = parent, border_color='black', border_width=2, width=width, height=height)
        self.area_display_values = area_display_values
        self.selected_cam_num = 1
        
        # self.rowconfigure(0, weight=1)
        
        self.lbl = customtkinter.CTkLabel(master=self, text="Area1 Zone Display", fg_color='grey', corner_radius=12,  width=180*5+50, height=30)
        # self.label_radio_group.grid(row=0, column=0, padx=10, pady=10, )
        self.lbl.grid(row=0, column=0, columnspan=5, padx=10, pady=10, sticky="nsew")
        self.radio_var = tkinter.IntVar(value=0)
        
        self.radio_button1 = customtkinter.CTkRadioButton(master = self, text='None', variable=self.radio_var, value=0,
                                                          command=self.radio_button_command, width=180, height=30)
        self.radio_button1.grid(row=1, column=0, pady=8, padx=8, sticky="nsew")
        
        self.radio_button2 = customtkinter.CTkRadioButton(master = self, text='Global In/Out', variable=self.radio_var, value=1,
                                                          command=self.radio_button_command, width=180, height=30)
        self.radio_button2.grid(row=1, column=1, pady=8, padx=8, sticky="nsew")
        # self.radio_button2.grid(row=1, column=1, pady=4, padx=4, sticky="n")
        
        self.radio_button3 = customtkinter.CTkRadioButton(master = self, text='Car Wash Wait', variable=self.radio_var, value=2,
                                                          command=self.radio_button_command, width=180, height=30)
        self.radio_button3.grid(row=1, column=2, pady=8, padx=8, sticky="nsew")
        
        self.radio_button4 = customtkinter.CTkRadioButton(master = self, text='3', variable=self.radio_var, value=3,
                                                          command=self.radio_button_command, width=180, height=30)
        self.radio_button4.grid(row=1, column=3, pady=8, padx=8, sticky="nsew")
        
        self.radio_button5 = customtkinter.CTkRadioButton(master = self, text='4', variable=self.radio_var, value=4,
                                                          command=self.radio_button_command, width=180, height=30)
        self.radio_button5.grid(row=1, column=4, pady=8, padx=8, sticky="nsew")
        
        self.set_selected_cam_num(1)
      
    def radio_button_command(self):
        # pass
        radio_var = self.radio_var.get()
        
        self.area_display_values[0].set(0)
        self.area_display_values[1].set(0)
        self.area_display_values[2].set(0)
        
        if self.selected_cam_num == 1:
            self.area_display_values[0].set(radio_var)
        elif self.selected_cam_num == 2:
            self.area_display_values[1].set(radio_var)
        elif self.selected_cam_num == 3:
            self.area_display_values[2].set(radio_var)
        
        # print('radio var 1 = ' , radio_var1)
        # self.area_display_values[0].set(radio_var1)
        # print('self.area_display_values[0] = ', self.area_display_values[0])
        # area1_zone_display_value.set()
        # self.area_display_values = radio_var1

    def set_selected_cam_num(self, idx):
        self.radio_var.set(0)
        self.selected_cam_num = idx
        # idx:1->4buttons, idx:2->3buttons, idx:3->5buttons
        if idx == 1:
            self.lbl.configure(text='Area1 Zone Display')
            self.radio_button4.configure(state='normal', text='place0')
            self.radio_button5.configure(state='disable', text='X')
            # self.radio_button5.grid_forget()
        elif idx == 2:
            self.lbl.configure(text='Area3 Zone Display')
            self.radio_button4.configure(state='disable', text='X')
            # self.radio_button4.grid_forget()
            self.radio_button5.configure(state='disable', text='X')
            # self.radio_button5.grid_forget()
        elif idx == 3:
            self.lbl.configure(text='Area4 Zone Display')
            self.radio_button4.configure(state='normal', text='Electric Charging Zone')
            self.radio_button4.grid(row=1, column=3, pady=8, padx=8)
            self.radio_button5.configure(state='normal', text='Car Interior Wash Zone')
            self.radio_button5.grid(row=1, column=4, pady=8, padx=8)
        
# area4: global in out , car washing wait , Electric charging zone , car interior washing zone
# area1: global in out , car wasing wait , place 0
# area3: global in out, car wasing wait