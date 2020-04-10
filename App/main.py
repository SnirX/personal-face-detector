import os
import tkinter
import cv2
from PIL import Image, ImageTk
from App.register import RegisterApp
from App.live_detect import LiveApp


class MainApp:
    REGISTER = os.path.join(os.getcwd(), 'register.py')
    LIVE = os.path.join(os.getcwd(), 'live_detect.py')

    def __init__(self, window, window_title):
        self.cap = cv2.VideoCapture(0)
        self.window = window
        self.window.title(window_title)
        image = cv2.imread(os.path.join('Images', 'main_image.PNG'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        self.canvas = tkinter.Canvas(window, width=image.shape[1], height=image.shape[0] + 50)
        self.canvas.pack()
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.canvas.create_image(0, 50, image=self.photo, anchor=tkinter.NW)

        title_frame = tkinter.Frame(window)
        title_frame.place(x=0, y=0, anchor="nw", width=image.shape[1] + 4, height=50)
        self.name_label = tkinter.Label(title_frame, text='Personal Face Detector')
        self.name_label.config(font=("Courier", 15))
        self.name_label.pack(side="left", padx=0, pady=10, expand=True)

        btn_frame = tkinter.Frame(window, background=self.from_rgb((117, 123, 129)))
        btn_frame.place(x=0, y=50, anchor="nw", width=image.shape[1] + 4, height=50)

        self.btn_register = tkinter.Button(btn_frame, text="Register", width=20, command=self.register,
                                           bg=self.from_rgb((52, 61, 70)), fg="white")
        self.btn_register.pack(side="left", padx=10, pady=10, expand=True)

        self.btn_live = tkinter.Button(btn_frame, text="Live Detection", width=20, command=self.live,
                                       bg=self.from_rgb((52, 61, 70)), fg="white")
        self.btn_live.pack(side="left", padx=10, pady=10, expand=True)

        self.window.mainloop()

    def register(self):
        RegisterApp(tkinter.Toplevel(), "Register", video_cap=self.cap)

    def live(self):
        LiveApp(tkinter.Toplevel(), cap=self.cap)

    def from_rgb(self, rgb):
        return "#%02x%02x%02x" % rgb


MainApp(tkinter.Tk(), "Personal Face Detector")
