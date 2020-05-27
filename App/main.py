import os
import tkinter
import cv2
from PIL import Image, ImageTk
from App.register import RegisterApp
from App.live_detect import LiveApp
from App.fool_model import FoolModel
from tkinter import messagebox
from App.Embedding.EmbeddingWrapper import EmbeddingWrapper


class MainApp:
    REGISTER = os.path.join(os.getcwd(), 'register.py')
    LIVE = os.path.join(os.getcwd(), 'live_detect.py')

    def __init__(self, window, window_title):
        self.embedding = EmbeddingWrapper()
        self.embedding.load_cropped_images()
        self.cap = None
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

        self.btn_register = tkinter.Button(btn_frame, text="Register", width=15, command=self.register,
                                           bg=self.from_rgb((52, 61, 70)), fg="white")
        self.btn_register.pack(side="left", padx=10, pady=10, expand=True)

        self.btn_live = tkinter.Button(btn_frame, text="Live Detection", width=15, command=self.live,
                                       bg=self.from_rgb((52, 61, 70)), fg="white")
        self.btn_live.pack(side="left", padx=10, pady=10, expand=True)

        self.btn_fool = tkinter.Button(btn_frame, text="Fool Me!", width=15, command=self.fool,
                                       bg=self.from_rgb((52, 61, 70)), fg="white")
        self.btn_fool.pack(side="left", padx=10, pady=10, expand=True)

        self.window.mainloop()

    def register(self):
        cv2.destroyAllWindows()
        if self.cap is not None:
            self.cap.release()
        try:
            self.cap = cv2.VideoCapture(0)
        except Exception as e:
            messagebox.showerror("Error connecting to camera", "Please Make sure another program not using the camera.")
            return None
        RegisterApp(tkinter.Toplevel(), "Register", video_cap=self.cap)

    def live(self):
        cv2.destroyAllWindows()
        if self.cap is not None:
            self.cap.release()
        try:
            self.cap = cv2.VideoCapture(0)
        except Exception as e:
            messagebox.showerror("Error connecting to camera", "Please Make sure another program not using the camera.")
            return None
        LiveApp(tkinter.Toplevel(), cap=self.cap)

    @staticmethod
    def fool():
        FoolModel(tkinter.Toplevel(), "Fool Model")

    @staticmethod
    def from_rgb(rgb):
        return "#%02x%02x%02x" % rgb


MainApp(tkinter.Tk(), "Personal Face Detector")
