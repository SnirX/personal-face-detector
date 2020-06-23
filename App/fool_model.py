import logging
import os
import tkinter
from tkinter import messagebox
from tkinter.filedialog import askopenfilename, IntVar, Checkbutton

import cv2
import torch
from PIL import Image, ImageTk
from torchvision import transforms

from App.AgeGender.FaceModelWrapper import FaceModelWrapper
from App.FoolMe.attacker import Attacker
from App.FoolMe.image_cropper import ImageCropper
from App.FoolMe.pgd import run_pgd
from App.register import MyVideoCapture


class FoolModel:
    DEFAULT_IMAGE_PATH = os.path.join('Images', 'No_picture_available.png')
    WIDTH = 700
    HEIGHT = 600

    def __init__(self, window, window_title):
        self.camera = get_camera()
        self.vid = MyVideoCapture(self.camera)
        self.face_model_wrapper = FaceModelWrapper()
        self.image_cropper = ImageCropper()
        self.attacker = Attacker()
        self.window = window
        self.window.geometry("{}x{}".format(self.WIDTH, self.HEIGHT))  # set the size of the app to be 500x500
        self.window.resizable(0, 0)  # Don't allow resizing in the x or y direction
        self.window.title(window_title)

        self.is_fool_mode = IntVar()

        btn_frame = tkinter.Frame(window, background=self.from_rgb((117, 123, 129)))
        btn_frame.place(x=0, y=0, anchor="nw", width=self.WIDTH, height=50)

        self.btn_file_select = tkinter.Button(btn_frame, text="Select", width=10, command=self.select_image,
                                              bg=self.from_rgb((52, 61, 70)), fg="white")
        self.btn_file_select.pack(side="left", padx=10, pady=10, expand=True)

        self.btn_predict = tkinter.Button(btn_frame, text="Predict", width=10, command=self.predict,
                                          bg=self.from_rgb((52, 61, 70)), fg="white")
        self.btn_predict.pack(side="left", padx=10, pady=10, expand=True)

        self.btn_fool = Checkbutton(btn_frame, text="Fool Model", variable=self.is_fool_mode,
                                    font=('courier', 15, 'bold'),
                                    highlightbackground=self.from_rgb((117, 123, 129)))
        self.btn_fool.pack(side="left", padx=10, pady=10, expand=True)

        image_frame = tkinter.Frame(self.window, background=self.from_rgb((117, 123, 129)))
        image_frame.place(x=100, y=75, anchor="nw", width=480, height=480)
        img = Image.open(self.DEFAULT_IMAGE_PATH).resize((480, 480), Image.ANTIALIAS)
        default_image = ImageTk.PhotoImage(img)
        self.image_gui = tkinter.Label(image_frame, image=default_image)
        self.image_gui.pack(expand="yes")

        self.window.mainloop()

    def from_rgb(self, rgb):
        return "#%02x%02x%02x" % rgb

    def select_image(self):
        filename = askopenfilename()
        self._update_image(path=filename)

    def predict(self):
        if self.is_fool_mode.get() == 1:
            messagebox.showinfo("Info", "You chose to fool me :)")
        else:
            messagebox.showinfo("Info", "You chose not to fool me :)")

    def _update_image(self, path):
        try:
            img = Image.open(path).resize((480, 480), Image.ANTIALIAS)
            # tk_img = ImageTk.PhotoImage(img) # TODO: original line
            cropped_image_as_tensor = self.get_image_as_tensor_from_camera()
            tuple2 = run_pgd(cropped_image_as_tensor)  # TODO: to return
            tk_img = ImageTk.PhotoImage(tuple2[0])  # TODO: to return
            self.image_gui.configure(image=tk_img)
            self.image_gui.photo_ref = tk_img
        except Exception as e:
            logging.error("failed to update image", e)
            messagebox.showerror("Error", "Failed fetching image.")

    def get_image_as_tensor_from_camera(self):
        ret, frame = self.camera.read()
        frame_face, faces, bboxes = self.face_model_wrapper.get_face_box(frame)
        if faces:
            for index in range(len(faces)):
                face = faces[index]
                bbox = bboxes[index]
                face_emb = transforms.ToTensor()(
                    Image.fromarray(cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (160, 160))))
                return face_emb
        raise RuntimeError("No faces in camera")


def get_camera():
    try:
        return cv2.VideoCapture(0)
    except Exception as e:
        messagebox.showerror("Error connecting to camera", "Please Make sure another program not using the camera.")
        return None
