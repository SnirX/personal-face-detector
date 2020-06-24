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
from App.Embedding.EmbeddingWrapper import EmbeddingWrapper
from App.FoolMe.exceptions.fool_me_exceptions import TooManyFacesException, NoFaceException, NoLabelException
from App.FoolMe.pgd import run_pgd


class FoolModel:
    DEFAULT_IMAGE_PATH = os.path.join('Images', 'No_picture_available.png')
    WIDTH = 700
    HEIGHT = 600

    def __init__(self, window, window_title):
        self.face_model_wrapper = FaceModelWrapper()
        self.embedding_wrapper = EmbeddingWrapper()
        self.window = window
        self.window.geometry("{}x{}".format(self.WIDTH, self.HEIGHT))  # set the size of the app to be 500x500
        self.window.resizable(0, 0)  # Don't allow resizing in the x or y direction
        self.window.title(window_title)

        btn_frame = tkinter.Frame(window, background=self.from_rgb((117, 123, 129)))
        btn_frame.place(x=0, y=0, anchor="nw", width=self.WIDTH, height=50)

        self.btn_file_select = tkinter.Button(btn_frame, text="Select", width=10, command=self.select_image,
                                              bg=self.from_rgb((52, 61, 70)), fg="white")
        self.btn_file_select.pack(side="left", padx=10, pady=10, expand=True)

        self.btn_predict = tkinter.Button(btn_frame, text="Predict", width=10, command=self.predict,
                                          bg=self.from_rgb((52, 61, 70)), fg="white")
        self.btn_predict.pack(side="left", padx=10, pady=10, expand=True)

        self.btn_fool = tkinter.Button(btn_frame, text="Fool Model", width=10, command=self.fool_model,
                                          bg=self.from_rgb((52, 61, 70)), fg="white")
        self.btn_fool.pack(side="left", padx=10, pady=10, expand=True)

        self.name_label = tkinter.Label(btn_frame, text='Target label:', background=self.from_rgb((117, 123, 129)))
        self.name_label.pack(side="left", padx=0, pady=10, expand=True)

        self.entry = tkinter.Entry(btn_frame, width=20)
        self.entry.pack(side="left", padx=2, pady=10)

        image_frame = tkinter.Frame(self.window, background=self.from_rgb((117, 123, 129)))
        image_frame.place(x=100, y=75, anchor="nw", width=480, height=480)
        img = Image.open(self.DEFAULT_IMAGE_PATH).resize((480, 480), Image.ANTIALIAS)
        default_image = ImageTk.PhotoImage(img)
        self.image_gui = tkinter.Label(image_frame, image=default_image)
        self.image_gui.pack(expand="yes")

        self.original_image = None
        self.fake_image = None
        self.fool_me = False
        self.fake_image_as_tensor = None
        self.cropped_image_as_tensor = None

        self.window.mainloop()

    def from_rgb(self, rgb):
        return "#%02x%02x%02x" % rgb

    def select_image(self):
        filename = askopenfilename()
        self._update_image(path=filename)

    def predict(self):
        if self.fool_me is True and self.fake_image_as_tensor is not None:
            predicted_name, score, _ = self.embedding_wrapper.who_am_i(tensor=self.fake_image_as_tensor)
            print("Prediction - {}, Score - {}".format(predicted_name, score))
            messagebox.showinfo("Info", "Prediction - {}".format(predicted_name))
        elif self.fool_me is False and self.original_image is not None:
            predicted_name, score, _ = self.embedding_wrapper.who_am_i(tensor=transforms.ToTensor()(self.original_image))
            print("Prediction - {}, Score - {}".format(predicted_name, score))
            messagebox.showinfo("Info", "Prediction - {}".format(predicted_name))
        else:
            messagebox.showerror("Error", "Please select image")

    def fool_model(self):
        try:
            if self.original_image is None:
                messagebox.showerror("Error", "Please select an image")
                return None
            if self.entry.get() == '':
                messagebox.showerror("Error", "You must enter a label to fool model")
                return None
            fake_image_as_tensor, score = run_pgd(transforms.ToTensor()(self.original_image), self.entry.get())
            self.fake_image_as_tensor = fake_image_as_tensor
            resized_image = transforms.ToPILImage()(fake_image_as_tensor).convert("RGB").resize((480, 480), Image.ANTIALIAS)
            self.fake_image = resized_image
            tk_img = ImageTk.PhotoImage(resized_image)
            self.image_gui.configure(image=tk_img)
            self.image_gui.photo_ref = tk_img
            self.fool_me = True
        except NoLabelException as e:
            logging.error("failed to update image, no such label in system: {}".format(self.entry.get()), e)
            messagebox.showerror("Error", "Please choose a label which exists in the system")
        except Exception as e:
            logging.error("failed to load image", e)
            messagebox.showerror("Error", "Failed to fool model model.")

    def _update_image(self, path):
        try:
            img = Image.open(path).resize((480, 480), Image.ANTIALIAS)
            self.cropped_image_as_tensor = self._get_cropped_image_as_tensor(img)
            self.original_image = transforms.ToPILImage()(self.cropped_image_as_tensor).convert("RGB")
            tk_img = ImageTk.PhotoImage(self.original_image.resize((480, 480), Image.ANTIALIAS))
            self.image_gui.configure(image=tk_img)
            self.image_gui.photo_ref = tk_img
            self.fool_me = False
        except NoFaceException as e:
            logging.error("failed to update image, no face in image", e)
            messagebox.showerror("Error", "Please choose image with a face :)")
        except TooManyFacesException as e:
            logging.error("failed to update image, more than 1 face in image", e)
            messagebox.showerror("Error", "Please choose an image with one face :)")
        except Exception as e:
            logging.error("failed to update image", e)
            messagebox.showerror("Error", "Failed fetching image.")

    def _get_cropped_image_as_tensor(self, image) -> torch.Tensor:
        frame_face, faces, bboxes = self.face_model_wrapper.get_face_box(image)
        if faces:
            if len(faces) > 1:
                raise TooManyFacesException()
            for index in range(len(faces)):
                face = faces[index]
                face_emb = transforms.ToTensor()(
                    Image.fromarray(cv2.resize(face, (160, 160))))
                return face_emb
        raise NoFaceException()
