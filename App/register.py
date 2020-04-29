import os
import tkinter
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from App.Embedding.EmbeddingWrapper import EmbeddingWrapper


class RegisterApp:

    def __init__(self, window, window_title, video_cap):
        self.embedding = EmbeddingWrapper()
        self.window = window
        self.window.title(window_title)

        self.vid = MyVideoCapture(video_cap)
        self.canvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        btn_frame = tkinter.Frame(window, background=self.from_rgb((117, 123, 129)))
        btn_frame.place(x=0, y=0, anchor="nw", width=self.vid.width + 4)

        self.name_label = tkinter.Label(btn_frame, text='Enter your name:', background=self.from_rgb((117, 123, 129)))
        self.name_label.pack(side="left", padx=0, pady=10, expand=True)

        self.entry = tkinter.Entry(btn_frame, width=30)
        self.entry.pack(side="left", padx=2, pady=10)

        self.btn_snapshot = tkinter.Button(btn_frame, text="Register", width=20, command=self.snapshot,
                                           bg=self.from_rgb((52, 61, 70)), fg="white")
        self.btn_snapshot.pack(side="left", padx=10, pady=10, expand=True)

        self.delay = 1
        self.update()

        self.window.mainloop()

    def snapshot(self):
        images = []
        if self.entry.get() == '':
            messagebox.showerror("Name Error", "You must enter name to register with!")
            return None
        elif os.path.isdir(os.path.join(self.embedding.cropped_images_dir, self.entry.get())):
            messagebox.showerror("Name already exists", "Error Registering. This name already exists.")
            return None
        for i in range(5):
            messagebox.showinfo("Notification", "Taking picture number {}, make sure to change positions.".format(i+1))
            ret, frame = self.vid.get_frame()
            if ret:
                images.append(Image.fromarray(frame))
            else:
                messagebox.showerror("Capture Error", "Error capturing image. Please try again.")
                return None
        self.embedding.register_person(name=self.entry.get(), imgs=images, batch=False)
        messagebox.showinfo("Success", "You're Registered successfully!")

    def update(self):
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

            self.window.after(self.delay, self.update)
        else:
            messagebox.showerror("Error connecting to camera", "Please Make sure another program not using the camera.")

    def from_rgb(self, rgb):
        return "#%02x%02x%02x" % rgb


class MyVideoCapture:
    def __init__(self, video_source):
        self.vid = video_source
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return None, None
