import cv2
import tkinter as tk
from tkinter import *
from tkinter.ttk import Frame
from PIL import Image, ImageTk


class LiveApp:
    def __init__(self, window, cap):
        self.white = "#ffffff"
        lightblue2 = "#adc5ed"
        font = "Constantia"
        self.fontButtons = (font, 12)
        max_width = 800
        max_height = 480

        self.window = window
        self.window.title('Live Detection')
        self.window.configure(bg=lightblue2)
        self.window.geometry('%dx%d+%d+%d' % (max_width, max_height, 0, 0))
        self.window.resizable(0, 0)

        self.mainFrame = Frame(self.window)
        self.mainFrame.place(x=20, y=20)

        # Capture video frames
        self.lmain = tk.Label(self.mainFrame)
        self.lmain.grid(row=0, column=0)
        self.cap = cap

        self.closeButton = Button(self.window, text="CLOSE", font=self.fontButtons, bg=self.white, width=20, height=1)
        self.closeButton.configure(command=lambda: self.window.destroy())
        self.closeButton.place(x=270, y=430)

        self.show_frame()  # Display
        self.window.mainloop()  # Starts GUI

    def show_frame(self):
        ret, frame = self.cap.read()

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        img = Image.fromarray(cv2image).resize((760, 400))
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
        self.lmain.after(1, self.show_frame)
