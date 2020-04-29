import cv2
import tkinter as tk
from tkinter import *
from tkinter.ttk import Frame
from PIL import Image, ImageTk
from tkinter import messagebox
from App.Embedding.EmbeddingWrapper import EmbeddingWrapper
from torchvision import transforms


def get_face_box(net, frame, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
    return frame_opencv_dnn, bboxes


class LiveApp:
    def __init__(self, window, cap):
        face_proto = "AgeGender/opencv_face_detector.pbtxt"
        face_model = "AgeGender/opencv_face_detector_uint8.pb"
        age_proto = "AgeGender/age_deploy.prototxt"
        age_model = "AgeGender/age_net.caffemodel"
        gender_proto = "AgeGender/gender_deploy.prototxt"
        gender_model = "AgeGender/gender_net.caffemodel"

        # Load network
        self.ageNet = cv2.dnn.readNet(age_model, age_proto)
        self.genderNet = cv2.dnn.readNet(gender_model, gender_proto)
        self.faceNet = cv2.dnn.readNet(face_model, face_proto)

        self.embedding = EmbeddingWrapper()

        self.padding = 20
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.genderList = ['Male', 'Female']
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

        # The frame of the camera photos
        self.lmain = tk.Label(self.mainFrame)
        self.lmain.grid(row=0, column=0)
        self.cap = cap

        self.closeButton = Button(self.window, text="CLOSE", font=self.fontButtons, bg=self.white, width=20, height=1)
        self.closeButton.configure(command=lambda: self.window.destroy())
        self.closeButton.place(x=270, y=430)

        self.show_frame()  # Display
        self.window.mainloop()  # Starts GUI

    def show_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame_face, bboxes = get_face_box(self.faceNet, frame)
                if bboxes:
                    for bbox in bboxes:
                        face = frame[max(0, bbox[1] - self.padding):min(bbox[3] + self.padding, frame.shape[0] - 1), max(
                            0, bbox[0] - self.padding):min(bbox[2] + self.padding, frame.shape[1] - 1)]

                        face_emb = transforms.ToTensor()(Image.fromarray(cv2.resize(cv2.cvtColor(
                            face, cv2.COLOR_BGR2RGB), (160, 160))))
                        output, min_score, all_scores = self.embedding.who_am_i(face_emb)

                        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
                        self.genderNet.setInput(blob)
                        genderPreds = self.genderNet.forward()
                        gender = self.genderList[genderPreds[0].argmax()]

                        self.ageNet.setInput(blob)
                        agePreds = self.ageNet.forward()
                        age = self.ageList[agePreds[0].argmax()]

                        label = "{},{},{}".format(output, gender, age)
                        cv2.putText(frame_face, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                                    2, cv2.LINE_AA)
                frame_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)

                img = Image.fromarray(frame_face).resize((760, 400))
                imgtk = ImageTk.PhotoImage(image=img)
                self.lmain.imgtk = imgtk
                self.lmain.configure(image=imgtk)
                self.lmain.after(1, self.show_frame)
            else:
                messagebox.showerror("Error connecting to camera", "Please Make sure another program not using the camera.")
        else:
            messagebox.showerror("Error connecting to camera", "Please Make sure you have a camera.")
