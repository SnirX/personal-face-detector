import os
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
class FaceModelWrapper(object):
    face_model = os.path.join(os.path.dirname(__file__), "opencv_face_detector_uint8.pb")
    face_proto = os.path.join(os.path.dirname(__file__), "opencv_face_detector.pbtxt")
    faceNet = cv2.dnn.readNet(face_model, face_proto)
    padding=20
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FaceModelWrapper, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    def get_face_box(self, frame, conf_threshold=0.7):
        if isinstance(frame,Image.Image):
            frame = np.asarray(frame)
        frame_opencv_dnn = frame.copy()
        frame_height = frame_opencv_dnn.shape[0]
        frame_width = frame_opencv_dnn.shape[1]
        blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        faces = []
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                face = frame[max(0, y1 - self.padding):min(y2 + self.padding, frame.shape[0] - 1), max(
                    0, x1 - self.padding):min(x2 + self.padding, frame.shape[1] - 1)]
                faces.append(face)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)

        return frame_opencv_dnn, faces, bboxes
