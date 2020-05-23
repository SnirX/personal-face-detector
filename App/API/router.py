from PIL import Image
from torchvision.transforms import transforms

from App.AgeGender.FaceModelWrapper import FaceModelWrapper
from App.Embedding.EmbeddingWrapper import EmbeddingWrapper
from fastapi import APIRouter, File, UploadFile
import numpy as np
import cv2 as cv2

router = APIRouter()

embedding_wrapper = EmbeddingWrapper()
faceNet = FaceModelWrapper()


def load_images(func):
    if embedding_wrapper.how_many_labels_were_loaded() == 0:
        func()
    else:
        return "Images were already loaded.."

    return "Done" if embedding_wrapper.how_many_labels_were_loaded() > 0 else "Failed"


@router.get("/load_crop_images")
def load_crop_images():
    return load_images(embedding_wrapper.load_cropped_images)


@router.get("/load_orig_images")
def load_orig_images():
    return load_images(embedding_wrapper.load_orig_images)


@router.get('/labels')
def list_labels():
    return embedding_wrapper.get_labels()


@router.get('/embedded')
def get_enbedded_vectors():
    return embedding_wrapper.get_name2vector()



@router.post('/predict')
# assumes that the picture include only one face !
async def predict_person(file: UploadFile = File(...)):
    print(file.filename)
    if not file.content_type.startswith("image"):
        return "File must be an image."

    nparr = np.frombuffer(await file.read(), np.uint8)
    image = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    Image.fromarray(image, 'RGB').show()
    _, faces, _ = faceNet.get_face_box(image)
    if len(faces) != 1:
        return "Image doesnt contain 1 face : Found {} faces.".format(len(faces))

    face = faces[0]
    face_emb = transforms.ToTensor()(Image.fromarray(cv2.resize(face, (160, 160))))
    output, min_score, all_scores = embedding_wrapper.who_am_i(face_emb)
    return (output,min_score)







