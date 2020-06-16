import torch
from facenet_pytorch import MTCNN

from Utils.torch_utils import get_torch_device


class ImageCropper(object):
    _device = get_torch_device()
    _mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=_device
    )
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ImageCropper, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    def crop_to_tensor(self, image):
        cropped = self._mtcnn(image)
        return torch.stack([cropped]).to(self._device)

