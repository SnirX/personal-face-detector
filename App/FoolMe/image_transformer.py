import torch
from torchvision import transforms

from Utils.torch_utils import get_torch_device


class ImageTransformer(object):

    device = get_torch_device()
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ImageTransformer, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def transform_img_to_tensor(self, image):
        image_as_tensor = transforms.toTensor()(image)
        stacked_tensor = torch.stack([image_as_tensor]).to(self.device)
        stacked_tensor.requires_grad = True
        stacked_tensor.retain_grad()
        return stacked_tensor
