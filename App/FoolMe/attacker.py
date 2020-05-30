import logging

import torch
import torch.nn as nn

from App.Embedding.EmbeddingWrapper import EmbeddingWrapper
from App.FoolMe.image_transformer import ImageTransformer
from Utils.torch_utils import get_torch_device


class Attacker(object):
    _instance = None
    embedding_wrapper: EmbeddingWrapper = EmbeddingWrapper()
    image_transformer: ImageTransformer = ImageTransformer()
    loss_fn = nn.CosineSimilarity()
    device = get_torch_device()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Attacker, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def attack(self, cropped_image, target_label, epsilon=0.01):
        image_as_tensor = self.image_transformer.transform_img_to_tensor(cropped_image)
        embedding = self.embedding_wrapper.get_embedding_by_tensor(image_as_tensor)
        victim_embeddings = self.embedding_wrapper.get_embeddings_by_label(target_label)
        mean_victim_embedding = self.embedding_wrapper.get_mean_embedding_of_embedding_set(victim_embeddings)
        return self._targeted_fast_gradient_sign_method(embedding, mean_victim_embedding, epsilon)

    def _targeted_fast_gradient_sign_method(self, embedding: torch.Tensor, victim_embedding: torch.Tensor, epsilon):
        loss = -self.loss_fn(embedding, victim_embedding)
        loss.backward(retain_graph=True)
        gradient_sign = embedding.grad.data.sign()
        image_with_noise = embedding + epsilon * gradient_sign
        clipped_image_with_noise = torch.clamp(image_with_noise, 0, 1)
        return clipped_image_with_noise
