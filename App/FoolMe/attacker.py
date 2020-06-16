import torch
import torch.nn as nn

from App.Embedding.EmbeddingWrapper import EmbeddingWrapper
from App.FoolMe.image_transformer import ImageTransformer
from Utils.torch_utils import get_torch_device, requires_and_retains_grad


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

    def attack(self, cropped_image_as_tensor, target_label, epsilon=0.02, epochs=2):
        victim_embeddings = self.embedding_wrapper.get_embeddings_by_label(target_label)
        mean_victim_embedding = self.embedding_wrapper.get_mean_embedding_of_embedding_set(victim_embeddings)
        return self._targeted_fast_gradient_sign_method(cropped_image_as_tensor, mean_victim_embedding, epsilon)

    # TODO: to implement
    def _projected_gradient_descent(self, embedding: torch.Tensor, victim_embedding: torch.Tensor, epsilon, epochs):
        images_with_noise = list()
        for epoch in range(epochs):
            image_with_noise = self._targeted_fast_gradient_sign_method(embedding, victim_embedding, epsilon)
            images_with_noise.append(image_with_noise)

    def _targeted_fast_gradient_sign_method(self, cropped_image_as_tensor: torch.Tensor, victim_embedding: torch.Tensor, epsilon):
        victim_embedding_1_dim_added = victim_embedding.unsqueeze(0)  # 512 to 1,512
        embedding = self.embedding_wrapper.get_embedding_by_tensor_without_detach(cropped_image_as_tensor)
        requires_and_retains_grad(cropped_image_as_tensor)
        loss = self.loss_fn(embedding, victim_embedding_1_dim_added)
        loss.backward(retain_graph=True)
        gradient_sign = cropped_image_as_tensor.grad.data.sign()
        image_with_noise = cropped_image_as_tensor + epsilon * gradient_sign
        clipped_image_with_noise = torch.clamp(image_with_noise, 0, 1)
        return clipped_image_with_noise
