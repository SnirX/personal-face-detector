import time

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

from App.Embedding.EmbeddingWrapper import EmbeddingWrapper
from App.FoolMe.exceptions.fool_me_exceptions import NoLabelException

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device).train(False)
embedding_wrapper = EmbeddingWrapper()


def run_pgd(source_tensor, target_label, epsilon=0.045, epochs=3):
    if not embedding_wrapper.is_label_exists(target_label):
        raise NoLabelException()

    targets_dict_tensors = embedding_wrapper.get_embeddings_by_label(target_label)
    average_vector = embedding_wrapper.get_mean_embedding_of_embedding_set(targets_dict_tensors).unsqueeze(0)

    image = transforms.ToPILImage()(source_tensor.squeeze(0)).convert("RGB")
    tensor = transforms.ToTensor()(image)
    stacked_tensor = torch.stack([tensor]).to(device)  # Tensor([1,2,3]) -> Tensor([[1,2,3]])

    start_time = time.time()
    is_first = True
    image_with_noise = stacked_tensor
    for epoch in range(epochs):
        print("target : {} , epsilon : {}, epoch : {}".format(target_label, epsilon, epoch + 1))
        image_with_noise = TFGSM(image_with_noise, resnet, average_vector, epsilon, requires_grad=is_first)
        is_first = False
    score = embedding_wrapper.get_distance_between_embeddings(average_vector, resnet(image_with_noise))
    print("Time took for pgd on target {} : {} seconds".format(target_label, time.time() - start_time))
    print("Score: {}".format(score))
    return image_with_noise.squeeze(0), score


def TFGSM(random_image: torch.Tensor, model, target_vector, epsilon, requires_grad=False):
    print("working..")
    if requires_grad:
        random_image.requires_grad = requires_grad
    random_image.retain_grad()
    ri_embedded_vector = model(random_image)
    cos = nn.CosineSimilarity()
    loss = cos(ri_embedded_vector, target_vector)
    # Calculate gradients of model in backward pass
    loss.backward(retain_graph=True)
    data_grad = random_image.grad.data
    # The sign of the data grad (positive/negative)
    sign_data_grad = data_grad.sign()
    # Create the new image by adjusting each pixel of the input image
    image_with_noise = random_image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range to the image pixels
    image_with_noise = torch.clamp(image_with_noise, 0, 1)
    # Return the perturbed image
    return image_with_noise
