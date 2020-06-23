import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from App.Embedding.EmbeddingWrapper import EmbeddingWrapper

embedding_wrapper = EmbeddingWrapper()
embedding_wrapper.load_cropped_images()


def run_pgd(source_tensor, target_label='Snir', epsilon=0.045, epochs=2):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device).train(False)

    random_imgs_path = 'images/random_people/'

    # Just random people..
    # random_imgs_with_label = load_data(random_imgs_path)
    random_imgs_with_label = list()
    random_imgs_with_label.append((transforms.ToPILImage()(source_tensor.squeeze(0)).convert("RGB"), 'shabi'))

    targets_dict = defaultdict(
        lambda: {"average_vector": torch.FloatTensor([[0] * 512]).to(device), "amount_of_vectors": 0})

    targets_dict_tensors = embedding_wrapper.get_embeddings_by_label(target_label)
    targets_dict[target_label]['average_vector'] = embedding_wrapper.get_mean_embedding_of_embedding_set(
        targets_dict_tensors)
    targets_dict[target_label]['amount_of_vectors'] = len(targets_dict_tensors)

    random_tensors = []
    for image, person_in_image in random_imgs_with_label:
        tensor = transforms.ToTensor()(image)
        stacked_tensor = torch.stack([tensor]).to(device)  # Tensor([1,2,3]) -> Tensor([[1,2,3]])
        random_tensors.append(stacked_tensor)

    random_people_labels = []
    for i, tpl in enumerate(random_imgs_with_label):
        random_people_labels.append(tpl[1])

    epsilons = [epsilon]
    target_embedded_vector2 = targets_dict.get(target_label).get('average_vector').unsqueeze(0)

    titles = random_people_labels
    pgd_scores = {
        target_label: {random_label: {eps: defaultdict(lambda: float) for eps in epsilons} for random_label in titles}
        for target_label in targets_dict.keys()}
    images_with_noise = []

    start_time = time.time()
    epsilon = epsilons.pop()
    is_first = True
    for epoch in range(epochs):
        print("target : {} , epsilon : {}, epoch : {}".format(target_label, epsilon, epoch + 1))
        for tensor in random_tensors:
            image_with_noise = TFGSM(tensor, resnet, target_embedded_vector2, epsilon, requires_grad=is_first)
    images_with_noise.append(image_with_noise)
    scores = []
    score = diff_between_tensors(target_embedded_vector2, resnet(image_with_noise))
    scores.append(round(score, 4))
    pgd_scores[target_label][titles[0]][epsilon][epoch + 1] = score
    draw_tensors(images_with_noise, (5, 10), scores)
    print("Time took for pgd on target {} : {} seconds".format(target_label, time.time() - start_time))
    return transforms.ToPILImage()(image_with_noise).convert("RGB"), score


def load_data(directory: str) -> list:
    # Directory should contain other directories as labels and inside each directory files for each person.
    # For example : label is taken from dir name mariel and inside dir mariel there are pictures of mariel
    dataset = datasets.ImageFolder(directory)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    imgs = []
    loader = DataLoader(dataset, collate_fn=lambda x: x[0])
    for x, y in loader:
        imgs.append((x, dataset.idx_to_class[y]))  # dataset.idx_to_class[y] -> label
    return imgs


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


def draw_tensors(tensors, figsize, titles=[]):
    fig = plt.figure(figsize=figsize)
    for i, tensor in enumerate(tensors):
        subplt = fig.add_subplot(1, len(tensors), i + 1)
        if titles:
            subplt.set_title(titles[i])
        # size of tensor we are getting is (1,3,160,160) -> (3,160,160)->(160,160,3)
        plt.imshow(np.asarray(tensor.detach()[0].permute(1, 2, 0)))
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def diff_between_tensors(orig: torch.Tensor, fake: torch.Tensor):
    orig_size = orig.size()
    fake_size = fake.size()
    assert orig_size == fake_size, "input tensors Must be in the same size ! orig={}, fake={}".format(orig_size,
                                                                                                      fake_size)
    return (orig - fake).norm().item()


if __name__ == "__main__":
    run_pgd()
