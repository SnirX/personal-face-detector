from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os


class EmbeddingWrapper():
    images_dir = "C:/Users/Mariel.Cherkassky/Desktop/College/DL/personal-face-detector/App/Embedding/dataset"
    cropped_images_dir = "{}/cropped".format(images_dir)
    registered_images_dir = "{}/orig".format(images_dir)
    test_images_dir = "{}/test".format(images_dir)

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )
        self.name2vector = {}  # the key is the name , the value is set of embeddeing vectors

    def __collate_fn(self, x):
        return x[0]

    # load into memory images that were already cropped
    def load_cropped_images(self):
        aligned = []
        names = []
        workers = 0 if os.name == 'nt' else 4
        dataset = datasets.ImageFolder(EmbeddingWrapper.cropped_images_dir)
        dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=self.__collate_fn, num_workers=workers)
        for x, y in loader:
            print("working on image of {}".format(dataset.idx_to_class[y]))
            x = transforms.ToTensor()(x)
            aligned.append(x)
            name = dataset.idx_to_class[y]
            names.append(name)

        aligned = torch.stack(aligned).to(self.device)
        embeddings = self.resnet(aligned).detach().cpu()
        self.save_vectors_in_mem(names, embeddings)

    def load_orig_images(self):
        aligned = []
        names = []
        workers = 0 if os.name == 'nt' else 4
        dataset = datasets.ImageFolder(EmbeddingWrapper.registered_images_dir)
        dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=self.__collate_fn, num_workers=workers)
        for x, y in loader:
            print("working on image of {}".format(dataset.idx_to_class[y]))
            x, prob = self.mtcnn(x, return_prob=True)  # mtcnn returns a tensor with size (3,160,160)
            if x is not None:
                print('Face detected with probability: {:8f}'.format(prob))
                aligned.append(x)
                name = dataset.idx_to_class[y]
                names.append(name)

                #  Save the cropped face
                dest_dir_path = "{}/{}/".format(EmbeddingWrapper.cropped_images_dir, name)
                os.makedirs(os.path.dirname(dest_dir_path), exist_ok=True)
                list_files = os.listdir(dest_dir_path)  # dir is your directory path
                number_files = len(list_files)
                save_image(x, "{}/{}.jpg".format(dest_dir_path, number_files + 1))
            else:
                print("face wasnt detected")

        aligned = torch.stack(aligned).to(self.device)
        embeddings = self.resnet(aligned).detach().cpu()
        self.save_vectors_in_mem(names, embeddings)

    def save_vectors_in_mem(self, names, embeddings):
        for index in range(embeddings.size()[0]):
            name = names[index]
            if name not in self.name2vector:
                self.name2vector[name] = set()
                self.name2vector[name].add(embeddings[index])
            else:
                self.name2vector[name].add(embeddings[index])

    def who_am_i(self, tensor: torch.Tensor):
        tensor = torch.stack([tensor]).to(self.device)
        scores = {key: {'score': 0, 'avg': 0, 'num_vectors': len(self.name2vector[key])} for key in
                  self.name2vector.keys()}
        embedded_tensor = self.resnet(tensor).detach().cpu()
        for key, embedded_vectors in self.name2vector.items():
            for embedded_vector in embedded_vectors:
                scores[key]['score'] = scores[key]['score'] + (embedded_tensor - embedded_vector).norm().item()

        min_avg = 1000
        min_name = ""
        for key in scores:
            avg = scores[key]['score'] / scores[key]['num_vectors']
            scores[key]['avg'] = avg
            if avg < min_avg:
                min_avg = avg
                min_name = key

        return min_name,min_avg,scores
