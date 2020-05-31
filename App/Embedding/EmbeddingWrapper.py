import os
import logging
import traceback

import cv2
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from App.AgeGender.FaceModelWrapper import FaceModelWrapper
from Utils.torch_utils import get_torch_device


class EmbeddingWrapper(object):
    images_dir = os.path.join(os.path.dirname(__file__), "dataset")
    cropped_images_dir = os.path.join(images_dir, "cropped")
    registered_images_dir = os.path.join(images_dir, "orig")
    test_images_dir = os.path.join(images_dir, "test")
    device = get_torch_device()
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device).train(False)
    faceNet = FaceModelWrapper()
    name2vector = {}  # the key is the name , the value is set of embeddeing vectors
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmbeddingWrapper, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    @staticmethod
    def collate_fn(x):
        return x[0]

    '''
    The function loads into memory embedded vectors that it generates from cropped images in cropped_images_dir
    '''

    # load into memory images that were already cropped
    def load_cropped_images(self):
        print("\n------- Starting loading cropped images--------")
        try:
            aligned = []
            names = []
            workers = 0 if os.name == 'nt' else 4
            dataset = datasets.ImageFolder(EmbeddingWrapper.cropped_images_dir)
            dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
            loader = DataLoader(dataset, collate_fn=EmbeddingWrapper.collate_fn, num_workers=workers)
            for x, y in loader:
                print("working on image of {}".format(dataset.idx_to_class[y]))
                x = transforms.ToTensor()(x)
                aligned.append(x)
                name = dataset.idx_to_class[y]
                names.append(name)

            self.___generate_embedding_vectors_and_save_in_mem(names, aligned)
        except Exception as e:
            print("error loading cropped images.")
            print(traceback.format_exc())

        print("------- Finished loading cropped images--------")

    '''

    '''

    def load_orig_images(self):
        print("\n------- Starting loading orig images--------")
        names, cropped_images = self.crop_orig_images(EmbeddingWrapper.registered_images_dir,
                                                      EmbeddingWrapper.cropped_images_dir)
        self.___generate_embedding_vectors_and_save_in_mem(names, cropped_images)
        print("------- Finished loading orig images--------")

    '''
    The functions crops all images in registered_images_dir directory and saves them in cropped_images_dir
    params : 
            imgs_src_dir - the dir that contains all directories of orig_images.
                           The directories structure must be : img_src_dir/person_name/img1 , img_src_dir/person_name/img2 etc ...
            imgs_dest_dir - the location the cropped images will be saved in.
                            The directories structure will be : img_dst_dir/person_name/img1, img_dst_dir/person_name/img2
    '''

    def crop_orig_images(self, imgs_src_dir, imgs_dst_dir):
        aligned = []
        names = []
        workers = 0 if os.name == 'nt' else 4
        dataset = datasets.ImageFolder(imgs_src_dir, )
        dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=EmbeddingWrapper.collate_fn, num_workers=workers)
        for x, y in loader:
            print("working on image of {}".format(dataset.idx_to_class[y]))

            _, faces, _ = self.faceNet.get_face_box(x)
            if faces:
                name = dataset.idx_to_class[y]
                names.append(name)
                x = faces[0]
                dest_dir_path = os.path.join(imgs_dst_dir, name)
                EmbeddingWrapper.save_image_on_disk(cv2.resize(x, (160, 160)), dest_dir_path)
                x = transforms.ToTensor()(Image.fromarray(cv2.resize(x, (160, 160))))
                print('Face was detected')
                aligned.append(x)


            else:
                print("face wasnt detected")

        return names, aligned

    @staticmethod
    def save_images_on_disk(imgs, dest):
        for img in imgs:
            EmbeddingWrapper.save_image_on_disk(img, dest)

    '''
    params : 
        img - PILImage obj
        dest - destination of img, without its name only until parent dir.
    '''

    @staticmethod
    def save_image_on_disk(img, dest):
        os.makedirs(dest, exist_ok=True)
        list_files = os.listdir(dest)  # dir is your directory path
        number_files = len(list_files)
        from PIL import Image
        img = Image.fromarray(img)
        img.save(os.path.join(dest, "{}.jpg".format(number_files + 1)))
        # save_image(img, os.path.join(dest, "{}.jpg".format(number_files + 1)))

    '''
    The functions gets as input cropped images and labels and saves embedded vectors in memory
    params : 
            names:list of labels
            cropped_images:list of cropped images
    '''

    def ___generate_embedding_vectors_and_save_in_mem(self, names, cropped_images: list):
        cropped_images = torch.stack(cropped_images).to(EmbeddingWrapper.device)
        embeddings = self.get_embedding_by_tensor(cropped_images)
        for index in range(embeddings.size()[0]):
            name = names[index]
            if name not in self.name2vector:
                self.name2vector[name] = set()
            # print("name : {} , vector : {}".format(name, embeddings[index]))
            self.name2vector[name].add(embeddings[index])
        return embeddings

    '''
    The function register new person into our memory db.
    params : 
        name - name of person
        imgs - list of images of the person
    '''

    def register_person(self, name, imgs: list):
        cropped_imgs = []
        dest = os.path.join(EmbeddingWrapper.cropped_images_dir, name)

        for img in imgs:
            _, faces, _ = self.faceNet.get_face_box(img)
            if faces and len(faces) == 1:
                face = faces[0]
                face = cv2.resize(face, (160, 160))
                cropped_imgs.append(transforms.ToTensor()(Image.fromarray(face)))
                EmbeddingWrapper.save_image_on_disk(img=face, dest=dest)

        if len(cropped_imgs) == 0 and faces:
            print("Didnt found faces in the images of the person,nothing to register")
            return None

        if len(faces) > 1:
            print("Please make sure that the images have only one face inside")
            return None

        print("Start calc vectors for {} images of {}".format(len(cropped_imgs), name))
        return self.___generate_embedding_vectors_and_save_in_mem([name] * len(cropped_imgs), cropped_imgs)

    '''
    The functions recognizes who is in the image (tensor)
    params : 
            tensor - tensor of image in dimensions (3,160,160)
    return : 
            min_name - the label of the person with min distance
            min_avg - the min avg distance
            scores - dict of scores 
    '''

    def who_am_i(self, tensor: torch.Tensor, threshold=0.8) -> (str, float, dict):
        assert tensor.size() == (3, 160, 160), "input tensor to function should be in the following dims : (3,160,160)"
        tensor = torch.stack([tensor]).to(EmbeddingWrapper.device)
        scores = {key: {'score': 0, 'avg': 0, 'num_vectors': len(self.name2vector[key])} for key in
                  self.name2vector.keys()}
        embedded_tensor = self.get_embedding_by_tensor(tensor)
        for key, embedded_vectors in self.name2vector.items():
            for embedded_vector in embedded_vectors:
                scores[key]['score'] = scores[key]['score'] +\
                                       self.get_distance_between_embeddings(embedded_tensor, embedded_vector)

        min_avg = 1000
        min_name = ""
        for key in scores:
            avg = scores[key]['score'] / scores[key]['num_vectors']
            scores[key]['avg'] = avg
            if avg < min_avg:
                min_avg = avg
                min_name = key

        if min_avg > threshold:
            min_name = "unknown"

        return min_name, min_avg, scores

    def get_embeddings_by_label(self, label):
        if label in self.name2vector.keys():
            return self.name2vector.get(label)
        logging.info("No embeddings for label - {}".format(label))
        return set()

    def get_embedding_by_tensor(self, tensor: torch.Tensor):
        return self.resnet(tensor).detach().cpu()

    def get_mean_embedding_of_embedding_set(self, embeddings_as_set: set):
        embeddings_sum = torch.FloatTensor([0] * 512).to(self.device)  # ResNet last layer is 512
        for embedding in embeddings_as_set:
            embeddings_sum += embedding
        return embeddings_sum / len(embeddings_as_set)

    @staticmethod
    def get_distance_between_embeddings(embedding_1, embedding_2):
        return (embedding_1 - embedding_2).norm().item()
