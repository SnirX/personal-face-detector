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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    def __init__(self):
        self.name2vector = {}  # the key is the name , the value is set of embeddeing vectors

    @staticmethod
    def collate_fn(x):
        return x[0]

    '''
    The function loads into memory embedded vectors that it generates from cropped images in cropped_images_dir
    '''

    # load into memory images that were already cropped
    def load_cropped_images(self):
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

    '''
    
    '''

    def load_orig_images(self):
        names, cropped_images = self.crop_orig_images(EmbeddingWrapper.registered_images_dir,
                                                      EmbeddingWrapper.cropped_images_dir)
        self.___generate_embedding_vectors_and_save_in_mem(names, cropped_images)

    '''
    The functions crops all images in registered_images_dir directory and saves them in cropped_images_dir
    params : 
            imgs_src_dir - the dir that contains all directories of orig_images.
                           The directories structure must be : img_src_dir/person_name/img1 , img_src_dir/person_name/img2 etc ...
            imgs_dest_dir - the location the cropped images will be saved in.
                            The directories structure will be : img_dst_dir/person_name/img1, img_dst_dir/person_name/img2
    '''

    @staticmethod
    def crop_orig_images(imgs_src_dir, imgs_dst_dir):
        aligned = []
        names = []
        workers = 0 if os.name == 'nt' else 4
        dataset = datasets.ImageFolder(imgs_src_dir, )
        dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=EmbeddingWrapper.collate_fn, num_workers=workers)
        for x, y in loader:
            print("working on image of {}".format(dataset.idx_to_class[y]))
            x, prob = EmbeddingWrapper.mtcnn(x, return_prob=True)  # mtcnn returns a tensor with size (3,160,160)
            if x is not None:
                print('Face detected with probability: {:8f}'.format(prob))
                aligned.append(x)
                name = dataset.idx_to_class[y]
                names.append(name)

                #  Save the cropped face
                dest_dir_path = "{}/{}/".format(imgs_dst_dir, name)
                os.makedirs(os.path.dirname(dest_dir_path), exist_ok=True)
                list_files = os.listdir(dest_dir_path)  # dir is your directory path
                number_files = len(list_files)
                save_image(x, "{}/{}.jpg".format(dest_dir_path, number_files + 1))
            else:
                print("face wasnt detected")

        return names, aligned

    '''
    The functions gets as input cropped images and labels and saves embedded vectors in memory
    params : 
            names:list of labels
            cropped_images:list of cropped images
    '''

    def ___generate_embedding_vectors_and_save_in_mem(self, names, cropped_images):
        cropped_images = torch.stack(cropped_images).to(EmbeddingWrapper.device)
        embeddings = EmbeddingWrapper.resnet(cropped_images).detach().cpu()
        for index in range(embeddings.size()[0]):
            name = names[index]
            if name not in self.name2vector:
                self.name2vector[name] = set()
                self.name2vector[name].add(embeddings[index])
            else:
                self.name2vector[name].add(embeddings[index])

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
        embedded_tensor = EmbeddingWrapper.resnet(tensor).detach().cpu()
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

        if min_avg > threshold:
            min_name = "unknown"

        return min_name, min_avg, scores
