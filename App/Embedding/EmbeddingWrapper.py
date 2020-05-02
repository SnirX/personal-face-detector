import traceback

from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import cv2


def get_face_box(net, frame, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
    return frame_opencv_dnn, bboxes

class EmbeddingWrapper(object):
    images_dir = os.path.join(os.path.dirname(__file__), "dataset")
    cropped_images_dir = os.path.join(images_dir, "cropped")
    registered_images_dir = os.path.join(images_dir, "orig")
    test_images_dir = os.path.join(images_dir, "test")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    face_proto = "AgeGender/opencv_face_detector.pbtxt"
    face_model = "AgeGender/opencv_face_detector_uint8.pb"
    faceNet = cv2.dnn.readNet(face_model, face_proto)
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
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
                EmbeddingWrapper.save_image_on_disk(x, dest_dir_path)

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
        save_image(img, os.path.join(dest, "{}.jpg".format(number_files + 1)))

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
            # print("name : {} , vector : {}".format(name, embeddings[index]))
            self.name2vector[name].add(embeddings[index])
        return embeddings

    '''
    The function register new person into our memory db.
    params : 
        name - name of person
        imgs - list of images of the preson
    '''

    def register_person(self, name, imgs: list, batch=False):
        cropped_imgs = []
        dest = os.path.join(EmbeddingWrapper.cropped_images_dir, name)
        if batch:
            cropped_imgs, prob = EmbeddingWrapper.mtcnn(imgs, return_prob=True)
            EmbeddingWrapper.save_images_on_disk(cropped_imgs, dest)
        else:
            for img in imgs:
                frame_face, bboxes = get_face_box(self.faceNet, img)
                if len(bboxes) == 1:
                    # cropped_img, prob = EmbeddingWrapper.mtcnn(img, return_prob=True)
                    face = img[max(0, bboxes[0][1] - 20):min(bboxes[0][3] + 20, img.shape[0] - 1), max(
                        0, bboxes[0][0] - 20):min(bboxes[0][2] + 20, img.shape[1] - 1)]
                    face = transforms.ToTensor()(Image.fromarray(cv2.resize(face, (160, 160))))
                    cropped_imgs.append(face)
                    EmbeddingWrapper.save_image_on_disk(img=face, dest=dest)
        if len(cropped_imgs) == 0:
            print("Didnt found faces in the images of the person,nothing to register")
            return None

        print("Start calc vectors for {} images of {}".format(len(cropped_imgs),name))
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
