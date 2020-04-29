import unittest

from PIL import Image
from torchvision import transforms
import pathlib
from App.Embedding.EmbeddingWrapper import EmbeddingWrapper
import os


class EmbeddingTests(unittest.TestCase):

    def test_load_orig_dataset(self):
        embedded_wrapper = EmbeddingWrapper()
        embedded_wrapper.load_orig_images()
        for name, value in embedded_wrapper.name2vector.items():
            assert len(value) != 0, "key {} exist but no embedded vectors found ".format(name)

    def test_load_cropped_dataset(self):
        embedded_wrapper = EmbeddingWrapper()
        embedded_wrapper.load_cropped_images()
        for name, value in embedded_wrapper.name2vector.items():
            assert len(value) != 0, "key {} exist but no embedded vectors found ".format(name)
        return embedded_wrapper

    def test_who_in_picture(self):
        embedded_wrapper = self.test_load_cropped_dataset()
        for subdir, dirs, files in os.walk(EmbeddingWrapper.test_images_dir):
            for filename in files:
                filepath = subdir + os.sep + filename
                image_tensor = transforms.ToTensor()(Image.open(filepath))
                output, min_score, all_scores = embedded_wrapper.who_am_i(image_tensor)
                path = pathlib.PurePath(subdir)
                label = path.name
                print("\n expected : [{}] \n got : [{}] \n score : {}".format(label, output, min_score))

    def test_crop_images(self):
        imgs_dst_dir = EmbeddingWrapper.cropped_images_dir
        EmbeddingWrapper.crop_orig_images(imgs_src_dir=EmbeddingWrapper.registered_images_dir,
                                          imgs_dst_dir=imgs_dst_dir)
        assert len(
            os.listdir(imgs_dst_dir)) != 0, "{} should have directry for each person but seems it is empty".format(
            imgs_dst_dir)

    def test_register_person(self, batch=False, name='mariel'):
        embedded_wrapper = EmbeddingWrapper()
        imgs = []
        imgs_path = os.path.join(EmbeddingWrapper.registered_images_dir, name, '')

        for subdir, dirs, files in os.walk(imgs_path):
            for filename in files:
                filepath = subdir + os.sep + filename
                imgs.append(Image.open(filepath))
        embedded_wrapper.register_person(batch=batch, name=name, imgs=imgs)
        assert len(
            embedded_wrapper.name2vector[name]) > 0, "Expected to have vectors inside name2vector but found nothing"

    def test_register_person_batch(self):
        self.test_register_person(batch=True, name='mariel_batch')
