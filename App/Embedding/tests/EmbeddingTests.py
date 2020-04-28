import glob
import unittest

from PIL import Image
from torchvision import transforms
import pathlib
from app.EmbeddingWrapper import EmbeddingWrapper
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
                output,min_score,all_scores = embedded_wrapper.who_am_i(image_tensor)
                path = pathlib.PurePath(subdir)
                label = path.name
                print("\n expected : [{}] \n got : [{}] \n score : {}".format(label,output,min_score))