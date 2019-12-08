import os
from random import sample

import imageio
import numpy as np
import scipy
import scipy.ndimage as ndimage
import torch


class DataLoader():
    """Dataloader class for UNet training
        ARGUMENTS:
            data_path (str):
            sampling (int):
            name (str):
            kwargs: argument to be passed to functions called in __init__"""

    def __init__(self, data_path, sampling=1, name='default_DataLoader', **kwargs):

        self.name = name

        self.data_path = data_path
        self.sampling_interval = sampling
        
        self.build_train_val_test(data_path, **kwargs)        


    def __repr__(self):
        return "{}".format(self.name)

    def __str__(self):
        return self.__repr__()


    def load_files(self, path):
        """Loads all the paths of the images for the dataset"""

        return [os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith('.png')]


    def build_train_val_test(self, data_path, test_frac=0.10, val_frac=0.10, exclusive_val=False, **kwargs):
        """Loads all the images then, splits them into train, test and validation sets"""

        ref_images = self.load_files(os.path.join(data_path, 'REF_FRAMES'))
        masks = self.load_files(os.path.join(data_path, 'MASKS'))

        if len(ref_images) != len(masks):
            print("Training images and target have different size, images may notbe paired properly")

        ref_images.sort()
        masks.sort()

        # Converts to string arrays for easier sampling
        ref_images = np.array(ref_images[::self.sampling_interval])
        masks = np.array(masks[::self.sampling_interval])

        n_images = ref_images.shape[0]
        test_ids = np.random.randint(0, n_images, int(n_images *test_frac))

        self.test_images = ref_images[test_ids]
        self.test_masks = masks[test_ids]

        ref_images = np.delete(ref_images, test_ids)
        masks = np.delete(masks, test_ids)

        val_ids = np.random.randint(0, ref_images.shape[0], int(n_images *(1 -test_frac) *val_frac))

        self.val_images = ref_images[val_ids]
        self.val_masks = masks[val_ids]

        if exclusive_val:
            ref_images = np.delete(ref_images, val_ids)
            masks = np.delete(masks, val_ids)

        self.train_images = ref_images
        self.train_masks = masks


    def load_batch(self, batch_size):
        """Returns a generator of images and training target
            PARAMS:
                batch_size (int): size of each batch yielded by the generator

            RETURNS:
                (generator(torch.Tensor)): returns a torch.Tensor of the batch dims = (batch_size, channels=1, image.shape[0], image.shape[1]) 
                """

        # max amount of batches
        self.n_batches = int(min(len(self.train_images), len(self.train_masks)) /batch_size)
        total_samples = self.n_batches *batch_size 

        # sampling images
        ind = np.random.choice(len(self.train_images), total_samples)

        for i in range(0, self.n_batches, batch_size):
            ref_img = [self.load_image(filename) for filename in self.train_images[ind[i:i+batch_size]]]
            mask = [self.load_image(filename, soften=True) for filename in self.train_masks[ind[i:i+batch_size]]]

            yield np.stack(ref_img, axis=0), np.stack(mask, axis=0)


    def load_test(self):
        """Returns a stack of all the test images"""

        ref_img = [self.load_image(filename) for filename in self.test_images]
        mask = [self.load_image(filename) for filename in self.test_masks]

        return np.stack(ref_img, axis=0), np.stack(mask, axis=0)


    def load_image(self, filepath, normalize=True, soften=False):
        """Loads image from path
            PARAMS:
                filepath (str): path where the image is located (supports .png, .jpg, ...)
                normalize (bool): whether to normalize the output to [0,1]
                soften (bool): if true applies a guassian blur to the loaded image

            RETURNS:
                (torch.Tensor): float Tensor form the image, dims = (channels=1,  image.shape[0], image.shape[1])
                """
                
        #loads image and add channel dim
        img = np.expand_dims(imageio.imread(filepath).astype('float32'), axis=0)

        # Normalize for Net input
        if normalize and not img.max() == 0:
            img -= img.min()
            img /= img.max()

        # Gaussian blur on the images (mask) gives better perf as training target
        if soften and (img != 0).any():
            img = ndimage.gaussian_filter(img, (0,2,2))
            img /= img.max()

        return img
