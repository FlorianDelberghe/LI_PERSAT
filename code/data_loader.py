import os
from random import sample

import imageio
import numpy as np
import scipy
import scipy.ndimage as ndimage
import torch


class DataLoader():

    def __init__(self, data_path, sampling=1, name='default_DataLoader'):

        self.name = name

        self.data_path = data_path
        self.sampling_interval = sampling
        
        # Sets the path for the training images
        self.ref_images = [os.path.join(data_path, 'REF_FRAMES', filename) for filename in os.listdir(os.path.join(self.data_path, 'REF_FRAMES')) if filename.endswith('.png')]
        self.masks = [os.path.join(data_path, 'MASKS', filename) for filename in os.listdir(os.path.join(self.data_path, 'MASKS')) if filename.endswith('.png')]

        self.ref_images.sort()
        self.masks.sort()

        # Converts to string arrays for easier sampling
        self.ref_images = np.array(self.ref_images)
        self.masks  =np.array(self.masks)


    def __repr__(self):
        return "{}".format(self.name)

    def __str__(self):
        return self.__str__()


    def load_files(self, path):
        """Loads all the paths of the files for the training"""
        raise NotImplementedError


    def load_batch(self, batch_size):
        """Returns a generator of images and training target
            PARAMS:
                batch_size (int): size of each batch yielded by the generator
            RETURNS:
                (generator(torch.Tensor)): returns a torch.Tensor of the batch dims = (batch_size, channels=1, image.shape[0], image.shape[1]) 
                """

        # max amount of batches
        self.n_batches = int(min(len(self.ref_images), len(self.masks)) /batch_size)
        total_samples = self.n_batches *batch_size 

        # sampling images
        ind = np.random.choice(len(self.ref_images), total_samples)

        for i in range(0, self.n_batches, self.sampling_interval):
            ref_img = [self.load_image(filename, True) for filename in self.ref_images[ind[i:i+batch_size]]]
            mask = [self.load_image(filename, True, soften=True) for filename in self.masks[ind[i:i+batch_size]]]

            yield torch.stack(ref_img, axis=0), torch.stack(mask, axis=0)


    def load_image(self, filepath, normalize=False, soften=False):
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
        if soften:
            img = ndimage.gaussian_filter(img, (0, 1,1))

        # Converts to float Tensor for training
        return torch.from_numpy(img).float()
