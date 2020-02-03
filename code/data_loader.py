import os
from random import sample

import imageio
import numpy as np
import scipy
import scipy.ndimage as ndimage


class DataLoader():
    """Dataloader class for UNet training, loads training images and masks from folders
        ARGUMENTS:
            data_path (str): path to the folder containing 'MASK' folder with training target and 'REF_FRAMES' folder with the training images
            sampling (int): sampling interval (>=1) of the images to reduce training set size
            name (str): name of the Loader for debugging purposes

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
        """Loads all the files of the images for the dataset"""

        return [os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith('.png')]


    def build_train_val_test(self, data_path, test_frac=0.10, val_frac=0.10, exclusive_val=True, **kwargs):
        """Loads all the images then, splits them into train, test and validation sets"""

        ref_images = self.load_files(os.path.join(data_path, 'REF_FRAMES'))
        masks = self.load_files(os.path.join(data_path, 'MASKS'))

        if len(ref_images) != len(masks):
            print("Training images and target have different size, images may not be paired properly")

        ref_images.sort()
        masks.sort()

        # Converts to string arrays for easier sampling
        ref_images = np.array(ref_images[::self.sampling_interval])
        masks = np.array(masks[::self.sampling_interval])

        n_images = min(ref_images.shape[0], masks.shape[0])
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
                (generator(np.array)): returns an array of the batch dims = (batch_size, channels=1, image.shape[0], image.shape[1]) 
        """

        # max amount of batches
        self.n_batches = int(min(len(self.train_images), len(self.train_masks)) //batch_size)
        total_samples = self.n_batches *batch_size 

        # sampling images
        ind = np.random.choice(len(self.train_images), total_samples)

        for i in range(self.n_batches):
            ref_img = [self.load_image(filename) for filename in self.train_images[ind[i:i+batch_size]]]
            mask = [self.load_image(filename, soften=True) for filename in self.train_masks[ind[i:i+batch_size]]]

            ref_img, mask = np.stack(ref_img, axis=0), np.stack(mask, axis=0)

            # kinda Data augmentation
            for i in range(batch_size):
                if i % 4 == 1:
                    ref_img[i] = ref_img[i,...,::-1,:]
                    mask[i] = mask[i,...,::-1,:]
                    continue

                if i % 4 == 2:
                    ref_img[i] = ref_img[i,...,::-1]
                    mask[i] = mask[i,...,::-1]
                    continue

                if i % 4 == 3:
                    ref_img[i] = ref_img[i,...,::-1,::-1]
                    mask[i] = mask[i,...,::-1,::-1]
                    continue

            yield ref_img, mask


    def load_test(self, size=None):
        """Returns a stack of all the test images"""

        if size is None:
            ref_img = [self.load_image(filename) for filename in self.test_images]
            mask = [self.load_image(filename) for filename in self.test_masks]

        else:
            if  size < 1 or size > len(self.test_images):
                raise ValueError(f"Tried to load {size} val images, must be in [1, {len(self.test_images)}]")
            
            ind = np.random.randint(0, len(self.val_images), size)

            ref_img = [self.load_image(filename) for filename in self.test_images[ind]]
            mask = [self.load_image(filename) for filename in self.test_masks[ind]]

        return np.stack(ref_img, axis=0), np.stack(mask, axis=0)

    
    def load_val(self, size=1):
        """Returns a random validation image"""

        if size < 1 or size > len(self.val_images):
            raise ValueError(f"Tried to load {size} val images, must be in [1, {len(self.val_images)}]")

        ind = np.random.randint(0, len(self.val_images), size)
        
        ref_img = [self.load_image(filename) for filename in self.val_images[ind]]
        mask = [self.load_image(filename) for filename in self.val_masks[ind]]

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
                
        # Loads image and add channel dim
        img = imageio.imread(filepath).astype('float32')
        img = np.expand_dims(img, axis=0)

        # Normalize for Net input
        if normalize and (img != 0).any():
            img -= img.min()
            img /= img.max()

        # Gaussian blur on the images (mask) gives better perf as training target
        if soften and (img != 0).any():
            img = ndimage.gaussian_filter(img, (0,2,2))
            img /= img.max()

        return img
