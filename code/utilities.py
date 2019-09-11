import os
import sys
import pickle

import numpy as np
import skimage.external.tifffile as tifffile


def progress_bar(pos, total, length=50, newline=True):
    """Creates a string with the progression of the algorithm
        PARAMS:
            pos, total (int/float): current algo progression with max value
            lenght (int): lenght in characters of the progress bar
            newline (bool): whether or not the string has a new line when the progression bar is Done

        RETURNS:
            progress_bar (str): [{# *pos}{- *(1-pos)}]
    """
    if pos == total:
        return '[{:s}]\n'.format('#'*length) if newline else '[{:s}]'.format('#'*length)

    rel_pos = int(pos/total*length)
    return("[{0:-<{length:d}s}]".format('#'*rel_pos, length=length))


def load_img_path(folder_path, file_type='.tif'):
    """Returns the filepaths of the images in the given folder"""

    imgs_paths = [os.path.join(folder_path, filename) 
                  for filename in os.listdir(folder_path) if filename.endswith(file_type)]
    return imgs_paths


def load_imgs(imgs_paths):
    """Returns images as numpy array from the imgs_path
        PARAMS:
            'imgs_paths': path of the images to load (relative or absolute)
                type str or list (changes type of the output)
        RETURNS:    
            'images': loaded images as numpy arrays 
                single array if imgs_paths in of type str
                list of arrays if imgs_paths in of type list
    """

    if isinstance(imgs_paths, str):
        print("Loading: {}".format(imgs_paths))
        images = tifffile.imread(imgs_paths)
    elif isinstance(imgs_paths, list):
        # Returns loaded images as a list
        images = []
        for filename in imgs_paths:
            print("Loading: {}".format(filename))
            images.append(tifffile.imread(filename))
    else:
        raise ValueError("Wrong type from imgs_paths: {}".format(type(imgs_paths)))

    return images


def save_stack(stack, how='pickle', filename='temp', out_dir='outputs', **kwargs):
    """Saves stacks of images
        PARAMS:
            stack (np.array):
            how (str)
            filename (str)
            out_dir (str)
    """

    def rescale(stack, bit_depth=16):

        assert bit_depth in [8, 16], "Wrong value for bit_depth: {}".format(bit_depth)

        # Rescales to [0, 1]
        stack = stack.astype('float32')
        stack -= stack.min()
        stack /= stack.max()

        # Rescales to [0, 2**bit_depth -1] (uint8 or int16)
        if bit_depth == 8:
            stack *= (2**bit_depth -1)
        else: 
            stack *= (2**(bit_depth-1) -1)

        return stack.astype('uint{:d}'.format(bit_depth))

    def save2png(stack, axis=0, one_in_x=10, fileprefix='', out_dir=''):
        """Saves stack to multiple png images for training"""
        slc = [slice(None)] *len(stack.shape)
        for i in range(0, stack.shape[axis], one_in_x):            
            slc[axis] = slice(i, i+1)
            print("\rSaving: {}_{}.png".format(fileprefix, i+1), end=' '*10)
            tifffile.imsave(os.path.join(out_dir, "{}_{}.png".format(fileprefix, i+1)), rescale(stack[slc], 8))

    try:
        os.makedirs(out_dir, exist_ok=True)
    except:
        print("Could not create dir: '{}'".format(out_dir))
        raise 

    if how == 'pickle':
        print("Saving image as {}.pkl".format(filename))
        pickle.dump(stack, open(os.path.join(out_dir, '{}.pkl'.format(filename)), 'wb'))

    elif how.lower() in ['tiff', 'tif']:
        print("Saving image as {}.tif".format(filename))
        tifffile.imsave(os.path.join(out_dir, '{}.tif'.format(filename)), rescale(stack, 16))

    elif how == 'png':
        save2png(stack, fileprefix=filename, out_dir=out_dir, **kwargs)

    else:
        raise NotImplementedError


