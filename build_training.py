"""
Imports bright field images to compute bacteria positions and saves frame + bacteria detection to folder to be used as
U-Net training data
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import skimage.external.tifffile as tifffile

import code.processing as processing
import code.utilities as utilities


DATA_PATH = "data/"
OUT_PATH = DATA_PATH+'bf_seg'


def rescale(stack):
    "Rescales stacks and images to [0, 255] uint8"
    stack -= stack.min()
    stack = stack /stack.max()
    return (stack *255).astype('uint8')


def main():

    imgs_paths = utilities.load_img_path(DATA_PATH+'/bright_field', '.tif')

    bf_stacks = (utilities.load_imgs(path) for path in imgs_paths)

    os.makedirs(os.path.join(OUT_PATH, 'REF_FRAMES/'), exist_ok=True)
    os.makedirs(os.path.join(OUT_PATH, 'MASKS/'), exist_ok=True)

    for i, bf_stack in enumerate(bf_stacks):
        
        mask = processing.bright_field_segmentation(bf_stack)
        mask = processing.coregister(mask, np.zeros((2,)), np.zeros((2,)))

        for j in range(0, bf_stack.shape[0], 10):
            print("\rSaving to stack_{}_{}.png".format(i+1, j+1), end=' '*5)
            tifffile.imsave(os.path.join(OUT_PATH, 'REF_FRAMES/', "stack_{}_{}.png".format(i+1, j+1)), rescale(bf_stack[j]))
            tifffile.imsave(os.path.join(OUT_PATH, 'MASKS/', "mask_{}_{}.png".format(i+1, j+1)), mask[j])
        print('')


if __name__ == "__main__":
    main()