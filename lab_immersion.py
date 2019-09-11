import os
import sys
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal
import skimage
import skimage.external.tifffile as tifffile
import skimage.feature
import scipy.ndimage.morphology as morphology
import torch
from scipy import fftpack
from skimage import exposure, measure
from skimage.filters import threshold_minimum, threshold_triangle
from skimage.feature import canny

import code.processing as processing
import code.utilities as utilities

# Some conv function from scipy/skimage output FutureWarnings
warnings.simplefilter(action='ignore', category=(FutureWarning))        

"""
TODO:
    -PCA feature extraction
    -Wavelets denoising
"""

DATA_PATH = "data/"
OUT_PATH = "outputs/"
FRAMERATE = 5


def main():

    #%% -------------------- ISCAT contrat enhancement -------------------- %%#

    imgs_paths = utilities.load_img_path(DATA_PATH+'ISCAT/')
    stack = utilities.load_imgs(imgs_paths[0])

    # Median filtering and Normalization 
    stack = processing.image_correction(stack)

    # Contrast enhancement
    stack = processing.enhance_contrast(stack, 'stretching', (1, 99))

    # Range of non filtered elements [px]
    min_size, max_size = 1, 13
    # Fourier filtering of image
    stack = processing.fft_filtering(stack, min_size, max_size, True, 'horizontal & vertical')
    stack -= stack.min()
    stack /= stack.max()
    
    utilities.save_stack(stack, 'tiff', 'fft_filtered', OUT_PATH)        


    fig = plt.figure()
    tifffile.imshow(stack, cmap='gray', figure=fig)
    plt.show()


if __name__ == '__main__':
    main()
