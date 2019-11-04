import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
import skimage.external.tifffile as tifffile
import scipy.ndimage as ndimage
import scipy.ndimage.morphology as morphology
from scipy import fftpack
from skimage import exposure

import code.utilities as utilities
from  code.utilities import progress_bar, save_stack

# Some conv function from scipy/skimage output FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def structural_element(shape, size):
    """Build a 3D structural element to be used for morphological operations
        PARAMS:
            shape (str): shape of the structural element support 'square', 'circle', 'cross', diamond'
            size (tuple): size in the 3 dimensions of the element
        RETURNS:
            element (np.array): structural element"""

    if shape == 'square':
        element = np.ones(size[1:])
        
    elif shape == 'cross':
        element = np.zeros(size[1:])
        element[size[1] //2, :] = 1
        element[:, size[2] //2] = 1
        
    elif shape == 'circle':
        element = np.zeros(size[1:])
        for i in range(size[1]):
            for j in range(size[2]):
                if (i -size[1] //2) **2 + (j -size[2] //2)**2 <= (size[1] //2) **2:
                    element[i, j] = 1
                    
    elif shape == 'diamond':
        element = np.zeros(size[1:])
        for i in range(size[1]):
            for j in range(size[2]):
                if ((i+1 -size[1] //2)  + (j+1 -size[2] //2)) < size[1] //2 :
                    element[i, j] = 1

    return np.stack([element] *size[0], axis=0)


def median_image(stack, method=None, sample_size=100, niter=100, progress=False):
    """Computes the median of a stack of images 
        PARAMS:
            stack (np.array): input array to compute the median of
            method (str): median computaion method, supports 'iter', 'bootstrap', default=None uses np.median
            sample_size (int): size of the sample for the iterative and bootstraping method
            niter (int): max iter for iterative method
            progress (bool): prints a progress bar
        RETURNS: 
            median_image (np.array): median of the input array
    """

    print("Computing median image with method: {}...".format(method), end='\n')

    stack_len = stack.shape[0]

    if method == 'iter':
        medians = []
        if progress: print("\r{}".format(progress_bar(0, stack_len)), end='')

        for i in range(0, stack_len, sample_size):
            # np.median() returns float64 for concerns of memory conservation, we convert to int16
            medians.append(np.median(stack[i:i +sample_size], axis=0).astype(np.int16))

            if progress: print("\r{}".format(progress_bar(i+1, stack_len)), end='')

        median = np.median(np.stack(medians, axis=0), axis=0).astype(np.int16)

    elif method == 'bootstrap':
        medians = []
        if progress: print("\r{}".format(progress_bar(0, niter)), end='')

        for i in range(niter):
            ind = np.random.choice(stack_len, sample_size)
            medians.append(np.median(stack[ind], axis=0).astype(np.int16))

            if progress: print("\r{}".format(progress_bar(i+1, niter)), end='')

        median = np.median(np.stack(medians, axis=0), axis=0).astype(np.int16)

    else:
        median = np.median(stack, axis=0)

    return median


def normalize_stack(stack, progress=False):
    """Normalizes stack slice by slice inplace operation
        PARAMS:
            stack (np.array): numpy array to be normalized along axis 0 (dtype=float32)
            progress (bool): print a progress bar if true
        RETURNS:
            stack (np.array): normalized array (dtype=float32)
    """

    print("Normalizing stack...", end='\n')

    if progress: print("\r{}".format(progress_bar(0, stack.shape[0])), end='')

    for i in range(stack.shape[0]):
        # Rescales stack to [0, 1]
        stack[i] += np.abs(stack[i].min()) if stack[i].min() < 0 else 0
        stack[i] /= np.abs(stack[i]).max()
        if progress: print("\r{}".format(progress_bar(i+1, stack.shape[0])), end='')
        
    return stack


def image_correction(image, save_as=None):
    """Compputes the median of the input stack, then substracts it and normalize to [0,1]"""

    # Makes sure that we will not be overflow errors
    assert image.min() >= 0 and image.max() <= 2**15 -1, "min:{}, max:{}".format(image.min(), image.max())
    image = image.astype('int16')

    median_img = median_image(image, 'bootstrap', sample_size=40, niter=20)

    print("Centering...")
    image -= median_img
    image = normalize_stack(image.astype('float32'))

    if save_as is None:
        return image
    else:
        raise NotImplementedError
        utilities.save_stack(image, how=save_as, filename=filename[:-4]+'_corr')


def enhance_contrast(stack, method='stretching', progress=True, **kwargs):
    """Enhaces image contrast defaults to contrast stretching with percentile = (2, 98)"""

    def enhance_contrast_img(image, method=method, percentile=(2, 98), clip_limit=0.009, **kwargs):
        """TODO: stacks with none dim would be easier"""

        if method == 'stretching':
            p_low, p_high = np.percentile(image.ravel(), percentile)
            contrast_img = exposure.rescale_intensity(image, in_range=(p_low, p_high))
            return contrast_img

        elif method == 'hist-equal':
            contrast_img = exposure.equalize_hist(image)
            return contrast_img

        elif method == 'adapt-hist-equal':
            """TODO: FIX THIS !"""
            raise NotImplementedError
            image[image == 0] = np.finfo(np.float32).eps
            contrast_img = exposure.equalize_adapthist(image, clip_limit)
            return contrast_img

        else:
            raise ValueError("'{}' is not a valid contrast rescaling method".format(method))


    print("Enhancing contrast with method: {}".format(method))
    #Contrast enhancment for stacks
    if len(stack.shape) == 3:

        contrast_stack = np.empty_like(stack)
        if progress: print("\r{}".format(progress_bar(0, stack.shape[0])), end='')

        if method == 'stretching':
            for i in range(stack.shape[0]):
                contrast_stack[i] = enhance_contrast_img(stack[i], method, **kwargs)
                if progress: print("\r{}".format(progress_bar(i+1, stack.shape[0])), end='')

        else:
            # Conversion to float image
            if stack.min() < 0 or stack.max() > 1:
                contrast_stack = stack.astype('float32') -stack.min()
                contrast_stack /= contrast_stack.max()
            else:
                contrast_stack = stack.astype('float32')

            for i in range(stack.shape[0]):
                contrast_stack[i] = enhance_contrast_img(contrast_stack[i], method, **kwargs)
                if progress: print("\r{}".format(progress_bar(i+1, stack.shape[0])), end='')
            
    # Contrast enhancment for single images
    elif len(stack.shape) == 2:
        contrast_stack = enhance_contrast_img(stack, method, **kwargs)

    return contrast_stack


def temporal_denoising(stack, method='mean', window_size=5, progress=True):

    temporal_denoise = np.empty(stack[window_size-1:,...].shape, dtype=np.uint16)
    print("Temporal denoising...")
    if progress: print("\r{}".format(progress_bar(0, stack.shape[0])), end='')
    if method == 'mean':
        for i in range(stack.shape[0]-window_size-1):
            temporal_denoise[i] = stack[i:i+window_size].mean(axis=0).astype(np.uint16)
            if progress: print("\r{}".format(progress_bar(i+1, stack.shape[0]-window_size-1)), end='')
        return temporal_denoise
    elif method == 'median':
        for i in range(stack.shape[0]-window_size-1):
            temporal_denoise[i] = np.median(stack[i:i+window_size], axis=0).astype(np.uint16)
            if progress: print("\r{}".format(progress_bar(i+1, stack.shape[0]-window_size-1)), end='')
        return temporal_denoise
    else:
        raise NotImplementedError


def fft_filtering(stack, min_size, max_size, keep_mean=True, remove_stripes='horizontal', progress=True, *args, **kwargs):
    """Frequency filtering of the image with FFT
        PARAMS:
            stack, min_size, max_size, keep_mean=True, remove_stripes='horizontal', progress=True
        RETURNS:
            filt_stack (np.array): fitered stack
            
        TODO: temporal (3D) FFT"""

    def circular_mask(size, radius, smooth=True,  sigma=40, center=None):
        """Creates a circle in the middle of an image"""

        mask = np.zeros(size)
        if center is None:
            center = (size[0]//2, size[1]//2)
        X, Y = np.ogrid[:size[0], :size[1]]
        dist = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
        mask[dist <= radius] = 1
        
        if smooth:
            mask = ndimage.gaussian_filter(mask, (radius)/5, mode='nearest')
        return mask

    def gaussian_kernel(size, sigma):
        """Build a gaussian kernel of shape (size, size) and standard dev sigma"""

        x, y = np.meshgrid(np.arange(-size[0]//2, size[0]//2), np.arange(-size[1]//2, size[1]//2))
        gaussian = 1 /(np.sqrt(2*np.pi) *sigma) *np.exp(-1/(2*sigma**2) *(x**2 + y**2) )
        return gaussian /gaussian.max()

    def center_fft(im_fft):
        """Rearanges the quadrant of the fft or the filters so that the zero frequency is in the middle"""

        H, W = im_fft.shape
        centered_fft = np.empty_like(im_fft)
        centered_fft[0:H//2,0:W//2] = im_fft[-H//2:,-W//2:]
        centered_fft[-H//2:,-W//2:] = im_fft[0:H//2,0:W//2]
        centered_fft[0:H//2,-W//2:] = im_fft[-H//2:,0:W//2]
        centered_fft[-H//2:,0:W//2] = im_fft[0:H//2,-W//2:]
        return centered_fft


    print("Filtering structure (small, large) = ({}, {}) [px]".format(min_size, max_size))

    # Computes fourier domain frequencies from min/max sizes of allowed structuring elements
    f_low, f_high = (stack.shape[1]/2)/max_size, (stack.shape[1]/2)/min_size

    # Band pass filter
    fft_filter = gaussian_kernel(stack[0].shape, f_high) - gaussian_kernel(stack[0].shape, f_low)
    fft_filter = center_fft(fft_filter)    

    if remove_stripes is not None:
        width = 5
        x = np.stack([np.arange(-(width//2), width//2 +1)] *stack.shape[-1], axis=-1)
        gauss_1d = np.exp(-0.5 *np.abs(x)**2 *2**2)
        # Currently sets lowest freq to 0
        gauss_1d /= gauss_1d.max()

        if 'horizontal' in remove_stripes:
            fft_filter[:, :(width //2) +1] = fft_filter[:, :(width //2) +1] *(1 -gauss_1d[(width //2):].T)
            fft_filter[:, -(width //2):] = fft_filter[:, -(width //2):] *(1 -gauss_1d[:(width //2)].T)

        if 'vertical' in remove_stripes:
            fft_filter[:(width //2) +1] = fft_filter[:(width //2) +1] *(1 -gauss_1d[(width //2):])
            fft_filter[-(width //2):] = fft_filter[-(width //2):] *(1 -gauss_1d[:(width //2)])
        
    if keep_mean:
        fft_filter[0, 0] = 1

    filt_stack = np.zeros(stack.shape)

    for i in range(stack.shape[0]):
        im_fft = fftpack.fft2(stack[i])
        filt_fft = im_fft *fft_filter
        filt_stack[i] = fftpack.ifft2(filt_fft).real
        if progress: print("\r{}".format(progress_bar(i+1, stack.shape[0])), end='')

    return filt_stack


def time_pooling(stack):

    win_size = 5
    stack_pool = np.zeros((stack.shape[0] -win_size, stack.shape[1], stack.shape[2]))
    for i in range(stack.shape[0] -win_size):
        print("\rimage: {}".format(i), end='')
        stack_pool[i] = stack[i:i+win_size].min(axis=0)

    return stack_pool


def bright_field_segmentation(image_stack, debug=False):
    """Segments cells out of bright field microscopy images
        PARAMS:
            image_stack (np.array): 
            debug (bool): True if diagnostics plots are needed
        RETURNS:
            mask (np.array): stack of maskw where the cells were detected
    """

    def struct_3D(t, x, y, shape='cube'):
        """Creates a 3D cross mask for binary morphology"""

        if shape == 'cube':
            mask = np.ones((t,x,y))
        elif shape == 'cross':
            mask = np.zeros((t, x, y))
            mask[t //2],  mask[x //2], mask[t //2] = 1, 1, 1
        else:
            print("Used cube as default structuring element")
            mask = struct_3D(t, x, y, shape='cube')
        return mask


    image_stack = enhance_contrast(image_stack, 'hist-equal')

    # Gradient computation and gaussian filtering
    mask = np.zeros_like(image_stack)
    for i in range(image_stack.shape[0]):
        mask[i] = skimage.filters.sobel(image_stack[i]) 
        mask[i] = ndimage.gaussian_filter(mask[i], 1.25)

    # Morphology
    print("Morphology...")
    mask = morphology.grey_dilation(mask, structure=np.ones((2, 1,1)))
    mask = morphology.binary_closing((mask > 1.35).astype('uint8'), structure=np.ones((3, 10,10)))
    mask = morphology.binary_opening(mask, structure=np.ones((1, 5,5)))

    if debug:
        fig, axes = plt.subplots(1, 2, num=1, figsize=(10, 5))
        tifffile.imshow(image_stack, cmap='gray', figure=fig, subplot=axes.ravel()[0])
        axes.ravel()[0].set_title("Original Image")
        tifffile.imshow(mask, cmap='gray', figure=fig, subplot=axes.ravel()[1])
        axes.ravel()[1].set_title("Bacteria Detection")
        plt.show()

    return mask.astype('uint8')


def coregister(stack, pixel_ratio=1.38, translation=np.zeros((3,)), rotation_angle=0.0):
    """Coregisters the BF image to ISCAT pixel space, translation and rotation (fixed pixel_size_ratio)
        PARAMS:
            stack (np.array): input array
            pixel_ratio (float): ratio of the pixel size for the zoom of the image
            translation (np.array): 1 by n_dim array with the amount of pixels by which to shift
            rotation_angle (float): angle (in radians) by which the array will be rotated (clockwise)     
        RETURNS:
            coreg_stack (np.array): transformed stack
    """


    print("Coregistering...")
    
    # pixsize_BF -> pix_size_ISCAT (changes array's shape)
    coreg_stack = ndimage.zoom(stack, (1, pixel_ratio, pixel_ratio), order=1, mode='nearest')

    if rotation_angle == 0 and (translation == 0.0).all():

        crop_ind = (coreg_stack.shape[1] -512) //2
        return coreg_stack[:,crop_ind:crop_ind+512,crop_ind:crop_ind+512]
    else:

        # affine_trans from ISCAT coord -> BF
        coreg_stack = ndimage.rotate(coreg_stack, rotation_angle *(360 /(2*np.pi)), axes=(1,2), reshape=False)
        # coreg_stack = ndimage.shift(coreg_stack, translation)
        crop_ind = (coreg_stack.shape[1] -512) //2
        coreg_stack = coreg_stack[:,crop_ind:crop_ind+512,crop_ind:crop_ind+512]

    return coreg_stack















































