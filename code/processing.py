import os
import pickle
import warnings

import imageio
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
import skimage.external.tifffile as tifffile
from skimage.morphology import skeletonize
import scipy.ndimage as ndimage
import scipy.ndimage.morphology as morphology
from scipy import fftpack
from skimage import exposure

import code.utilities as utilities
from  code.utilities import progress_bar

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


def median_image(stack, method=None, progress=False, **kwargs):
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

    # Defaults parameters for median computation
    niter = kwargs['niter'] if 'niter' in kwargs.keys() else 100
    sample_size = kwargs['sample_size'] if 'sample_size' in kwargs.keys() else 40

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


def normalize_stack(stack):
    """Normalizes stack slice by slice inplace operation
        PARAMS:
            stack (np.array): numpy array to be normalized along axis 0 (dtype=float32)
            progress (bool): print a progress bar if true
        RETURNS:
            stack (np.array): normalized array (dtype=float32)
    """

    print("Normalizing stack...", end='\n')

    for i in range(stack.shape[0]):
        # Rescales stack to [0, 1]
        stack[i] -= stack[i].min()
        stack[i] /= np.abs(stack[i].max())
        
    return stack


def image_correction(image):
    """Computes the median of the input stack, then substracts it and normalize to [0,1]"""

    # Makes sure that we will not be overflow errors
    assert image.min() >= 0 and image.max() <= 2**15 -1, "min:{}, max:{}".format(image.min(), image.max())
    # Cast to signedint for substraction
    image = image.astype('int16')

    median_img = median_image(image, 'bootstrap', sample_size=40, niter=20)

    print("Centering...")
    image -= median_img
    image = normalize_stack(image.astype('float32'))

    return image


def enhance_contrast(stack, method='stretching', axis=0, progress=True, **kwargs):
    """Enhaces contrast of image stack slice by slice, defaults to contrast stretching with percentile = (2, 98)
        ARGS:
            stack (np.array): input image
            method (str): contrast enhancement method in ['stretching', 'hist-equal', 'adapt-hist-equal']
            axis (int): time axis of the stack along which to iterate
            progress (bool): show progress as a progress bar
        KWARGS:
            percentile (tuple(int)): for 'stretching' saturation percentile of the stack
        
        RETURNS:
            contrast_img (np.array): contrast enhanced stack
            """

    def enhance_frame_contrast(image, method=method, percentile=(2, 98), clip_limit=0.009, **kwargs):
        """TODO: stacks with none dim would be easier"""

        if method == 'stretching':
            p_low, p_high = np.percentile(image.ravel(), percentile)
            contrast_img = exposure.rescale_intensity(image, in_range=(p_low, p_high))
            return contrast_img

        elif method == 'hist-equal':
            if image.min() < 0 or image.max() > 1:
                contrast_image = (image -image.min()).astype('float32')
                contrast_image /= contrast_image.max()

            contrast_img = exposure.equalize_hist(image)
            return contrast_img

        elif method == 'adapt-hist-equal':
            """TODO: FIX THIS! or don't, takes too long anyway :p"""
            raise NotImplementedError
            image[image == 0] = np.finfo(np.float32).eps
            contrast_img = exposure.equalize_adapthist(image, clip_limit)
            return contrast_img

        else:
            raise ValueError("'{}' is not a valid contrast rescaling method".format(method))


    print("Enhancing contrast with method: {}".format(method))

    #Contrast enhancement for stacks
    if len(stack.shape) == 3:
        contrast_stack = np.empty_like(stack)

        if progress: print("\r{}".format(progress_bar(0, stack.shape[0])), end='')

        if method == 'stretching':
            slc = [slice(None)] *len(stack.shape)
            
            for i in range(stack.shape[axis]):
                slc[axis] = slice(i, i+1, 1)
                contrast_stack[slc] = enhance_frame_contrast(stack[slc], method, **kwargs)
                if progress: print("\r{}".format(progress_bar(i+1, stack.shape[axis])), end='')

        else:
            # Conversion to float image
            if stack.min() < 0 or stack.max() > 1:
                contrast_stack = stack.astype('float32') -stack.min()
                contrast_stack /= contrast_stack.max()

            else:
                contrast_stack = stack.astype('float32')

            for i in range(stack.shape[axis]):
                contrast_stack[i] = enhance_frame_contrast(contrast_stack[i], method, **kwargs)
                if progress: print("\r{}".format(progress_bar(i+1, stack.shape[0])), end='')
            
    # Contrast enhancment for single images
    elif len(stack.shape) == 2:
        contrast_stack = enhance_frame_contrast(stack, method, **kwargs)

    return contrast_stack


def temporal_denoising(stack, method='mean', window_size=5, progress=True):
    """Denoising method with mean and median temporal filters"""

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
                stack (np.array): input image stack
                min_size (int): min size of features for high pass
                max_size (int): max size of features for low pass
                keep_mean (bool): keeps the mean value of the image (zero frequency component)
                remove_stripes (str): gaussian stripe to remove high frequency in x or y direction, string conaining either 'horizontal' or 'vertical' or both
                progress (bool): displays progress bar in terminal
                
            RETURNS:
                filt_stack (np.array): fitered stack
    """

    def gaussian_kernel(size, sigma):
        """Build a gaussian kernel of shape (size, size) and standard dev sigma"""

        x, y = np.meshgrid(np.arange(-size[0]//2, size[0]//2), np.arange(-size[1]//2, size[1]//2))
        gaussian = 1 /(np.sqrt(2*np.pi) *sigma) *np.exp(-1/(2*sigma**2) *(x**2 + y**2) )
        return gaussian /gaussian.max()


    def center_fft(im_fft):
        """Rearanges the quadrant of the fft or the filters so that the zero frequency is in the middlen instead of [0,0]"""

        H, W = im_fft.shape
        centered_fft = np.empty_like(im_fft)
        centered_fft[0:H//2,0:W//2] = im_fft[-H//2:,-W//2:]
        centered_fft[-H//2:,-W//2:] = im_fft[0:H//2,0:W//2]
        centered_fft[0:H//2,-W//2:] = im_fft[-H//2:,0:W//2]
        centered_fft[-H//2:,0:W//2] = im_fft[0:H//2,-W//2:]
        return centered_fft


    print(f"Filtering structure {min_size} < _ < {max_size} [px]")

    # Computes fourier domain frequencies from min/max sizes of allowed structuring elements
    f_low, f_high = (stack.shape[1]/2)/max_size, (stack.shape[1]/2)/min_size

    # Band pass filter
    fft_filter = gaussian_kernel(stack[0].shape, f_high) - gaussian_kernel(stack[0].shape, f_low)
    fft_filter = center_fft(fft_filter)    

    if remove_stripes is not None:
        width = 5
        x = np.stack([np.arange(-(width//2), width//2 +1)] *stack.shape[-1], axis=-1)
        gauss_1d = np.exp(-0.5 *np.abs(x)**2 *2 **2)
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
    
    # GPU processing ?
    for i in range(stack.shape[0]):
        stack_fft = fftpack.fft2(stack[i])
        filt_fft = stack_fft *fft_filter
        filt_stack[i] = fftpack.ifft2(filt_fft).real
        if progress: print("\r{}".format(progress_bar(i+1, stack.shape[0])), end='')

    return filt_stack


def bright_field_segmentation(stack, debug=False):
    """Segments cells out of bright field microscopy images
        PARAMS:
            stack (np.array): 
            debug (bool): True if diagnostics plots are needed
        RETURNS:
            mask (np.array): stack of maskw where the cells were detected
    """

    stack = enhance_contrast(stack, 'hist-equal')

    # Gradient computation and gaussian filtering
    mask = np.empty_like(stack)
    for i in range(stack.shape[0]):
        mask[i] = ndimage.gaussian_filter(mask[i], 1.5)
        mask[i] = skimage.filters.sobel(stack[i]) 

    # Morphology
    print("Morphology...")
    # mask = morphology.grey_erosion(mask, structure=np.ones((2, 1,1)))
    print(mask.min(), mask.max())
    mask = morphology.binary_closing((mask < 20 /255).astype('uint8'), structure=np.ones((2, 2,2)))
    mask = morphology.binary_erosion(mask, structure=structural_element('circle', (2,5,5)))
    mask = morphology.binary_opening(mask, structure=structural_element('circle', (2,5,5)))
    mask = morphology.binary_closing(mask, structure=structural_element('circle', (3,10,10)))
    # mask = morphology.binary_closing((mask > 1.35).astype('uint8'), structure=structural_element('square', (3, 10, 10)))
    # mask = morphology.binary_opening(mask, structure=np.ones((1, 5,5)))



    # stack = enhance_contrast(stack, 'hist-equal')

    # # Gradient computation and gaussian filtering
    # mask = np.zeros_like(stack)
    # for i in range(stack.shape[0]):
    #     mask[i] = skimage.filters.sobel(stack[i]) 
    #     mask[i] = ndimage.gaussian_filter(mask[i], 1.5)
    
    # # Morphology
    # print("Morphology...")
    # mask = morphology.binary_dilation(mask < 0.01, structure=structural_element('circle', (3,10,10)))
    # mask = morphology.binary_closing(mask, structure=structural_element('cross', (3,13,13)))
    # mask = morphology.binary_opening(mask, structure=structural_element('circle', (1,15,15)))
    
    imageio.mimsave('stack.gif', np.concatenate([(stack *255).astype('uint8'), ((mask) *255).astype('uint8')], axis=-1))
    raise KeyboardInterrupt
    
    if debug:
        imageio.mimsave('brigth_field_seg.gif', np.concatenate([(stack *255).astype('uint8'), ((mask) *255).astype('uint8')], axis=-1))

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
    
    # pix_size_BF -> pix_size_ISCAT (changes array's shape)
    coreg_stack = ndimage.zoom(stack, (1, pixel_ratio, pixel_ratio), order=1, mode='nearest')

    if rotation_angle == 0.0 and (translation == 0.0).all():
        crop_ind = (coreg_stack.shape[1] -512) //2
        return coreg_stack[:,crop_ind:crop_ind+512,crop_ind:crop_ind+512]

    else:
        # affine_trans from ISCAT coord -> BF
        coreg_stack = ndimage.rotate(coreg_stack, rotation_angle *(360 /(2*np.pi)), axes=(1,2), reshape=False)
        # coreg_stack = ndimage.shift(coreg_stack, translation)
        crop_ind = (coreg_stack.shape[1] -512) //2
        coreg_stack = coreg_stack[:,crop_ind:crop_ind+512,crop_ind:crop_ind+512]

    return coreg_stack