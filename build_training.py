"""
Imports bright field images to compute bacteria positions and saves frame + bacteria detection to folder to be used as
U-Net training data
"""
import code.processing as processing
import code.utilities as utilities
from code.models import UNet
import glob
import os
import re
import sys
import time

import numpy as np
import scipy.ndimage.morphology as morphology
import skimage.external.tifffile as tifffile
import torch

DATA_PATH = "data/"
NETWORK_PATH = "/mnt/plabNAS/Lorenzo/iSCAT/iSCAT Data/Lorenzo/"


def rescale(stack):
    "Rescales stacks and images to [0, 255] uint8"
    
    stack -= stack.min()
    stack = stack /stack.max()
    return (stack *255).astype('uint8')


def normalize(array, inplace=True):
    """Normalizes images to [0, 1]"""
    
    array -= array.min()
    array /= array.max()

    if not inplace:
        return array


def build_brigth_field_training(filepaths):
    """"""
    
    OUT_PATH = DATA_PATH+'bf_seg/'
    bf_stacks = (utilities.load_imgs(path) for path in filepaths)

    os.makedirs(os.path.join(OUT_PATH, 'REF_FRAMES/'), exist_ok=True)
    os.makedirs(os.path.join(OUT_PATH, 'MASKS/'), exist_ok=True)

    for i, bf_stack in enumerate(bf_stacks):
        
        print(bf_stack.shape[1:])
        
        bf_stack = bf_stack[::4]
        mask = processing.bright_field_segmentation(bf_stack)
        bf_stack = processing.coregister(bf_stack, 1.38, np.zeros((3,)), 0.0)
        mask = processing.coregister(mask, 1.38, np.zeros((3,)), 0.0)
    
        for j in range(0, bf_stack.shape[0], 2):
            if bf_stack[j].shape == mask[j].shape:
                if mask[j].max() == 0: continue

                print("\rSaving to stack_{}_{}.png".format(i+1, j+1), end=' '*5)                
                tifffile.imsave(os.path.join(OUT_PATH, 'REF_FRAMES/', "stack_{}_{}.png".format(i+1, j+1)), rescale(bf_stack[j]))
                tifffile.imsave(os.path.join(OUT_PATH, 'MASKS/', "mask_{}_{}.png".format(i+1, j+1)), mask[j]*255)
            else:
                print("Error, shape: {}, {}".format(bf_stack[j].shape, mask[j].shape))
                break

        print('')


def build_iscat_training(bf_filepaths, iscat_filepaths, sampling=4):
    """"""

    OUT_PATH = DATA_PATH+'iscat_seg/'

    # Range of non filtered elements [px]
    min_size, max_size = 1, 13

    os.makedirs(os.path.join(OUT_PATH, 'REF_FRAMES/'), exist_ok=True)
    os.makedirs(os.path.join(OUT_PATH, 'MASKS/'), exist_ok=True)

    iscat_stacks = (utilities.load_imgs(path) for path in iscat_filepaths)
    bf_stacks = (utilities.load_imgs(path) for path in bf_filepaths)

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        print("Running on: {:s}".format(torch.cuda.get_device_name(device)))
        cuda = torch.device('cuda')
    else:
        print("No CUDA device found")
        sys.exit(1)

    unet = UNet(1, 1, "basic UNet", device=cuda, data_path="data/iscat_seg")
    unet.load_state_dict(torch.load('outputs/saved_models/basic UNet.pth'))

    for i, (bf_stack, iscat_stack) in enumerate(zip(bf_stacks, iscat_stacks)):

        bf_stack = bf_stack[::sampling].astype('float32')
        normalize(bf_stack)
        # Samples iscat image to correct for the difference in framefate
        iscat_stack = iscat_stack[::sampling *int(metadata['iscat_fps'] /metadata['tirf_fps'])]
        
        torch_stack = torch.from_numpy(bf_stack).unsqueeze(1).cuda()
        mask = unet.predict_stack(torch_stack).detach().squeeze().cpu().numpy() > 0.05
        mask = processing.coregister(mask, 1.38)
        mask = morphology.grey_erosion(mask *255, structure=processing.structural_element('circle', (3,5,5)))
        mask = morphology.grey_closing(mask, structure=processing.structural_element('circle', (3,7,7)))
        mask = (mask > 50).astype('uint8')

        # Median filtering and Normalization 
        iscat_stack = processing.image_correction(iscat_stack)

        # Contrast enhancement
        iscat_stack = processing.enhance_contrast(iscat_stack, 'stretching', (1, 99))
        
        # Fourier filtering of image
        iscat_stack = processing.fft_filtering(iscat_stack, min_size, max_size, True, 'horizontal & vertical')
        iscat_stack = processing.enhance_contrast(iscat_stack, 'stretching', (10, 90))

        # tifffile.imsave(os.path.join("outputs/bf_stack.gif"), rescale(bf_stack))

        # tifffile.imsave(os.path.join("outputs/iscat_train.gif"), np.concatenate([rescale(iscat_stack), mask*255], axis=2))

        for j in range(0, iscat_stack.shape[0], 8):
            if iscat_stack[j].shape == mask[j].shape:
                # Doesn't save images without detected cells
                if mask[j].max() == 0: continue

                print("\rSaving to stack_{}_{}.png".format(i+1, j+1), end=' '*5)                
                tifffile.imsave(os.path.join(OUT_PATH, 'REF_FRAMES/', "stack_{}_{}.png".format(i+1, j+1)), rescale(iscat_stack[j]))
                tifffile.imsave(os.path.join(OUT_PATH, 'MASKS/', "mask_{}_{}.png".format(i+1, j+1)), mask[j]*255)
            else:
                print("Error, shape: {}, {}".format(iscat_stack[j].shape, mask[j].shape))
                break

        print('')
        

def get_experiments_metadata(path):

    metadata = {}
    with open(path +os.path.basename(os.path.normpath(path)), 'rt', encoding='cp1252') as f:
        for line in f:  

            if line.strip().lower().startswith('details'):
                next(f)
            if line.lower().startswith("iscat magnification"):
                metadata['iscat_magnification'] = int(next(f).strip().strip('x'))

            if line.lower().startswith("tirf magnification"):
                metadata['tirf_magnification'] = int(next(f).strip().strip('x'))

            if line.lower().startswith("fps iscat"):
                metadata['iscat_fps'] = int(next(f).strip())

            if line.lower().startswith("fps tirf"):
                metadata['tirf_fps'] = int(next(f).strip())

    return metadata


def clean_data_paths(path_list, extra_patterns=[]):
    """Grabs only the paths that contain valid data"""

    patterns = ['test', 'fluo', 'beads', 'nm', '.![tif]$']
    for path in path_list:
        for p in patterns:
            if not re.search(p, path, re.IGNORECASE) == None:
                print("Removed ", path)
                path_list.remove(path)

        for p in extra_patterns:
            if not re.search(p, path, re.IGNORECASE) == None:
                print("Removed ", path)
                path_list.remove(path)

    return path_list


def main():

    DATA_PATH = NETWORK_PATH +"2018/décembre/04/PilH-FliC-_Agar_Microchannel_0_Flo/"
    metadata = get_experiments_metadata(DATA_PATH)

    bf_img_paths = glob.glob(DATA_PATH +"cam1/event[0-9]_tirf/*.tif")
    bf_img_paths = clean_data_paths(bf_img_paths)
    bf_img_paths.sort()

    iscat_img_paths = glob.glob(DATA_PATH +"cam1/event[0-9]/*PreNbin*.tif")
    iscat_img_paths.sort()

    print(bf_img_paths, '\n\n', iscat_img_paths)

    build_brigth_field_training(bf_img_paths)

    build_iscat_training(bf_img_paths, iscat_img_paths, 1)


if __name__ == "__main__":
    main()
