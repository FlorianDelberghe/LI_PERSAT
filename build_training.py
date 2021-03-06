"""
Imports bright field images to compute bacteria positions and saves frame + bacteria detection to folder to be used as
U-Net training data
"""
import glob
import os
import re
import sys
import time

import imageio
import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.morphology as morphology
from scipy.ndimage import gaussian_filter
import skimage.external.tifffile as tifffile
from skimage.draw import line_aa
import torch

from code import utilities, processing
from code.models import UNetCell


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


def parse_pili_coords(filepath):
    """
    PARAMS:
        paths (list): list of the paths to the log files with the pilis positions
    """

    pili_coords = {}
    stack_dims = None

    with open(filepath, 'rt') as f:
        for line in f:
            if line.lower().startswith('image dims') and stack_dims is None:
                stack_dims = tuple(map(lambda x: int(x), line.strip().split('\t')[1:]))
                pili_coords['dims'] = stack_dims
                continue

            if line.lower().startswith("slice "):
                slc = line.strip().split()[1]
                if slc not in pili_coords.keys():   
                    pili_coords[slc] = []
                pili_coords[slc].append(next(f).strip().split(', '))
                pili_coords[slc].append(next(f).strip().split(', '))

    return pili_coords


def get_pili_masks(pili_coords_paths):
    """Returns generator for the masks of the pili training set from a list of  dictionaries of their positions (dict as return by the build_iscat_pili_training() function)
    
        ARGS:
            pili_coors_path (list(dic)):

        RETURNS:
            pili_masks (generator(np.array)): genretor of the masks generated by build_mask()"""

    def build_mask(coord_dict):
        """Creates a stack of the pili masks for the training target from a dictionary
            ARGS:
                coord_dict (dic):                            
            RETURNS:
                pili_mask (np.array): zero array with 255 value (then gaussian blured) at the position of the pili"""

        pili_mask = np.squeeze(np.zeros(coord_dict['dims']), axis=(2,4))
        pili_mask = np.swapaxes(pili_mask, 0,2)

        for slc in coord_dict.keys():
            if slc == 'dims': continue

            for i in range(0, len(coord_dict[slc][0]), 2):
                rr, cc, _ = line_aa(int(coord_dict[slc][1][i + 0]),
                                    int(coord_dict[slc][0][i + 0]),
                                    int(coord_dict[slc][1][i + 1]),
                                    int(coord_dict[slc][0][i + 1]))

                try:
                    pili_mask[int(slc)-1, rr, cc] = 1
                except IndexError:
                    pili_mask[int(slc)-1, rr-1, cc-1] = 1
        
        pili_mask = (gaussian_filter(pili_mask, (0,.7,.7)) > 0.2) *255

        return (pili_mask /pili_mask.max()).astype('uint8')


    pili_coords = (parse_pili_coords(pili_coords_path) for pili_coords_path in pili_coords_paths)
    pili_masks = (build_mask(coord_dict) for coord_dict in pili_coords)
    
    return pili_masks


def build_brigth_field_training(filepaths, sampling=4):
    """Creates bright field training data and target in data/bf_seg/[REF_FRAMES / MASKS] for the bright field cell segmentation task
    
        ARGS:
            filepath (list(str)): filepaths of all the images to input as returned by utilitiues.load_data_paths()
            sampling (int): sampling interval of the saved images (lower storage footprint)        
    """
    
    OUT_PATH = DATA_PATH+'bf_seg/'
    bf_stacks = (utilities.load_imgs(path) for path in filepaths)

    os.makedirs(os.path.join(OUT_PATH, 'REF_FRAMES/'), exist_ok=True)
    os.makedirs(os.path.join(OUT_PATH, 'MASKS/'), exist_ok=True)

    for i, bf_stack in enumerate(bf_stacks):
        
        print(bf_stack.shape[1:])
        
        # bf_stack = bf_stack[::sampling]

        # Change scale from (384, 384) to (512,512)
        if bf_stack.shape[1:] != (512,512):
            bf_stack = processing.coregister(bf_stack, 1.38, np.zeros((3,)), 0.0)

        print(bf_stack.shape)
        mask = processing.bright_field_segmentation(bf_stack)        
    
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
    """Creates iscat training data and target in data/iscat_seg/[REF_FRAMES / MASKS] for the iSCAT cell segmentation task
    
        ARGS:
            bf_filepaths (list(str)): filepaths of all the bright field images to input as returned by utilitiues.load_data_paths()            
            iscat_filepaths (list(str)): filepaths of all the iscat images to input as returned by utilitiues.load_data_paths()
            sampling (int): sampling interval of the saved images (lower storage footprint)
    """

    OUT_PATH = DATA_PATH+'iscat_seg/'
    os.makedirs(os.path.join(OUT_PATH, 'REF_FRAMES/'), exist_ok=True)
    os.makedirs(os.path.join(OUT_PATH, 'MASKS/'), exist_ok=True)

    # Range of non filtered elements [px]
    min_size, max_size = 1, 13

    iscat_stacks = (utilities.load_imgs(path) for path in iscat_filepaths)
    bf_stacks = (utilities.load_imgs(path) for path in bf_filepaths)

    # Returns the metadata of the exwperiments such as frame rate
    metadatas = get_experiments_metadata(iscat_filepaths)

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        print("Running on: {:s}".format(torch.cuda.get_device_name(device)))
        cuda = torch.device('cuda')
    else:
        # Doesn't run on CPU only machines comment if no GPU
        print("No CUDA device found")
        sys.exit(1)

    unet = UNetCell(1, 1, device=cuda, bilinear_upsampling=False)
    unet.load_state_dict(torch.load('outputs/saved_models/bf_unet.pth'))

    for i, (bf_stack, iscat_stack, metadata) in enumerate(zip(bf_stacks, iscat_stacks, metadatas)):
        if i < 45: continue

        bf_stack = bf_stack.astype('float32')
        print(bf_stack.shape)
        if bf_stack.shape[1:] != iscat_stack.shape[1:]:
            bf_stack = processing.coregister(bf_stack, 1.38)
            print(bf_stack.shape)

        normalize(bf_stack)

        # Samples iscat image to correct for the difference in framefate
        iscat_stack = iscat_stack[::sampling *int(metadata['iscat_fps'] /metadata['tirf_fps'])]
        
        torch_stack = torch.from_numpy(bf_stack).unsqueeze(1).cuda()
        mask = unet.predict_stack(torch_stack).detach().squeeze().cpu().numpy() > 0.05
        mask = morphology.grey_erosion(mask *255, structure=processing.structural_element('circle', (3,5,5)))
        mask = morphology.grey_closing(mask, structure=processing.structural_element('circle', (3,7,7)))
        mask = (mask > 50).astype('uint8')

        # Median filtering and normalization 
        iscat_stack = processing.image_correction(iscat_stack)

        # Contrast enhancement
        iscat_stack = processing.enhance_contrast(iscat_stack, 'stretching', percentile=(1, 99))
        
        # Fourier filtering of image
        iscat_stack = processing.fft_filtering(iscat_stack, min_size, max_size, True)
        iscat_stack = processing.enhance_contrast(iscat_stack, 'stretching', percentile=(3, 97))

        for j in range(0, min(iscat_stack.shape[0], mask.shape[0]), sampling):
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
        

def build_iscat_fluo_training(iscat_filepaths, fluo_filepaths):
    """Creates iscat training data and target in data/iscat_seg/[REF_FRAMES / MASKS] for the iSCAT cell segmentation task with fluorent images used as training
    
        ARGS:
            iscat_filepaths (list(str)): filepaths of all the iSCAT images to input as returned by utilities.load_data_paths()
            fluo_filepaths (list(str)): filepaths of all the fluo images to input as returned by utilities.load_data_paths()
    """

    print("Building iSCAT/fluo dataset...")

    OUT_PATH = DATA_PATH +'iscat_fluo/'
    os.makedirs(os.path.join(OUT_PATH, 'REF_FRAMES/'), exist_ok=True)
    os.makedirs(os.path.join(OUT_PATH, 'MASKS/'), exist_ok=True)

    fluo_tirf_imgs = (tifffile.imread(fluo_tirf_img_path) for fluo_tirf_img_path in fluo_filepaths)
    fluo_iscat_imgs = (tifffile.imread(fluo_iscat_img_path) for fluo_iscat_img_path in iscat_filepaths)

    min_size, max_size = 1, 13

    for i, (tirf, iscat) in enumerate(zip(fluo_tirf_imgs, fluo_iscat_imgs)):
        tirf = processing.coregister(tirf, 1.38)
        tirf = ndimage.median_filter(tirf, 5)
        tirf = tirf > 300
        tirf = ndimage.binary_closing(tirf, processing.structural_element('circle', (1,10,10))).astype('float32')
        
        iscat = processing.image_correction(iscat)
        iscat = processing.enhance_contrast(iscat, 'stretching', percentile=(1, 99))
        iscat = processing.fft_filtering(iscat, min_size, max_size, True)
        iscat = processing.enhance_contrast(iscat, 'stretching', percentile=(3, 97))
        
        # Discards the first image
        for j in range(1, tirf.shape[0]):
            try:
                if (tirf != 0).any():
                    imageio.imsave(OUT_PATH +'REF_FRAMES/' +"iscat_{}_{}.png".format(i+1, j+1), rescale(iscat[int(j *iscat.shape[0] /tirf.shape[0])]))
                    imageio.imsave(OUT_PATH +'MASKS/' +"fluo_{}_{}.png".format(i+1, j+1), rescale(tirf[j]))
            
            except IndexError:
                # In case the dimensions don't match -> goes to next slice
                print("IndexError")
                continue


def build_iscat_pili_training(iscat_filepaths, pili_coords_filepaths):
    """Creates the training set and target for the pili training task
        ARGS:
                iscat_filepaths (list(str)): filepaths of all the iSCAT images to input as returned by utilitiues.load_data_paths()
                pili_coords_filepaths (list(str)): filepaths of all the pili cordinates as computed by the included imageJ plugin
    """

    OUT_PATH = DATA_PATH+'iscat_pili/'
    os.makedirs(os.path.join(OUT_PATH, 'REF_FRAMES/'), exist_ok=True)
    os.makedirs(os.path.join(OUT_PATH, 'MASKS/'), exist_ok=True)

    ref_iscat_imgs = (tifffile.imread(filepath) for filepath in iscat_filepaths)
    pili_masks = get_pili_masks(pili_coords_filepaths)

    # Preprocessing
    min_size, max_size = 1, 13
    iscat_stacks = (processing.image_correction(iscat_stack) for iscat_stack in ref_iscat_imgs)
    iscat_stacks = (processing.enhance_contrast(iscat_stack, 'stretching', percentile=(1, 99)) for iscat_stack in iscat_stacks)
    iscat_stacks = (processing.fft_filtering(iscat_stack, min_size, max_size, True) for iscat_stack in iscat_stacks)
    iscat_stacks = (processing.enhance_contrast(iscat_stack, 'stretching', percentile=(3, 97)) for iscat_stack in iscat_stacks)

    for i, (iscat, pili) in enumerate(zip(iscat_stacks, pili_masks)):
        for j in range(pili.shape[0]):
            if (pili[j] != 0).any() and not (j <= 1 or j >= pili[j].shape[0] -2):
                imageio.imsave(OUT_PATH +'REF_FRAMES/' +"ref_iscat_{}_{}.png".format(i+1, j+1), rescale(iscat[j]))
                imageio.imsave(OUT_PATH +'MASKS/' +"pili_{}_{}.png".format(i+1, j+1), pili[j])


def build_hand_segmentation_tirf_testval():
    """Creates dataset of manually segmented tirf images for validation and testing"""

    OUT_PATH = DATA_PATH+'hand_seg_tirf/'
    os.makedirs(os.path.join(OUT_PATH, 'REF_FRAMES/'), exist_ok=True)
    os.makedirs(os.path.join(OUT_PATH, 'MASKS/'), exist_ok=True)

    ROOT_TEST_PATH = "data/hand-segmentation/"    
    tirf_files = glob.glob(os.path.join(ROOT_TEST_PATH, 'tirf/*.tif'))
    cell_seg_files = glob.glob(os.path.join(ROOT_TEST_PATH, 'cell_seg/*.txt'))

    tirf_files.sort(); cell_seg_files.sort();

    for i, (tirf, cell_seg) in enumerate(zip(tirf_files, cell_seg_files)):
        
        tirf_stack = tifffile.imread(tirf)
        mask_cell = utilities.cell_mask_from_segmentation(cell_seg)

        tirf_stack = processing.coregister(tirf_stack, 1.38, np.zeros((3,)), 0.0) if tirf_stack.shape[1:] != (512,512) else tirf_stack
        mask_cell = processing.coregister(mask_cell, 1.38, np.zeros((3,)), 0.0) if mask_cell.shape[1:] != (512,512) else mask_cell

        for j in range(0, tirf_stack.shape[0], 4):
            print("\rSaving to stack_{}_{}.png".format(i+1, j+1), end=' '*5)                
            tifffile.imsave(os.path.join(OUT_PATH, 'REF_FRAMES/', "stack_{}_{}.png".format(i+1, j+1)), rescale(tirf_stack[j]))
            tifffile.imsave(os.path.join(OUT_PATH, 'MASKS/', "mask_{}_{}.png".format(i+1, j+1)), mask_cell[j].astype('uint8'))

        print('')


def build_hand_segmentation_iscat_testval():
    """Creates dataset of manually segmented iSCAT images for validation and testing"""

    OUT_PATH_CELL = DATA_PATH+'hand_seg_iscat_cell/'
    os.makedirs(os.path.join(OUT_PATH_CELL, 'REF_FRAMES/'), exist_ok=True)
    os.makedirs(os.path.join(OUT_PATH_CELL, 'MASKS/'), exist_ok=True)

    OUT_PATH_PILI = DATA_PATH+'hand_seg_iscat_pili/'
    os.makedirs(os.path.join(OUT_PATH_PILI, 'REF_FRAMES/'), exist_ok=True)
    os.makedirs(os.path.join(OUT_PATH_PILI, 'MASKS/'), exist_ok=True)

    ROOT_TEST_PATH = "data/hand-segmentation/"    
    iscat_files = glob.glob(os.path.join(ROOT_TEST_PATH, 'iSCAT/*.tif'))
    cell_seg_files = glob.glob(os.path.join(ROOT_TEST_PATH, 'cell_seg/*.txt'))
    pili_seg_files = glob.glob(os.path.join(ROOT_TEST_PATH, 'pili_seg/*.txt'))

    iscat_files.sort(); cell_seg_files.sort(); pili_seg_files.sort();

    for i, (iscat, cell_seg, pili_seg) in enumerate(zip(iscat_files,cell_seg_files, pili_seg_files)):
        
        # Loading tirf and iSCAT images
        iscat_stack = tifffile.imread(iscat)

        # iSCAT preprocessing
        iscat_stack = processing.image_correction(iscat_stack)
        iscat_stack = processing.enhance_contrast(iscat_stack, 'stretching', percentile=(1, 99))
        iscat_stack = processing.fft_filtering(iscat_stack, 1, 13, True)
        iscat_stack = processing.enhance_contrast(iscat_stack, 'stretching', percentile=(3, 97))

        # Loading ground truth masks
        mask_cell = utilities.cell_mask_from_segmentation(cell_seg)
        mask_pili = utilities.pili_mask_from_segmentation(pili_seg)

        for j in range(0, iscat_stack.shape[0], 8):
            print("\rSaving to stack_{}_{}.png".format(i+1, j+1), end=' '*5)                
            tifffile.imsave(os.path.join(OUT_PATH_CELL, 'REF_FRAMES/', "stack_{}_{}.png".format(i+1, j+1)), rescale(iscat_stack[j]))
            tifffile.imsave(os.path.join(OUT_PATH_CELL, 'MASKS/', "mask_{}_{}.png".format(i+1, j+1)), mask_cell[j //2].astype('uint8'))

        print('')
        for j in range(iscat_stack.shape[0]):
            if not (mask_pili != 0).any(): continue

            print("\rSaving to stack_{}_{}.png".format(i+1, j+1), end=' '*5)                
            tifffile.imsave(os.path.join(OUT_PATH_PILI, 'REF_FRAMES/', "stack_{}_{}.png".format(i+1, j+1)), rescale(iscat_stack[j]))
            tifffile.imsave(os.path.join(OUT_PATH_PILI, 'MASKS/', "mask_{}_{}.png".format(i+1, j+1)), mask_pili[j].astype('uint8'))

        print('')


def get_experiments_metadata(paths):
    """Returns the metadata of the experiments from the file at the root of the experiment folder
        ARGS:
            paths (list(str)): root paths of the experiments
        RETURNS:
            metadata (list(dict)): dict with the magnification and fps information of the experiments
    """
    metadatas = []
    for path in paths:
        metadata = {}
        path = '/'.join(path.split('/')[:-3])+'/'

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

        metadatas.append(metadata)

    return metadatas


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
    """Builds the training data for the given dataset given by the first command line argument
        >>> python build_training.py <dataset>
    """

    if len(sys.argv) > 1:
        if str(sys.argv[1]) == 'bright_field':
            bf_img_paths = utilities.load_data_paths('bright field', '/cam1/event[0-9]_tirf/*.tif')
            iscat_img_paths = utilities.load_data_paths('bright field', '/cam1/event[0-9]/*PreNbin*.tif')

            bf_img_paths.sort()
            iscat_img_paths.sort()

            # /!\ Issues w/ bf_seg do not run /!\
            # build_brigth_field_training(bf_img_paths[:5])

        if str(sys.argv[1]) == 'iscat':
            bf_img_paths = utilities.load_data_paths('bright field', '/cam1/event[0-9]_tirf/*.tif')
            iscat_img_paths = utilities.load_data_paths('bright field', '/cam1/event[0-9]/*PreNbin*.tif')

            bf_img_paths.sort()
            iscat_img_paths.sort()

            build_iscat_training(bf_img_paths, iscat_img_paths, sampling=1)
        
        if str(sys.argv[1]) == 'fluo':
            fluo_tirf_paths = utilities.load_data_paths('fluo', '/cam1/event[0-9]_tirf/*.tif')
            fluo_iscat_paths = utilities.load_data_paths('fluo', '/cam1/event[0-9]/*PreNbin*.tif')

            fluo_tirf_paths.sort()
            fluo_iscat_paths.sort()

            build_iscat_fluo_training(fluo_iscat_paths, fluo_tirf_paths)

        if str(sys.argv[1]) == 'pili':
            DATA_PATH = NETWORK_PATH +"2018/décembre/04/PilH-FliC-_Agar_Microchannel_0_Flo/"

            iscat_img_paths = glob.glob(DATA_PATH +"cam[0-9]/event[0-9]/*PreNbin*.tif")
            iscat_img_paths.sort()

            iscat_coords_paths = glob.glob(DATA_PATH +"cam[0-9]/event[0-9]/*.txt")
            iscat_coords_paths.sort()

            build_iscat_pili_training(iscat_img_paths[:len(iscat_coords_paths)], iscat_coords_paths)
   

if __name__ == "__main__":
    main()
