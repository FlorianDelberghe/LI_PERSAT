
import glob
import os
import pickle
import sys
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.morphology as morphology
import scipy.signal
import skimage
import skimage.external.tifffile as tifffile
import skimage.feature
import skimage.filters
import skimage.io
import torch
from matplotlib import cm
from scipy import fftpack
from scipy.ndimage import gaussian_filter
from skimage import exposure, measure
from skimage.filters import threshold_minimum, threshold_triangle
from skimage.morphology import skeletonize
from skimage.transform import hough_line, hough_line_peaks

import warnings
from code import data_loader, models, processing, utilities
import build_training
from code.data_loader import DataLoader


def rescale(stack, normalize=True):
    """Rescales stacks and images to [0, 255] uint8"""
    
    if normalize:
        stack -= stack.min()
        stack = stack /stack.max()

    return (stack *255).astype('uint8')


def import_test_data():
    """Imports the test data in one of the 4 given folders as a generator yielding the stacks one at a time"""

    DATA_PATH = ['/mnt/plabNAS/Lorenzo/iSCAT/iSCAT Data/Lorenzo/2019/novembre/20/fliC-_PaQa_Gasket_0/',
                 '/mnt/plabNAS/Lorenzo/iSCAT/iSCAT Data/Lorenzo/2020/janvier/16/fliC-_PaQa_solid_4/',
                 '/mnt/plabNAS/Lorenzo/iSCAT/iSCAT Data/Lorenzo/2018/décembre/04/PilH-FliC-_Agar_Microchannel_0_Flo/',
                 '/mnt/plabNAS/Lorenzo/iSCAT/iSCAT Data/Lorenzo/2020/janvier/17/pilH-fliC-_PaQa_solid_0/']

    DATA_PATH = DATA_PATH[-1]

    tirf_paths = glob.glob(DATA_PATH +'cam1/event[0-9]_tirf/*.tif')
    iscat_paths = glob.glob(DATA_PATH +'cam1/event[0-9]/*PreNbin*.tif')

    fps_iscat, fps_tirf = 100, 50

    for i, (iscat_path, tirf_path) in enumerate(zip(iscat_paths, tirf_paths)):

        iscat = tifffile.imread(iscat_path)[::int(fps_iscat /fps_tirf)]
        tirf = tifffile.imread(tirf_path)

        iscat = processing.image_correction(iscat)
        iscat = processing.enhance_contrast(iscat, 'stretching', percentile=(1, 99))
        iscat = processing.fft_filtering(iscat, 1, 13, True)
        iscat = processing.enhance_contrast(iscat, 'stretching', percentile=(3, 97))
        iscat = iscat.astype('float32')

        tirf = processing.coregister(tirf, 1.38)

        # For fluo tirf images
        # tirf = ndimage.median_filter(tirf, 5)
        # tirf = (tirf > .9 *skimage.filters.threshold_triangle(tirf.ravel())).astype('uint8')
        # tirf = ndimage.binary_opening(tirf, processing.structural_element('circle', (1,10,10))).astype('uint8')
        # tirf = predict_cell_detection(tirf)

        tirf = tirf.astype('float32')
        
        iscat -= iscat.min()
        iscat /= iscat.max()
        
        tirf -= tirf.min()
        tirf /= tirf.max()

        yield iscat, tirf


def predict_cell_detection(stack, sigma=2, cell_diameter=20):
    """Finer and slightly denoised predictions compared to simple thresholding"""
    
    # For handling of stacks of images
    if len(stack.shape) >= 3:
        cell_pred = np.empty(stack.shape, dtype='uint8')
        for i in range(stack.shape[0]):
            cell_pred[i] = predict_cell_detection(stack[i])

        return cell_pred

    # Sizes in nm
    cell_size = 600
    pixel_size = 63
    cell_pixel_size = cell_size /pixel_size *1.38
    valid_stack = stack > 1 /(sigma *np.sqrt(2 *np.pi))
    valid_stack = morphology.binary_closing(valid_stack, processing.structural_element('circle', (1,10,10)).squeeze())
    skeleton = skeletonize(valid_stack)
    cell_pred = morphology.binary_dilation(skeleton, processing.structural_element('circle', (1,int(cell_pixel_size),int(cell_pixel_size))).squeeze())

    return cell_pred.astype('uint8')


def test_cell_detection():
    """"""

    device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    cuda = torch.device('cuda')

    unet_tirf = models.UNetCell(1, 1, device=device, name="testing_UNet", bilinear_upsampling=False)
    unet_tirf.load_state_dict(torch.load('outputs/saved_models/bf_unet.pth'))
    unet_tirf.eval()

    unet_iscat = models.UNetCell(1, 1, device=device, name="testing_UNet")
    unet_iscat.load_state_dict(torch.load('outputs/saved_models/iscat_unet_augment_before_fluo.pth'))
    unet_iscat.eval()

    for i, (iscat, tirf) in enumerate(import_test_data()):
        with torch.no_grad():
            iscat_torch = torch.from_numpy(iscat).float().cuda().unsqueeze(1)
            tirf_torch = torch.from_numpy(tirf).float().cuda().unsqueeze(1)

            pred_iscat = unet_iscat.predict_stack(iscat_torch)
            pred_iscat = pred_iscat.detach().cpu().squeeze().numpy()

            pred_tirf = unet_tirf.predict_stack(tirf_torch)
            pred_tirf = pred_tirf.detach().cpu().squeeze().numpy()

        out_iscat = np.stack([rescale(iscat, False)] *3, -1)
        out_tirf = np.stack([rescale(tirf, False)] *3, -1)

        out_iscat[...,1][pred_iscat > .6] = 0
        out_tirf[...,1][pred_tirf > .6] = 0
                
        imageio.mimsave(f'outputs/cell_detect/detect_cell_iscat_{i+1}.gif', out_iscat)
        imageio.mimsave(f'outputs/cell_detect/detect_cell_tirf_{i+1}.gif', out_tirf)


def test_pili_detection():
    """""" 

    # Setting up the UNets for cell detection
    device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    cuda = torch.device('cuda')

    unet_cell = models.UNetCell(1, 1, device=device, name="testing_UNet")
    unet_cell.load_state_dict(torch.load('outputs/saved_models/iscat_unet_augment_before_fluo.pth'))
    unet_cell.eval()

    unet_pili = models.UNetPili(1, 1, device=device, name="testing_UNet")
    unet_pili.load_state_dict(torch.load('outputs/saved_models/pili_unet_augment_16_channels_170.pth'))
    unet_pili.eval()

    for j, (iscat, tirf) in enumerate(import_test_data()):
        with torch.no_grad():

            img_torch = torch.from_numpy(iscat).float().cuda().unsqueeze(1)

            pred_cell_stack = unet_cell.predict_stack(img_torch)
            pred_cell_stack = pred_cell_stack.detach().cpu().squeeze().numpy()

            pred_pili_stack = unet_pili.predict_stack(img_torch)
            pred_pili_stack = pred_pili_stack.detach().cpu().squeeze().numpy()

        # tirf = tirf > np.percentile(tirf.ravel(), 90)
        # tirf = ndimage.binary_opening(tirf, processing.structural_element('circle', (1,8,8)))

        # pred_cell = predict_cell_detection(pred_cell_stack) 

        out = np.stack([rescale(iscat)] *3, axis=-1)
        out[...,1][pred_cell_stack > .6] = 0
        out[...,1][pred_pili_stack > .55] = 255

        out = np.concatenate([np.stack([rescale(iscat)] *3, axis=-1), out], axis=-2)
        imageio.mimsave(f'outputs/pili_detect/detect_pili_{j+1}.gif', out)
        
        continue

        for i in range(iscat.shape[0]):
            print(f"\r{i+1:0>4d}", end=' '*5)

            pred_cell = predict_cell_detection(pred_cell_stack[i])
            # pred_cell = pred_cell_stack[i] > 1 /(2 *np.sqrt(2 *np.pi))
            pred_pili = pred_pili_stack[i]

            out = np.stack([rescale(iscat[i])] *3, axis=2)
            out[...,0][pred_cell != 0] = rescale(pred_cell, False)[pred_cell != 0]
            out[...,1][pred_cell != 0] = 0
            out[...,2][pred_cell != 0] = 0

            out[...,1][pred_pili > .55] = rescale(pred_pili > .55, False)[pred_pili > .55]

            out = np.concatenate([np.stack([rescale(iscat[i])] *3, axis=2), out, np.stack([rescale(tirf[i])] *3, axis=2), np.stack([rescale(pred_pili)] *3, axis=2)], axis=1)
            imageio.imsave(f'outputs/temp/detect_pili_{j+1}_{i+1}.png', out)
            # imageio.mimsave(f'outputs/temp/test_{i+1}.gif', np.concatenate([rescale(iscat), rescale(tirf)], axis=1), fps=30)


def main():

    # Checks for an available graphics card 
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        print("Running on: {:s}".format(torch.cuda.get_device_name(device)))
        cuda = torch.device('cuda')
    else:
        print("No CUDA device found")
        sys.exit(1)    

    # test_cell_detection()
    test_pili_detection()


if __name__ == '__main__':
    main()
