import argparse
import os
import glob
import sys

import numpy as np
import cv2
import scipy.ndimage.morphology as morphology
import torch
import skimage.external.tifffile as tifffile
from imageio import mimread, mimsave

import code.processing as processing
from code.models import UNetCell, UNetPili


def cell_mask_from_segmentation(filepath):

    def parse_cell_loc(filepath):
        """Parses ImageJ macro's output files for cell's location
            ARGS:
                filepath (str): filepath to the cell segmentation
            RETURN:
                cell_loc (dict): contains slices as keys and list of lists of cell contour coordinates as values, first key is images dimensions
        """
        cell_loc = {}
        with open(filepath, 'rt') as f:
            for line in f:
                
                if line.strip().startswith('Image dims'):
                    cell_loc['image_dims'] = tuple(map(int, line.strip().split('\t')[1:]))

                if line.strip().startswith('Slice'):
                    current_slice = int(line.strip().split()[1])
                    cell_loc[current_slice] = [[], []]
                    line = next(f)

                if line.startswith('Cell'):
                    cell_loc[current_slice][0].append(list(map(lambda x: int(float(x))-1, (next(f).strip().split(', ')))))
                    cell_loc[current_slice][1].append(list(map(lambda x: int(float(x))-1, (next(f).strip().split(', ')))))

        return cell_loc

    def build_cell_mask(cell_loc):
        """Returns a stack of the cells's positions in a mask stack with the same dimensions as the original image
            ARGS:
                cell_loc (dict): dictionary of the contours coordinates of the cell as returned by parse_cell_loc()
            RETURNS:
                mask (np.array): mask array as uint8 true is 255 false is 0
        """
        contours = np.zeros(tuple(filter(lambda x: x > 1, cell_loc['image_dims'])), dtype='uint8')
        contours = np.moveaxis(contours, -1, 0)
        
        for key, values in cell_loc.items():
            if key == 'image_dims': continue

            cells_x, cells_y = values
            
            for cell_x, cell_y in zip(cells_x, cells_y):
                temp_cont = np.zeros(contours.shape[1:3])
                cv2.drawContours(temp_cont, [np.array([[x, y] for x, y in zip(cell_x, cell_y)])], -1, (255, 255, 255), thickness=cv2.FILLED)
                contours[key][temp_cont != 0] = 255

        contours = interp_lin(contours)

        return  contours 
    
    def interp_lin(stack):
        """Pixel wise linear time interpolation"""
        for i in range(0, stack.shape[0]-4, 4):
            for j in range(1, 4, 1):
                stack[i+j] = stack[i] *(1 -j/4) + stack[i+4] *(j/4)

        stack[stack <= 255 /2] = 0
        return stack

    def interp_morpho(stack):
        """Interpolation of missing data with temporal dilation"""
        stack = morphology.binary_dilation(stack != 0, structure=processing.structural_element('square', (5,1,1)))
        stack = morphology.binary_closing(stack != 0, structure=processing.structural_element('circle', (3,3,3)))

        return (stack *255).astype('uint8')


    cell_loc = parse_cell_loc(filepath)
    mask = build_cell_mask(cell_loc)

    return mask
    

def pili_mask_from_segmentation(filepath):
    
    def parse_pili_loc(filepath):
        """Parses ImageJ macro's output files for pili location
            ARGS:
                filepath (str): filepath to the pili segmentation
            RETURN:
                pili_loc (dict): contains slices as keys and pili coordinates as values, first key is images dimensions
        """
        pili_loc = {}
        with open(filepath, 'rt') as f:
            for line in f:
                
                if line.strip().startswith('Image dims'):
                    pili_loc['image_dims'] = tuple(map(int, line.strip().split('\t')[1:]))
                    continue
                
                if line.strip().startswith('Slice'):
                    current_slice = int(line.strip().split()[1])
                    pili_loc[current_slice] = [[], []]
                    pili_loc[current_slice][0].extend(list(map(lambda x: int(float(x))-1, (next(f).strip().split(', ')))))
                    pili_loc[current_slice][1].extend(list(map(lambda x: int(float(x))-1, (next(f).strip().split(', ')))))
                 
        return pili_loc

    def build_pili_mask(pili_loc):
        """Returns a stack of the pili's positions in a mask stack with the same dimensions as the original image
            ARGS:
                pili_loc (dict): dictionary of the coordinates of the pili as returned by parse_pili_loc()
            RETURNS:
                mask (np.array): mask array as uint8 true is 255 false is 0
        """
        mask = np.zeros(tuple(filter(lambda x: x > 1, pili_loc['image_dims'])), dtype='uint8')
        mask = np.moveaxis(mask, -1, 0)
        
        for key, values in pili_loc.items():
            if key == 'image_dims': continue

            coords_x, coords_y = values
            
            pili_start = [(coords_x[i], coords_y[i]) for i in range(0, len(coords_x), 2)]
            pili_end = [(coords_x[i+1], coords_y[i+1]) for i in range(0, len(coords_x), 2)]

            tmp_slc = np.zeros(mask.shape[1:3])
            for start, end in zip(pili_start, pili_end):
                cv2.line(tmp_slc, start, end, color=(255, 255, 255), thickness=5)
            mask[key][tmp_slc != 0] = 255

        return  mask 


    pili_loc = parse_pili_loc(filepath)
    mask = build_pili_mask(pili_loc)

    return mask


def compute_metrics(pred, ground_truth):
    """Computes commom metrics for segmentation evaluation
        ARGS:
            pred (np.array): boolean array of the detection
            ground_truth (np.array): boolean array of the detection
        RETURNS:
            accuracy, recall, precision, f1_score, iou (float)
    """      
    # Compute every combination of true/false positive/negative
    tp = np.logical_and(pred, ground_truth)
    fp = np.logical_and(pred, np.logical_not(ground_truth))
    tn = np.logical_and(np.logical_not(pred), np.logical_not(ground_truth))
    fn = np.logical_and(np.logical_not(pred), ground_truth)

    # Epsilon to prevent zero division
    eps = np.finfo(np.float32).eps

    # Compute statistical metrics
    accuracy = np.logical_or(tp, tn).sum() / ground_truth.size
    recall = tp.sum() / (np.logical_or(tp, fn).sum() +eps)
    precision = tp.sum() / (np.logical_or(tp, fp).sum() +eps)
    f1_score = 2 * tp.sum() / (2 * tp.sum() + np.logical_or(fp, fn).sum() +eps)

    # Compute Intersection over Union (IoU) metric
    iou = np.logical_and(pred, ground_truth).sum() / (np.logical_or(pred, ground_truth).sum() +eps)

    return accuracy, recall, precision, f1_score, iou


def main():
    # Checks for an available graphics card 
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        print("Running on GPU {:d}: {:s}".format(device, torch.cuda.get_device_name(device)))
    else:
        print("No CUDA device found")
        device = 'cpu'    
        
    # Loading paths to test images and groud truth
    ROOT_TEST_PATH = "data/hand-segmentation/"
    
    iscat_files = glob.glob(os.path.join(ROOT_TEST_PATH, 'iSCAT/*.tif'))
    tirf_files = glob.glob(os.path.join(ROOT_TEST_PATH, 'tirf/*.tif'))
    cell_seg_files = glob.glob(os.path.join(ROOT_TEST_PATH, 'cell_seg/*.txt'))
    pili_seg_files = glob.glob(os.path.join(ROOT_TEST_PATH, 'pili_seg/*.txt'))

    iscat_files.sort(); tirf_files.sort(); cell_seg_files.sort(); pili_seg_files.sort()


    # Loading UNet models
    unet_tirf = UNetCell(1, 1, device=device, bilinear_upsampling=False)
    unet_tirf.load_state_dict(torch.load('outputs/saved_models/bf_unet.pth'))
    unet_tirf.eval()

    unet_iscat = UNetCell(1, 1, device=device)
    unet_iscat.load_state_dict(torch.load('saved_models/iscat_unet_augment_before_fluo.pth'))
    unet_iscat.eval()

    unet_pili = UNetPili(1, 1, device=device)
    unet_pili.load_state_dict(torch.load('saved_models/pili_unet_augment_16_channels_170.pth'))
    unet_pili.eval()


    # Iterating over the test files
    for i, (iscat, tirf, cell_seg, pili_seg) in enumerate(zip(iscat_files, tirf_files, cell_seg_files, pili_seg_files)):
        
        # Loading tirf and iSCAT images
        iscat_stack = tifffile.imread(iscat)
        tirf_stack = tifffile.imread(tirf)

        # iSCAT preprocessing
        iscat_stack = processing.image_correction(iscat_stack)
        iscat_stack = processing.enhance_contrast(iscat_stack, 'stretching', percentile=(1, 99))
        iscat_stack = processing.fft_filtering(iscat_stack, 1, 13, True)
        iscat_stack = processing.enhance_contrast(iscat_stack, 'stretching', percentile=(3, 97))

        # Loading ground truth masks
        mask_cell = cell_mask_from_segmentation(cell_seg).astype('bool')
        mask_pili = pili_mask_from_segmentation(pili_seg).astype('bool')

        # Predicting stacks
        with torch.no_grad():
            torch_tirf = torch.from_numpy((tirf_stack / tirf_stack.max()).astype('float32')).to(device=device)
            torch_iscat = torch.from_numpy((iscat_stack / iscat_stack.max()).astype('float32')).to(device=device)

            pred_cell_tirf = unet_tirf.predict_stack(torch_tirf.unsqueeze(1)).squeeze().cpu().numpy()
            pred_cell_iscat = unet_iscat.predict_stack(torch_iscat.unsqueeze(1)).squeeze().cpu().numpy()
            pred_pili_iscat = unet_pili.predict_stack(torch_iscat.unsqueeze(1)).squeeze().cpu().numpy()

        # Computing metrics of models
        print(f"Image {i+1} metrics:")
        print("Cell_detect (tirf): accuracy={:.3f}, recall={:.3f}, precision={:.3e}, F1 score={:.3e}, IoU={:.3f}".format(*compute_metrics(pred_cell_tirf >= .6, mask_cell)))
        print("Cell_detect (iSCAT): accuracy={:.3f}, recall={:.3f}, precision={:.3e}, F1 score={:.3e}, IoU={:.3f}".format(*compute_metrics(pred_cell_iscat[::2] >= .6, mask_cell)))
        print("Pili_detect (iSCAT): accuracy={:.3f}, recall={:.3f}, precision={:.3e}, F1 score={:.3e}, IoU={:.3e}".format(*compute_metrics(pred_cell_iscat >= .55, mask_pili)))

        # Saving prediction and ground truth
        out_tirf = np.stack([np.concatenate([tirf_stack /tirf_stack.max() *255] *2, axis=2)] *3, axis=-1).astype('uint8')
        out_iscat = np.stack([np.concatenate([iscat_stack /iscat_stack.max() *255] *2, axis=2)] *3, axis=-1).astype('uint8')

        out_tirf[...,:out_tirf.shape[2] //2, 1][mask_cell != 0] = 255
        out_tirf[...,out_tirf.shape[2] //2:, 1][pred_cell_tirf >= .6] = 255

        out_iscat[::2,:,:out_iscat.shape[2] //2, 1][mask_cell != 0] = 255
        out_iscat[...,:out_iscat.shape[2] //2, 0][mask_pili != 0] = 255

        out_iscat[...,:,out_iscat.shape[2] //2:, 1][pred_cell_iscat >= .6] = 255
        out_iscat[...,out_iscat.shape[2] //2:, 0][pred_pili_iscat >= .55] = 255

        # Ground truth on the left, net prediction on the right
        mimsave(f'outputs/tirf_truth_pred_{i+1}.gif', out_tirf.astype('uint8'), fps=20)
        mimsave(f'outputs/iscat_truth_pred_{i+1}.gif', out_iscat.astype('uint8'), fps=20)

        
if __name__ == '__main__':
    main()