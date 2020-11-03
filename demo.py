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
from code.utilities import cell_mask_from_segmentation, pili_mask_from_segmentation

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
    unet_tirf.load_state_dict(torch.load('saved_models/bf_unet.pth'))
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