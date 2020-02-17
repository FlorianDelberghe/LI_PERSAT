import os
import sys

import numpy as np
import torch

from code import models, data_loader
from code.data_loader import DataLoader


def train_cell_detection(device, net_name, data_path, refine_w_fluo=False):
    """Trains a model for cell detection, can be used for bright field and iSCAT images"""

    if net_name is None: net_name = 'cell_UNet'
    if data_path is None: data_path =  'data/iscat_seg_temp'

    data_loader = DataLoader('data/iscat_seg_temp', sampling=8)
    unet = models.UNetCell(1, 1, name="iscat_UNet_augment", device=device, data_loader=data_loader, bilinear_upsampling=True)

    unet.train_model(epochs=40, batch_size=12, lr=0.0002, gamma=0.95)
    unet.save_state_dir('outputs/saved_models/', "{}_before_fluo.pth".format(unet.name.lower()))

    if refine_w_fluo:
        # Freezes the encoding part of the UNet
        for layer in [unet.down_conv1, unet.down_conv2, unet.down_conv3, unet.down_conv4, unet.down_conv5]:
            layer.requires_grad = False
        
        # Updates data loader w/ finer dataset
        data_loader = DataLoader('data/iscat_fluo', sampling=1)
        unet.data_loader = data_loader

        unet.train_model(epochs=30, batch_size=5, lr=0.0001, gamma=0.90)
        unet.save_state_dir('outputs/saved_models/', "{}_after_fluo.pth".format(unet.name.lower()))

    
def train_pili_detection(device, net_name, data_path):
    """Trains a model for pili detection in iSCAT images"""

    if net_name is None: net_name = 'pili_UNet'
    if data_path is None: data_path =  'data/iscat_seg_temp'

    data_loader = DataLoader('data/iscat_pili', sampling=1)
    unet = models.UNetPili(1, 1, name="pili_UNet_augment_16_channels_sigout", device=device, data_loader=data_loader)

    unet.train_model(epochs=450, batch_size=8, lr=0.0002, gamma=0.98)


def main():
    """Lauches the code to train a UNet
        From command line:
        >>> python train_model.py <pili/cell> <net_name> <data_path>
    """

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        print(f"Running on: {torch.cuda.get_device_name(device)}")

        cuda = torch.device('cuda')
    else:
        print("No CUDA device found")
        sys.exit(1)

    if len(sys.argv) > 1:
        net_name = sys.argv[2] if len(sys.argv) > 2 else None
        path = sys.argv[3] if len(sys.argv) > 3 else None
        
        if str(sys.argv[1]) == 'cell':
            train_cell_detection(cuda, net_name, path)
            
        if str(sys.argv[1]) == 'pili':
            train_pili_detection(cuda, net_name, path)
    else:
        print("Missing argument, which network to train")


if __name__ == '__main__':
    main()