import os
import sys

import numpy as np
import torch

from code import models
from code import data_loader
from code.data_loader import DataLoader


def main():

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        print(f"Running on: {torch.cuda.get_device_name(device)}")

        cuda = torch.device('cuda')
    else:
        print("No CUDA device found")
        sys.exit(1)

    data_loader = DataLoader('data/iscat_seg', sampling=2)
    unet = models.UNetCell(1, 1, name="iscat_UNet", device=cuda, data_loader=data_loader)

    unet.train_model(epochs=40, batch_size=10, lr=0.002, gamma=0.92)


    data_loader = DataLoader('data/iscat_fluo', sampling=1)
    unet.data_loader = data_loader

    unet.train_model(epochs=30, batch_size=5, lr=5e-5, gamma=0.90)

if __name__ == '__main__':
    main()