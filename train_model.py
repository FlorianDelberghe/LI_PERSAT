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
        print("Running on: {:s}".format(torch.cuda.get_device_name(device)))

        cuda = torch.device('cuda')
    else:
        print("No CUDA device found")
        sys.exit(1)

    unet = models.UNet(1, 1, "iSCAT_UNet_2", device=cuda, data_path="data/iscat_seg")

    unet.train_model(epochs=20, batch_size=10)

if __name__ == '__main__':
    main()