import os
from code.data_loader import DataLoader
from code.utilities import progress_bar

import imageio
import numpy as np
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


class DoubleConv(nn.Module):
    """Includes 2 following convolutional layers"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_):
        x = self.dconv(input_)
        return x


class DownConv(nn.Module):
    """Convolutional layer using DoubleConv -> reduce image size"""

    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input_):
        x = self.conv(input_)
        x_pool = self.pool(x)
        return x_pool, x


class UpConv(nn.Module):
    """Up convolution with transposed convolution
    TODO: try with bilin interpolation and 1x1 conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UpConv, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, input_, skip_input):
        x = self.up(input_)
        # /!\ maybe padd because of size differences
        x = torch.cat((x, skip_input), dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """Mother class for the UNets used for cell and pili segmentation"""

    def __init__(self, name):
        super(UNet, self).__init__()

        self.name = name

    def __repr__(self):
        return f"{self.name}"

    def __str__(self):
        return self.__repr__()


class UNetCell(UNet):
    """UNet model for cell segmentation"""

    def __init__(self, in_channels, out_channels, device, name='UNet', data_loader=None, *args, **kwargs):
        super(UNetCell, self).__init__(name)

        self.conv_ch = 8
        self.name = name

        self.data_loader = data_loader

        self.device = device

        # ---- UNET architecture ---- #
        # Down convolutions
        self.down_conv1 = DownConv(in_channels, self.conv_ch)
        self.down_conv2 = DownConv(self.conv_ch, self.conv_ch*2)
        self.down_conv3 = DownConv(self.conv_ch*2, self.conv_ch*4)
        self.down_conv4 = DownConv(self.conv_ch*4, self.conv_ch*8)
        self.down_conv5 = DownConv(self.conv_ch*8, self.conv_ch*16)
        # Up convolutions
        self.up_conv1 = UpConv(self.conv_ch*16, self.conv_ch*8)
        self.up_conv2 = UpConv(self.conv_ch*8, self.conv_ch*4)
        self.up_conv3 = UpConv(self.conv_ch*4, self.conv_ch*2)
        self.up_conv4 = UpConv(self.conv_ch*2, self.conv_ch)
        self.out_conv = nn.Conv2d(self.conv_ch, out_channels, 1, padding=0, stride=1)
        # Last 1x1 conv layer ?

        # moves the model to the requiered device and converts to the appropiate type
        self.to(device=self.device, dtype=torch.float32)


    def forward(self, input_):
        """"Computes forward pass of UNet"""

        x, conv1 = self.down_conv1(input_)
        x, conv2 = self.down_conv2(x)
        x, conv3 = self.down_conv3(x)        
        x, conv4 = self.down_conv4(x)        
        _, conv5 = self.down_conv5(x)     

        x = self.up_conv1(conv5, conv4)        
        x = self.up_conv2(x, conv3)        
        x = self.up_conv3(x, conv2)
        x = self.up_conv4(x, conv1)
        x = self.out_conv(x)

        # Apply sigmoid at the output when not training (for the training, better stability with BCELogitLoss)
        if not self.training:
            x = torch.sigmoid(x)

        return x


    def train_model(self, epochs=100, batch_size=10, lr=0.002, gamma=0.95):
        """Trains the model for given epochs and batch_size and saves model at each epoch
            PARAMS:
                epochs (int): number of training rounds on the data
                batch_size (int): silmultaneous images to train on"""

        # Better numerical stability than BCELoss(sigmoid)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))

        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        
        print("Training on {:d} images...".format(len(self.data_loader.train_images)))
        print("\r[{:0>2d}/{}]{}  loss: {:.6}".format(0, epochs, progress_bar(0, epochs), 0.0), end=' '*10)
        

        for e in range(epochs):
            print('LR: {:.7f}'.format(scheduler.get_lr()[0]), end=' '*10)

            for batch_i, (train_input, train_target) in enumerate(self.data_loader.load_batch(batch_size)):    
                
                # Gets data to the model's device
                train_input = torch.from_numpy(train_input).to(device=self.device)
                train_target = torch.from_numpy(train_target).to(device=self.device)

                output = self.forward(train_input)
                loss = criterion(output, train_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if not batch_i % 5:
                    print("\r[{:0>2d}/{}]{}  loss: {:.6f}".format(
                        e +1, epochs, 
                        progress_bar((e *self.data_loader.n_batches) +batch_i +1, epochs *self.data_loader.n_batches,
                                    newline=False),
                        loss.item()), end='\t')

            # Generate test images in eval() mode
            self.eval()
            _, axes = plt.subplots(1, 3, figsize=(60,20))
            axes[0].imshow(train_input.cpu().numpy()[0].squeeze(), cmap='gray')
            axes[0].set_title("train_input")
            axes[1].imshow((self(train_input)).cpu().detach().numpy()[0].squeeze(), cmap='gray')
            axes[1].set_title("output")
            axes[2].imshow(train_target.cpu().numpy()[0].squeeze(), cmap='gray')
            axes[2].set_title("train_target")
            plt.savefig("outputs/fig_{:d}.png".format(e+1))
            plt.close()
            self.train() 

            scheduler.step()

            if e > 0:
                self.save_state_dir('outputs/saved_models', "{}.pth".format(self.name.lower()))
        else:
            print('')
            test_input, test_target = self.data_loader.load_test()
            torch_input = torch.from_numpy(test_input).to(device=self.device)

            test_pred = self.predict_stack(torch_input).squeeze().detach().cpu().numpy()
            
            imageio.mimsave('outputs/test_pred.gif', np.concatenate([(test_input.squeeze() *255).astype('uint8'), (test_pred *255).astype('uint8'), (test_target.squeeze() *255).astype('uint8')], axis=2))


    def predict_stack(self, stack, sampling=10):
        """Predicts a whole stack of images iterating on axis=0 every 'sampling' images
            PARAMS:
                stack (torch.Tensor): stack of 2D images dims=(n_images, y,x)
                sampling (int): sampling of the images on the zero-th axis"""

        # Use .no_grad() to save memory
        with torch.no_grad():
            pred_stack = torch.empty_like(stack)

            # Updates the running mean of the BN layers
            # for i in range(0,10):
            #     self(stack[(i*sampling):(i*sampling)+sampling])

            #TODO: add eval() and debug
            self.eval()

            for i in range(0, stack.shape[0], sampling):
                pred_stack[i:i+sampling] = self(stack[i:i+sampling])

            self.train()

        return pred_stack


    def save_state_dir(self, folderpath, filename):
        """Saves the current state of the network to a given path
            PARAMS:
                folderpath (str): folder to save the model to 
                filename (str): name of the file"""

        if not os.path.isdir(folderpath):
            try:
                os.makedirs(folderpath, exist_ok=True)
            except:
                print ("Creation of the directory %s failed" % 'saved_models')
                raise
            else:
                print ("Successfully created the directory %s " % 'saved_models')

        torch.save(self.state_dict(), os.path.join(folderpath, filename))


class UNetPili(UNet):
    """UNet model for pili segmentation"""

    def __init__(self, in_channels, out_channels, name, device, data_loader=None, *args, **kwargs):
        super(UNetPili, self).__init__(name)

        self.conv_ch = 8
        self.name = name

        self.data_loader = data_loader

        self.device = device

        # ---- UNET architecture ---- #
        # Down convolutions
        self.down_conv1 = DownConv(in_channels, self.conv_ch)
        self.down_conv2 = DownConv(self.conv_ch, self.conv_ch*2)
        self.down_conv3 = DownConv(self.conv_ch*2, self.conv_ch*4)
        self.down_conv4 = DownConv(self.conv_ch*4, self.conv_ch*8)
        self.down_conv5 = DownConv(self.conv_ch*8, self.conv_ch*16)
        # Up convolutions
        self.up_conv1 = UpConv(self.conv_ch*16, self.conv_ch*8)
        self.up_conv2 = UpConv(self.conv_ch*8, self.conv_ch*4)
        self.up_conv3 = UpConv(self.conv_ch*4, self.conv_ch*2)
        self.up_conv4 = UpConv(self.conv_ch*2, self.conv_ch)
        self.out_conv = nn.Conv2d(self.conv_ch, out_channels, 1, padding=0, stride=1)
        # Last 1x1 conv layer ?

        # moves the model to the requiered device and converts to the appropiate type
        self.to(device=self.device, dtype=torch.float32)


    def forward(self, input_):
        """"Computes forward pass of UNet"""

        x, conv1 = self.down_conv1(input_)
        x, conv2 = self.down_conv2(x)
        x, conv3 = self.down_conv3(x)        
        x, conv4 = self.down_conv4(x)        
        _, conv5 = self.down_conv5(x)     

        x = self.up_conv1(conv5, conv4)        
        x = self.up_conv2(x, conv3)        
        x = self.up_conv3(x, conv2)
        x = self.up_conv4(x, conv1)
        x = self.out_conv(x)

        # Apply sigmoid at the output when not training (for the training, better stability with BCELogitLoss)
        if not self.training:
            x = torch.sigmoid(x)

        return x


    def train_model(self, epochs=100, batch_size=10):
        """Trains the model for given epochs and batch_size and saves model at each epoch
            PARAMS:
                epochs (int): number of training rounds on the data
                batch_size (int): silmultaneous images to train on"""

        # Better numerical stability than BCELoss(sigmoid)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))
        
        print("Training on {:d} images...".format(len(self.data_loader.train_images)))
        print("\r[{}/{}]{}  loss: {:.6}".format(0, epochs, progress_bar(0, epochs), 0.0), end=' '*10)

        for e in range(epochs):
            for batch_i, (train_input, train_target) in enumerate(self.data_loader.load_batch(batch_size)):    
                
                # Gets data to the model's device
                train_input = torch.from_numpy(train_input).to(device=self.device)
                train_target = torch.from_numpy(train_target).to(device=self.device)

                output = self.forward(train_input)
                loss = criterion(output, train_target) 

                l1_reg = torch.tensor(0).float().to(self.device)
                for param in self.parameters():
                    l1_reg += torch.norm(param, 1)

                optimizer.zero_grad()
                (loss + 1e-5*l1_reg).backward()
                optimizer.step()
                
                print("\r[{}/{}]{}  loss: {:.6f}".format(
                    e +1, epochs, 
                    progress_bar((e*self.data_loader.n_batches) +batch_i +1, epochs *self.data_loader.n_batches,
                                 newline=False),
                    loss.item()), end=' '*10)

            # Generate test images in eval() mode
            self.eval()
            fig, axes = plt.subplots(1, 3, figsize=(60,20))
            axes[0].imshow(train_input.cpu().numpy()[0].squeeze(), cmap='gray')
            axes[0].set_title("train_input")
            axes[1].imshow((self(train_input)).cpu().detach().numpy()[0].squeeze(), cmap='gray')
            axes[1].set_title("output")
            axes[2].imshow(train_target.cpu().numpy()[0].squeeze(), cmap='gray')
            axes[2].set_title("train_target")
            plt.savefig(f"outputs/fig_{e+1:d}.png")
            plt.close()
            self.train()

            if e > 0 and not (e+1) %5:
                self.save_state_dir('outputs/saved_models', "{}.pth".format(self.name.lower()))
        else:
            print('')
            self.eval()
            test_img, test_mask = self.data_loader.load_test()
            test_pred = self(torch.from_numpy(test_img).to(self.device)).detach().cpu().numpy().squeeze()
            imageio.mimsave('outputs/test_pred.gif', np.concatenate([(np.squeeze(test_img, 1) * 255).astype('uint8'),
                                                       (test_pred * 255).astype('uint8'),
                                                       (np.squeeze(test_mask, 1) * 255).astype('uint8')], axis=2))


    def predict_stack(self, stack, batch_size=10):
        """Predicts a whole stack of images iterating on axis=0 every 'sampling' images
        prevents memory for possibly very large stacks
            PARAMS:
                stack (torch.Tensor): stack of 2D images dims=(n_images, y,x)
                sampling (int): sampling of the images on the zero-th axis"""

        print("Predicting stack...")
        # Use .no_grad() to save memory
        with torch.no_grad():
            self.eval()
            pred_stack = torch.empty_like(stack)

            # Updates the running mean of the BN layers
            for i in range(0,10):
                self(stack[(i*batch_size):(i*batch_size)+batch_size])

            #TODO: add eval() and debug
            # self.eval()

            for i in range(0, stack.shape[0], batch_size):
                print("\r{}".format(progress_bar(i, stack.shape[0], newline=False)), end='')
                pred_stack[i:i+batch_size] = torch.sigmoid(self(stack[i:i+batch_size]))

            print("\r{}".format(progress_bar(stack.shape[0], stack.shape[0])), end='')
            self.train()

        return pred_stack


    def save_state_dir(self, folderpath, filename):
        """Saves the current state of the network to a given path
            PARAMS:
                folderpath (str): folder to save the model to 
                filename (str): name of the file"""

        if not os.path.isdir(folderpath):
            try:
                os.makedirs(folderpath, exist_ok=True)
            except:
                print ("Creation of the directory %s failed" % 'saved_models')
                raise
            else:
                print ("Successfully created the directory %s " % 'saved_models')

        torch.save(self.state_dict(), os.path.join(folderpath, filename))




