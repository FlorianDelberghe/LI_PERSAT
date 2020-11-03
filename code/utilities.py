import os
import sys
import glob
import re

import numpy as np
import skimage.external.tifffile as tifffile
import cv2


def progress_bar(pos, total, length=50, newline=True):
    """Creates a string with the progression of the algorithm
        PARAMS:
            pos, total (int/float): current algo progression with max value
            lenght (int): lenght in characters of the progress bar
            newline (bool): whether or not the string has a new line when the progression bar is Done

        RETURNS:
            progress_bar (str): [{# *pos}{- *(1-pos)}]
    """

    if pos == total:
        return '[{:s}]\n'.format('#'*length) if newline else '[{:s}]'.format('#'*length)

    rel_pos = int(pos/total*length)
    return("[{0:-<{length:d}s}]".format('#'*rel_pos, length=length))


def load_img_path(folder_path, file_type='.tif'):
    """Returns the filepaths of the images in the given folder"""

    return glob.glob("{}*{}".format(folder_path, file_type))


def load_imgs(imgs_paths):
    """Returns images as numpy array from the imgs_paths
        PARAMS:
            imgs_paths (str / list(str)): path of the images to load (relative or absolute)
                type str or list (changes type of the output)
        RETURNS:    
            images (np.array / list(np.array)): loaded images as numpy arrays 
                single array if imgs_paths in of type str
                list of arrays if imgs_paths in of type list
    """

    if isinstance(imgs_paths, str):
        print("Loading: {}".format(imgs_paths))
        images = tifffile.imread(imgs_paths)

    elif isinstance(imgs_paths, list):
        # Returns loaded images as a list
        images = []
        for filename in imgs_paths:
            print("Loading: {}".format(filename))
            images.append(tifffile.imread(filename))
    else:
        raise ValueError("Wrong type from imgs_paths: {}".format(type(imgs_paths)))

    return images


def load_data_paths(dataset, 
                    pattern='/cam1/event[0-9]_tirf/*PreNbin*.tif',
                    file_mnt="mnt/plabNAS/"):
    """Loads the paths of curated data sets
        ARGUMENTS:
            dataset (str): the name of the dataset that we want to load
            pattern (str): file pattern to be searched by glob
            file_mnt (str): where the labNAS is mounted
    """

    files = []

    with open('data/curated_data.txt', 'rt') as f:
        for line in f:
            if line.lower().startswith("*{}".format(dataset.lower())):
                for line in f:
                    # the dataest is finished
                    if line.lower().startswith('*'):
                        break
                    # empty line
                    if not line.strip():
                        continue
                    else:
                        files.append(re.sub(r'\\', r'/', '/' +file_mnt +line.strip()))
    
    files = [glob.glob(file+ pattern) for file in files]
    files = [item for sublist in files for item in sublist]

    return files


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

