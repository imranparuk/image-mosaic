# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:03:36 2018

@author: imranparuk
"""

from PIL import Image
import numpy
from skimage import io, color

import os
from os import listdir
from os.path import isfile, join

from pathlib import Path

import sys
from sys import argv

import argparse
from argparse import ArgumentParser


#used to show that program is still running
import spinner

class image_mosiac():
    """
    This class creates image mosaics in python using numpy and PIL (Pillow) 
    -An input image is used to create a new image from 'tiles' (other images)    
    """
    def __init__(self, input_image_file, tiles_files, num_tiles, comparison_mode=1, transform=1):
        """
        Python class constructor
            - initilizes all class objects
        """
        self.transform = transform
        self.comparison_mode = comparison_mode
        
        if not(len(num_tiles) > 1):
            tile_max_px = numpy.sqrt(num_tiles)
            self.num_tiles_x = tile_max_px
            self.num_tiles_y = tile_max_px
        else:
            self.num_tiles_x = num_tiles[0]
            self.num_tiles_y = num_tiles[1]
       
        self.tiles_files = tiles_files
        self.N = float(4.0 / 29.0)

        self.input_image = Image.open(input_image_file)
        self.np_input_image = numpy.array(self.input_image)

        self.scaled_input_image = self.scaleInputImage(self.input_image, self.np_input_image)
        self.np_scaled_input_image = numpy.array(self.scaled_input_image)
               
        self.tile_max_x_px = int(self.np_scaled_input_image.shape[1]/self.num_tiles_x)
        self.tile_max_y_px = int(self.np_scaled_input_image.shape[0]/self.num_tiles_y)
        
        self.conversion_matrix = numpy.array( [[0.4124564,  0.3575761,  0.1804375],
                                       [0.2126729,  0.7151522,  0.0721750],
                                       [0.0193339,  0.1191920,  0.9503041]])
    
        self.illumination_matrix = numpy.array([95.047, 100.0, 108.883])
        
        self.tile_dict = {}
        self.populate_tile_dict()
        self.iterate_through_array()
  
    def scale_input_image(self, image, np_image):
        """
        This function scales in input image to ensure you can fit all the tiles in
        it takes the current size of 'image', tries to fit as many tiles in, then
        rounds that number off to a whole number and creates an optiumum size 
        which is then used to resize the original image (maintaining original aspect ratio)
        """
        xdiv = (np_image.shape[1]/self.num_tiles_x)
        ydiv = (np_image.shape[0]/self.num_tiles_y)
        
        #int (round down always) or round (actual rounding to nearest int)
        adjustScalex = int(self.num_tiles_x) * int(xdiv)
        adjustScaley = int(self.num_tiles_y) * int(ydiv)
        
        size = (adjustScalex, adjustScaley)
        return image.resize(size, Image.ANTIALIAS)
            
    def scipy_rgb_to_lab(self, rgb):
        """
        This function is used to convert RGB from the color-space to the CIE*Lab color-space
        with the use of SkImage image processing libaray. Its used as a comparison to the results of
        'rgb_to_lab'
        """
        np_rgb = numpy.array([[rgb]])
        lab = color.rgb2lab(np_rgb/255)
        return lab


    def convert_rgb_to_srgb(self, rgb_val):
        '''
        This function converts from the RGB color-space to the sRGB color space
        Transformation formulae from: 
            https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
                
            V' = {   V/12.92                   if (v < 0.04545)
                 {   ((V+0.055)/1.055)^2.4     else
        '''
        in_col_np = numpy.array(rgb_val)
        in_col_np_norm = in_col_np/255
        return numpy.array([ numpy.power(((value + 0.055) / 1.055), 2.4) if value > 0.04045 else value / 12.92 for value in in_col_np_norm])
    
       
    def convert_srgb_to_lab(self, srgb_val):
        '''
        This function converts sRGB to the CIE-XYZ, then to the CIE-*Lab color spaces respectively. 
        following the algorithm described in these links:
            http://www.easyrgb.com/en/math.php
            https://stackoverflow.com/a/5021831
            https://en.wikipedia.org/wiki/CIELAB_color_space#Forward_transformation
            
        Assumed: Under Illuminant D65 with normalization Y = 100
        '''
        srgbval_norm = srgb_val.dot(100)

        dot_prod = self.conversion_matrix.dot(srgbval_norm)
        dot_prod_illumin = numpy.array([dot_prod[0]/self.illumination_matrix[0], dot_prod[1]/self.illumination_matrix[1], dot_prod[2]/self.illumination_matrix[2] ])
        
        XYZ = [ (numpy.power(value, 1/3) if (value > 0.008856) else (7.787 * value) + (16 / 116) ) for value in dot_prod_illumin]
    
        L = (116 * XYZ[1]) - 16
        a = 500 * (XYZ[0] - XYZ[1])
        b = 200 * (XYZ[1] - XYZ[2])   
    
        return numpy.array([L,a,b])
    
    def my_rgb_to_lab(self, input_color):
        '''
        this function converts from colors from the RGB color-space to the CIE-Lab color space
        '''
        srgb = self.convert_rgb_to_srgb(input_color)
        return  self.convert_srgb_to_lab(srgb)
    
    def transform_matrix_convert(self, np_arr):
        """
        Transforms RGB numpy matrix to LAB numpy matrix
        
        transform done based on user requirements, this is set in 'self.transform'
        [1] general RGB 
        [2] RGB -> sRGB
        [3] RGB -> LAB (using my own implimentation)
        [4] RGB -> LAB (using built in library)
        """
        # todo: fix this, its ugly
        size = np_arr.shape
        x = size[0]
        y = size[1]
        for i in range(0, x-1):
            for j in range(0, y-1):
                temp_arr = np_arr[i, j]
                if self.transform == 1:
                    temp_arr_transform = temp_arr
                elif self.transform == 2:
                    temp_arr_transform = self.convert_rgb_to_srgb(tempArr)
                elif self.transform == 3:
                    temp_arr_transform = self.my_rgb_to_lab(tempArr)
                else:
                    temp_arr_transform = self.scipy_rgb_to_lab(tempArr)
                np_arr[i, j] = temp_arr_transform
        return np_arr

    @staticmethod
    def get_eucledian_dist(x, y):
        """
        Calculates the Euclidean Distance based off the following equasion
        #E_d(x,y) = sqrt( square(Ax-Ay) + square(Bx-By) + square(Cx-Cy) ).
        """
        return numpy.linalg.norm(x-y)
    
    def max_function(self, input_dict, cmp):
        """
        This custom max function returns a dictionary of item in 'input_dict'
        which has the lowest Euclidean distance to 'cmp'. It returns a dictionary
        with one item, which is the dictionary item which the lowest distance
        """
        max_item = {}
            
        for key, value in input_dict.items():   
            exld = self.get_eucledian_dist(value[1], cmp)
            try:
                list(max_item.values())[0]
            except IndexError:
                max_item.update({key:exld})
            else:
                new_exld = self.get_eucledian_dist(value[1], cmp)
                if new_exld < list(max_item.values())[0]:
                    max_item.clear()
                    max_item.update({key:exld})
        return max_item
    
    def populate_tile_dict(self):
        """
        Firstly, this function converts the list of image files into image objects as well as numpy objects
        it also creates small tiles that will fit in the max tile size that will later be used to create the
        final image
        Secondly, based on the mode different means are calculated
        Mode 1: Mean is of the transformed small tile images 
        Mode 2: Mean is of the transformed large tile images
        """
        for file in self.tiles_files:
            tile_pic = Image.open(file)
            tile_np = numpy.array(tile_pic)
            
            size = (self.tile_max_x_px, self.tile_max_y_px)
            tile_pic_mini = tile_pic.resize(size)
            
            tile_np_mini = numpy.array(tile_pic_mini)
            
            if self.comparison_mode == 1:
                title_np_ave = numpy.mean(self.transform_matrix_convert(tile_np_mini.copy()), axis=(0, 1))
            else:
                title_np_ave = numpy.mean(self.transform_matrix_convert(tile_np.copy()), axis=(0, 1))

            self.tile_dict[file] = [tile_np_mini, title_np_ave]
    
    def iterate_through_array(self):
        """
        This function is used to iterate through the main image matrix pixels. 
        A sub-matrix is extracted and the mean of its image domain is calculated
        
        if matrix A is below:
        
        |0  1  2  3 |     -> each element is extracted and a tile matched to it via
        |4  5  6  7 |        the custom max_function using the max Euclidean distace
        |8  9  10 11|        after which, this elemet is replaced by the matrix of 
        |12 13 14 15|        the matched tile
        
        """
        xdiv = self.tile_max_x_px
        ydiv = self.tile_max_y_px
        
        for i in numpy.arange(xdiv-1, self.np_scaled_input_image.shape[1]+1,xdiv):
            for j in numpy.arange(ydiv-1, self.np_scaled_input_image.shape[0],ydiv):
                int_i = int(i)
                int_j = int(j)
                
                sub_matrix = self.np_scaled_input_image[int_j-(ydiv-1):int_j, int_i-(xdiv-1):int_i].copy()
                sub_average = numpy.mean(self.transform_matrix_convert(sub_matrix), axis=(0, 1))
       
                maximum = self.max_function(self.tile_dict, sub_average)
                maxKey = list(maximum.keys())[0]
                        
                self.np_scaled_input_image[int_j-(ydiv-1):int_j+1, int_i-(xdiv-1):int_i+1] = self.tile_dict[maxKey][0]
    
    def return_image(self):
        """
        This function is used to return a resulting image after all the image processing is done.
        It returns a standard RGB image that can be saved.
        """
        return Image.fromarray(self.np_scaled_input_image, 'RGB')   
    

class ModeAction(argparse.Action):
    def __call__(self, parser_, namespace, values, option_string=None):
        if values not in range(1, 5):
            parser_.error("{0} is not a valid mode. Refer to github doc -> https://github.com/imranparuk/image-mosaic"
                          .format(option_string))
        setattr(namespace, self.dest, values)


class TransformAction(argparse.Action):
    def __call__(self, parser_, namespace, values, option_string=None):
        if values not in range(1, 5):
            parser_.error("{0} is not a valid transform. Refer to github doc -> https://github.com/imranparuk/image-"
                          "mosaic".format(option_string))
        setattr(namespace, self.dest, values)
         
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-m','--mode',action=ModeAction, type=int)
    parser.add_argument('-trans', '-tr', '--transform', action=TransformAction,type=int)
    parser.add_argument('-t', '--tiles', type=int, nargs='+')
    parser.add_argument('-fi', '--file_in', type = str, default='1_in.jpg')
    parser.add_argument('-fo', '--file_out', type = str, default='1_out.jpg')

    parser.add_argument('-d', '--dir', type = str, default='../images/')

    args = vars(parser.parse_args())

    mode = args['mode']
    transform = args['transform']
    num_tiles = args['tiles']
    target_image = args['file']
    target_dir = args['dir']

    # todo: fix this, still ugly
    path = Path(target_dir)
    tiles_dir = path / Path('reference_files/DB')
    output_dir = path / Path("out")
    in_file = path / Path("../images") / Path(target_image)
    out_file = output_dir / Path("out_1.jpg")

    print("Input Image File:= {0}".format(in_file))
    print("Output file:= {0}".format(out_file))
    print("Tile Images Directory:= {0}".format(tiles_dir))
    print("Output Directory:= {0}".format(output_dir))

    if (len(num_tiles) > 1):
        num_tiles_tot = num_tiles[0] * num_tiles[1]
        print("Number of Tiles:= {0}".format(num_tiles_tot))
    else:
        print("Number of Tiles:= {0}".format(*num_tiles))

    print()
    
    sys.stdout.write("Program Running: ")
    spinner = spinner.Spinner()
    spinner.start()

    temp_tile_path = Path(os.path.abspath(str(tiles_dir)))

    files = [str(temp_tile_path / f) for f in listdir(str(temp_tile_path)) if isfile(str(temp_tile_path / Path(f)))]

    image_mos = image_mosiac(target_image_file, files, num_tiles, mode, transform)

    image = image_mos.return_image()
    image.show()

    with open(str(out_file), 'w') as f:
        image.save(f)
    
    spinner.stop()
    print("Done.")

    #eof















