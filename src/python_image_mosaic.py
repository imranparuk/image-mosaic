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
    def __init__(self, inputImageFile, tilesFiles, numTiles, comparisonMode=1, transform=1):
        """
        Python class constructor
            - initilizes all class objects
        """
        self.transform = transform
        self.comparisonMode = comparisonMode
        
        if not(len(numTiles) > 1):
            tile_max_px = numpy.sqrt(numTiles)
            self.num_tiles_x = tile_max_px
            self.num_tiles_y = tile_max_px
        else:
            self.num_tiles_x = numTiles[0]
            self.num_tiles_y = numTiles[1]
       
        self.tilesFiles = tilesFiles
        self.N = float(4.0 / 29.0)

        self.input_image = Image.open(inputImageFile)
        self.np_input_image = numpy.array(self.input_image)

        self.scaled_input_image = self.scaleInputImage(self.input_image, self.np_input_image)
        self.np_scaled_input_image = numpy.array(self.scaled_input_image)
        
        print(self.np_scaled_input_image.shape)
        
        self.tile_max_x_px = int(self.np_scaled_input_image.shape[1]/self.num_tiles_x)
        self.tile_max_y_px = int(self.np_scaled_input_image.shape[0]/self.num_tiles_y)
        
        self.conversion_matrix = numpy.array( [[0.4124564,  0.3575761,  0.1804375],
                                       [0.2126729,  0.7151522,  0.0721750],
                                       [0.0193339,  0.1191920,  0.9503041]])
    
        self.illumination_matrix = numpy.array([95.047, 100.0, 108.883])
        
        self.tileDict = {}
        self.populateTileDict()
        self.iterateThroughArray()
  
    def scaleInputImage(self, image, np_image):
        """
        This function scales in input image to ensure you can fit all the tiles in
        it takes the current size of 'image', tries to fit as many tiles in, then
        rounds that number off to a whole number and creates an optiumum size 
        which is then used to resize the original image (maintaining original aspect ratio)
        """
        xdiv = (np_image.shape[1]/self.num_tiles_x)
        ydiv = (np_image.shape[0]/self.num_tiles_y)

        adjustScalex = int(self.num_tiles_x * xdiv)
        adjustScaley = int(self.num_tiles_y * ydiv)
        
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


    def convert_rgb_to_srgb(self, rgbVal):
        '''
        This function converts from the RGB color-space to the sRGB color space
        Transformation formulae from: 
            https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
                
            V' = {   V/12.92                   if (v < 0.04545)
                 {   ((V+0.055)/1.055)^2.4     else
        '''
        in_col_np = numpy.array(rgbVal)
        in_col_np_norm = in_col_np/255
        return numpy.array([ numpy.power(((value + 0.055) / 1.055), 2.4) if value > 0.04045 else value / 12.92 for value in in_col_np_norm])
    
       
    def convert_srgb_to_lab(self, srgbVal):
        '''
        This function converts sRGB to the CIE-XYZ, then to the CIE-*Lab color spaces respectively. 
        following the algorithm described in these links:
            http://www.easyrgb.com/en/math.php
            https://stackoverflow.com/a/5021831
            https://en.wikipedia.org/wiki/CIELAB_color_space#Forward_transformation
            
        Assumed: Under Illuminant D65 with normalization Y = 100
        '''
        srgbVal_norm = srgbVal.dot(100)

        dot_prod = self.conversion_matrix.dot(srgbVal_norm)
        dot_prod_illumin = numpy.array([dot_prod[0]/self.illumination_matrix[0], dot_prod[1]/self.illumination_matrix[1], dot_prod[2]/self.illumination_matrix[2] ])
        
        XYZ = [ (numpy.power(value, 1/3) if (value > 0.008856) else (7.787 * value) + (16 / 116) ) for value in dot_prod_illumin]
    
        L = (116 * XYZ[1]) - 16
        a = 500 * (XYZ[0] - XYZ[1])
        b = 200 * (XYZ[1] - XYZ[2])   
    
        return numpy.array([L,a,b])
    
    def my_rgb_to_lab(self, inputColor):
        '''
        this function converts from colors from the RGB color-space to the CIE-Lab color space
        '''
        srgb = self.convert_rgb_to_srgb(inputColor)
        return  self.convert_srgb_to_lab(srgb)
    
    def transformMatrixConvert(self, numpyArr):
        """
        Transforms RGB numpy matrix to LAB numpy matrix
        
        transform done based on user requirements, this is set in 'self.transform'
        [1] general RGB 
        [2] RGB -> sRGB
        [3] RGB -> LAB (using my own implimentation)
        [4] RGB -> LAB (using built in library)
        """
        size = numpyArr.shape
        x = size[0]
        y = size[1]
        for i in range(0, x-1):
            for j in range(0, y-1):
                tempArr = numpyArr[i, j]        
                if (self.transform == 1):
                    tempArrTransform = tempArr
                elif (self.transform == 2):
                    tempArrTransform = self.convert_rgb_to_srgb(tempArr)
                elif (self.transform == 3):
                    tempArrTransform = self.my_rgb_to_lab(tempArr)
                else:
                    tempArrTransform = self.scipy_rgb_to_lab(tempArr)
                numpyArr[i, j] = tempArrTransform
        return numpyArr            
    
    def getEuclideanDistance(self, x, y):
        """
        Calculates the Euclidean Distance based off the following equasion
        #E_d(x,y) = sqrt( square(Ax-Ay) + square(Bx-By) + square(Cx-Cy) ).
        """
        return numpy.linalg.norm(x-y)
    
    def MaxFunction(self, input_dict, cmp):
        """
        This custom max function returns a dictionary of item in 'input_dict'
        which has the lowest Euclidean distance to 'cmp'. It returns a dictionary
        with one item, which is the dictionary item which the lowest distance
        """
        maxItem = {}
            
        for key, value in input_dict.items():   
            exld = self.getEuclideanDistance(value[1], cmp)
            try:
                list(maxItem.values())[0]
            except IndexError:
                maxItem.update({key:exld})
            else:
                new_exld = self.getEuclideanDistance(value[1], cmp) 
                if (new_exld < list(maxItem.values())[0] ):
                    maxItem.clear()
                    maxItem.update({key:exld})
        return maxItem
    
    def populateTileDict(self):
        """
        Firstly, this function converts the list of image files into image objects as well as numpy objects
        it also creates small tiles that will fit in the max tile size that will later be used to create the
        final image
        Secondly, based on the mode different means are calculated
        Mode 1: Mean is of the transformed small tile images 
        Mode 2: Mean is of the transformed large tile images
        """
        for file in self.tilesFiles:
            tile_pic = Image.open(file)
            tile_np = numpy.array(tile_pic)
            
            size = (self.tile_max_x_px, self.tile_max_y_px)
            tile_pic_mini = tile_pic.resize(size)
            
            tile_np_mini = numpy.array(tile_pic_mini)
            
            if (self.comparisonMode == 1):    
                title_np_ave = numpy.mean(self.transformMatrixConvert(tile_np_mini.copy()), axis=(0, 1))
            else:
                title_np_ave = numpy.mean(self.transformMatrixConvert(tile_np.copy()), axis=(0, 1))

            
            self.tileDict[file] = [tile_np_mini, title_np_ave]
    
    def iterateThroughArray(self):
        """
        This function is used to iterate through the main image matrix pixels. 
        A sub-matrix is extracted and the mean of its image domain is calculated
        
        if matrix A is below:
        
        |0  1  2  3 |     -> each element is extracted and a tile matched to it via
        |4  5  6  7 |        the custom MaxFunction using the max Euclidean distace
        |8  9  10 11|        after which, this elemet is replaced by the matrix of 
        |12 13 14 15|        the matched tile
        
        """
        xdiv = self.tile_max_x_px
        ydiv = self.tile_max_y_px
        
        print(self.np_scaled_input_image.shape[1])
        print(numpy.arange(19,750,20))
        
        for i in numpy.arange(xdiv-1,self.np_scaled_input_image.shape[1]+1,xdiv):
            for j in numpy.arange(ydiv-1,self.np_scaled_input_image.shape[0],ydiv):
                int_i = int(i)
                int_j = int(j)
                
                sub_matrix =self.np_scaled_input_image[int_j-(ydiv-1):int_j+1, int_i-(xdiv-1):int_i+1].copy()
                sub_average = numpy.mean(self.transformMatrixConvert(sub_matrix) , axis=(0, 1)) 
       
                maximum = self.MaxFunction(self.tileDict, sub_average)  
                maxKey = list(maximum.keys())[0]
                        
                #self.np_scaled_input_image[int_i-(xdiv-1):int_i+1, int_j-(ydiv-1):int_j+1] = self.tileDict[(maxKey)][0]
                self.np_scaled_input_image[int_j-(ydiv-1):int_j+1, int_i-(xdiv-1):int_i+1] = self.tileDict[(maxKey)][0]
    def returnImage(self):
        """
        This function is used to return a resulting image after all the image processing is done.
        It returns a standard RGB image that can be saved.
        """
        return Image.fromarray(self.np_scaled_input_image, 'RGB')   
    
    
####################################################################################################################################################
       
class isModeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
         if values not in range(1, 5):
            parser.error("{0} is not a valid mode. Refer to github doc -> https://github.com/imranparuk/image-mosaic".format(option_string))
            #raise argparse.ArgumentError("Minimum bandwidth is 12")

         setattr(namespace, self.dest, values)
         
class isTransformAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
         if values not in range(1, 5):
            parser.error("{0} is not a valid transform. Refer to github doc -> https://github.com/imranparuk/image-mosaic".format(option_string))
            #raise argparse.ArgumentError("Minimum bandwidth is 12")

         setattr(namespace, self.dest, values)
         
if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('-m','--mode',action=isModeAction, type=int)
    parser.add_argument('-trans', '-tr', '--transform', action=isTransformAction,type=int)
    parser.add_argument('-t', '--tiles', type=int, nargs='+')
    parser.add_argument('-f', '--file', type = str)   
    args = vars(parser.parse_args())

    #& user defined if need be. 
    mode = args['mode']
    transform = args['transform']
    numberOfTiles = args['tiles']
    mainImageTarget = args['file']
    #&
        
    try:
        wk_dir =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    else:
        wk_dir =  os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
        
    mainImageSubdir = "\\Example Output\\"
    tileImagesSubdir = "\\DB\\"
    path = os.path.normpath(wk_dir)
    output_image = "out_" + mainImageTarget

    print("Target Image:= {0}".format(path+mainImageSubdir+mainImageTarget))
    print("Tile Images Directory:= {0}".format(path + tileImagesSubdir))
    print("Number of Tiles:= {0}".format(numberOfTiles))
    
    sys.stdout.write("Program Running: ")
    spinner = spinner.Spinner()
    spinner.start()
    
    files = [path + tileImagesSubdir + f for f in listdir(path + tileImagesSubdir) if isfile(join(path + tileImagesSubdir, f))]
    mainImagePath = path+mainImageSubdir+mainImageTarget

    image_mos = image_mosiac(mainImagePath, files, numberOfTiles, mode, transform)

    image = image_mos.returnImage()
    with open(path+mainImageSubdir+output_image, 'w') as f:
        image.save(f)
    
    spinner.stop()
    print("Done.")

    print("Output file:= {0}".format(path+mainImageSubdir+output_image))

    image.show()
    #eof















