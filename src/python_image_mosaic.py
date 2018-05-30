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


class image_mosiac():
    """
    This class creates image mosaics in python using numpy and PIL (Pillow) 
    -An input image is used to create a new image from 'tiles' (other images)    
    """
    def __init__(self, inputImageFile, tilesFiles, numTiles):
        self.tile_max_px = numpy.sqrt(numTiles)
        
        self.tilesFiles = tilesFiles
        
        self.input_image = Image.open(inputImageFile)
        self.np_input_image = numpy.array(self.input_image)
    
    def scaleInputImage(self, image, np_image):
        xdiv = (np_image.shape[0]/self.tile_max_px)
        ydiv = (np_image.shape[1]/self.tile_max_px)
        
        adjustScalex = int(self.tile_max_px * xdiv)
        adjustScaley = int(self.tile_max_px * ydiv)
        
        size = (adjustScalex, adjustScaley)
        return image.resize(size, Image.ANTIALIAS)
    
    def func(self, t):
        if (t > 0.008856):
            return numpy.power(t, 1/3.0)
        else:
            return 7.787 * t + 16 / 116.0
        
    def rgb_to_lab(self, rgb):
        np_rgb = numpy.array(rgb)
        
        normalized_rgb = np_rgb / 255
    
        cie = numpy.dot(self.conversion_matrix, normalized_rgb)
        
        cie[0] = cie[0] /0.950456
        cie[2] = cie[2] /1.088754 
    
        L = 116 * numpy.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1]
        a = 500*(self.func(cie[0]) - self.func(cie[1]))
        b = 200*(self.func(cie[1]) - self.func(cie[2]));
        
        return [L, a, b]
    
    def transformMatrixRGBtoLAB(self, numpyArr):
        size = numpyArr.shape
        x = size[0]
        y = size[1]

        for i in range(0, x-1):
            for j in range(0, y-1):
                tempArr = numpyArr[i, j]
                tempArrTransform = self.rgb2lab(tempArr) 
                numpyArr[i, j] = tempArrTransform
    
        return numpyArr 
    
    def getEuclideanDistance(self, x, y):
        return numpy.linalg.norm(x-y)
    
    def MaxFunction(self, input_dict, cmp):
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
        for file in self.tilesFiles:
            tile_pic = Image.open(file)
            tile_np = numpy.array(tile_pic)
            
            size = (self.tile_max_x_px, self.tile_max_y_px)
            tile_pic_mini = tile_pic.resize(size)
            tile_np_mini = numpy.array(tile_pic_mini)
            
            title_np_ave = numpy.mean(tile_np_mini, axis=(0, 1))

            
            self.tileDict[file] = [tile_np_mini, title_np_ave]
    
    def iterateThroughArray(self):
        xdiv = self.tile_max_x_px
        ydiv = self.tile_max_y_px
        
        for i in numpy.arange(xdiv-1,self.np_scaled_input_image.shape[0],xdiv):
            for j in numpy.arange(ydiv-1,self.np_scaled_input_image.shape[1],ydiv):
                int_i = int(i)
                int_j = int(j)
                
                sub_matrix =self.np_scaled_input_image[int_i-(xdiv-1):int_i+1, int_j-(ydiv-1):int_j+1].copy()
                sub_average = numpy.mean(sub_matrix, axis=(0, 1))
       
                maximum = self.MaxFunction(self.tileDict, sub_average)  
                maxKey = list(maximum.keys())[0]
        
                self.np_scaled_input_image[int_i-(xdiv-1):int_i+1, int_j-(ydiv-1):int_j+1] = self.tileDict[(maxKey)][0]


    
    
if __name__ == "__main__":
    try:
        wk_dir =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    else:
        wk_dir =  os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
        
    mainImageSubdir = "\\Example Output\\"
    tileImagesSubdir = "\\DB\\"
    mainImageTarget = "1_in.jpg"

    path = os.path.normpath(wk_dir)
    output_image = "out_" + mainImageTarget
    
    files = [path + tileImagesSubdir + f for f in listdir(path + tileImagesSubdir) if isfile(join(path + tileImagesSubdir, f))]
    mainImagePath = path+mainImageSubdir+mainImageTarget
