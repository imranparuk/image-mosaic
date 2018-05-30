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
        pass
    
    def getEuclideanDistance(self, x, y):
        pass
    
    def populateTileDict(self):
        pass
    
    def iterateThroughArray(self):
        pass


    
    
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
