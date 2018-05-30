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

#used to show that program is still running
import spinner


class image_mosiac():
    """
    This class creates image mosaics in python using numpy and PIL (Pillow) 
    -An input image is used to create a new image from 'tiles' (other images)    
    """
    def __init__(self, inputImageFile, tilesFiles, numTiles, comparisonMode, useSkImage=False):
        """
        Python class constructor
            - initilizes all class objects
        """
        self.useSkImage = useSkImage
        self.comparisonMode = comparisonMode
        self.tile_max_px = numpy.sqrt(numTiles)
        
        self.tilesFiles = tilesFiles
        
        self.input_image = Image.open(inputImageFile)
        self.np_input_image = numpy.array(self.input_image)

        self.scaled_input_image = self.scaleInputImage(self.input_image, self.np_input_image)
        self.np_scaled_input_image = numpy.array(self.scaled_input_image)
        
        self.tile_max_x_px = int(self.np_scaled_input_image.shape[0]/self.tile_max_px)
        self.tile_max_y_px = int(self.np_scaled_input_image.shape[1]/self.tile_max_px)
        
        self.conversion_matrix  = [[0.412453, 0.357580, 0.180423],
                                   [0.212671, 0.715160, 0.072169],
                                   [0.019334, 0.119193, 0.950227]]
        
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
        xdiv = (np_image.shape[0]/self.tile_max_px)
        ydiv = (np_image.shape[1]/self.tile_max_px)
        
        adjustScalex = int(self.tile_max_px * xdiv)
        adjustScaley = int(self.tile_max_px * ydiv)
        
        size = (adjustScalex, adjustScaley)
        return image.resize(size, Image.ANTIALIAS)
        
    def func(self, t):
        """
        used by 'rgb_to_lab' function
        """
        if (t > 0.008856):
            return numpy.power(t, 1/3.0)
        else:
            return 7.787 * t + 16 / 116.0
    
    def simple_rgb_to_lab(self, rgb):
        """
        Stolen Code:
            https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
        
        used to convert RGB from the color-space to the CIE*Lab color-space
        """
        np_rgb = numpy.array(rgb)
        
        normalized_rgb = np_rgb / 255
    
        cie = numpy.dot(self.conversion_matrix, normalized_rgb)
        
        cie[0] = cie[0] /0.950456
        cie[2] = cie[2] /1.088754 
    
        L = 116 * numpy.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1]
        a = 500*(self.func(cie[0]) - self.func(cie[1]))
        b = 200*(self.func(cie[1]) - self.func(cie[2]));
        
        return [L, a, b]
    
    def scipy_rgb_to_lab(self, rgb):
        """
        This function is used to convert RGB from the color-space to the CIE*Lab color-space
        with the use of SkImage image processing libaray. Its used as a comparison to the results of
        'rgb_to_lab'
        """
        np_rgb = numpy.array([[rgb]])
        lab = color.rgb2lab(np_rgb/255)
        return lab
    
    def robust_rgb2lab(self, inputColor):
        """
        Stolen Code:
            https://gist.github.com/manojpandey/f5ece715132c572c80421febebaf66ae
        
        used to convert RGB from the color-space to the CIE*Lab color-space
        """
        num = 0
        RGB = [0, 0, 0]
        for value in inputColor:
            value = float(value) / 255
            if value > 0.04045:
                value = ((value + 0.055) / 1.055) ** 2.4
            else:
                value = value / 12.92   
            RGB[num] = value * 100
            num = num + 1
        XYZ = [0, 0, 0, ]
        X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
        Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
        Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
        XYZ[0] = round(X, 4)
        XYZ[1] = round(Y, 4)
        XYZ[2] = round(Z, 4)
        # Observer= 2°, Illuminant= D65
        XYZ[0] = float(XYZ[0]) / 95.047         # ref_X =  95.047
        XYZ[1] = float(XYZ[1]) / 100.0          # ref_Y = 100.000
        XYZ[2] = float(XYZ[2]) / 108.883        # ref_Z = 108.883  
        num = 0
        for value in XYZ:  
            if value > 0.008856:
                value = value ** (0.3333333333333333)
            else:
                value = (7.787 * value) + (16 / 116)   
            XYZ[num] = value
            num = num + 1   
        Lab = [0, 0, 0]  
        L = (116 * XYZ[1]) - 16
        a = 500 * (XYZ[0] - XYZ[1])
        b = 200 * (XYZ[1] - XYZ[2])   
        Lab[0] = round(L, 4)
        Lab[1] = round(a, 4)
        Lab[2] = round(b, 4)  
        return numpy.array(Lab)
    
    def transformMatrixRGBtoLAB(self, numpyArr):
        """
        Transforms RGB numpy matrix to LAB numpy matrix
        """
        size = numpyArr.shape
        x = size[0]
        y = size[1]

        for i in range(0, x-1):
            for j in range(0, y-1):
                tempArr = numpyArr[i, j]
                tempArrTransform = self.robust_rgb2lab(tempArr) if (self.useSkImage == False) else self.scipy_rgb_to_lab(tempArr)
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
        Mode 1: Mean is of the small tile images 
        Mode 2: Mean is of the transformed small tile images
        Mode 3: Mean is of large tile images
        Mode 4: Mean is of the transformed large tile images
        """
        for file in self.tilesFiles:
            tile_pic = Image.open(file)
            tile_np = numpy.array(tile_pic)
            
            size = (self.tile_max_x_px, self.tile_max_y_px)
            tile_pic_mini = tile_pic.resize(size)
            tile_np_mini = numpy.array(tile_pic_mini)
            
            if (self.comparisonMode == 1):         
                title_np_ave = numpy.mean(tile_np_mini, axis=(0, 1))
            elif (self.comparisonMode == 2):
                title_np_ave = numpy.mean(self.transformMatrixRGBtoLAB(tile_np_mini.copy()), axis=(0, 1))
            elif (self.comparisonMode == 3):
                title_np_ave = numpy.mean(tile_np, axis=(0, 1))
            else:
                title_np_ave = numpy.mean(self.transformMatrixRGBtoLAB(tile_np.copy()), axis=(0, 1))
            
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
        
        for i in numpy.arange(xdiv-1,self.np_scaled_input_image.shape[0],xdiv):
            for j in numpy.arange(ydiv-1,self.np_scaled_input_image.shape[1],ydiv):
                int_i = int(i)
                int_j = int(j)
                
                sub_matrix =self.np_scaled_input_image[int_i-(xdiv-1):int_i+1, int_j-(ydiv-1):int_j+1].copy()
                sub_average = numpy.mean(sub_matrix, axis=(0, 1)) if ((self.comparisonMode%2) == 1) else numpy.mean(self.transformMatrixRGBtoLAB(sub_matrix) , axis=(0, 1)) 
       
                maximum = self.MaxFunction(self.tileDict, sub_average)  
                maxKey = list(maximum.keys())[0]
        
                self.np_scaled_input_image[int_i-(xdiv-1):int_i+1, int_j-(ydiv-1):int_j+1] = self.tileDict[(maxKey)][0]
                
    def returnImage(self):
        """
        This function is used to return a resulting image after all the image processing is done.
        It returns a standard RGB image that can be saved.
        """
        return Image.fromarray(self.np_scaled_input_image, 'RGB')        

if __name__ == "__main__":
    try:
        wk_dir =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    else:
        wk_dir =  os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
        
    mainImageSubdir = "\\Example Output\\"
    tileImagesSubdir = "\\DB\\"
    path = os.path.normpath(wk_dir)
    
    ##& -> User Defined Variables
    mainImageTarget = "1_in.jpg"
    numberOfTiles = 1600
    mode = 2    
    useSkImage = False
    ##&
    
    output_image = "out_" + mainImageTarget

    print("Target Image:= {0}".format(path+mainImageSubdir+mainImageTarget))
    print("Tile Images Directory:= {0}".format(path + tileImagesSubdir))
    print("Number of Tiles:= {0}".format(numberOfTiles))
    
    
    sys.stdout.write("Program Running: ")
    spinner = spinner.Spinner()
    spinner.start()
    
    files = [path + tileImagesSubdir + f for f in listdir(path + tileImagesSubdir) if isfile(join(path + tileImagesSubdir, f))]
    mainImagePath = path+mainImageSubdir+mainImageTarget

    image_mos = image_mosiac(mainImagePath, files, numberOfTiles, mode, useSkImage)

    image = image_mos.returnImage()
    with open(path+mainImageSubdir+output_image, 'w') as f:
        image.save(f)
    
    spinner.stop()
    print("Done.")

    print("Output file:= {0}".format(path+mainImageSubdir+output_image))

    image.show()
















