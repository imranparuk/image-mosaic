# -*- coding: utf-8 -*-
"""
Created on Tue May 29 19:23:09 2018

@author: imranparuk
"""
from PIL import Image
import os
from os import walk
import numpy
import collections
import math

pathy = "c:\\Users\\2307221\\Documents\\work\\personal\\pineapple\\"
path = os.path.normpath(pathy)
#mydir = os.path.dirname(os.path.abspath(__file__))
#wk_dir = wk_dir = os.path.dirname(os.path.realpath('__file__'))




main_pic = Image.open(path+"\\Example Output\\1_in.jpg")
main_pix = numpy.array(main_pic)

print(main_pix.shape)

no_tiles = 20
xdiv = int(main_pix.shape[0]/no_tiles)
ydiv = int(main_pix.shape[1]/no_tiles)

adjustScalex = no_tiles * xdiv
adjustScaley = no_tiles * ydiv


print(adjustScalex)
print(adjustScaley)

size = (adjustScalex, adjustScaley)

main_pic.thumbnail(size, Image.ANTIALIAS)
main_pic.save(path+"\\Example Output\\1_in_scaled.jpg", "JPEG")

main_pix = numpy.array(main_pic)
main_average = numpy.mean(main_pic, axis=(0, 1))

print(main_pix.shape)

xdiv = int(main_pix.shape[0]/no_tiles)
ydiv = int(main_pix.shape[1]/no_tiles)

print(xdiv)
print(ydiv)
print(main_pix.shape[0])




#%%
N = float(4.0 / 29.0)

def fInv(x):
    if (x > (6.0/29.0)):
        return math.pow(x, 3)
    else:
        return (108.0 / 841.0) * (x- N)
#        if (x > 6.0 / 29.0) {
#            return x*x*x;
#        } else {
#            return (108.0 / 841.0) * (x - N);
#        }
def f(x):
    if (x > (216.0/24389.0)):
        return numpy.cbrt(x)
    else:
        return (841.0/108.0)*x + N
    
#   private static double f(double x) {
#        if (x > 216.0 / 24389.0) {
#            return Math.cbrt(x);
#        } else {
#            return (841.0 / 108.0) * x + N;
#        }
#    }
def toCIEXYZ(rgbVal):
    
    i = (rgbVal[0] + 16) * (1.0/116.0)
    X = fInv(i + rgbVal[1]*(1.0/500.0))
    Y = fInv(i)
    Z = fInv(i - rgbVal[2]*(1.0/200.0))
    
    return numpy.array([X,Y,Z])
#        double i = (colorvalue[0] + 16.0) * (1.0 / 116.0);
#        double X = fInv(i + colorvalue[1] * (1.0 / 500.0));
#        double Y = fInv(i);
#        double Z = fInv(i - colorvalue[2] * (1.0 / 200.0));
#        return new float[] {(float) X, (float) Y, (float) Z};
    
def fromCIEXYZ(ciexyzVal):
    l = f(ciexyzVal[1])
    L = 116.0*l - 16.0
    a = 500.0*(f(ciexyzVal[0]) - l)
    b = 200.0*(l- f(ciexyzVal[2]))
    
    return numpy.array([L,a,b])
    
# @Override
#    public float[] fromCIEXYZ(float[] colorvalue) {
#        double l = f(colorvalue[1]);
#        double L = 116.0 * l - 16.0;
#        double a = 500.0 * (f(colorvalue[0]) - l);
#        double b = 200.0 * (l - f(colorvalue[2]));
#        return new float[] {(float) L, (float) a, (float) b};
#    }
def func(t):
    if (t > 0.008856):
        return numpy.power(t, 1/3.0)
    else:
        return 7.787 * t + 16 / 116.0

#Conversion Matrix
matrix = [[0.412453, 0.357580, 0.180423],
          [0.212671, 0.715160, 0.072169],
          [0.019334, 0.119193, 0.950227]]

def rgb_to_lab(rgb):
    np_rgb = numpy.array(rgb)
    
    normalized_rgb = np_rgb / 255

    cie = numpy.dot(matrix, normalized_rgb)
    
    cie[0] = cie[0] /0.950456
    cie[2] = cie[2] /1.088754 

    L = 116 * numpy.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1]
    a = 500*(func(cie[0]) - func(cie[1]))
    b = 200*(func(cie[1]) - func(cie[2]));
    
    #from skimage import io, color
    #lab = color.rgb2lab(rgb)
    
    return [L, a, b]
    #return numpy.array([L,a,b])

#%%

#d(x,y) = sqrt( (Rx-Ry) + (Gx-Gy) + (Bx-By) ).
def getEuclDistance(x, y):
    return numpy.linalg.norm(x-y)
     
def myMax(input_dict, cmp):
    
    maxItem = {}
    for key, value in input_dict.items():
        
        exld = getEuclDistance(value[2], cmp)
        #print("key: {0} , val: {1}".format(key, exld))

        try:
            list(maxItem.values())[0]

        except IndexError:
            maxItem.update({key:exld})

        else:
            new_exld = getEuclDistance(value[2], cmp) 
            if (new_exld < list(maxItem.values())[0] ):
                maxItem.clear()
                maxItem.update({key:exld})
    #print("max {0}".format(maxItem))
    return maxItem



#%%


    
    
    

def transformMat(numpyArr):
    
    size = numpyArr.shape
    x = size[0]
    y = size[1]
    #print(y)
    for i in range(0, x-1):
        for j in range(0, y-1):
            tempArr = numpyArr[i, j]
            tempArrTransform = rgb_to_lab(tempArr)
            numpyArr[i, j] = tempArrTransform

    return numpyArr         

def un_transformMat(numpyArr):
    
    size = numpyArr.shape
    x = size[0]
    y = size[1]
    #print(y)
    for i in range(0, x-1):
        for j in range(0, y-1):
            tempArr = numpyArr[i, j]
            tempArrTransform = fromCIEXYZ(tempArr)
            numpyArr[i, j] = tempArrTransform

    return numpyArr     
            
#%%

print("yo")
files = []
for (dirpath, dirnames, filenames) in walk(path + "\\DB"):
    files.extend(filenames)
    break

tile_dict = {}

for file in files:
    tile_pic = Image.open(path+"\\DB\\" + file)
    tile_np = numpy.array(tile_pic)
    title_np_ave = numpy.mean(tile_np, axis=(0, 1))
    
    size = (xdiv, ydiv)
    tile_pic_mini = tile_pic.resize(size)
    tile_np_mini = numpy.array(tile_pic_mini)

    tile_np_trans = transformMat(tile_np) 
    tile_np_trans_ave = numpy.mean(tile_np_trans, axis=(0, 1))
    
    tile_dict[file] = [tile_np_mini, title_np_ave, tile_np_trans_ave]



#%%
    
sub_matrix_dict = {}

print("here")
for i in numpy.arange(xdiv-1,main_pix.shape[0],xdiv):
    for j in numpy.arange(ydiv-1,main_pix.shape[1],ydiv):
        int_i = int(i)
        int_j = int(j)
        
        sub_matrix =main_pix[int_i-(xdiv-1):int_i+1, int_j-(ydiv-1):int_j+1].copy()
        sub_average = numpy.mean(sub_matrix, axis=(0, 1))
        
        sub_average_trans = numpy.mean(transformMat(sub_matrix) , axis=(0, 1)) 

        maximum = myMax(tile_dict, sub_average_trans)  
        maxKey = list(maximum.keys())[0]

        main_pix[int_i-(xdiv-1):int_i+1, int_j-(ydiv-1):int_j+1] = tile_dict[(maxKey)][0]

    
    
#%%
#un_transformMat(main_pix)     
img = Image.fromarray(main_pix, 'RGB')
img.show()

#print(sub_matrix_dict)











