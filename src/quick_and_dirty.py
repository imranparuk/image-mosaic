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

#print(main_pix)
print(main_pix.shape)

no_tiles = 60
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

#print(main_pix)
print(main_pix.shape)

xdiv = int(main_pix.shape[0]/no_tiles)
ydiv = int(main_pix.shape[1]/no_tiles)

print(xdiv)
print(ydiv)
print(main_pix.shape[0])
#mypix = pix[0:xdiv, 0:ydiv].copy()
    
#print(mypix)
#print(mypix.shape)

#transformMat(main_pix)


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
    #print(normalized_rgb)
    cie = numpy.dot(matrix, normalized_rgb)
    
    cie[0] = cie[0] /0.950456
    cie[2] = cie[2] /1.088754 

    L = 116 * numpy.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1]
    a = 500*(func(cie[0]) - func(cie[1]))
    b = 200*(func(cie[1]) - func(cie[2]));
    
    from skimage import io, color
    lab = color.rgb2lab(rgb)
    
    return lab
    #return numpy.array([L,a,b])

def rgb_to_XYZ(rgb):
    
    np_rgb = numpy.array(rgb)
    
    np_rgb_norm = np_rgb / 255
    
    var_R = np_rgb_norm[0]
    var_G = np_rgb_norm[1]
    var_B = np_rgb_norm[2]
    
    if ( var_R > 0.04045 ):
        var_R = numpy.power(( ( var_R + 0.055 ) / 1.055 ) , 2.4)
    else:
        var_R = var_R / 12.92
    if ( var_G > 0.04045 ): 
        var_G = numpy.power(( ( var_G + 0.055 ) / 1.055 ) , 2.4)
    else:
        var_G = var_G / 12.92
    if ( var_B > 0.04045 ):
        var_B = numpy.power(( ( var_B + 0.055 ) / 1.055 ) , 2.4)
    else:
        var_B = var_B / 12.92
    
    var_R = var_R * 100
    var_G = var_G * 100
    var_B = var_B * 100
    
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505
    
    return numpy.array([X, Y, Z])

rgb_to_XYZ([40,0,0])
#%%
   
def getEuclDistance(x, y):
    return numpy.linalg.norm(x-y)
     

    
    
#d(x,y) = sqrt( (Rx-Ry) + (Gx-Gy) + (Bx-By) ).

def myMax(input_dict, cmp):
    
    maxItem = {}
    for key, value in input_dict.items():
        
       # myvalue = transformMat(value)
        exld = getEuclDistance(value, cmp)
        #print(exld)
        #print(value)
        #print(cmp)
        #print("abc: {0}".format(exld))
        
        
        try:
            list(maxItem.values())[0]

        except IndexError:
            maxItem.update({key:exld})

        else:
            new_exld = getEuclDistance(value, cmp) 
            if (new_exld < list(maxItem.values())[0] ):
                maxItem.clear()
                maxItem.update({key:exld})
                
    #print("def: {0}".format(maxItem))

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
            tempArrTransform = rgb_to_XYZ(tempArr)
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


files = []
for (dirpath, dirnames, filenames) in walk(path + "\\DB"):
    files.extend(filenames)
    break

#print(files)
image_dict = {}
file_dict = {}
file_dict_transform = {}


for file in files:
    file_pic = Image.open(path+"\\DB\\" + file)
    
    size = (xdiv, ydiv)
    file_pic = file_pic.resize(size)
    
    
    
    file_pix = numpy.array(file_pic)
    #img = Image.fromarray(file_pix, 'RGB')
    #img.show()
    #break
    #transformMat(file_pix)
    
    image_dict[file] = file_pix
    test = transformMat(file_pix.copy())
    #file_dict_transform[file] = test
    
    file_average = numpy.mean(test, axis=(0, 1))
    
    file_dict[file] = file_average

    #file_size = (20, 20)
    #file_pic.thumbnail(file_size, Image.ANTIALIAS)
    
    #temp_pix1 = numpy.array(file_pic)
    
    #print(temp_pix1.shape)
    #print(temp_pix2.shape)

#print(image_dict)

transformMat(main_pix)




#%%
    
sub_matrix_dict = {}


for i in numpy.arange(xdiv-1,main_pix.shape[0],xdiv):
    for j in numpy.arange(ydiv-1,main_pix.shape[1],ydiv):
        int_i = int(i)
        int_j = int(j)
        
        sub_matrix =main_pix[int_i-(xdiv-1):int_i+1, int_j-(ydiv-1):int_j+1].copy()
        #print(sub_matrix.shape) # = image_dict[(maxKey)]
        


        #transformMat(sub_matrix)
        #print( "i: {0} j:{1}".format(int_i,int_i-(xdiv-1)))
        #print(sub_matrix)
        sub_average = numpy.mean(sub_matrix, axis=(0, 1))
        
        maximum = myMax(file_dict, numpy.array(sub_average))  
        maxKey = list(maximum.keys())[0]
        
        main_pix[int_i-(xdiv-1):int_i+1, int_j-(ydiv-1):int_j+1] = image_dict[(maxKey)]
        #print(maxKey)
        #print(image_dict[(maxKey)].shape)

#        maxKey = list(maximum.keys())[0]
#        
#        #print(sub_matrix)
#        transformMat(sub_matrix)
#        
#        sub_average = numpy.mean(sub_matrix, axis=(0, 1))
#        maximum = myMax(file_dict_transform, numpy.array(sub_average))  #compare `int` version of each item
#        maxKey = list(maximum.keys())[0]
#        
#        print(maxKey)
#        print(image_dict[(maxKey)].shape)
##        
#        #temp_average = numpy.mean(sub_matrix, axis=(0, 1))
#        #sub_matrix_dict[(int_i-xdiv, int_j-ydiv)] = (sub_matrix, temp_average)
#        
#        transformMat(sub_matrix)
#        
#        
#        

#        
#        
#        #print(image_dict[(maxKey)])
#        #print(sub_matrix)
#        #print("#")
    
    
#%%
#un_transformMat(main_pix)     
img = Image.fromarray(main_pix, 'RGB')
img.show()

#print(sub_matrix_dict)

