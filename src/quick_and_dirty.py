# -*- coding: utf-8 -*-
"""
Created on Tue May 29 19:23:09 2018

@author: imranparuk
"""


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