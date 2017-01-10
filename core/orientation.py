# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:07:43 2015
Based on:
http://bigwww.epfl.ch/demo/orientation/index.html
@author: konopczynski
"""

import numpy as np
from scipy import ndimage


def compute_harris_response(image):
    """ compute the Harris corner detector response function 
        for each pixel in the image"""
    Wxx,Wxy,Wyy=StrTensor2D(image)
    #determinant and trace
    Wdet = Wxx * Wyy - Wxy**2
    Wtr  = Wxx + Wyy

    return Wdet / Wtr

def StrTensor2D(I,sigma=1.0):
    s = round(3.0*sigma)
    x=np.mgrid[-s:s+1]

    Gauss =       np.exp(-(x**2)/(2*sigma**2)) * 1/(np.sqrt(2*np.pi)*sigma)
    DGx1D = - x * np.exp(-(x**2)/(2*sigma**2)) * 1/(np.sqrt(2*np.pi)*sigma**3)
    
    Dx1 = ndimage.convolve1d(I,DGx1D,axis=0)
    Dx  = ndimage.convolve1d(Dx1,Gauss,axis=1)

    Dy1 = ndimage.convolve1d(I,DGx1D,axis=1)
    Dy  = ndimage.convolve1d(Dy1,Gauss,axis=0)
    
    Ixx1 = Dx*Dx
    Iyy1 = Dy*Dy
    Ixy1 = Dx*Dy    
    
    Ixx2 = ndimage.convolve1d(Ixx1,Gauss,axis=0)
    Ixx  = ndimage.convolve1d(Ixx2,Gauss,axis=1)
    
    Iyy2 = ndimage.convolve1d(Iyy1,Gauss,axis=0)
    Iyy  = ndimage.convolve1d(Iyy2,Gauss,axis=1)
    
    Ixy2 = ndimage.convolve1d(Ixy1,Gauss,axis=0)
    Ixy  = ndimage.convolve1d(Ixy2,Gauss,axis=1)

    return [Ixx,Ixy,Iyy]

def Orientation2D(Ixx,Ixy,Iyy):
    theta = 0.5 * np.arctan(2* Ixy/(Iyy-Ixx) )
    return theta


def Coherency_J2D(Ixx,Ixy,Iyy):
    C = np.sqrt((Iyy-Ixx)**2 + 4*Ixy**2) / (Ixx + Iyy)
    return C

def Energy2D(Ixx,Ixy,Iyy):
    E=Ixx+Iyy
    return E

def StrTensor3D(I,sigma=1.0):
    # Make kernel coordinates
    s = round(3*sigma)
    x=np.mgrid[-s:s+1]  
    
    Gauss =       np.exp(-(x**2)/(2*sigma**2)) * 1/(np.sqrt(2*np.pi)*sigma)
    DGx1D = - x * np.exp(-(x**2)/(2*sigma**2)) * 1/(np.sqrt(2*np.pi)*sigma**3)
    
    Dx1 = ndimage.convolve1d(I,  DGx1D,axis=0)
    Dx2 = ndimage.convolve1d(Dx1,Gauss,axis=1)
    Dx  = ndimage.convolve1d(Dx2,Gauss,axis=2)

    Dy1 = ndimage.convolve1d(I,  DGx1D,axis=1)
    Dy2 = ndimage.convolve1d(Dy1,Gauss,axis=0)
    Dy  = ndimage.convolve1d(Dy2,Gauss,axis=2)
    
    Dz1 = ndimage.convolve1d(I,  DGx1D,axis=2)
    Dz2 = ndimage.convolve1d(Dz1,Gauss,axis=0)
    Dz  = ndimage.convolve1d(Dz2,Gauss,axis=1)
    
    Ixx1 = Dx*Dx
    Iyy1 = Dy*Dy
    Izz1 = Dz*Dz
    Ixy1 = Dx*Dy    
    Ixz1 = Dx*Dz
    Iyz1 = Dy*Dz
    
    Ixx2 = ndimage.convolve1d(Ixx1,Gauss,axis=0)
    Ixx3 = ndimage.convolve1d(Ixx2,Gauss,axis=1)
    Ixx  = ndimage.convolve1d(Ixx3,Gauss,axis=2)

    Iyy2 = ndimage.convolve1d(Iyy1,Gauss,axis=0)
    Iyy3 = ndimage.convolve1d(Iyy2,Gauss,axis=1)
    Iyy  = ndimage.convolve1d(Iyy3,Gauss,axis=2)
    
    Izz2 = ndimage.convolve1d(Izz1,Gauss,axis=0)
    Izz3 = ndimage.convolve1d(Izz2,Gauss,axis=1)
    Izz  = ndimage.convolve1d(Izz3,Gauss,axis=2)
    
    Ixy2 = ndimage.convolve1d(Ixy1,Gauss,axis=0)
    Ixy3 = ndimage.convolve1d(Ixy2,Gauss,axis=1)
    Ixy  = ndimage.convolve1d(Ixy3,Gauss,axis=2)
    
    Ixz2 = ndimage.convolve1d(Ixz1,Gauss,axis=0)
    Ixz3 = ndimage.convolve1d(Ixz2,Gauss,axis=1)
    Ixz  = ndimage.convolve1d(Ixz3,Gauss,axis=2)
    
    Iyz2 = ndimage.convolve1d(Iyz1,Gauss,axis=0)
    Iyz3 = ndimage.convolve1d(Iyz2,Gauss,axis=1)
    Iyz  = ndimage.convolve1d(Iyz3,Gauss,axis=2)

    return [Ixx1,Iyy1,Izz1,Ixy1,Ixz1,Iyz1]

def det3d_B(Bxx,Byy,Bzz,Bxy,Bxz,Byz):
    pp1 = Bxx*( Byy*Bzz - Byz*Byz )
    pp2 = Bxy*( Bxy*Bzz - Byz*Bxz )
    pp3 = Bxz*( Bxy*Byz - Byy*Bxz )
    det = pp1 - pp2 + pp3     
    return det

def eigv3D_Vectorized(Dxx,Dyy,Dzz,Dxy,Dxz,Dyz):
    # |Dxx Dxy Dxz|   |a b c|
    # |Dyx Dyy Dyz| = |d e f|
    # |Dzx Dzy Dzz|   |g h i|
    ii,jj,kk=np.shape(Dxx)
    Vp1 = Dxy**2 + Dxz**2 + Dyz**2
    Vq  = (Dxx+Dyy+Dzz)/3
    Vp2 = (Dxx-Vq)**2 + (Dyy-Vq)**2 + (Dzz-Vq)**2 + 2*Vp1
    Vp  = np.sqrt(Vp2 / 6)
    # B = (1 / p) * (A - q * I)
    Bxx = (1/Vp) * (Dxx-Vq)
    Byy = (1/Vp) * (Dyy-Vq)
    Bzz = (1/Vp) * (Dzz-Vq)
    Bxy = (1/Vp) * Dxy
    Bxz = (1/Vp) * Dxz
    Byz = (1/Vp) * Dyz
    # r = det(B) / 2
    Vr = det3d_B(Bxx,Byy,Bzz,Bxy,Bxz,Byz)/2
    phi = np.arccos(Vr) / 3
    if np.isnan(phi).any():
        phi[Vr <= -1] = np.pi/3
        phi[Vr >= 1] = 0
    v1 = Vq + 2 * Vp * np.cos(phi)
    v3 = Vq + 2 * Vp * np.cos(phi + (2*np.pi/3))
    # since trace(A) = v1 + v2 + v3
    v2 = 3 * Vq - v1 - v3
    #|v3|>|v1|>|v2|
    return [v1,v2,v3]

def SortEigen3D(u1,u2,u3,mode=3):
    v1=u1.copy()
    v2=u2.copy()
    v3=u3.copy()
    if (mode==1):
        # Order and return actual eigenvalues
        # Order: v1<=v2<=v3
        # Return: [v1,v2,v3]
        check1=-(v1<=v2)
        v1[check1],v2[check1] = v2[check1],v1[check1]
        check2=-(v2<=v3)
        v2[check2],v3[check2] = v3[check2],v2[check2]
        check1=-(v1<=v2)
        v1[check1],v2[check1] = v2[check1],v1[check1]
    elif (mode==2):
        # Order and return only the magnitudes
        # Order: |v1|<=|v2|<=|v3|
        # Return: [|v1|,|v2|,|v3|]
        v1=np.abs(v1)
        v2=np.abs(v2)
        v3=np.abs(v3)
        check1=-(v1<=v2)
        v1[check1],v2[check1] = v2[check1],v1[check1]
        check2=-(v2<=v3)
        v2[check2],v3[check2] = v3[check2],v2[check2]
        check1=-(v1<=v2)
        v1[check1],v2[check1] = v2[check1],v1[check1]
    elif (mode==3):
        # Order the magnitudes return the actual values
        # Order: |v1|<=|v2|<=|v3|
        # Return: [v1,v2,3]
        # np.max(check1)
        # np.max(check2)
        check1=-((np.abs(v1)<=np.abs(v2)))
        v1[check1],v2[check1] = v2[check1],v1[check1]
        check2=-((np.abs(v2)<=np.abs(v3)))
        v2[check2],v3[check2] = v3[check2],v2[check2]
        check1=-((np.abs(v1)<=np.abs(v2)))
        v1[check1],v2[check1] = v2[check1],v1[check1]
    else:
        print ("Wrong Mode")
        return 0
    return v1,v2,v3

def Energy3D(Ixx,Iyy,Izz):
    E=Ixx+Iyy+Izz
    return E
