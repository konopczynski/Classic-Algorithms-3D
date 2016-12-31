# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 12:19:32 2016
Based on:
http://www.mathworks.com/matlabcentral/fileexchange/24409-hessian-based-frangi-vesselness-filter/content/imgaussian.m
@author: konopczynski
"""
import numpy as np
from scipy import ndimage
from numpy import linalg as LA

def Hessian2D_separable(I,Sigma=1.0):
    # Make kernel coordinates
    s = round(3*Sigma)
    x=np.mgrid[-s:s+1]
    Gauss =  np.exp(-(x**2)/(2*Sigma**2)) * 1/(np.sqrt(2*np.pi)*Sigma)
    DGxx1D = -(Sigma**2 - x**2) * np.exp(-(x**2)/(2*Sigma**2)) * 1/(np.sqrt(2*np.pi)*Sigma**5)
    DGx1D = - x * np.exp(-x**2/(2*Sigma**2)) * 1/(np.sqrt(2*np.pi)*Sigma**3)
    
    Dx1 = ndimage.convolve1d(I,DGxx1D,axis=0)
    Dxx = ndimage.convolve1d(Dx1,Gauss,axis=1)
    
    Dy1 = ndimage.convolve1d(I,DGxx1D,axis=1)
    Dyy = ndimage.convolve1d(Dy1,Gauss,axis=0)
    
    Dx  = ndimage.convolve1d(I, DGx1D,axis=0)
    Dxy = ndimage.convolve1d(Dx,DGx1D,axis=1)
    return [Dxx,Dxy,Dyy]

def Hessian3D_Separable(I,Sigma=1.0):
    # Make kernel coordinates
    s = round(3*Sigma)
    x=np.mgrid[-s:s+1]  
    
    Gauss =  np.exp(-(x**2)/(2*Sigma**2)) * 1/(np.sqrt(2*np.pi)*Sigma)
    DGxx1D = -(Sigma**2 - x**2) * np.exp(-(x**2)/(2*Sigma**2)) * 1/(np.sqrt(2*np.pi)*Sigma**5)
    DGx1D = - x * np.exp(-x**2/(2*Sigma**2)) * 1/(np.sqrt(2*np.pi)*Sigma**3)
    
    Dx1 = ndimage.convolve1d(I, DGxx1D,axis=0)
    Dx2 = ndimage.convolve1d(Dx1,Gauss,axis=1)
    Dxx = ndimage.convolve1d(Dx2,Gauss,axis=2)

    Dy1 = ndimage.convolve1d(I, DGxx1D,axis=1)
    Dy2 = ndimage.convolve1d(Dy1,Gauss,axis=0)
    Dyy = ndimage.convolve1d(Dy2,Gauss,axis=2)

    Dz1 = ndimage.convolve1d(I, DGxx1D,axis=2)
    Dz2 = ndimage.convolve1d(Dz1,Gauss,axis=0)
    Dzz = ndimage.convolve1d(Dz2,Gauss,axis=1)

    Dxy1= ndimage.convolve1d(I,    DGx1D,axis=0)
    Dxy2= ndimage.convolve1d(Dxy1, DGx1D,axis=1)
    Dxy = ndimage.convolve1d(Dxy2, Gauss,axis=2)

    Dxz1= ndimage.convolve1d(I,    DGx1D,axis=0)
    Dxz2= ndimage.convolve1d(Dxz1, DGx1D,axis=2)
    Dxz = ndimage.convolve1d(Dxz2, Gauss,axis=1)

    Dyz1= ndimage.convolve1d(I,    DGx1D,axis=1)
    Dyz2= ndimage.convolve1d(Dyz1, DGx1D,axis=2)
    Dyz = ndimage.convolve1d(Dyz2, Gauss,axis=0)
    return [Dxx,Dyy,Dzz,Dxy,Dxz,Dyz]

def eigv2D_Vectorized(Dxx,Dxy,Dyy):
    # Compute the eigenvectors of J, v1 and v2
    tmp = np.sqrt((Dxx - Dyy)**2 + 4*Dxy**2)
    # Compute the eigenvalues
    v1 = 0.5*(Dxx + Dyy + tmp)
    v2 = 0.5*(Dxx + Dyy - tmp)
    return [v1,v2]

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

def SortEigen2D(u1,u2,mode=3):
    v1=u1.copy()
    v2=u2.copy()    
    if (mode==1):
        # Order and return actual eigenvalues
        # Order: v1<=v2
        # Return: [v1,v2]
        check=-(v1<=v2)
        v1[check],v2[check] = v2[check],v1[check]
    elif (mode==2):
        # Order and return only the magnitudes
        # Order: |v1|<=|v2|
        # Return: [|v1|,|v2|]
        v1=np.abs(v1)
        v2=np.abs(v2)
        check=-(v1<=v2)
        v1[check],v2[check] = v2[check],v1[check]
    elif (mode==3):
        # Order the magnitudes return the actual values
        # Order: |v1|<=|v2|
        # Return: [v1,v2]
        check=-(np.abs(v1)<=np.abs(v2))
        v1[check],v2[check] = v2[check],v1[check]
    else:
        print ("Wrong Mode")
        return 0
    return v1,v2
    # Sort eigen values by absolute value abs(Lambda1)<abs(Lambda2)
    # check=abs(v1)>abs(v2)
    check = (abs(v1)>abs(v2))
    v1[check],v2[check] = v2[check],v1[check]
    return v1,v2

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

def checkV3D(v1,v2,v3):
    check1 = (abs(v1)<=abs(v2))
    check2 = (abs(v2)<=abs(v3))
    B=np.min(check1) & np.min(check2)
    return B

###############################################################################
################################# OLD FUNCTIONS ###############################
###############################################################################

def Hessian2D(I,Sigma=1.0):
    # Bruteforce implementation
    # The convolution step should use the separability of Gaussian kernel 
    # also, s is always positive - it should be used in the equation.  
    
    # Make kernel coordinates
    s = round(3*Sigma)
    [X,Y] = np.mgrid[-s:s+1, -s:s+1]
    
    # Build the gaussian 2nd derivatives filters
    #DGaussxx = 1/(2*pi*Sigma^4) * (X.^2/Sigma^2 - 1) .* exp(-(X.^2 + Y.^2)/(2*Sigma^2));
    DGaussxx = 1/(2*np.pi*Sigma**4) * (X**2/Sigma**2 - 1) * np.exp(-(X**2 + Y**2)/(2*Sigma**2))
    #DGaussxy = 1/(2*pi*Sigma^6) * (X .* Y)           .* exp(-(X.^2 + Y.^2)/(2*Sigma^2));
    DGaussxy = 1/(2*np.pi*Sigma**6) * (X * Y)           * np.exp(-(X**2 + Y**2)/(2*Sigma**2))
    #DGaussyy = DGaussxx';
    DGaussyy = DGaussxx.T

    Dxx = ndimage.convolve(I,DGaussxx)
    Dxy = ndimage.convolve(I,DGaussxy)
    Dyy = ndimage.convolve(I,DGaussyy)
    #H=np.concatenate((Dxx,Dxy,Dyy),axis=1)
    return [Dxx,Dxy,Dyy]

def eigv2D(Dxx,Dxy,Dyy):
    ii,jj=np.shape(Dxx)
    v1 = np.zeros(shape=(ii, jj))
    v2 = np.zeros(shape=(ii, jj))
    for i in range(ii):
        for j in range(jj):
            H=np.array([[Dxx[i][j],Dxy[i][j]],[Dxy[i][j],Dyy[i][j]]])
            v1[i][j],v2[i][j] = LA.eigvals(H)
            # Eigenvalues must be of such form: |v1|<|v2|            
            if np.abs(v1[i][j]) > np.abs(v2[i][j]):
                v1[i][j], v2[i][j] = v2[i][j], v1[i][j]
                if v1[i][j] == 0:
                    v1[i][j] = np.finfo(float).eps
    return v1,v2

def Hessian3D(I,Sigma=1.0):
    # Make kernel coordinates
    s = round(3*Sigma)
    [X,Y,Z] = np.mgrid[-s:s+1, -s:s+1, -s:s+1]    
    
    xx_const = 1/(np.sqrt(8) * np.pi**(3/2.0) *Sigma**4 * np.sqrt(Sigma**2))
    exp_const = np.exp(-(X**2 + Y**2 + Z**2)/(2*Sigma**2))
    xy_const = np.sqrt(Sigma**2) * 1/( np.sqrt(8) * np.pi**(1.5) * Sigma**8 )
    
    DGxx= (X**2/Sigma**2 - 1) * xx_const * exp_const
    DGyy= (Y**2/Sigma**2 - 1) * xx_const * exp_const
    DGzz= (Z**2/Sigma**2 - 1) * xx_const * exp_const
    
    DGxy= (X*Y) * xy_const * exp_const
    DGxz= (X*Z) * xy_const * exp_const
    DGyz= (Y*Z) * xy_const * exp_const
    
    Dxx = ndimage.convolve(I,DGxx)
    Dyy = ndimage.convolve(I,DGyy)
    Dzz = ndimage.convolve(I,DGzz)
    Dxy = ndimage.convolve(I,DGxy)
    Dxz = ndimage.convolve(I,DGxz)
    Dyz = ndimage.convolve(I,DGyz)
    return [Dxx,Dyy,Dzz,Dxy,Dxz,Dyz]

def eigv3D_ImageJ(Dxx,Dyy,Dzz,Dxy,Dxz,Dyz):
    fhxx = Dxx
    fhxy = Dxy
    fhxz = Dxz
    fhyy = Dyy
    fhyz = Dyz
    fhzz = Dzz
    
    a = -(fhxx + fhyy + fhzz)
    b = fhxx*fhyy + fhxx*fhzz + fhyy*fhzz - fhxy*fhxy - fhxz*fhxz - fhyz*fhyz
    c = fhxx*(fhyz*fhyz - fhyy*fhzz) + fhyy*fhxz*fhxz + fhzz*fhxy*fhxy - 2*fhxy*fhxz*fhyz
    q = (a*a - 3*b)/9
    r = (a*a*a - 4.5*a*b + 13.5*c)/27
    # sqrtq = (q > 0) ? Math.sqrt(q) : 0
    sqrtq = np.sqrt(q)
    sqrtq3 = sqrtq*sqrtq*sqrtq
    
    rsqq3 = r/sqrtq3
    # angle = (rsqq3*rsqq3 <= 1) ? Math.acos(rsqq3) : Math.acos(rsqq3 < 0 ? -1 : 1)
    angle = np.arccos(rsqq3)
    
    h1 = -2*sqrtq*np.cos(angle/3) - a/3
    h2 = -2*sqrtq*np.cos((angle + 2*np.pi)/3) - a/3
    h3 = -2*sqrtq*np.cos((angle - 2*np.pi)/3) - a/3
    return h1,h2,h3

def Cr8Hes(Dxx,Dyy,Dzz,Dxy,Dxz,Dyz,indx):
    i=indx[0]
    j=indx[1]
    k=indx[2]
    H=np.array([[ Dxx[i][j][k], Dxy[i][j][k], Dxz[i][j][k] ],
                [ Dxy[i][j][k], Dyy[i][j][k], Dyz[i][j][k] ],
                [ Dxz[i][j][k], Dyz[i][j][k], Dzz[i][j][k] ]])
    return H

def eig3x3(A):
    # https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
    # Given a real symmetric 3x3 matrix A, compute the eigenvalues
    p1 = A[0,1]**2 + A[0,2]**2 + A[1,2]**2
    if (p1==0):
        #A is diagonal
        eig1 = A[0,0]
        eig2 = A[1,1]
        eig3 = A[2,2]
    else:
        q=np.trace(A)/3
        p2 = (A[0,0] - q)**2 + (A[1,1] - q)**2 + (A[2,2] - q)**2 + 2 * p1
        p = np.sqrt(p2 / 6)
        # I is the 3x3 identity matrix
        I=np.identity(3)
        B = (1 / p) * (A - q * I)
        r = np.linalg.det(B)/2
        #r = det3d(B)/2
        # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
        # but computation error can leave it slightly outside this range.
        if (r <= -1):
            print ("r<=-1")
            phi = np.pi/3
        elif(r >= 1):
            print ("r>=1")
            phi = 0
        else:
            #print ("else")
            phi = np.arccos(r)/3
        # the eigenvalues satisfy eig3 <= eig2 <= eig1
        eig1 = q + 2. * p * np.cos(phi)
        eig3 = q + 2. * p * np.cos(phi + (2*np.pi/3))
        # since trace(A) = eig1 + eig2 + eig3
        eig2 = 3. * q - eig1 - eig3
    return [eig1, eig2, eig3]

def eigv3D(Dxx,Dyy,Dzz,Dxy,Dxz,Dyz):
    ii,jj,kk=np.shape(Dxx)
    v1 = np.zeros(shape=(ii, jj, kk))
    v2 = np.zeros(shape=(ii, jj, kk))
    v3 = np.zeros(shape=(ii, jj, kk))
    for i in range(ii):
        print(i)
        for j in range(jj):
            for k in range(kk):
                H = Cr8Hes(Dxx,Dyy,Dzz,Dxy,Dxz,Dyz,(i,j,k))
                v1[i][j][k],v2[i][j][k],v3[i][j][k] = eig3x3(H)
    return [v1,v2,v3]

def det3d_B_2(Bxx,Byy,Bzz,Bxy,Bxz,Byz):
    pp1 = Bxx*Byy*Bzz
    pp2 = Bxy*Byz*Bxz
    pp3 = Bxz*Bxy*Byz
    pp4 = Bxz*Byy*Bxz
    pp5 = Bxy*Bxy*Bzz
    pp6 = Bxx*Byz*Byz
    det = pp1 + pp2 + pp3 -pp4-pp5-pp6     
    return det

def det3d(A):
    p1 = A[0,0]*( A[1,1]*A[2,2] - A[1,2]*A[2,1] )
    p2 = A[0,1]*( A[1,0]*A[2,2] - A[1,2]*A[2,0] )
    p3 = A[0,2]*( A[1,0]*A[2,1] - A[1,1]*A[2,0] )
    det = p1 - p2 + p3     
    return det
