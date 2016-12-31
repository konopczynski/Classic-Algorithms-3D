# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 12:20:29 2016
Based on Frangi Vesselness Filter
@author: konopczynski
"""
import numpy as np
from core.hessians import Hessian2D_separable, eigv2D_Vectorized, SortEigen2D # for 2D
from core.hessians import Hessian3D_Separable, eigv3D_Vectorized, SortEigen3D, checkV3D # for 3D


def Frangi2D(I,L1,L2,ves_color):
    beta = 0.5
    beta = 2*beta**2
    # Lambda1(Lambda1==0) = eps;
    Rb = (np.abs(L1)/np.abs(L2))
    S = np.sqrt(L1**2 + L2**2)
    c = np.abs(np.max(I) - np.min(I))/np.max(S)
    c = 2*c**2
    # Compute the output image
    O = np.exp(-Rb**2/ (beta)) * (1 - np.exp(-S**2/c))
    if ves_color=='bright':
        check = (L2>0)
        O[check]=0
    elif ves_color=='dark':
        check = (L2<0)
        O[check]=0
    check2=np.isnan(O)
    O[check2]=0
    return O

def ScaledFrangi2D(I,sigmas=[1.0],ves_color='bright'):
    O=[]
    for s in sigmas:
        print(s)
        Dxx,Dxy,Dyy=Hessian2D_separable(I,s)
        # Correct for scale
        Dxx = (s**2)*Dxx
        Dxy = (s**2)*Dxy
        Dyy = (s**2)*Dyy
        Lambda1,Lambda2=eigv2D_Vectorized(Dxx,Dxy,Dyy)
        Lambda1,Lambda2=SortEigen2D(Lambda1,Lambda2)
        O.append(Frangi2D(I,Lambda1,Lambda2,ves_color))
    if len(sigmas) > 1:
        M = np.zeros(shape=(np.shape(I)[0], np.shape(I)[1]))
        for i in range(len(sigmas)):
            M = np.maximum(M,O[i])
    else:
        M=O[0]
    return M

def ScaledFrangi3D(I,sigmas=[1.0],ves_color='bright'):
    #defaultoptions
    FrangiAlpha=0.5
    FrangiBeta=0.5
    A = 2*FrangiAlpha**2
    B = 2*FrangiBeta**2
    #FrangiC=500
    O=[]    
    for s in sigmas:
        print(s)
        #Calculate 3D hessian
        [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] = Hessian3D_Separable(I,s)
          
        if(s>0):
            #Correct for scaling
            c=s**2
            Dxx = c*Dxx
            Dxy = c*Dxy
            Dxz = c*Dxz
            Dyy = c*Dyy
            Dyz = c*Dyz
            Dzz = c*Dzz

        #Calculate eigen values
        #TODO: assure, there are no NaNs in v1,v2,v3
        v1,v2,v3=eigv3D_Vectorized(Dxx,Dyy,Dzz,Dxy,Dxz,Dyz)
        v1[np.isnan(v1)]=-5000
        v2[np.isnan(v2)]=-5000
        v3[np.isnan(v3)]=-5000
        v1,v2,v3=SortEigen3D(v1,v2,v3,3)
        if checkV3D(v1,v2,v3):
            print("|v1|<=|v2|<=|v3|")
        else:
            print("v1 v2 and v3 have wrong order")
        #[Lambda1,Lambda2,Lambda3]=eigv3D(Dxx,Dxy,Dxz,Dyy,Dyz,Dzz)
        # TODO: Free memory: clear Dxx Dyy Dzz Dxy Dxz Dyz
        
        # Calculate absolute values of eigen values
        LambdaAbs1=np.abs(v1)
        LambdaAbs2=np.abs(v2)
        LambdaAbs3=np.abs(v3)
        
        # The Vesselness Features
        Ra=LambdaAbs2/LambdaAbs3
        Rb=LambdaAbs1/np.sqrt(LambdaAbs2*LambdaAbs3)
    
        # Second order structureness. S = sqrt(sum(L^2[i])) met i =< D
        S = np.sqrt(LambdaAbs1**2 + LambdaAbs2**2 + LambdaAbs3**2)
        FrangiC = np.abs(np.max(I) - np.min(I))/np.max(S)
        C = 2*FrangiC**2
        # TODO:  Free memory: clear LambdaAbs1 LambdaAbs2 LambdaAbs3
        
        # Compute Vesselness function
        expRa = (1 - np.exp(-(Ra**2/A)))
        expRb =      np.exp(-(Rb**2/B))
        expS  = (1 - np.exp(-  S**2/C))
        # TODO:  Free memory: clear S A B C Ra Rb
    
        # Compute Vesselness function
        Voxel_data = expRa * expRb * expS
        # TODO:  Free memory: clear expRa expRb expRc
        
        #BlackWhite=False
        if(ves_color=='bright'):
            print('bright')
            Voxel_data[v2 > 0]=0
            Voxel_data[v3 > 0]=0
        elif(ves_color=='dark'):
            print('dark')
            Voxel_data[v2 < 0]=0
            Voxel_data[v3 < 0]=0
        elif(ves_color=='all'):
            print('all')
        O.append(Voxel_data)
    if len(sigmas) > 1:
        M = np.zeros(shape=(np.shape(I)[0], np.shape(I)[1], np.shape(I)[2]))
        for i in range(len(sigmas)):
            M = np.maximum(M,O[i])
    else:
        M=O[0]
    return M
