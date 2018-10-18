#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:34:47 2018
Example on how to use Sader's Formulation implemented in Python
@author: loch
"""

import Sader_Implementation_1D_Rectangular_Cantilevers as Sader
import matplotlib.pyplot as plt
import numpy as np
## Define Parameters
Re_Scaling = 10 ;T_Scaling = 0.5; L = 10e-3; b = 1e-3;
h = 1e-4; E = 165e9; rho = 1000;  Nx = 201; Nn = 10; f_initial = 0.1;
f_final = 5e4; Nw = 5000; T = 300; Plots_Inside = False; Save = False;

W, dW, f_p, f_v, f_R, Q, f, X, fp_indexes = Sader.Sader_Scaling_Parameters(Re_Scaling, T_Scaling, L, b, h, E, rho, Nx, Nn, f_initial, f_final, Nw, T, Plots_Inside, Save)
## Figures
plt.figure()
plt.title('Slope of displacement at x=L')
plt.semilogy(f/f_v[0],dW[:,-1],'-b')
plt.semilogy(f_p/f_v[0],dW[fp_indexes,-1],'rx')
plt.xlim([0.01, f[-1]/f_v[0]])
plt.ylabel('dW_dx')
plt.xlabel('$\omega/\omega_{vac}$')
plt.tight_layout()
plt.show()


plt.figure()
plt.title('Displacement at x=L')
plt.semilogy(f/f_v[0],W[:,-1],'-b')
plt.semilogy(f_p/f_v[0],W[fp_indexes,-1],'rx')
plt.xlim([0.01, f[-1]/f_v[0]])
plt.ylabel('W')
plt.xlabel('$\omega/\omega_{vac}$')
plt.tight_layout()
plt.show()

W_fp = W[fp_indexes,:]
W_norm = np.sqrt(W_fp) / np.abs(np.sqrt(W_fp[:,-1]))[:,np.newaxis]

plt.figure()
for jj in range(3):
   plt.subplot(3,1,jj+1)
   plt.title('%1.0f Mode' %(jj+1))
   plt.plot(X,W_norm[jj,:])
   plt.xlim([0, L])
   plt.tight_layout()
   if jj < 2:
       plt.xticks([])
plt.show()

F = 3e4
index = np.argmin(np.abs(np.array(f)-F)) 
plt.figure()
plt.title('Displacement at frequency F = %4.1f' %F)
plt.plot(X,W[index,:],'-b')
plt.xlim([0, L])
plt.ylabel('W')
plt.xlabel('$X(m)$')
plt.tight_layout()
plt.show()

print('Quality factor is %2.2f.' %Q)

## Define Parameters for analysis without the scaling parameters
rho_c = 2328 ; eta = 8.9e-4 ; L = 15e-3; b = 1e-3;
h = 1e-4; E = 165e9; rho = 1000;  Nx = 201; Nn = 10; f_initial = 0.1;
f_final = 2e5; Nw = 5000; T = 300; Plots_Inside = False; Save = False;

W, dW, f_p, f_v, f_R, Q, f, X, fp_indexes, Re_Scaling, T_Scaling = Sader.Sader_Material_Parameters(rho_c, eta, L, b, h, E, rho, Nx, Nn, f_initial, f_final, Nw, T, Plots_Inside, Save)
## Figures

plt.figure()
plt.title('Slope of displacement at x=L')
plt.semilogy(f/f_v[0],dW[:,-1],'-b')
plt.semilogy(f_p/f_v[0],dW[fp_indexes,-1],'rx')
plt.xlim([0.01, f[-1]/f_v[0]])
plt.ylabel('dW_dx')
plt.xlabel('$\omega/\omega_{vac}$')
plt.tight_layout()
plt.show()


plt.figure()
plt.title('Displacement at x=L')
plt.semilogy(f/f_v[0],W[:,-1],'-b')
plt.semilogy(f_p/f_v[0],W[fp_indexes,-1],'rx')
plt.xlim([0.01, f[-1]/f_v[0]])
plt.ylabel('W')
plt.xlabel('$\omega/\omega_{vac}$')
plt.tight_layout()
plt.show()

W_fp = W[fp_indexes,:]
W_norm = np.sqrt(W_fp) / np.abs(np.sqrt(W_fp[:,-1]))[:,np.newaxis]

plt.figure()
plt.title('Modes')
for jj in range(3):
   plt.subplot(3,1,jj+1)
   plt.title('%1.0f Mode' %(jj+1))
   plt.plot(X,W_norm[jj,:])
   plt.xlim([0, L])
   plt.tight_layout()
   if jj < 2:
       plt.xticks([])
plt.show()

F = 3e4
index = np.argmin(np.abs(np.array(f)-F)) 
plt.figure()
plt.title('Displacement at frequency F = %4.1f' %F)
plt.plot(X,W[index,:],'-b')
plt.xlim([0, L])
plt.ylabel('W')
plt.xlabel('$X(m)$')
plt.tight_layout()
plt.show()

print('Quality factor is %2.2f.' %Q)