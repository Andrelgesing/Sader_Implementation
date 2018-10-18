"""
Created on Wed Oct 17 10:58:18 2018
Package created as the implementation of Sader's methods for 1D rectangular cantilevers
@author: Loch
"""

def Sader_Scaling_Parameters(Re_Scaling = 1 ,T_Scaling = 5, L = 10e-3, b = 1e-3, h = 1e-4, E = 165e9, rho = 1000,  Nx = 201,Nn = 10,f_initial = 0.1, f_final = 5e4, Nw = 5000, T = 300, Plots_Inside = False, Save = False):
    """
    Function created as the implementation of Sader's methods for 1D rectangular cantilevers
    In this function you enter with the scaling parameters Re_Scaling, and T_Scaling, if you would like to enter with material properties, refer to the Sader_Material_Parameters function.
    All parameters must be scalars and in the SI (m, Pa, kg/m^3, and so on)
    
    L is the Length of the cantilever, b is width of the cantilever, h is the height of the cantilever, E is the Youngs Module, 
    and rho the Fluid density. 
    Nx, Nw are spatial and frequency discretization, and Nn is the number of coefficients. 
    f_init and f_final are the limits of the frequency domain, note that f_final should be large enough for the Alpha integration to converge. 
    T is the temperature in Kelvin.
    Plots_Inside determines whether you are plotting, and saving if you will save these figures.
    
    This function returns W, dW, f_p, f_v, f_R, Q, f, X
    W is the displacement as a function of f and X
    dW is the slopt of the displacement as a function of f and X
    f_p are the peak frequencies
    f_v are the vacum ressonances
    f_R are the ressonance frequencies neglecting dissipative effecs
    Q is the first ressonance quality factor
    f is the frequency discretization
    X is the spatial discretization    
    
    Example:
        W, dW, f_p, f_v, f_R, Q, f, X, fp_indexes = Sader_Scaling_Parameters()
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.signal
    kb =  1.38064852 * 1e-23
    I = b*h**3/12 # Inertial moment
    rho_c = rho*b/(T_Scaling*h)
    mu = rho_c*b*h # Mass per length unit
    X = np.linspace(0, L, Nx)
    C = Find_N_roots(Nn,Plots_Inside,Save)
    f_v = np.square(C)/ L**2 * np.sqrt(E*I / mu) 
    f = np.logspace(np.log10(f_initial), np.log10(f_final), Nw, endpoint=True, base=10.0)
    eta = C[0]**2/(8*np.sqrt(3))*h*(b/L)**2*rho/Re_Scaling*np.sqrt(E/rho_c)
    Re = rho*f*b**2/(4*eta)
    Gamma_Cant = Gamma_Function(Re,Plots_Inside, Save)
    B =  np.sqrt(f/f_v[0]) * C[0] * (1 + np.pi*rho*b**2 / 4 / mu * Gamma_Cant)**(1/4) 
    Phi=[]
    Alpha =[]
    dPhi_dx = []
    for cn in C:
        Phi.append((np.cos(cn*X/L) - np.cosh(cn*X/L) + (np.cos(cn)+np.cosh(cn)) / (np.sin(cn)+np.sinh(cn)) \
                        * (np.sinh(cn*X/L)-np.sin(cn*X/L))))
        Alpha.append(2*np.sin(cn)*np.tan(cn)/(cn*(cn**4 - B**4)*(np.sin(cn)+np.sinh(cn))))
        dPhi_dx.append((-np.sin(cn*X/L) - np.sinh(cn*X/L) + (np.cos(cn)+np.cosh(cn)) / (np.sin(cn)+np.sinh(cn)) \
                  * (np.cosh(cn*X/L)-np.cos(cn*X/L)))*cn)
    Alpha = np.array(Alpha)
    Phi = np.array(Phi)
    dPhi_dx = np.array(dPhi_dx)
    df=[]      
    for ww in range(Nw):
        if ww==0:
            df.append((f[ww+1]-f[ww])/2)
        elif ww==Nw-1:
            df.append((f[ww]-f[ww-1])/2)
        else:
            df.append((f[ww+1]-f[ww-1])/2)  
    Int_Alpha = np.sum(np.abs(Alpha)**2*df,axis=1)
    Div1 = C**4*Int_Alpha
    Fac1 = np.abs(Alpha)**2/Div1[:,np.newaxis]
    H = 3/2*f_v[0] * np.pi*(Fac1[:,:,None] * dPhi_dx[:,np.newaxis]**2).sum(axis=0)
    dW = H*2*kb*T/f_v[0]
    fp_indexes = scipy.signal.find_peaks(np.array(H[:,-1]))[0]
    f_p = f[fp_indexes]
    W = (Fac1[:,:,None] * Phi[:,np.newaxis]**2).sum(axis=0)
    W_fp = W[fp_indexes,:]
    W_norm = np.sqrt(W_fp) / np.abs(np.sqrt(W_fp[:,-1]))[:,np.newaxis]
    f_plot = f/f_v[0]
    f_R = (1+np.pi*T_Scaling/4)**(-0.5)*f_v[0]
    index = np.argmin(np.abs(np.array(f)-f_R))    
    Q = (4/(np.pi*T_Scaling)+np.real(Gamma_Cant[index]))/np.imag(Gamma_Cant[index]) 
    for ii in range(20):
        index = np.argmin(np.abs(np.array(f)-f_R))    
        Q = (4/(np.pi*T_Scaling) + np.real(Gamma_Cant[index]))/np.imag(Gamma_Cant[index])
        f_R = (1+np.pi*T_Scaling/4*np.real(Gamma_Cant[index]))**(-0.5)*f_v[0] 
    if Plots_Inside:
        plt.figure()
        plt.semilogy(f_plot,H[:,-1],'-b')
        plt.semilogy(f_p/f_v[0],H[fp_indexes,-1],'rx')
        plt.xlim([0.01, f_plot[-1]])
        plt.ylabel('H')
        plt.xlabel('$\omega/\omega_{vac}$')
        plt.tight_layout()
        plt.show()
        
    if Plots_Inside:
       plt.figure()
       for jj in range(3):
           plt.subplot(3,1,jj+1)
           plt.plot(X,W_norm[jj,:])
           plt.xlim([0, L])
           if jj < 2:
                plt.xticks([])
       plt.tight_layout()
       plt.show()
    return W, dW, f_p, f_v, f_R, Q, f, X, fp_indexes


def Sader_Material_Parameters(rho_c = 2328 , eta = 8.9e-4 , L = 10e-3, b = 1e-3, h = 1e-4, E = 165e9, rho = 1000,  Nx = 201,Nn = 10,f_initial = 0.1, f_final = 5e4, Nw = 5000, T = 300, Plots_Inside = True, Save = False):
    """
    Function created as the implementation of Sader's methods for 1D rectangular cantilevers
    In this function you enter with the material properties, if you would like to enter scaling parameters Re_Scaling, and T_Scaling, refer to the Sader_Scaling_Parameters function.
    All parameters must be scalars and in the SI (m, Pa, kg/m^3, and so on)
    
    rho_c is the cantilever density, and eta is the fluid's Dynamic viscosity.
    L is the Length of the cantilever, b is width of the cantilever, h is the height of the cantilever, E is the Youngs Module, 
    and rho the Fluid density. 
    Nx, Nw are spatial and frequency discretization, and Nn is the number of coefficients. 
    f_init and f_final are the limits of the frequency domain, note that f_final should be large enough for the Alpha integration to converge. 
    T is the temperature in Kelvin.
    Plots_Inside determines whether you are plotting, and saving if you will save these figures.
    
    This function returns W, dW, f_p, f_v, f_R, Q, f, X
    W is the displacement as a function of f and X
    dW is the slopt of the displacement as a function of f and X
    f_p are the peak frequencies
    f_v are the vacum ressonances
    f_R are the ressonance frequencies neglecting dissipative effecs
    Q is the first ressonance quality factor
    f is the frequency discretization
    X is the spatial discretization    
    Re_Scaling is the scaling Reynolds Number
    T_Scaling represents the ratio of added apparent mass
    
    Example:
        W, dW, f_p, f_v, f_R, Q, f, X, Re_Scaling, T_Scaling = Sader_Material_Parameters()
    """    
    import numpy as np
    import scipy.signal
    import matplotlib.pyplot as plt
    kb =  1.38064852 * 1e-23
    I = b*h**3/12 # Inertial moment
    T_Scaling = rho*b/(rho_c*h)
    mu = rho_c*b*h # Mass per length unit
    X = np.linspace(0, L, Nx)
    C = Find_N_roots(Nn,Plots_Inside,Save)
    f_v = np.square(C)/ L**2 * np.sqrt(E*I / mu) 
    f = np.logspace(np.log10(f_initial), np.log10(f_final), Nw, endpoint=True, base=10.0)
    Re_Scaling = C[0]**2/(8*np.sqrt(3))*h*(b/L)**2*rho/eta*np.sqrt(E/rho_c)
    Re = rho*f*b**2/(4*eta)
    Gamma_Cant = Gamma_Function(Re,Plots_Inside,Save)
    B =  np.sqrt(f/f_v[0]) * C[0] * (1 + np.pi*rho*b**2 / 4 / mu * Gamma_Cant)**(1/4) 
    Phi=[]
    Alpha =[]
    dPhi_dx = []
    for cn in C:
        Phi.append((np.cos(cn*X/L) - np.cosh(cn*X/L) + (np.cos(cn)+np.cosh(cn)) / (np.sin(cn)+np.sinh(cn)) \
                        * (np.sinh(cn*X/L)-np.sin(cn*X/L))))
        Alpha.append(2*np.sin(cn)*np.tan(cn)/(cn*(cn**4 - B**4)*(np.sin(cn)+np.sinh(cn))))
        dPhi_dx.append((-np.sin(cn*X/L) - np.sinh(cn*X/L) + (np.cos(cn)+np.cosh(cn)) / (np.sin(cn)+np.sinh(cn)) \
                  * (np.cosh(cn*X/L)-np.cos(cn*X/L)))*cn)
    Alpha = np.array(Alpha)
    Phi = np.array(Phi)
    dPhi_dx = np.array(dPhi_dx)
    df=[]      
    for ww in range(Nw):
        if ww==0:
            df.append((f[ww+1]-f[ww])/2)
        elif ww==Nw-1:
            df.append((f[ww]-f[ww-1])/2)
        else:
            df.append((f[ww+1]-f[ww-1])/2)  
    Int_Alpha = np.sum(np.abs(Alpha)**2*df,axis=1)
    Div1 = C**4*Int_Alpha
    Fac1 = np.abs(Alpha)**2/Div1[:,np.newaxis]
    H = 3/2*f_v[0] * np.pi*(Fac1[:,:,None] * dPhi_dx[:,np.newaxis]**2).sum(axis=0)
    dW = H*2*kb*T/f_v[0]
    fp_indexes = scipy.signal.find_peaks(np.array(H[:,-1]))[0]
    f_p = f[fp_indexes]
    W = (Fac1[:,:,None] * Phi[:,np.newaxis]**2).sum(axis=0)
    W_fp = W[fp_indexes,:]
    W_norm = np.sqrt(W_fp) / np.abs(np.sqrt(W_fp[:,-1]))[:,np.newaxis]
    f_plot = f/f_v[0]
    f_R = (1+np.pi*T_Scaling/4)**(-0.5)*f_v[0]
    index = np.argmin(np.abs(np.array(f)-f_R))    
    Q = (4/(np.pi*T_Scaling)+np.real(Gamma_Cant[index]))/np.imag(Gamma_Cant[index]) 
    for ii in range(20):
        index = np.argmin(np.abs(np.array(f)-f_R))    
        Q = (4/(np.pi*T_Scaling) + np.real(Gamma_Cant[index]))/np.imag(Gamma_Cant[index])
        f_R = (1+np.pi*T_Scaling/4*np.real(Gamma_Cant[index]))**(-0.5)*f_v[0] 
    if Plots_Inside:
        plt.figure()
        plt.semilogy(f_plot,H[:,-1],'-b')
        plt.semilogy(f_p/f_v[0],H[fp_indexes,-1],'rx')
        plt.xlim([0.01, f_plot[-1]])
        plt.ylabel('H')
        plt.xlabel('$\omega/\omega_{vac}$')
        plt.tight_layout()
        plt.show()
        
    if Plots_Inside:
       plt.figure()
       for jj in range(3):
           plt.subplot(3,1,jj+1)
           plt.plot(X,W_norm[jj,:])
           plt.xlim([0, L])
           if jj < 2:
                plt.xticks([])
       plt.tight_layout()
       plt.show()
    return W, dW, f_p, f_v, f_R, Q, f, X, fp_indexes, Re_Scaling, T_Scaling,

def Find_N_roots(Nn=20,Plots_Inside=False, Save = False): 
    """
    Function to find Nn roots of the transcedental Equation 1+Cos(c)Cosh(c) =  0
    Enter with the number of roots (coefficients) you want (Nn), followed by True if you want to plot this result, and True if you want to save the plot.
    Standard values are Nn=20, and Plots equals False
    This function returns the vector C, corresponding to the first Nn coefficients of the transcedental equation.
    Example:
        Nn = 20
        C = Find_N_roots(Nn, True)
    """
    import scipy.optimize
    import numpy as np
    import matplotlib.pyplot as plt
    def func_Coef(C):
        return np.cos(C)*np.cosh(C) + 1
    ii=2
    Delta_C = 1     
    C = list(scipy.optimize.root(func_Coef, ii).x)
    while len(C)<Nn:
        if func_Coef(ii) > 0 and func_Coef(ii+Delta_C) <0:
            C.append(scipy.optimize.brenth(func_Coef,ii,ii+Delta_C))         
        elif func_Coef(ii) < 0 and func_Coef(ii+Delta_C)>0:
            C.append(scipy.optimize.brenth(func_Coef,ii,ii+Delta_C))
        ii += Delta_C
    C=np.array(C)    
    if Plots_Inside:
        name = plt.figure()
        C_range = np.linspace(0,C[-1]+Delta_C,201) 
        plt.title('# Plot of roots C')
        plt.plot(C_range,func_Coef(C_range),'-b', label = 'Function');
        plt.yscale('symlog')
        plt.plot(C,np.zeros(len(C))*1e-4,'ro',label = 'Root')
        plt.legend()
        plt.xlabel('C')
        plt.ylabel('Value')
        plt.xlim([0, C[-1]])
        if Save:
            name.savefig('PDF/PDF_Roots.pdf', bbox_inches='tight')
        plt.tight_layout()
        plt.show()
    return C

def Gamma_Function(Re,Plots_Inside=False, Save = False):
    """
    Function to determinate the Gamma Hydrodynamic function of circular beams, and extrapolate to rectangular cantilevers.
    The function returns only the cantilever hydrodynamic funtion.
    Enter with the Reynolds number (as vector or np.array), followed by True if you want to plot this result, and True if you want to save the plot.
    Example:
        Re=np.logspace(-2,2,endpoint = True, base = 10)
        Gamma_Cant = Gamma_Function(Re,True, False)
    """
    import numpy as np
    import scipy.special

    import matplotlib.pyplot as plt
    Gamma_Circ = []
    for ii in range(len(Re)):
        # Equation 18 - Tau for circular beams
        Gamma_Circ.append(1 + 4*1j*scipy.special.kv(1, -1j*np.sqrt(1j*Re[ii])) / \
                (np.sqrt(1j*Re[ii])*scipy.special.kv(0, -1j*np.sqrt(1j*Re[ii]))))
    # Equation 22
    Tau = np.log10(Re)
    
    #Equation 21a - Real part of Omega
    Omega_Real = (0.91324 - 0.48274*Tau + 0.46842*Tau**2 - 0.12886*Tau**3 \
              + 0.044055*Tau**4 - 0.0035117*Tau**5 + 0.00069085* Tau**6)\
              / (1 - 0.56964*Tau + 0.48690*Tau**2 - 0.13444 * Tau**3\
                 + 0.045155*Tau**4 - 0.0035862 * Tau**5 + 0.00069085*Tau**6)
              # Equation 21b - Imaginary part of Omega
    Omega_imag = (-0.024134 - 0.029256 *Tau + 0.016294 * Tau**2 - 0.00010961*Tau**3\
              + 0.000064577*Tau**4 - 0.000045510*Tau**5)\
            / (1 - 0.59702*Tau + 0.55182*Tau**2 - 0.18357*Tau**3\
               + 0.079156 * Tau**4 - 0.014369*Tau**5 + 0.0028361*Tau**6)              

    Omega = Omega_Real + 1j*Omega_imag
    Gamma_Cant = Omega*Gamma_Circ
    if Plots_Inside:
#       KEYS = {'Real','Imag'}
#       Re_sader = {None:None}
#       Gamma = {None:None}
#       for key in KEYS:
#           csv_file = '/home/loch/PhD/Python/Sader/CSV_Files/Gamma_'+key+'.csv'
#           df = pd.read_csv(csv_file, header=None, index_col = False, names=['Re', 'Gamma'])
#           Re_sader[key] = df.Re
#           Gamma[key] = df.Gamma
       name = plt.figure()
       plt.title('$\Gamma_\mathrm{cant}$')
       plt.loglog(Re,np.real(Gamma_Cant),'-b', label = ('$\Gamma_\mathrm{cant}$'))
       plt.loglog(Re,np.imag(Gamma_Cant),'-k', label = ('$\Gamma_\mathrm{cant}$'))
#       for key in KEYS:
#           plt.loglog(Re_sader[key], Gamma[key], '--y', label = (key + '$\Gamma_\mathrm{cant}$ [ref]'))
       plt.xlim([0.01, 100])
       plt.ylim([0.3, 120])
       plt.xlabel('Re')
       plt.ylabel('$\Gamma$')
       plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
       if Save:
           name.savefig('PDF_Hydro_Function.pdf', bbox_inches='tight')
       plt.show()   
    return Gamma_Cant

