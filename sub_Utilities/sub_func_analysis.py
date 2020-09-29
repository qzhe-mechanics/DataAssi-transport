import numpy as np
# import sobol_seq  # require https://pypi.org/project/sobol_seq/
from pyDOE import lhs   # The experimental design package for python; Latin Hypercube Sampling (LHS)

# Generate coordinates vector for uniform grids over a 2D rectangles. Order starts from left-bottom, row-wise, to right-up
def rectspace(a,b,c,d,nx,ny):
    x = np.linspace(a,b,nx)
    y = np.linspace(c,d,ny)
    [X,Y] = np.meshgrid(x,y)
    Xm = np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1)
    return Xm

def rectspace_dis(lb,ub,N,len_ratio=None,adjust=None,rand_rate=None):
    
    if len_ratio:
        ny = np.sqrt(N/len_ratio).astype(int)
        nx = (N/ny).astype(int)
        N_new = nx * ny
    else:
        ny = np.sqrt(N/2).astype(int)
        nx = (N/ny).astype(int)
        N_new = nx * ny        

    a, b, c, d = lb[0], ub[0],lb[1],ub[1]
    
    if adjust:
        a = a + adjust
        b = b - adjust
        c = c + adjust
        d = d - adjust

    Xm = rectspace(a,b,c,d,nx,ny)
    if rand_rate:
        Xm[:,0:1] = Xm[:,0:1] + np.random.normal(0,rand_rate,(N_new,1))
        Xm[:,1:2] = Xm[:,1:2] + np.random.normal(0,rand_rate,(N_new,1))        
    return Xm, N_new

  
def find_nearestl2(XM, x):
    # array = np.asarray(array)
    XM0 = XM - x
    norm_XM0 = np.linalg.norm(XM0,axis=1)
    idx = norm_XM0.argmin()
    return XM[idx,:], idx


def find_nearestVec(XM, xm):
#	(num_y,num_x) = xm.shape
    idx = []
    (num_y,num_x) = xm.shape
    xm_new = np.zeros((num_y,num_x))
    for i in range(0,num_y):
        X_i, idx_i = find_nearestl2(XM, xm[i,:])
        xm_new[i,:] = X_i
        idx.append(idx_i)
    return idx, xm_new 


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
