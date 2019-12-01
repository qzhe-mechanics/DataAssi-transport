"""
Code: version 6.5

@author: Qizhi He @ PNNL (qizhi.he@pnnl.gov)

Reference:

QiZhi He, David Brajas-Solano, Guzel Tartakovsky, Alexandre M. Tartakovsky, "Physics-Informed Neural Networks for Multiphysics Data Assimilation with Application to Subsurface Transport"

Correspondence: Alexandre.Tartakovsky@pnnl.gov
"""

import os
import sys
sys.path.insert(0, 'subcode_MPINN/')
# sys.path.insert(0, 'sub_Utilities/') 

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs                       # The experimental design package for python; Latin Hypercube Sampling (LHS)
from scipy.interpolate import griddata
import time               
import tensorflow as tf
from class_PINN_probl import *
# import matplotlib.axes as plx

sys_len = len(sys.argv)
print(sys_len)

'''System Input'''
path_f   = sys.argv[1]      if len(sys.argv) > 1 else './Output/temp/pro1_ts3_opt2i1k_lr1e3_temp_k48h48f200/'
seed_num = int(sys.argv[2]) if len(sys.argv) > 2 else 111
N_s      = int(sys.argv[3]) if len(sys.argv) > 3 else 1    # Initialization of NNs
N_k      = int(sys.argv[4]) if len(sys.argv) > 4 else 48   # Measurement of Conductivity
N_h      = int(sys.argv[5]) if len(sys.argv) > 5 else 48   # Measurement of Pressure head
N_f      = int(sys.argv[6]) if len(sys.argv) > 6 else 200   # Collocations
N_c      = int(sys.argv[7]) if len(sys.argv) > 7 else 100   # Measurement of Concentration
N_fc     = int(sys.argv[8]) if len(sys.argv) > 8 else 400   # Collocations
# Parameter to control the loss function
para_k   = float(sys.argv[9])  if len(sys.argv) > 9  else 1.0
para_h   = float(sys.argv[10]) if len(sys.argv) > 10 else 1.0
para_kh  = float(sys.argv[11]) if len(sys.argv) > 11 else 1.0
para_c   = float(sys.argv[12]) if len(sys.argv) > 12 else 1.0  
para_khc = float(sys.argv[13]) if len(sys.argv) > 13 else 1.0
# System control
flag_data = int(sys.argv[14])  if len(sys.argv) > 14  else 12 
IF_plot   = int(sys.argv[15])  if len(sys.argv) > 15  else 1
# NN Architech
type_NNk  = int(sys.argv[16])  if len(sys.argv) > 16  else 31
type_NNh  = int(sys.argv[17])  if len(sys.argv) > 17  else 31
type_NNc  = int(sys.argv[18])  if len(sys.argv) > 18  else 31
# NN Activation function (1: tanh; 2.ReLu;) (Haven't find a way to implement...)
type_actk  = int(sys.argv[19])  if len(sys.argv) > 19 else 1  
type_acth  = int(sys.argv[20])  if len(sys.argv) > 20 else 1
type_actc  = int(sys.argv[21])  if len(sys.argv) > 21 else 1
# Optimizer
# type_op: 1: L-BFGS ; 2. Adam; 3. BFGS - Adam 4: Other
type_op    = int(sys.argv[22])  if len(sys.argv) > 22 else 6 
batchs     = int(sys.argv[23])  if len(sys.argv) > 23 else 0        # number of batchs
num_epochs = int(sys.argv[24])  if len(sys.argv) > 24 else 1000     # number of itrations
learn_rate  = float(sys.argv[25])  if len(sys.argv) > 25 else 0.001 # learning rate (default for Reg:0.00001;PINN 0.0001)
# Regularization
type_reg   = int(sys.argv[26])   if len(sys.argv) > 26 else 0       # 1: L1 Reg (default)
coe_reg    = float(sys.argv[27]) if len(sys.argv) > 27 else 1e-8    # default (1e-8)
# Sequentially Multi-stage open
flag_solv  = sys.argv[28]  if len(sys.argv) > 28 else 'standard'    # 0: standard.. 1: sequent
# 1: CAD; 2: Darcy; 31-35: Regression;
flag_pro   = int(sys.argv[29])  if len(sys.argv) > 29 else 1      
sys_id = 29
sys_id += 1  
type_mea    = int(sys.argv[sys_id]) if len(sys.argv) > sys_id else 22 # seems 22 the best
sys_id += 1
seed_mea = int(sys.argv[sys_id])    if len(sys.argv) > sys_id else 111 # 1127
sys_id += 1
flag_lsty = int(sys.argv[sys_id])   if len(sys.argv) > sys_id else 1 # 1: default for the version
sys_id += 1
conv_opt   = float(sys.argv[sys_id])   if len(sys.argv) > sys_id else 0.001 # The conv coefficient for opt7

if flag_data == 12:
    path_data = './Data_mea/data1_sin_smooth/'       # The periodic data
elif flag_data == 41:
    path_data = './Data_mea/data_smooth_k02_normal/' # Correlation lenght = 0.2
elif flag_data == 42:
    path_data = './Data_mea/data_smooth_k05_normal/' # Correlation lenght = 0.5
elif flag_data == 43:
    path_data = './Data_mea/data_smooth_k10_normal/' # Correlation lenght = 1.0

if not os.path.exists(path_f):
    os.makedirs(path_f)

path_fig = path_f+'figures'+'/'
if not os.path.exists(path_fig):
    os.makedirs(path_fig)

f         = open(path_f+'record.out', 'a+')
f1        = open(path_f+'record_data.out', 'a+')
f1_loss   = open(path_f+'record_loss.out', 'a+')
f1_loss2  = open(path_f+'record_loss2.out', 'a+')
f2_weight = open(path_f+'record2_weight.out', 'a+')
f2_bias   = open(path_f+'record2_bias.out', 'a+')

print("test: Nk: {}, Nh: {}, Nf: {}, N_c: {}, N_fc: {}"\
    .format(N_k, N_h, N_f, N_c, N_fc), file=f)
print("seed: {}, # Initial: {}".format(seed_num,N_s),file=f)
print("Measurement Seed: {}, type: {}".format(seed_mea,type_mea),file=f)
print("para: para_k: {0}, para_h: {1}, para_kh: {2}, para_c: {3}, para_khc: {4}"\
    .format(para_k, para_h, para_kh, para_c, para_khc), file=f)
print("Problem & data set: {} {}".format(flag_pro,flag_data),file=f)
print("NN architecture: {} {} {}".format(type_NNk, type_NNh, type_NNc), file=f)
print("NN activation: {} {} {}".format(type_actk, type_acth, type_actc), file=f)   
print("Optimier: {} {} {} {} {} {}".format(type_op,batchs,num_epochs,learn_rate,flag_solv,conv_opt),file=f)
print("Regularization & loss type: {} {}, lsty = {}".format(type_reg,coe_reg,flag_lsty),file=f)

tf.reset_default_graph()

if __name__ == "__main__": 
    
    ### PREPROCESS
    """
    'star' denotes the measurement grids
    """

    ## Define Parameters
    Nx = 256
    Ny = 128
    N_test  = Nx * Ny    
    coe_L1, coe_L2  = 1, 0.5 # (m)
    coe_q    = 1 # (m/hr)
    coe_H2   = 0 # (m)
    coe_C0   = 1
    coe_phi  = 0.317
    coe_Dw   = 0.09
    coe_tau  = coe_phi**(1/3) # Dw * tau
    coe_aL, coe_aT   = 0.01, 0.001 # (m)

    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 0.5])

    if flag_data == 12:     
        # Load Coordinates
        dataset_c = np.loadtxt(path_data+'plot_matlab_sin_50d.dat', dtype=float) # Load solute concentration

        X         = dataset_c[:,[0,2]]   # Coordinates of all nodes (x,z)
        
        # Load solute concentration, C(x,z)
        C_star    = dataset_c[:,4:5]   # Concentration
        
        # Load hydraulic head, h(x,z)
        dataset_h = np.loadtxt(path_data+'plot_head.dat', dtype=float) # Load hydraulic head
        # h_star    = dataset_h[:,4:5]
        h_star    = dataset_h[:,2:3]

        Yl_h, Yr_h = h_star.min(0), h_star.max(0)
        h_star = h_star - Yl_h
        
        # Load conductive coefficient, k(x,z)
        dataset_k = np.loadtxt(path_data+'ksx_ijk.dat', dtype=float)
        k_star    = dataset_k[:][:,np.newaxis]
        
    elif flag_data == 41 or flag_data == 42 or flag_data == 43:
        if flag_data == 41:
            dataset_k = np.loadtxt(path_data+'smooth_field_02_normal.txt', dtype=float)
            dataset_all = np.loadtxt(path_data+'plot_matlab_smooth_field_02_normal50d.dat', dtype=float)
        elif flag_data == 42:
            dataset_k = np.loadtxt(path_data+'smooth_field_05_normal.txt', dtype=float)
            dataset_all = np.loadtxt(path_data+'plot_matlab_smooth_field_05_normal50d.dat', dtype=float)
        elif flag_data == 43:
            dataset_k = np.loadtxt(path_data+'smooth_field_10_normal.txt', dtype=float)
            dataset_all = np.loadtxt(path_data+'plot_matlab_smooth_field_10_normal50d.dat', dtype=float)
        
        X         = dataset_all[:,[0,2]]   # Coordinates of all nodes (x,z)
        h_star    = dataset_all[:,4:5]
        Yl_h, Yr_h = h_star.min(0), h_star.max(0)
        h_star    = h_star - Yl_h
        
        C_star    = dataset_all[:,6:7]   # Concentration

        # Load conductive coefficient, k(x,z)
        k_star    = dataset_k[:][:,np.newaxis]      
    
    # Measurement locations
    X_star    = X
    N = X_star.shape[0]  # the column size of u_star, total nodes
    
    if N_test != N:
        print('Input Error: N')
        input()

    ######################################################################
    ######################## Noiseless Training Data #####################
    ######################################################################
    
    # ====================================================================
    ## Training data (Domain)
    # ====================================================================

    #Latin Hypercube sampling
    #make obs deterministic for different net initialization seeds and 
    # increasing as you add more points

    if type_mea == 1: #
        '''Select Measurement Points'''
        seed_num_temp = seed_num
        # For K, h, C
        np.random.seed(seed_num_temp)
        idx_k = np.random.choice(N, N_k, replace=False)
        np.random.seed(seed_num_temp)
        idx_h = np.random.choice(N, N_h, replace=False)
        np.random.seed(seed_num_temp)
        idx_c = np.random.choice(N, N_c, replace=False)
        # '''Select Collocation points'''
        np.random.seed(seed_num_temp)
        X_f = lb + (ub-lb)*lhs(2, N_f)
        np.random.seed(seed_num_temp)
        X_fc = lb + (ub-lb)*lhs(2, N_fc) 

    elif type_mea == 2 or type_mea == 22 or type_mea == 21: # Uniformly random (Hierarchical verified)
        np.random.seed(seed_mea) 
        idx_k = np.random.choice(N,N_k,replace=False)
        idx_h = np.random.choice(N,N_h,replace=False) # Realized a typo for v6s1
        idx_c = np.random.choice(N,N_c,replace=False)

        # '''Select Collocation points'''
        if type_mea == 2 or type_mea == 22:
            X_f0  = lb + (ub-lb)*np.random.uniform(0.,1.,(N_f,2))
            X_fc0 = lb + (ub-lb)*np.random.uniform(0.,1.,(N_fc,2))

            if type_mea == 22:
                idx_f, _ = find_nearestVec(X, X_f0)
                idx_fc0, _ = find_nearestVec(X, X_fc0)
                X_f = X[idx_f,:]
                X_fc = X[idx_fc0,:]
            elif type_mea == 2:
                X_f = X_f0
                X_fc = X_fc0

        elif type_mea == 21:   
            # '''Select Collocation points''' (Not Hierarchical)
            X_f, N_f = rectspace_dis(lb,ub,N_f,len_ratio=2,adjust=0.00)
            X_fc, N_fc = rectspace_dis(lb,ub,N_fc,len_ratio=2,adjust=0.00)


    X_k = X[idx_k,:]              
    Y_k = k_star[idx_k]

    X_h = X[idx_h,:]
    Y_h = h_star[idx_h,:]

    X_c = X[idx_c,:]
    Y_c = C_star[idx_c,:]
        
    Y_f = np.zeros((N_f, 1))   
    Y_fc = np.zeros((N_fc, 1))

    # ====================================================================
    ## Training data (B.C.) for h
    # ====================================================================

    # Dirichlet boundaries
    xb2, hb2 = X[Nx-1:N:Nx,:], h_star[Nx-1:N:Nx,:]
    X_hbD = xb2
    Y_hbD = hb2
    
    # Neumann boundaries
    xb1 = np.zeros((Nx,2))
    xb1[:,0:1] = lb[0] + (ub[0]-lb[0])*lhs(1, Nx)
    
    xb3 = np.zeros((Nx,2))
    xb3[:,1:2] = ub[1]  # y-coordinate
    xb3[:,0:1] = xb1[:,0:1]  # x-coordinate

    X_hbN = np.concatenate([xb1, xb3], axis = 0)   # approximating value
    Y_hbN = np.zeros((X_hbN.shape[0], 1))   
        
    # Speical Neumann boundaries 
    xb0 = np.zeros((Ny,2))
    xb0[:,1:2] = lb[1] + (ub[1]-lb[1])*lhs(1, Ny)
    
    X_hbNs = xb0
    Y_hbNs = np.ones((X_hbNs.shape[0],1))*coe_q
    
    # ====================================================================
    ## Training data (B.C.) for Concentration (C)  
    # =============================================================================
    # Dirichlet boundaries (edge 0 left)
    xb0, cb0 = X[::Nx,:], C_star[::Nx,:]
    X_cbD = xb0
    Y_cbD = cb0
                
    # Neumann boundaries normal to x (edge 2)
    xb2 = np.zeros((Ny,2))
    xb2[:,0:1] = ub[0]  # y-coordinate
    xb2[:,1:2] = lb[1] + (ub[1]-lb[1])*lhs(1, Ny)
    
    X_cbN1 = xb2
    Y_cbN1 = np.zeros((X_cbN1.shape[0], 1))  
    
    # Neumann boundaries normal to y (edge 1,3)
    X_cbN2 = X_hbN    # edge 1, 3
    Y_cbN2 = np.zeros((X_cbN2.shape[0], 1))  
    
    # =============================================================================
    
    # Randomize neural-nets
    if N_s > 1:
        np.random.seed(seed_num) # reset        
        tf_seed_set = np.random.randint(0,2000,N_s)
        print('rand NNs seeds {}'.format(tf_seed_set))
        print('rand NNs seeds {}'.format(tf_seed_set),file=f)

    for i_loop in range(0,N_s):

        if N_s == 1:
            rand_seed_tf_i = seed_num
        elif N_s > 1:
            rand_seed_tf_i = tf_seed_set[i_loop]

        path_tf = path_f+'tf_model'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'/'
        if not os.path.exists(path_tf):
            os.makedirs(path_tf)

        ######################################################################
        ### Physics-informed Neural Network
        
        # Create model

        layers_k = sub_NN_type(type_NNk)
        layers_h = sub_NN_type(type_NNh)
        layers_c = sub_NN_type(type_NNc)
        
        # tf.set_random_seed(rand_seed_tf_i)
        print("\n",file=f)
        print("seed_tf: {}".format(rand_seed_tf_i),file=f)

        if_plot_rl2 = 1

        if flag_pro == 1:
            model = PINN_CAD_seq(X_h, Y_h, X_k, Y_k, X_c, Y_c, X_f, Y_f, X_fc, Y_fc,
                                  X_hbD, Y_hbD, X_hbN, Y_hbN, X_hbNs, Y_hbNs,
                                  X_cbD, Y_cbD, X_cbN1, Y_cbN1, X_cbN2, Y_cbN2,
                                  layers_h, layers_k, layers_c, lb, ub,
                                  coe_phi,coe_Dw,coe_tau,coe_aL,coe_aT,
                                  f,path_tf,f1_loss,f1_loss2,f2_weight,f2_bias,
                                  para_k,para_h,para_kh,para_c,para_khc,
                                  type_op,type_reg,coe_reg,rand_seed_tf_i,learn_rate,flag_solv,
                                  flag_pro,if_plot_rl2,flag_lsty,conv_opt,
                                  ref_xm = X_star,ref_k = k_star, ref_h = h_star, ref_c = C_star)                
        elif flag_pro == 2:       
                model = PINN_Darcy(X_h, Y_h, X_k, Y_k, X_c, Y_c, X_f, Y_f, X_fc, Y_fc,
                                      X_hbD, Y_hbD, X_hbN, Y_hbN, X_hbNs, Y_hbNs,
                                      X_cbD, Y_cbD, X_cbN1, Y_cbN1, X_cbN2, Y_cbN2,
                                      layers_h, layers_k, layers_c, lb, ub,
                                      coe_phi,coe_Dw,coe_tau,coe_aL,coe_aT,
                                      f,path_tf,f1_loss,f1_loss2,f2_weight,f2_bias,
                                      para_k,para_h,para_kh,para_c,para_khc,
                                      type_op,type_reg,coe_reg,rand_seed_tf_i,learn_rate,flag_solv,
                                      flag_pro,if_plot_rl2,flag_lsty,conv_opt,
                                      ref_xm = X_star,ref_k = k_star, ref_h = h_star, ref_c = C_star)

        elif flag_pro == 31 or flag_pro == 32 or flag_pro == 33 or flag_pro == 34 or flag_pro == 35:
            model = PINN_Reg(X_h, Y_h, X_k, Y_k, X_c, Y_c, X_f, Y_f, X_fc, Y_fc,
                                  X_hbD, Y_hbD, X_hbN, Y_hbN, X_hbNs, Y_hbNs,
                                  X_cbD, Y_cbD, X_cbN1, Y_cbN1, X_cbN2, Y_cbN2,
                                  layers_h, layers_k, layers_c, lb, ub,
                                  coe_phi,coe_Dw,coe_tau,coe_aL,coe_aT,
                                  f,path_tf,f1_loss,f1_loss2,f2_weight,f2_bias,
                                  para_k,para_h,para_kh,para_c,para_khc,
                                  type_op,type_reg,coe_reg,rand_seed_tf_i,learn_rate,flag_solv,
                                  flag_pro,if_plot_rl2,flag_lsty,conv_opt,
                                  ref_xm = X_star,ref_k = k_star, ref_h = h_star, ref_c = C_star)
        
        '''Train Neural Network'''
        num_print_opt = 100
        start_time = time.time()                
        model.train(type_op,batchs,num_epochs,flag_solv,num_print_opt)
        model.out_loss_components()

        elapsed = time.time() - start_time 
        print('Training time: %.4f' % (elapsed))
        print('Training time: {0: .4f}'.format(elapsed),file=f)

        # Predict at test points
        k_pred = model.predict_k(X_star)
        h_pred = model.predict_h(X_star)
        c_pred = model.predict_c(X_star)
        
        # Relative L2 error
        # same as calling k1, h1, c1 = model.error_test_l2(X_star)
        error_k = np.linalg.norm(k_star - k_pred, 2)/np.linalg.norm(k_star, 2)
        error_h = np.linalg.norm(h_star - h_pred, 2)/np.linalg.norm(h_star, 2)
        error_c = np.linalg.norm(C_star - c_pred, 2)/np.linalg.norm(C_star, 2)

        print('L2 Error k: %e' % (error_k))   
        print('L2 Error h: %e' % (error_h))
        print('L2 Error c: %e' % (error_c)) 

        print('Error k: {0: e}'.format(error_k),file=f)     
        print('Error h: {0: e}'.format(error_h),file=f)
        print('Error c: {0: e}'.format(error_c),file=f)   

        print('{0: .4f} {1: e} {2: e} {3: e}'.format(elapsed,error_k,error_h,error_c),file=f1)


        # Record
        f_k_pred  = open(path_tf+'record_k_pred.out', 'a+')
        f_h_pred  = open(path_tf+'record_h_pred.out', 'a+')
        f_c_pred  = open(path_tf+'record_c_pred.out', 'a+')

        mat_k = np.matrix(k_pred)
        for line in mat_k:
            np.savetxt(f_k_pred, line, fmt='%.8f')
        mat_h = np.matrix(h_pred)
        for line in mat_h:
            np.savetxt(f_h_pred, line, fmt='%.8f')
        mat_c = np.matrix(c_pred)
        for line in mat_c:
            np.savetxt(f_c_pred, line, fmt='%.8f')
        f_k_pred.close()
        f_h_pred.close()
        f_c_pred.close()

        # # Output
        # if type_op == 1:
        #     num_print_opt = 1
        # For BFGS, still output every num_print_opt 
        if batchs != 0: # When use batch; rloss_batch output every time.
            plt.close('all')
            plt.semilogy(model.rloss_batch)
            plt.xlabel('Iteration', fontsize=14)
            plt.ylabel('$L_{batch}$', fontsize=14)
            path_fig_save = path_tf +'loss_batch_'+str(type_op)+'_'+str(batchs)+'_'+str(num_epochs)
            plt.savefig(path_fig_save+'.png',dpi=200)

            f_rloss  = open(path_tf+'record_rloss_batch.out', 'a+')
            mat_rloss = np.matrix(model.rloss_batch)
            for line in mat_rloss:
                np.savetxt(f_rloss, line, fmt='%.8f')
            f_rloss.close()

        if batchs != 0:
            plt.close('all')
            num_lx = len(model.rloss)
            xm_ls = np.arange(0, num_print_opt * num_lx, num_print_opt)
            plt.semilogy(xm_ls,model.rloss)
            plt.xlabel('Iteration', fontsize=14)
            plt.ylabel('$Loss$', fontsize=14)
            path_fig_save = path_tf +'loss_full_'+str(type_op)+'_'+str(batchs)+'_'+str(num_epochs)
            plt.savefig(path_fig_save+'.png',dpi=200)
        elif batchs == 0:         
            plt.close('all')
            plt.semilogy(model.rloss) # Every iteration
            plt.xlabel('Iteration', fontsize=14)
            plt.ylabel('$Loss$', fontsize=14)
            path_fig_save = path_tf +'loss_'+str(type_op)+'_'+str(batchs)+'_'+str(num_epochs)
            plt.savefig(path_fig_save+'.png',dpi=200)
        
        f_rloss  = open(path_tf+'record_rloss.out', 'a+')
        mat_rloss = np.matrix(model.rloss)
        for line in mat_rloss:
            np.savetxt(f_rloss, line, fmt='%.8f')
        f_rloss.close()

        plt.close('all')
        num_lx = len(model.rloss_k)
        # xm_ls2 = np.linspace(0, num_print_opt * num_lx, num_lx)
        xm_ls = np.arange(0, num_print_opt * num_lx, num_print_opt)
        plt.semilogy(xm_ls,model.rloss_k)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('$L_k$', fontsize=14)
        path_fig_save = path_tf +'loss_k_'+str(type_op)+'_'+str(batchs)+'_'+str(num_epochs)
        plt.savefig(path_fig_save+'.png',dpi=200)

        f_rloss  = open(path_tf+'record_rloss_k.out', 'a+')
        mat_rloss = np.matrix(model.rloss_k)
        for line in mat_rloss:
            np.savetxt(f_rloss, line, fmt='%.8f')
        f_rloss.close()

        plt.close('all')
        num_lx = len(model.rloss_h)
        xm_ls = np.arange(0, num_print_opt * num_lx, num_print_opt)
        plt.semilogy(xm_ls,model.rloss_h)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('$L_h$', fontsize=14)
        path_fig_save = path_tf +'loss_h_'+str(type_op)+'_'+str(batchs)+'_'+str(num_epochs)
        plt.savefig(path_fig_save+'.png',dpi=200)

        f_rloss  = open(path_tf+'record_rloss_h.out', 'a+')
        mat_rloss = np.matrix(model.rloss_h)
        for line in mat_rloss:
            np.savetxt(f_rloss, line, fmt='%.8f')
        f_rloss.close()

        plt.close('all')
        num_lx = len(model.rloss_c)
        xm_ls = np.arange(0, num_print_opt * num_lx, num_print_opt)
        plt.semilogy(xm_ls,model.rloss_c)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('$L_c$', fontsize=14)
        path_fig_save = path_tf +'loss_c_'+str(type_op)+'_'+str(batchs)+'_'+str(num_epochs)
        plt.savefig(path_fig_save+'.png',dpi=200)
        f_rloss  = open(path_tf+'record_rloss_c.out', 'a+')
        mat_rloss = np.matrix(model.rloss_c)
        for line in mat_rloss:
            np.savetxt(f_rloss, line, fmt='%.8f')
        f_rloss.close()


        plt.close('all')
        num_lx = len(model.rloss_pde_f)
        xm_ls = np.arange(0, num_print_opt * num_lx, num_print_opt)
        plt.semilogy(xm_ls,model.rloss_pde_f)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('$L_f$', fontsize=14)
        path_fig_save = path_tf +'loss_f_'+str(type_op)+'_'+str(batchs)+'_'+str(num_epochs)
        plt.savefig(path_fig_save+'.png',dpi=200)   

        f_rloss  = open(path_tf+'record_rloss_pde_f.out', 'a+')
        mat_rloss = np.matrix(model.rloss_pde_f)
        for line in mat_rloss:
            np.savetxt(f_rloss, line, fmt='%.8f')
        f_rloss.close()

        if flag_pro != 2: # Only Darcy net not output this
            plt.close('all')
            num_lx = len(model.rloss_pde_fc)
            xm_ls = np.arange(0, num_print_opt * num_lx, num_print_opt)
            plt.semilogy(xm_ls,model.rloss_pde_fc)
            plt.xlabel('Iteration', fontsize=14)
            plt.ylabel('$L_{fc}$', fontsize=14)
            path_fig_save = path_tf +'loss_fc_'+str(type_op)+'_'+str(batchs)+'_'+str(num_epochs)
            plt.savefig(path_fig_save+'.png',dpi=200)     
            
            f_rloss  = open(path_tf+'record_rloss_pde_fc.out', 'a+')
            mat_rloss = np.matrix(model.rloss_pde_fc)
            for line in mat_rloss:
                np.savetxt(f_rloss, line, fmt='%.8f')
            f_rloss.close()                   

        if if_plot_rl2 == 1:
            plt.close('all')
            num_lx = len(model.rl2_k)
            xm_ls = np.arange(0, num_print_opt * num_lx, num_print_opt)
            plt.semilogy(xm_ls,model.rl2_k)
            # plt.ylim([0.001,5])
            plt.xlabel('Iteration', fontsize=14)
            plt.ylabel('Relative error $\epsilon^{K}$', fontsize=14)
            path_fig_save = path_tf +'rl2_k_'+str(type_op)+'_'+str(batchs)+'_'+str(num_epochs)
            plt.savefig(path_fig_save+'.png',dpi=200)

            f_rloss  = open(path_tf+'record_rl2_k.out', 'a+')
            mat_rloss = np.matrix(model.rl2_k)
            for line in mat_rloss:
                np.savetxt(f_rloss, line, fmt='%.8f')
            f_rloss.close()

            plt.close('all')
            num_lx = len(model.rl2_h)
            xm_ls = np.arange(0, num_print_opt * num_lx, num_print_opt)
            plt.semilogy(xm_ls,model.rl2_h)
            # plt.ylim([0.001,5])
            plt.xlabel('Iteration', fontsize=14)
            plt.ylabel('Relative error $\epsilon^{h}$', fontsize=14)
            path_fig_save = path_tf +'rl2_h_'+str(type_op)+'_'+str(batchs)+'_'+str(num_epochs)
            plt.savefig(path_fig_save+'.png',dpi=200)
            f_rloss  = open(path_tf+'record_rl2_h.out', 'a+')
            mat_rloss = np.matrix(model.rl2_h)
            for line in mat_rloss:
                np.savetxt(f_rloss, line, fmt='%.8f')
            f_rloss.close()

            plt.close('all')
            num_lx = len(model.rl2_c)
            xm_ls = np.arange(0, num_print_opt * num_lx, num_print_opt)
            plt.semilogy(xm_ls,model.rl2_c)
            # plt.ylim([0.001,5])
            plt.xlabel('Iteration', fontsize=14)
            plt.ylabel('Relative error $\epsilon^{C}$', fontsize=14)
            path_fig_save = path_tf +'rl2_c_'+str(type_op)+'_'+str(batchs)+'_'+str(num_epochs)
            plt.savefig(path_fig_save+'.png',dpi=200)
            f_rloss  = open(path_tf+'record_rl2_c.out', 'a+')
            mat_rloss = np.matrix(model.rl2_c)
            for line in mat_rloss:
                np.savetxt(f_rloss, line, fmt='%.8f')
            f_rloss.close()

        plt.close('all')

        #completely reset tensorflow
        tf.reset_default_graph()

        ######################################################################
        ############################# Plotting ###############################
        ######################################################################  
        
        if IF_plot == 1:
            nn = 200
            x = np.linspace(lb[0], ub[0], nn)
            y = np.linspace(lb[1], ub[1], nn)
            XX, YY = np.meshgrid(x,y)
            
            K_plot = griddata(X_star, k_pred.flatten(), (XX, YY), method='cubic')
            H_plot = griddata(X_star, h_pred.flatten(), (XX, YY), method='cubic')
            C_plot = griddata(X_star, c_pred.flatten(), (XX, YY), method='cubic')
            
            K_error = griddata(X_star, np.abs(k_star-k_pred).flatten(), (XX, YY), method='cubic')
            H_error = griddata(X_star, np.abs(h_star-h_pred).flatten(), (XX, YY), method='cubic')
            C_error = griddata(X_star, np.abs(C_star-c_pred).flatten(), (XX, YY), method='cubic')

            if flag_pro == 1 or flag_pro == 2 or flag_pro == 31 or flag_pro == 34 or flag_pro == 35:
                fig = plt.figure(1)
                plt.pcolor(XX, YY, K_plot, cmap='viridis')
                plt.plot(X_k[:,0], X_k[:,1], 'ko', markersize = 1.0)
                plt.clim(np.min(k_star), np.max(k_star))
                plt.jet()
                plt.colorbar()
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('$x_1$', fontsize=16)
                plt.ylabel('$x_2$', fontsize=16)
                plt.title('$K(x_1,x_2)$', fontsize=16)
                # plt.axes().set_aspect('equal')
                fig.tight_layout()
                plt.axis('equal')
                path_fig_save = path_fig+'map_k_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_pred'
                # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
                fig.savefig(path_fig_save+'.png',dpi=300)
                fig.clf()

                fig = plt.figure(2)
                plt.pcolor(XX, YY, K_error, cmap='viridis')
                plt.jet()
                plt.colorbar()
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('$x_1$', fontsize=16)
                plt.ylabel('$x_2$', fontsize=16)
                plt.title('Absolute error: $K$', fontsize=16) 
                fig.tight_layout()
                plt.axis('equal') # https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/axis_equal_demo.html
                path_fig_save = path_fig+'map_k_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_errors'
                # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
                fig.savefig(path_fig_save+'.png',dpi=300)
                fig.clf()

            if flag_pro == 1 or flag_pro == 2 or flag_pro == 32 or flag_pro == 34 or flag_pro == 35:
                fig = plt.figure(3)
                plt.pcolor(XX, YY, H_plot, cmap='viridis')
                plt.plot(X_h[:,0], X_h[:,1], 'ko', markersize = 1.0)
                plt.clim(np.min(h_star), np.max(h_star))
                plt.jet()
                plt.colorbar()
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('$x_1$', fontsize=16)
                plt.ylabel('$x_2$', fontsize=16)
                plt.title('$h(x_1,x_2)$', fontsize=16)
                fig.tight_layout()
                plt.axis('equal')
                path_fig_save = path_fig+'map_h_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_pred'
                # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
                fig.savefig(path_fig_save+'.png',dpi=300)
                fig.clf()

                fig = plt.figure(4)
                plt.pcolor(XX, YY, H_error, cmap='viridis')
                plt.jet()
                plt.colorbar()
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('$x_1$', fontsize=16)
                plt.ylabel('$x_2$', fontsize=16)
                plt.title('Absolute error: $h$', fontsize=16) 
                fig.tight_layout()
                plt.axis('equal')
                path_fig_save = path_fig+'map_h_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_errors'
                # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
                fig.savefig(path_fig_save+'.png',dpi=300)
                fig.clf()

            if flag_pro == 1 or flag_pro == 33 or flag_pro == 35:
                fig = plt.figure(5)
                plt.pcolor(XX, YY, C_plot, cmap='viridis')
                plt.plot(X_c[:,0], X_c[:,1], 'ko', markersize = 1.0)
                plt.clim(np.min(C_star), np.max(C_star))
                plt.jet()
                plt.colorbar()
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('$x_1$', fontsize=16)
                plt.ylabel('$x_2$', fontsize=16)
                plt.title('$C(x_1,x_2)$', fontsize=16)
                fig.tight_layout()
                plt.axis('equal')
                path_fig_save = path_fig+'map_c_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_pred'
                # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
                fig.savefig(path_fig_save+'.png',dpi=300)
                fig.clf()

                fig = plt.figure(6)
                plt.pcolor(XX, YY, C_error, cmap='viridis')
                plt.jet()
                plt.colorbar()
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('$x_1$', fontsize=16)
                plt.ylabel('$x_2$', fontsize=16)
                plt.title('Absolute error: $C$', fontsize=16) 
                fig.tight_layout()
                plt.axis('equal')
                path_fig_save = path_fig+'map_c_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_errors'
                # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
                fig.savefig(path_fig_save+'.png',dpi=300)
                fig.clf()

            # ------ New ------
            if flag_pro == 1 or flag_pro == 2:
                v1_pred, v2_pred = model.predict_v(X_star)
                V1_plot = griddata(X_star, v1_pred.flatten(), (XX, YY), method='cubic')
                V2_plot = griddata(X_star, v2_pred.flatten(), (XX, YY), method='cubic')

                fig = plt.figure(7)
                plt.pcolor(XX, YY, V1_plot, cmap='viridis')
                plt.plot(X_k[:,0], X_k[:,1], 'ko', markersize = 1.0)
                # plt.clim(np.min(k_star), np.max(k_star))
                plt.jet()
                plt.colorbar()
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('$x_1$', fontsize=16)
                plt.ylabel('$x_2$', fontsize=16)
                plt.title('$v_1(x_1,x_2)$', fontsize=16)
                fig.tight_layout()
                plt.axis('equal')
                path_fig_save = path_fig+'map_v1_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_pred'
                # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
                fig.savefig(path_fig_save+'.png',dpi=300)
                fig.clf()

                fig = plt.figure(8)
                plt.pcolor(XX, YY, V2_plot, cmap='viridis')
                plt.plot(X_k[:,0], X_k[:,1], 'ko', markersize = 1.0)
                # plt.clim(np.min(k_star), np.max(k_star))
                plt.jet()
                plt.colorbar()
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('$x_1$', fontsize=16)
                plt.ylabel('$x_2$', fontsize=16)
                plt.title('$v_2(x_1,x_2)$', fontsize=16)
                fig.tight_layout()
                plt.axis('equal')
                path_fig_save = path_fig+'map_v2_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_pred'
                # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
                fig.savefig(path_fig_save+'.png',dpi=300)
                fig.clf()

            if flag_pro == 1 or flag_pro == 2:
                f_pred = model.predict_f(X_star)
                f_plot = griddata(X_star, f_pred.flatten(), (XX, YY), method='cubic')

                fig = plt.figure(9)
                plt.pcolor(XX, YY, f_plot, cmap='viridis')
                plt.plot(X_f[:,0], X_f[:,1], 'ko', markersize = 1.0)
                # plt.clim(np.min(k_star), np.max(k_star))
                plt.jet()
                plt.colorbar()
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('$x_1$', fontsize=16)
                plt.ylabel('$x_2$', fontsize=16)
                plt.title('$f(x_1,x_2)$', fontsize=16)
                fig.tight_layout()
                plt.axis('equal')
                path_fig_save = path_fig+'map_f_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_pred'
                # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
                fig.savefig(path_fig_save+'.png',dpi=300)
                fig.clf()

            if flag_pro == 1:
                fc_pred = model.predict_fc(X_star)
                fc_plot = griddata(X_star, fc_pred.flatten(), (XX, YY), method='cubic')
                fc_pred_log = np.log10(np.absolute(fc_pred))
                fc_plot_log = griddata(X_star, fc_pred_log.flatten(), (XX, YY), method='cubic') 

                fig = plt.figure(10)
                plt.pcolor(XX, YY, fc_plot, cmap='viridis')
                plt.plot(X_fc[:,0], X_fc[:,1], 'ko', markersize = 1.0)
                # plt.clim(np.min(k_star), np.max(k_star))
                plt.jet()
                plt.colorbar()
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('$x_1$', fontsize=16)
                plt.ylabel('$x_2$', fontsize=16)
                plt.title('$f_C(x_1,x_2)$', fontsize=16)
                fig.tight_layout()
                plt.axis('equal')
                path_fig_save = path_fig+'map_fc_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_pred'
                # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
                fig.savefig(path_fig_save+'.png',dpi=300)
                fig.clf()

                fig = plt.figure(11)
                plt.pcolor(XX, YY, fc_plot_log, cmap='viridis', vmin=-2, vmax = 2)
                plt.plot(X_fc[:,0], X_fc[:,1], 'ko', markersize = 1.0)
                # plt.clim(np.min(k_star), np.max(k_star))
                plt.jet()
                plt.colorbar()
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('$x_1$', fontsize=16)
                plt.ylabel('$x_2$', fontsize=16)
                plt.title('$log(f_C(x_1,x_2))$', fontsize=16)
                fig.tight_layout()
                plt.axis('equal')
                path_fig_save = path_fig+'map_fclog_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_pred'
                # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
                fig.savefig(path_fig_save+'.png',dpi=300)
                fig.clf()

    print("\n",file=f)
    f.close()
    f1.close()
    f1_loss.close()
    f1_loss2.close()
    f2_weight.close()
    f2_bias.close()





