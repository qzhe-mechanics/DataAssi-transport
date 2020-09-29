# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:25:22 2019

@author: Qizhi He
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm # Instantly make your loops show a smart progress meter - just wrap any iterable with tqdm(iterable), and you're done!

def sub_normalization(X,Xl,Xr,scale_coe):
    len = Xr - Xl
    X_nor = 2.0 * scale_coe * (X - Xl)/len - scale_coe  # Mapped to [-scale_coe,scale_coe]
    return X_nor

def sub_batch(num_all, batch, type):
    if type == 21 or type == 22 or type == 23:
        if num_all < batch:
            num_batch = num_all
        else:
            num_batch = batch
    elif type == 2:
        if num_all < batch * 16:
            num_batch = 16
        else:
            num_batch = int(num_all/batchs)
    return num_batch

class model_Darcy:
    # Initialize the class
    def __init__(self, X_h, Y_h, X_k, Y_k, X_c, Y_c, X_f, Y_f, X_fc, Y_fc,
                 X_hbD, Y_hbD, X_hbN, Y_hbN, X_hbNs, Y_hbNs,
                 X_cbD, Y_cbD, X_cbN1, Y_cbN1, X_cbN2, Y_cbN2,
                 layers_h, layers_k, layers_c, lb, ub,
                 coe_phi,coe_Dw,coe_tau,coe_aL,coe_aT,
                 f,path_tf,f1_loss,f1_loss2,f2_weight,f2_bias,
                 para_k,para_h,para_kh,para_c,para_khc,
                 type_op,type_reg,coe_reg,rand_seed_tf,learn_rate,flag_solv,
                 flag_pro,if_plot_rl2,flag_lsty,conv_opt,
                 ref_xm = None,ref_k = None, ref_h = None, ref_c = None):
        
        ## v6s3 @ 2019.09.16
        self.flag_pro = flag_pro
        ## v6s4 @ 2019.10.02
        self.conv_opt = conv_opt
        ## @ 2019.09.25
        self.flag_lsty = flag_lsty

        if if_plot_rl2 == 0:  # v6s3 @ 2019.09.16
            self.if_record_rl2 = 0
        elif if_plot_rl2 == 1:
            self.if_record_rl2 = 1
            self.ref_xm = ref_xm
            self.ref_k = ref_k
            self.ref_h = ref_h
            self.ref_c = ref_c
        
        self.rl2_k = []
        self.rl2_h = []
        self.rl2_c = []        

        ## Record Input
        self.phi = coe_phi
        self.Dw  = coe_Dw
        self.tau = coe_tau
        self.aL  = coe_aL
        self.aT  = coe_aT

        self.lb = lb
        self.ub = ub

        # New
        self.para_k = para_k
        self.para_h = para_h
        self.para_kh = para_kh
        self.para_c = para_c
        self.para_khc = para_khc

        self.f  = f  
        self.f1_loss = f1_loss
        self.f1_loss2 = f1_loss2
        self.f2_weight = f2_weight
        self.f2_bias = f2_bias
        self.path_tf = path_tf

        self.type_reg = type_reg 

        #########################################################################
        ### Scaling & Normalization
        #########################################################################

        ## Scaling for input Coordinates and "Auto Grad"
        # ------------------------------------------------------------------
        scale_coe = 0.5   # a positive number denote the distance from center to one-end
        scale_X   = 2 * scale_coe / (ub-lb)

        self.scale_coe = scale_coe
        self.scale_X   = scale_X
      
        X_k    = sub_normalization(X_k,lb,ub,scale_coe)

        X_h    = sub_normalization(X_h,lb,ub,scale_coe)   
        X_f    = sub_normalization(X_f,lb,ub,scale_coe)   
        X_hbD  = sub_normalization(X_hbD,lb,ub,scale_coe)   
        X_hbN  = sub_normalization(X_hbN,lb,ub,scale_coe)   
        X_hbNs = sub_normalization(X_hbNs,lb,ub,scale_coe)  

        X_c    = sub_normalization(X_c,lb,ub,scale_coe)
        X_fc   = sub_normalization(X_fc,lb,ub,scale_coe)
        X_cbD  = sub_normalization(X_cbD,lb,ub,scale_coe)   
        X_cbN1 = sub_normalization(X_cbN1,lb,ub,scale_coe)   
        X_cbN2 = sub_normalization(X_cbN2,lb,ub,scale_coe)    
        # ------------------------------------------------------------------
        

        ## Normalization for NNs Output

        # For k
        coe_nor_k = max(Y_k.min(), Y_k.max(), key=abs)

        # For h
        coe_nor_hbD = max(Y_hbD.min(), Y_hbD.max(), key=abs)
        coe_nor_h = max(Y_h.min(), Y_h.max(), key=abs)
        if coe_nor_hbD > coe_nor_h:
            coe_nor_h = coe_nor_hbD

        # For c
        coe_nor_cbD = max(Y_cbD.min(), Y_cbD.max(), key=abs)
        coe_nor_c = max(Y_c.min(), Y_c.max(), key=abs)
        if coe_nor_cbD > coe_nor_c:
            coe_nor_c = coe_nor_cbD


        self.coe_nor_k = coe_nor_k
        self.coe_nor_h = coe_nor_h
        self.coe_nor_c = coe_nor_c

        print('k nor: {0:e}, h nor {1:e}, c nor {2:e}'.format(coe_nor_k,coe_nor_h,coe_nor_c),file=f)

        ## Standard Deviation
        Yl_k, Yr_k = Y_k.min(0), Y_k.max(0)
        Yk_std     = np.std(Y_k)
        if Yk_std <= 1e-2:
            print('Deviation of k {} is too small, reset to 1'.format(Yk_std))
            print('Deviation of k {} is too small, reset to 1'.format(Yk_std),file=f)
            Yk_std = 1.0
            
        self.Yk_std = Yk_std

        Yl_h, Yr_h = Y_h.min(0), Y_h.max(0)
        Yh_std     = np.std(Y_h)
        if Yh_std <= 1e-2:
            print('Deviation of h {} is too small, reset to 1'.format(Yh_std))
            print('Deviation of h {} is too small, reset to 1'.format(Yh_std),file=f)
            Yh_std = 1.0        
        self.Yh_std = Yh_std

        Yl_c, Yr_c = Y_c.min(0), Y_c.max(0)
        Yc_std     = np.std(Y_c)
        if Yc_std <= 1e-2:
            print('Deviation of C {} is too small, reset to 1'.format(Yc_std))
            print('Deviation of C {} is too small, reset to 1'.format(Yc_std),file=f)
            Yc_std = 1.0           
        
        self.Yc_std = Yc_std

        print('k std: {0:e}, h: std {1:e}, c: std {2:e}'.format(Yk_std,Yh_std,Yc_std),file=f)
        # ------------------------------------------------------------------
        
        ## Input self (with normalization)
        self.x1_k = X_k[:,0:1]
        self.x2_k = X_k[:,1:2]
        self.Y_k  = Y_k
        
        self.x1_h = X_h[:,0:1]
        self.x2_h = X_h[:,1:2]
        self.Y_h  = Y_h

        self.x1_c = X_c[:,0:1]
        self.x2_c = X_c[:,1:2]
        self.Y_c  = Y_c
        
        self.x1_f = X_f[:,0:1]
        self.x2_f = X_f[:,1:2]
        self.Y_f = Y_f
        self.X_f = X_f

        self.x1_fc = X_fc[:,0:1]
        self.x2_fc = X_fc[:,1:2]
        self.Y_fc  = Y_fc
        self.X_fc  = X_fc
        
        self.x1_hbD = X_hbD[:,0:1]
        self.x2_hbD = X_hbD[:,1:2]
        self.Y_hbD  = Y_hbD
        
        self.x1_hbN = X_hbN[:,0:1]
        self.x2_hbN = X_hbN[:,1:2]
        self.Y_hbN  = Y_hbN
        
        self.x1_hbNs = X_hbNs[:,0:1]
        self.x2_hbNs = X_hbNs[:,1:2]
        self.Y_hbNs  = Y_hbNs

        self.x1_cbD = X_cbD[:,0:1]
        self.x2_cbD = X_cbD[:,1:2]
        self.Y_cbD  = Y_cbD
        
        self.x1_cbN1 = X_cbN1[:,0:1]
        self.x2_cbN1 = X_cbN1[:,1:2]
        self.Y_cbN1  = Y_cbN1
        
        self.x1_cbN2 = X_cbN2[:,0:1]
        self.x2_cbN2 = X_cbN2[:,1:2]
        self.Y_cbN2  = Y_cbN2
        
        self.layers_k = layers_k
        self.layers_h = layers_h
        self.layers_c = layers_c

        #########################################################################
        ### Initialize network weights and biases  
        #########################################################################
        
        ## Define placeholders and computational graph
        # Domain
        self.x1_k_tf = tf.placeholder(tf.float32, shape=(None, self.x1_k.shape[1]))
        self.x2_k_tf = tf.placeholder(tf.float32, shape=(None, self.x2_k.shape[1]))
        self.Yk_tf   = tf.placeholder(tf.float32, shape=(None, self.Y_k.shape[1]))  
        
        self.x1_h_tf = tf.placeholder(tf.float32, shape=(None, self.x1_h.shape[1]))
        self.x2_h_tf = tf.placeholder(tf.float32, shape=(None, self.x2_h.shape[1]))        
        self.Yh_tf   = tf.placeholder(tf.float32, shape=(None, self.Y_h.shape[1]))
        
        self.x1_f_tf = tf.placeholder(tf.float32, shape=(None, self.x1_f.shape[1]))
        self.x2_f_tf = tf.placeholder(tf.float32, shape=(None, self.x2_f.shape[1]))        
        self.Yf_tf   = tf.placeholder(tf.float32, shape=(None, self.Y_f.shape[1]))

        self.x1_c_tf = tf.placeholder(tf.float32, shape=(None, self.x1_c.shape[1]))
        self.x2_c_tf = tf.placeholder(tf.float32, shape=(None, self.x2_c.shape[1]))        
        self.Yc_tf   = tf.placeholder(tf.float32, shape=(None, self.Y_c.shape[1]))

        self.x1_fc_tf = tf.placeholder(tf.float32, shape=(None, self.x1_fc.shape[1]))
        self.x2_fc_tf = tf.placeholder(tf.float32, shape=(None, self.x2_fc.shape[1]))        
        self.Yfc_tf   = tf.placeholder(tf.float32, shape=(None, self.Y_fc.shape[1]))
        
        # B.C.
        self.x1_hbD_tf = tf.placeholder(tf.float32, shape=(None, self.x1_hbD.shape[1]))
        self.x2_hbD_tf = tf.placeholder(tf.float32, shape=(None, self.x2_hbD.shape[1]))        
        self.YhbD_tf   = tf.placeholder(tf.float32, shape=(None, self.Y_hbD.shape[1]))
        
        self.x1_hbN_tf = tf.placeholder(tf.float32, shape=(None, self.x1_hbN.shape[1]))
        self.x2_hbN_tf = tf.placeholder(tf.float32, shape=(None, self.x2_hbN.shape[1]))        
        self.YhbN_tf   = tf.placeholder(tf.float32, shape=(None, self.Y_hbN.shape[1]))
        
        self.x1_hbNs_tf = tf.placeholder(tf.float32, shape=(None, self.x1_hbNs.shape[1]))
        self.x2_hbNs_tf = tf.placeholder(tf.float32, shape=(None, self.x2_hbNs.shape[1]))        
        self.YhbNs_tf   = tf.placeholder(tf.float32, shape=(None, self.Y_hbNs.shape[1]))
        
        self.x1_cbD_tf = tf.placeholder(tf.float32, shape=(None, self.x1_cbD.shape[1]))
        self.x2_cbD_tf = tf.placeholder(tf.float32, shape=(None, self.x2_cbD.shape[1]))        
        self.YcbD_tf   = tf.placeholder(tf.float32, shape=(None, self.Y_cbD.shape[1]))
        
        self.x1_cbN1_tf = tf.placeholder(tf.float32, shape=(None, self.x1_cbN1.shape[1]))
        self.x2_cbN1_tf = tf.placeholder(tf.float32, shape=(None, self.x2_cbN1.shape[1]))        
        self.YcbN1_tf   = tf.placeholder(tf.float32, shape=(None, self.Y_cbN1.shape[1]))
        
        self.x1_cbN2_tf = tf.placeholder(tf.float32, shape=(None, self.x1_cbN2.shape[1]))
        self.x2_cbN2_tf = tf.placeholder(tf.float32, shape=(None, self.x2_cbN2.shape[1]))        
        self.YcbN2_tf   = tf.placeholder(tf.float32, shape=(None, self.Y_cbN2.shape[1]))

        self.x1_v_tf = tf.placeholder(tf.float32, shape=(None, self.x1_h.shape[1]))
        self.x2_v_tf = tf.placeholder(tf.float32, shape=(None, self.x2_h.shape[1]))

        tf.set_random_seed(rand_seed_tf)
        self.weights_k, self.biases_k = self.initialize_NN_regu_k(layers_k)
        tf.set_random_seed(rand_seed_tf)
        self.weights_h, self.biases_h = self.initialize_NN_regu_h(layers_h)
        tf.set_random_seed(rand_seed_tf)
        self.weights_c, self.biases_c = self.initialize_NN_regu_c(layers_c)

        '''Add regularization terms'''
        if type_reg == 0:
            self.reg_term = 0.0
        elif type_reg == 1:
            regularizer = tf.contrib.layers.l1_regularizer(coe_reg)
            self.reg_term = tf.contrib.layers.apply_regularization(regularizer,weights_list=None)
        elif type_reg == 2:
            regularizer = tf.contrib.layers.l2_regularizer(coe_reg)
            self.reg_term = tf.contrib.layers.apply_regularization(regularizer,weights_list=None)
        
        '''Define Neural Net'''
        ## Evaluate prediction

        self.k_pred = self.net_k(self.x1_k_tf, self.x2_k_tf)
        self.h_pred = self.net_h(self.x1_h_tf, self.x2_h_tf)  
        self.c_pred = self.net_c(self.x1_c_tf, self.x2_c_tf)

        self.f_pred   = self.net_f(self.x1_f_tf, self.x2_f_tf)
        self.fc_pred  = self.net_fc(self.x1_fc_tf, self.x2_fc_tf)

        self.hbD_pred   = self.net_hbD(self.x1_hbD_tf, self.x2_hbD_tf) 
        self.hbN_pred   = self.net_hbN(self.x1_hbN_tf, self.x2_hbN_tf)
        self.hbNs_pred  = self.net_hbNs(self.x1_hbNs_tf, self.x2_hbNs_tf)

        self.cbD_pred  = self.net_cbD(self.x1_cbD_tf, self.x2_cbD_tf)
        self.cbN1_pred = self.net_cbN1(self.x1_cbN1_tf, self.x2_cbN1_tf)
        self.cbN2_pred = self.net_cbN2(self.x1_cbN2_tf, self.x2_cbN2_tf)

        # --- new ----
        self.v1_pred = self.net_v1(self.x1_v_tf, self.x2_v_tf)
        self.v2_pred = self.net_v2(self.x1_v_tf, self.x2_v_tf)

        ## Evaluate loss      

        self.coe_nor_k2 = 1/coe_nor_k**2
        self.coe_nor_h2 = 1/coe_nor_h**2
        self.coe_nor_c2 = 1/coe_nor_c**2

        self.coe_std_k2 = 1/Yk_std**2
        self.coe_std_h2 = 1/Yh_std**2
        self.coe_std_c2 = 1/Yc_std**2

        # Output each component of loss 

        self.loss_k      = tf.reduce_mean(tf.square(self.Yk_tf - self.k_pred))
        self.loss_h      = tf.reduce_mean(tf.square(self.Yh_tf - self.h_pred))
        self.loss_c      = tf.reduce_mean(tf.square(self.Yc_tf - self.c_pred))

        self.loss_pde_D  = tf.reduce_mean(tf.square(self.YhbD_tf - self.hbD_pred))

        self.loss_pde_f  = tf.reduce_mean(tf.square(self.Yf_tf - self.f_pred))
        self.loss_pde_N  = tf.reduce_mean(tf.square(self.YhbN_tf - self.hbN_pred))
        self.loss_pde_Ns = tf.reduce_mean(tf.square(self.YhbNs_tf - self.hbNs_pred))
        
        self.loss = self.loss_function() + self.reg_term

        self.loss_hx  = self.loss_h + self.loss_pde_D
        self.loss_fx  = self.coe_nor_k2 * (self.coe_nor_h2 * self.loss_pde_f + self.loss_pde_N + self.coe_nor_h2 * self.loss_pde_Ns)

        # Record loss
        self.rloss = []
        self.rloss_batch = []
        self.rloss_k = []
        self.rloss_h = []
        self.rloss_c = []

        self.rloss_pde_D = []
        self.rloss_pde_cD = []

        self.rloss_pde_f  = []
        self.rloss_pde_N  = []
        self.rloss_pde_Ns = []
        
        '''Define optimizer'''
        var_opt   = self.var_function()

        if type_op == 1: # L-BFGS-B
            var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_k")
            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_h")

            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                    var_list=var_opt,
                                                                    method  = 'L-BFGS-B', 
                                                                    options = {'maxiter': 50000,
                                                                              'maxfun': 50000,
                                                                              'maxcor': 50,
                                                                              'maxls': 50,
                                                                              'ftol' : 1.0 * np.finfo(float).eps})
            self.optimizer_h = tf.contrib.opt.ScipyOptimizerInterface( self.loss_hx,
                                                                    var_list=var_list2,
                                                                    method  = 'L-BFGS-B', 
                                                                    options = {'maxiter': 50000,
                                                                              'maxfun': 50000,
                                                                              'maxcor': 50,
                                                                              'maxls': 50,
                                                                              'ftol' : 1.0e-10})

        elif type_op == 2 or type_op == 21 or type_op == 22:
            # Adam
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss,var_list=var_opt) # Verified, useful.
        elif type_op == 7 or type_op == 71 or type_op == 72:

            if type_op == 7 or type_op == 72:
                self.optimizer_1 = tf.train.AdamOptimizer(learning_rate=learn_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss,var_list=var_opt)
            elif type_op == 71:
                global_step = tf.Variable(0, trainable=False)
                boundaries = [20000, 40000, 60000,100000]
                values = [learn_rate, 0.75*learn_rate, 0.5*learn_rate,0.25*learn_rate,0.2*learn_rate]
                decay_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries,values)
                self.optimizer_1 = tf.train.AdamOptimizer(learning_rate=decay_rate).minimize(self.loss,var_list=var_opt,global_step=global_step)
        

            self.optimizer_2 = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                    var_list=var_opt,
                                                                    method  = 'L-BFGS-B', 
                                                                    options = {'maxiter': 50000,
                                                                            'maxfun': 50000,
                                                                            'maxcor': 50,
                                                                            'maxls': 50,
                                                                            'ftol' : 1.0 * np.finfo(float).eps})                                                        


        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) #standard setting

        ## Initialize Tensorflow variables (Need intiaalization to activate the function Variable and Placeholder)
        init = tf.global_variables_initializer() # Returns an Op that initializes global variables.
        self.sess.run(init)    # Output tensors and metadata obtained when executing a session.


    ''' 
    ---------------------------------------------
    ## Initialize network weights and biases using Xavier initialization
    ---------------------------------------------
    '''
    def initialize_NN_regu_k(self, layers): 
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = np.sqrt(2./(in_dim + out_dim))
            return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)   
        
        with tf.variable_scope("var_k"):
            weights = []
            biases = []
            num_layers = len(layers)
            for l in range(0,num_layers-1):
                W = xavier_init(size=[layers[l], layers[l+1]])
                b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
                weights.append(W)
                biases.append(b) 
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)  

            tf.add_to_collection(tf.GraphKeys.WEIGHTS, W) 
        return weights, biases

    def initialize_NN_regu_h(self, layers): 
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = np.sqrt(2./(in_dim + out_dim))
            return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)   
        
        with tf.variable_scope("var_h"):
            weights = []
            biases = []
            num_layers = len(layers)
            for l in range(0,num_layers-1):
                W = xavier_init(size=[layers[l], layers[l+1]])
                b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
                weights.append(W)
                biases.append(b) 
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)  
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, W) 
        return weights, biases

    def initialize_NN_regu_c(self, layers): 
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = np.sqrt(2./(in_dim + out_dim))
            return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)   
        
        with tf.variable_scope("var_c"):
            weights = []
            biases = []
            num_layers = len(layers)
            for l in range(0,num_layers-1):
                W = xavier_init(size=[layers[l], layers[l+1]])
                b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
                weights.append(W)
                biases.append(b) 
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)  

            tf.add_to_collection(tf.GraphKeys.WEIGHTS, W) 
        return weights, biases


    ## Forward Pass      
    def forward_pass(self, H, layers, weights, biases):
        num_layers = len(layers)
        for l in range(0,num_layers-2): # Note: Not include num_layers-2. Given layers =5, here l = 0,1,2
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b)) # Note: Tanh map to [-1,1], the last layer without activation..

        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
    
    # Callback to print the loss at every optimization step
    def callback(self, loss,loss_k,loss_h,loss_f):
        num_print_opt = self.num_print_opt
        it = self.index_opt
        ########## record loss ############
        self.rloss.append(loss)
        ########## record loss ############

        if it % num_print_opt == 0:
            print('It={0:d}, Loss: {1:.3e}, Loss_k: {2:.3e}, Loss_h: {3: .3e}, Loss_f: {4:.3e}'.format(it,loss, loss_k, loss_h, loss_f))
            ########## record loss ############
            self.rloss_k.append(loss_k)
            self.rloss_h.append(loss_h)
            self.rloss_pde_f.append(loss_f)
            ########## record loss ############
        self.index_opt += 1

    def callback_fast(self, loss,loss_k,loss_h):
        num_print_opt = self.num_print_opt
        it = self.index_opt
        ########## record loss ############
        self.rloss.append(loss)
        ########## record loss ############

        if it % num_print_opt == 0:
            print('It={0:d}, Loss: {1:.3e}, Loss_k: {2:.3e}, Loss_h: {3: .3e}'.format(it,loss, loss_k, loss_h))
            ########## record loss ############
            self.rloss_k.append(loss_k)
            self.rloss_h.append(loss_h)
            ########## record loss ############
        self.index_opt += 1

    # Trains the model by minimizing the loss using L-BFGS
    def train(self,type_op,batchs,nIter,flag_solv,num_print_opt): 
        f = self.f
        f2_weight = self.f2_weight
        f2_bias = self.f2_bias
        path_tf = self.path_tf

        # Define a dictionary for associating placeholders with data
        tf_dict = {self.x1_k_tf: self.x1_k, self.x2_k_tf: self.x2_k, self.Yk_tf: self.Y_k,
                   self.x1_h_tf: self.x1_h, self.x2_h_tf: self.x2_h, self.Yh_tf: self.Y_h,
                   self.x1_c_tf: self.x1_c, self.x2_c_tf: self.x2_c, self.Yc_tf: self.Y_c,
                   self.x1_f_tf: self.x1_f, self.x2_f_tf: self.x2_f, self.Yf_tf: self.Y_f,
                   self.x1_fc_tf: self.x1_fc, self.x2_fc_tf: self.x2_fc, self.Yfc_tf: self.Y_fc,
                   self.x1_hbD_tf: self.x1_hbD, self.x2_hbD_tf: self.x2_hbD, self.YhbD_tf: self.Y_hbD,
                   self.x1_hbN_tf: self.x1_hbN, self.x2_hbN_tf: self.x2_hbN, self.YhbN_tf: self.Y_hbN,
                   self.x1_hbNs_tf: self.x1_hbNs, self.x2_hbNs_tf: self.x2_hbNs, self.YhbNs_tf: self.Y_hbNs,
                   self.x1_cbD_tf: self.x1_cbD, self.x2_cbD_tf: self.x2_cbD, self.YcbD_tf: self.Y_cbD,
                   self.x1_cbN1_tf: self.x1_cbN1, self.x2_cbN1_tf: self.x2_cbN1, self.YcbN1_tf: self.Y_cbN1,
                   self.x1_cbN2_tf: self.x1_cbN2, self.x2_cbN2_tf: self.x2_cbN2, self.YcbN2_tf: self.Y_cbN2}
        
        if type_op == 1:
            # Call SciPy's L-BFGS otpimizer
            self.index_opt = 1
            self.num_print_opt = num_print_opt            

            if flag_solv == 'standard':
                self.optimizer.minimize(self.sess, 
                                        feed_dict = tf_dict,         
                                        fetches = [self.loss,self.loss_k,self.loss_h], 
                                        loss_callback = self.callback_fast)
                loss_value,loss_k,loss_h,loss_f = self.sess.run([self.loss,self.loss_k,self.loss_h,self.loss_pde_f],feed_dict=tf_dict)
                print('train BFGS, epoch: {}, loss: {}, loss_k: {}, loss_h: {}, loss_f: {}'.\
                    format(self.index_opt,loss_value,loss_k,loss_h,loss_f),file=f)
                error_k, error_h, _ = self.error_test_l2(self.ref_xm)
                print('*** rl2_k: {0:.3e}, rl2_h: {1:.3e}'.format(error_k,error_h),file=f)

            elif flag_solv == 'sequent':
                idex = 1
                self.optimizer_h.minimize(self.sess, 
                                        feed_dict = tf_dict,         
                                        fetches = [self.loss,self.loss_k,self.loss_h], 
                                        loss_callback = self.callback_fast)

                loss_end, loss_end_loss_k, loss_end_loss_h = self.sess.run([self.loss,self.loss_k,self.loss_h],feed_dict=tf_dict)
                print('train: #{}, loss: {}, loss_k: {}, loss_h: {}'.\
                    format(idex,loss_end,loss_end_loss_k,loss_end_loss_h),file=f)
                idex += 1

                '''Train loss'''
                self.optimizer.minimize(self.sess, 
                           feed_dict = tf_dict,         
                           fetches = [self.loss,self.loss_k,self.loss_h],
                           loss_callback = self.callback_fast)

        elif type_op == 7 or type_op == 71 or type_op == 72:
            self.num_print_opt = num_print_opt
            if batchs == 0: # Vanilla
                # Run Adam optimizer
                start_time = time.time()

                for it in range(nIter):  # If nIter = 0, not run this sess
                    _, loss_value = self.sess.run([self.optimizer_1, self.loss], tf_dict)
                          
                    ########## record loss ############
                    self.rloss.append(loss_value)
                    ########## record loss ############

                    # Print
                    if it % num_print_opt == 0:
                        loss_loss_k,loss_loss_h,loss_loss_c,loss_pde_f = self.sess.run([self.loss_k,self.loss_h,self.loss_c,self.loss_pde_f],feed_dict=tf_dict)
                        
                        elapsed = time.time() - start_time
                        print('It: {0: d}, Time: {1: .2f} , Loss: {2: .3e}, Loss_k: {3: .3e}, Loss_h: {4: .3e}, Loss_f: {5: .3e}'.format(it, elapsed, loss_value, loss_loss_k, loss_loss_h, loss_pde_f)) 
                
                        ########## record loss ############
                        self.rloss_k.append(loss_loss_k)
                        self.rloss_h.append(loss_loss_h)
                        self.rloss_c.append(loss_loss_c)
                        self.rloss_pde_f.append(loss_pde_f)
                        ########## record loss ############

                        if self.if_record_rl2 == 1:
                            error_k, error_h, error_c = self.error_test_l2(self.ref_xm)
                            print('** rl2_k: {0:.3e}, rl2_h: {1:.3e}, rl2_c: {2: .3e}'.format(error_k,error_h,error_c))

                            self.rl2_k.append(error_k)
                            self.rl2_h.append(error_h)
                            self.rl2_c.append(error_c)

                        start_time = time.time()

                    # if loss_value < 0.001:
                    if loss_value < self.conv_opt:
                        break  

            loss_value, loss_k, loss_h, loss_f = self.sess.run([self.loss,self.loss_k, self.loss_h,self.loss_pde_f],feed_dict=tf_dict)
            print('train Adam, epoch: {}, loss: {}, loss_k: {}, loss_h: {}, loss_f: {}'.\
                format(it,loss_value,loss_k,loss_h,loss_f),file=f)
            error_k, error_h, _ = self.error_test_l2(self.ref_xm)
            print('** rl2_k: {0:.3e}, rl2_h: {1:.3e}'.format(error_k,error_h),file=f)    

            '''BFGS'''
            self.index_opt = 1
            self.optimizer_2.minimize(self.sess, 
                       feed_dict = tf_dict,         
                       fetches = [self.loss,self.loss_k,self.loss_h],
                       loss_callback = self.callback_fast)
            
            loss_value, loss_k,loss_h,loss_f = self.sess.run([self.loss,self.loss_k,self.loss_h,self.loss_pde_f],feed_dict=tf_dict)
            print('train BFGS, epoch: {}, loss: {}, loss_k: {}, loss_h: {}, loss_f: {}'.\
                format(self.index_opt,loss_value,loss_k,loss_h,loss_f),file=f)

        elif type_op == 2 or type_op == 21 or type_op == 22:
            if batchs == 0: # Vanilla
                # Run Adam optimizer
                start_time = time.time()

                # for it in range(nIter):  # If nIter = 0, not run this sess
                for it in tqdm(range(nIter)):  # If nIter = 0, not run this sess
                    _, loss_value = self.sess.run([self.optimizer, self.loss], tf_dict)
                          
                    ########## record loss ############
                    self.rloss.append(loss_value)
                    ########## record loss ############

                    # Print
                    if it % num_print_opt == 0:
                        loss_loss_k,loss_loss_h,loss_loss_c,loss_pde_f = self.sess.run([self.loss_k,self.loss_h,self.loss_c,self.loss_pde_f],feed_dict=tf_dict)
                        
                        elapsed = time.time() - start_time
                        print('It: {0: d}, Time: {1: .2f} , Loss: {2: .3e}, Loss_k: {3: .3e}, Loss_h: {4: .3e}, Loss_f: {5: .3e}'.format(it, elapsed, loss_value, loss_loss_k, loss_loss_h, loss_pde_f)) 
                
                        ########## record loss ############
                        self.rloss_k.append(loss_loss_k)
                        self.rloss_h.append(loss_loss_h)
                        self.rloss_c.append(loss_loss_c)
                        self.rloss_pde_f.append(loss_pde_f)
                        ########## record loss ############

                        if self.if_record_rl2 == 1:
                            error_k, error_h, error_c = self.error_test_l2(self.ref_xm)
                            print('** rl2_k: {0:.3e}, rl2_h: {1:.3e}, rl2_c: {2: .3e}'.format(error_k,error_h,error_c))

                            self.rl2_k.append(error_k)
                            self.rl2_h.append(error_h)
                            self.rl2_c.append(error_c)

                        start_time = time.time()

                    if loss_value < self.conv_opt:
                        break

            elif batchs != 0:   # The minibatch for Adam is complicated since the different size for input                
                
                if type_op == 2 or type_op == 21:                    
                    batch_size_k = sub_batch(self.x1_k.shape[0], batchs, type_op)
                    batch_size_h = sub_batch(self.x1_h.shape[0], batchs, type_op)
                    batch_size_hbD = sub_batch(self.x1_hbD.shape[0], batchs, type_op)

                elif type_op == 22: # Keep all measurement data
                    batch_size_k = self.x1_k.shape[0]
                    batch_size_h = self.x1_h.shape[0]
                    batch_size_hbD = self.x1_hbD.shape[0]


                batch_size_f = sub_batch(self.x1_f.shape[0], batchs, type_op)
                batch_size_hbN = sub_batch(self.x1_hbN.shape[0], batchs, type_op) # two edge...
                batch_size_hbNs = sub_batch(self.x1_hbNs.shape[0], batchs, type_op) # one edge..

                # Run Adam optimizer
                start_time = time.time()
                for it in range(nIter):  # If nIter = 0, not run this sess

                    randidx_k = np.random.randint(int(self.x1_k.shape[0]), size=batch_size_k)
                    randidx_h = np.random.randint(int(self.x1_h.shape[0]), size=batch_size_h)
                    randidx_f = np.random.randint(int(self.x1_f.shape[0]), size=batch_size_f)

                    randidx_hbD = np.random.randint(int(self.x1_hbD.shape[0]), size=batch_size_hbD)
                    randidx_hbN = np.random.randint(int(self.x1_hbN.shape[0]), size=batch_size_hbN)
                    randidx_hbNs = np.random.randint(int(self.x1_hbNs.shape[0]), size=batch_size_hbNs)

                    tf_dict_batch = {self.x1_k_tf: self.x1_k[randidx_k,:], self.x2_k_tf: self.x2_k[randidx_k,:], self.Yk_tf: self.Y_k[randidx_k,:],
                               self.x1_h_tf: self.x1_h[randidx_h,:], self.x2_h_tf: self.x2_h[randidx_h,:], self.Yh_tf: self.Y_h[randidx_h,:],
                               self.x1_f_tf: self.x1_f[randidx_f,:], self.x2_f_tf: self.x2_f[randidx_f,:], self.Yf_tf: self.Y_f[randidx_f,:],
                               self.x1_hbD_tf: self.x1_hbD[randidx_hbD,:], self.x2_hbD_tf: self.x2_hbD[randidx_hbD,:], self.YhbD_tf: self.Y_hbD[randidx_hbD,:],
                               self.x1_hbN_tf: self.x1_hbN[randidx_hbN,:], self.x2_hbN_tf: self.x2_hbN[randidx_hbN,:], self.YhbN_tf: self.Y_hbN[randidx_hbN,:],
                               self.x1_hbNs_tf: self.x1_hbNs[randidx_hbNs,:], self.x2_hbNs_tf: self.x2_hbNs[randidx_hbNs,:], self.YhbNs_tf: self.Y_hbNs[randidx_hbNs,:]}
                    
                    _, loss_batch = self.sess.run([self.optimizer, self.loss], tf_dict_batch)

                    ########## record loss ############
                    # self.rloss.append(loss_full)
                    self.rloss_batch.append(loss_batch) # batch...
                    ########## record loss ############
                    
                    # Print
                    if it % num_print_opt == 0:
                        loss_k_bch, loss_h_bch, loss_fx_bch = self.sess.run([self.loss_k, self.loss_h, self.loss_fx], tf_dict_batch)
                        elapsed = time.time() - start_time
                        
                        # Mini-batch
                        print('It: {0: d}, Time: {1: .2f} , Loss: {2: .3e}, Loss_k: {3: .3e}, Loss_h: {4: .3e}, Loss_fx: {5: .3e}'.format(it, elapsed, loss_batch, loss_k_bch, loss_h_bch, loss_fx_bch)) 

                        # Full-batch
                        loss_value, loss_k, loss_h, loss_fx = self.sess.run([self.loss,self.loss_k,self.loss_h, self.loss_fx],feed_dict=tf_dict)
                        print('* Full Batch - Loss: {0: .3e}, Loss_k: {1: .3e}, Loss_h: {2: .3e}, Loss_fx: {3: .3e}'.format(loss_value, loss_k, loss_h, loss_fx)) 
                        
                        ########## record loss ############
                        self.rloss_k.append(loss_k)
                        self.rloss_h.append(loss_h)
                        self.rloss_pde_f.append(loss_fx)
                        self.rloss.append(loss_value)
                        # self.rloss_pde_f.append(loss_f)
                        # self.rloss_pde_fc.append(loss_fc)
                        ########## record loss ############ 

                        if self.if_record_rl2 == 1:
                            error_k, error_h, error_c = self.error_test_l2(self.ref_xm)
                            print('** rl2_k: {0:.3e}, rl2_h: {1:.3e}, rl2_c: {2: .3e}'.format(error_k,error_h,error_c))

                            self.rl2_k.append(error_k)
                            self.rl2_h.append(error_h)
                            self.rl2_c.append(error_c)

                        start_time = time.time()

                    if loss_value < self.conv_opt:
                        break

            loss_value, loss_k, loss_h, loss_f = self.sess.run([self.loss,self.loss_k, self.loss_h,self.loss_pde_f],feed_dict=tf_dict)
            print('train Adam, epoch: {}, loss: {}, loss_k: {}, loss_h: {}, loss_f: {}'.\
                format(it,loss_value,loss_k,loss_h,loss_f),file=f)

        print ("End of learning process")  
        loss_end = self.sess.run(self.loss,feed_dict=tf_dict)
        print('loss: {0: e}'.format(loss_end),file=f)
        if self.type_reg != 0:
            loss_reg = self.sess.run(self.reg_term)
            print('loss_reg: {0: e}'.format(loss_reg),file=f)

    def out_loss_components(self):
        f = self.f
        f1_loss = self.f1_loss
        f1_loss2 = self.f1_loss2

        # Define a dictionary for associating placeholders with data
        tf_dict = {self.x1_k_tf: self.x1_k, self.x2_k_tf: self.x2_k, self.Yk_tf: self.Y_k,
                   self.x1_h_tf: self.x1_h, self.x2_h_tf: self.x2_h, self.Yh_tf: self.Y_h,
                   self.x1_c_tf: self.x1_c, self.x2_c_tf: self.x2_c, self.Yc_tf: self.Y_c,
                   self.x1_f_tf: self.x1_f, self.x2_f_tf: self.x2_f, self.Yf_tf: self.Y_f,
                   self.x1_fc_tf: self.x1_fc, self.x2_fc_tf: self.x2_fc, self.Yfc_tf: self.Y_fc,
                   self.x1_hbD_tf: self.x1_hbD, self.x2_hbD_tf: self.x2_hbD, self.YhbD_tf: self.Y_hbD,
                   self.x1_hbN_tf: self.x1_hbN, self.x2_hbN_tf: self.x2_hbN, self.YhbN_tf: self.Y_hbN,
                   self.x1_hbNs_tf: self.x1_hbNs, self.x2_hbNs_tf: self.x2_hbNs, self.YhbNs_tf: self.Y_hbNs,
                   self.x1_cbD_tf: self.x1_cbD, self.x2_cbD_tf: self.x2_cbD, self.YcbD_tf: self.Y_cbD,
                   self.x1_cbN1_tf: self.x1_cbN1, self.x2_cbN1_tf: self.x2_cbN1, self.YcbN1_tf: self.Y_cbN1,
                   self.x1_cbN2_tf: self.x1_cbN2, self.x2_cbN2_tf: self.x2_cbN2, self.YcbN2_tf: self.Y_cbN2}

        loss_end_loss_k = self.sess.run(self.loss_k,feed_dict=tf_dict)
        loss_end_loss_h = self.sess.run(self.loss_h,feed_dict=tf_dict)
        loss_end_loss_f = self.sess.run(self.loss_pde_f,feed_dict=tf_dict)
        loss_end_loss_D = self.sess.run(self.loss_pde_D,feed_dict=tf_dict)
        loss_end_loss_N = self.sess.run(self.loss_pde_N,feed_dict=tf_dict)
        loss_end_loss_Ns = self.sess.run(self.loss_pde_Ns,feed_dict=tf_dict)

        loss_end_loss_c   = self.sess.run(self.loss_c,feed_dict=tf_dict)


        print('Loss_k: {0: .3e}, Loss_h: {1: .3e}, Loss_f: {2: .3e}, Loss_D: {3: .3e}, Loss_N: {4: .3e}, Loss_Ns: {5: .3e}'\
            .format(loss_end_loss_k,loss_end_loss_h,loss_end_loss_f,loss_end_loss_D,loss_end_loss_N,loss_end_loss_Ns),file=f)
                     
        print('{0:.3e} {1:.3e} {2:.3e} {3:.3e} {4:.3e} {5:.3e}'\
            .format(loss_end_loss_k, loss_end_loss_h, loss_end_loss_f,loss_end_loss_D,loss_end_loss_N,loss_end_loss_Ns),file=f1_loss)
        

    # Forward pass for k
    def net_k(self, x1, x2):
        k = self.forward_pass(tf.concat([x1, x2], 1),  # tf.concat(,0 or 1) 0: row-wise; 1: column-wise
                              self.layers_k,
                              self.weights_k, 
                              self.biases_k)
        return k
    
    # Forward pass for h
    def net_h(self, x1, x2):
        h = self.forward_pass(tf.concat([x1, x2], 1),
                              self.layers_h,
                              self.weights_h, 
                              self.biases_h)
        return h

    # Forward pass for c
    def net_c(self, x1, x2):
        c = self.forward_pass(tf.concat([x1, x2], 1),
                              self.layers_c,
                              self.weights_c, 
                              self.biases_c)
        return c

    # Forward pass for f (s3) # Better results in sequential learning.
    def net_f(self, x1, x2):
        k = self.net_k(x1, x2)
        h = self.net_h(x1, x2)
        beta = self.scale_X

        h_x1 = tf.gradients(h, x1)[0] * beta[0]
        h_x2 = tf.gradients(h, x2)[0] * beta[1]
        f_1 = tf.gradients(k * h_x1, x1)[0] * beta[0]
        f_2 = tf.gradients(k * h_x2, x2)[0] * beta[1]
        f = f_1 + f_2
        return f

    # Forward pass for fc
    def net_fc(self, x1, x2):
        Dw  = self.Dw
        tau = self.tau
        aL  = self.aL
        aT  = self.aT
        phi = self.phi
        beta = self.scale_X

        k  = self.net_k(x1, x2)
        h  = self.net_h(x1, x2)
        c  = self.net_c(x1, x2)

        # Use Darcy velocity
        v1  = self.net_v1(x1, x2)
        v2  = self.net_v2(x1, x2)

        c_x1 = tf.gradients(c, x1)[0] * beta[0]
        c_x2 = tf.gradients(c, x2)[0] * beta[1]

        ### LHS
        fc_L = v1*c_x1 + v2*c_x2
        
        ### RHS
        v_l2  = tf.sqrt(v1**2+v2**2)
        
        ## ---------------------------------   
        coe_L = phi*Dw*tau + aL * v_l2
        coe_T = phi*Dw*tau + aT * v_l2  
        f_1 = tf.gradients(coe_L*c_x1, x1)[0] * beta[0] 
        f_2 = tf.gradients(coe_T*c_x2, x2)[0] * beta[1]
        fc_R = f_1 + f_2
        ## ---------------------------------   
        

        fc = fc_L - fc_R

        return fc

    def net_hbD(self, x1, x2):
        h = self.net_h(x1, x2)
        return h  
    
    def net_hbN(self, x1, x2):
        h = self.net_h(x1, x2)
        beta = self.scale_X
        g_2 = tf.gradients(h, x2)[0] * beta[1]
        return g_2
    
    def net_hbNs(self, x1, x2):
        h = self.net_h(x1, x2)
        k = self.net_k(x1, x2)
        beta = self.scale_X
        g_1 = - k * tf.gradients(h, x1)[0] * beta[0]
        return g_1    

    def net_cbD(self, x1, x2):
        c = self.net_c(x1, x2)
        return c 
    
    def net_cbN1(self, x1, x2):
        beta = self.scale_X
        c = self.net_c(x1, x2)
        g_1 = tf.gradients(c, x1)[0] * beta[0]
        return g_1
    
    def net_cbN2(self, x1, x2):
        beta = self.scale_X
        c = self.net_c(x1, x2)
        g_2 = tf.gradients(c, x2)[0] * beta[1]
        return g_2

    ## Evaluates predictions at test points           
    def predict_k(self, X_star):
        # Center around the origin
        # X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Normalization
        lb = self.lb
        ub = self.ub
        scale_coe = self.scale_coe
        X_star = sub_normalization(X_star,lb,ub,scale_coe)   
        # Predict
        tf_dict = {self.x1_k_tf: X_star[:,0:1], self.x2_k_tf: X_star[:,1:2]}    
        k_star = self.sess.run(self.k_pred, tf_dict) 
        return k_star
    
    # Evaluates predictions at test points           
    def predict_h(self, X_star): 
        # Center around the origin
        # X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        lb = self.lb
        ub = self.ub
        scale_coe = self.scale_coe
        X_star = sub_normalization(X_star,lb,ub,scale_coe)   
        # Predict
        tf_dict = {self.x1_h_tf: X_star[:,0:1], self.x2_h_tf: X_star[:,1:2]}    
        h_star = self.sess.run(self.h_pred, tf_dict) 
        return h_star

    def predict_c(self, X_star): 
        # Center around the origin
        lb = self.lb
        ub = self.ub
        scale_coe = self.scale_coe
        X_star = sub_normalization(X_star,lb,ub,scale_coe)  

        # Predict
        tf_dict = {self.x1_c_tf: X_star[:,0:1], self.x2_c_tf: X_star[:,1:2]}    
        c_star = self.sess.run(self.c_pred, tf_dict) 
        return c_star
    
    # Evaluates predictions at test points           
    def predict_f(self, X_star): 
        # Center around the origin
        # X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        lb = self.lb
        ub = self.ub
        scale_coe = self.scale_coe
        X_star = sub_normalization(X_star,lb,ub,scale_coe)   
        # Predict
        tf_dict = {self.x1_f_tf: X_star[:,0:1], self.x2_f_tf: X_star[:,1:2]}    
        f_star = self.sess.run(self.f_pred, tf_dict) 
        return f_star  

    # Evaluates predictions at test points           
    def predict_fc(self, X_star): 
        lb = self.lb
        ub = self.ub
        scale_coe = self.scale_coe
        X_star = sub_normalization(X_star,lb,ub,scale_coe)
        # Predict
        tf_dict = {self.x1_fc_tf: X_star[:,0:1], self.x2_fc_tf: X_star[:,1:2]}    
        fc_star = self.sess.run(self.fc_pred, tf_dict) 
        return fc_star  

# New ------------------
    def net_v1(self, x1, x2):
        # phi = self.phi
        beta = self.scale_X

        k  = self.net_k(x1, x2)
        h  = self.net_h(x1, x2)
        v1 = -k*tf.gradients(h, x1)[0] * beta[0]
        return v1

    def net_v2(self, x1, x2):
        # phi = self.phi
        beta = self.scale_X
        k  = self.net_k(x1, x2)
        h  = self.net_h(x1, x2)
        v2 = -k*tf.gradients(h, x2)[0] * beta[1]
        return v2

    # New
    def predict_v(self, X_star): 
        # Center around the origin
        lb = self.lb
        ub = self.ub
        scale_coe = self.scale_coe
        X_star = sub_normalization(X_star,lb,ub,scale_coe)  

        # Predict
        tf_dict = {self.x1_v_tf: X_star[:,0:1], self.x2_v_tf: X_star[:,1:2]}    
        v1_star, v2_star = self.sess.run([self.v1_pred, self.v2_pred], tf_dict) 
        return v1_star, v2_star
    
    def error_test_l2(self,X_star):
        k_star = self.ref_k
        h_star = self.ref_h
        C_star = self.ref_c

        k_pred = self.predict_k(X_star)
        h_pred = self.predict_h(X_star)
        c_pred = self.predict_c(X_star)

        # Relative L2 error
        error_k = np.linalg.norm(k_star - k_pred, 2)/np.linalg.norm(k_star, 2)
        error_h = np.linalg.norm(h_star - h_pred, 2)/np.linalg.norm(h_star, 2)
        error_c = np.linalg.norm(C_star - c_pred, 2)/np.linalg.norm(C_star, 2)

        return error_k, error_h, error_c