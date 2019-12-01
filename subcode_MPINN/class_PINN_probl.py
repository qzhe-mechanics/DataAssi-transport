"""
Created on 6/10/2019

@author: Qizhi He @ PNNL

Note: Consider Regression, Darcy, and Advection-Dispersion problem for Multiphysics subground flow problem
"""

import tensorflow as tf
from class_tf_models_Reg      import *
from class_tf_models_Darcy    import *
from class_tf_models_CAD      import *

##################################################
### Regression
##################################################
class PINN_Reg(model_Reg):
    def loss_function(self):

        # Record scaling factors for loss
        para_k = self.para_k
        para_h = self.para_h
        para_c = self.para_c

        coe_std_k2 = self.coe_std_k2
        coe_std_h2 = self.coe_std_h2
        coe_std_c2 = self.coe_std_c2  

        coe_nor_k2 = self.coe_nor_k2
        coe_nor_h2 = self.coe_nor_h2
        coe_nor_c2 = self.coe_nor_c2

        if self.flag_pro == 31:
            res = self.loss_k
        elif self.flag_pro == 32:
            res = self.loss_h
        elif self.flag_pro == 33:
            res = self.loss_c
        elif self.flag_pro == 34:
            # loss_0907_type0
            res = para_k * self.loss_k + para_h * self.loss_h
        elif self.flag_pro == 35:
            if self.flag_lsty == 1: # Default
                # loss_0909_type0 (the best)
                res = para_k * self.loss_k + para_h * self.loss_h + para_c  * self.loss_c
            elif self.flag_lsty == 15:
                # sca1 (with only coe_nor: default in v6s2)
                res = para_k * self.loss_k + \
                      para_h * (coe_nor_h2 / coe_nor_k2) * self.loss_h + \
                      para_c  * (coe_nor_c2 / coe_nor_k2) * self.loss_c
        return res

    def var_function(self):
        if self.flag_pro == 31:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_k")
        elif self.flag_pro == 32:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_h")
        elif self.flag_pro == 33:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_c")
        elif self.flag_pro == 34:
            var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_k")
            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_h")
            var_list = var_list1 + var_list2
        elif self.flag_pro == 35:
            var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_k")
            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_h")
            var_list3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_c")
            var_list = var_list1 + var_list2 + var_list3
        return var_list

##################################################
### Darcy (k,h,f)
##################################################
class PINN_Darcy(model_Darcy):
  def var_function(self):
    var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_k")
    var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_h")
    var_list = var_list1 + var_list2
    # var_list3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_c")
    # var_list = var_list1 + var_list2 + var_list3
    return var_list

  def loss_function(self):
    para_k = self.para_k
    para_h = self.para_h
    para_c = self.para_c
    para_kh = self.para_kh
    para_khc = self.para_khc

    coe_std_k2 = self.coe_std_k2
    coe_std_h2 = self.coe_std_h2
    coe_std_c2 = self.coe_std_c2  

    coe_nor_k2 = self.coe_nor_k2
    coe_nor_h2 = self.coe_nor_h2
    coe_nor_c2 = self.coe_nor_c2

    if self.flag_lsty == 1: # Default
        # sca1 (with only coe_nor: default in v6s2)
        res = para_k * self.loss_k + \
            para_h * (self.loss_h + self.loss_pde_D) + \
            para_kh * coe_nor_k2 * (coe_nor_h2 * self.loss_pde_f + self.loss_pde_N + coe_nor_h2 * self.loss_pde_Ns)
    elif self.flag_lsty == 15:
        # sca1 (with only coe_nor: default in v6s2)
        res = para_k * self.loss_k + \
            para_h * (coe_nor_h2 / coe_nor_k2) * (self.loss_h + self.loss_pde_D) + \
            para_kh * coe_nor_k2 * (coe_nor_h2 * self.loss_pde_f + self.loss_pde_N + coe_nor_h2 * self.loss_pde_Ns)
    return res


##################################################
### CAD
##################################################
class PINN_CAD_seq(model_CAD_seq):
    def var_function(self):
        var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_k")
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_h")
        var_list3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "var_c")
        var_list = var_list1 + var_list2 + var_list3
        return var_list

    def loss_function(self):
        para_k = self.para_k
        para_h = self.para_h
        para_c = self.para_c
        para_kh = self.para_kh
        para_khc = self.para_khc

        coe_std_k2 = self.coe_std_k2
        coe_std_h2 = self.coe_std_h2
        coe_std_c2 = self.coe_std_c2  

        coe_nor_k2 = self.coe_nor_k2
        coe_nor_h2 = self.coe_nor_h2
        coe_nor_c2 = self.coe_nor_c2
    
        if self.flag_lsty == 1: # Default
            # sca1
            res = para_k * self.loss_k + \
                para_h * (self.loss_h + self.loss_pde_D) + \
                para_kh * coe_nor_k2 * (coe_nor_h2 * self.loss_pde_f + self.loss_pde_N + coe_nor_h2 * self.loss_pde_Ns) + \
                para_c  * (self.loss_c + self.loss_pde_cD) + \
                para_khc * coe_nor_c2 * (coe_nor_k2 * coe_nor_h2 * self.loss_pde_fc + self.loss_pde_cN1 + self.loss_pde_cN2)
        elif self.flag_lsty == 15:
            # sca5
            res = para_k * self.loss_k + \
                para_h * (coe_nor_h2 / coe_nor_k2) * (self.loss_h + self.loss_pde_D) + \
                para_kh * coe_nor_k2 * (coe_nor_h2 * self.loss_pde_f + self.loss_pde_N + coe_nor_h2 * self.loss_pde_Ns) + \
                para_c  * (coe_nor_c2 / coe_nor_k2) * (self.loss_c + self.loss_pde_cD) + \
                para_khc * coe_nor_c2 * (coe_nor_k2 * coe_nor_h2 * self.loss_pde_fc + self.loss_pde_cN1 + self.loss_pde_cN2)                    
        return res

    def loss_function_khr(self):
        para_k = self.para_k
        para_h = self.para_h
        para_kh = self.para_kh

        coe_std_k2 = self.coe_std_k2
        coe_std_h2 = self.coe_std_h2

        coe_nor_k2 = self.coe_nor_k2
        coe_nor_h2 = self.coe_nor_h2
    
        # sca1 (with only coe_nor)
        res  = para_k * self.loss_k + \
                    para_h * (self.loss_h + self.loss_pde_D) + \
                    para_kh * coe_nor_k2 * (coe_nor_h2 * self.loss_pde_f + self.loss_pde_N + coe_nor_h2 * self.loss_pde_Ns)
    
        return res


def sub_NN_type(type_NN):
    if type_NN == 1:
        layers = [2,50,50,1]     # (~2500)
    elif type_NN == 11:
        layers = [2,40,40,1]     # (~1600)
    elif type_NN == 12:
        layers = [2,20,20,1]     # (~400)
    elif type_NN == 2:
        layers = [2,50,50,50,1]  # (~5000)
    elif type_NN == 21:
        layers = [2,40,40,40,1]  # (~3200)
    elif type_NN == 22:
        layers = [2,20,20,20,1]  # (~800)
    elif type_NN == 30:
        layers = [2,50,50,50,50,1]  # (~7500)
    elif type_NN == 31:
        layers = [2,40,40,40,40,1]  # (~4800)
    elif type_NN == 32:
        layers = [2,20,20,20,20,1]  # (~1200)
    elif type_NN == 40:
        layers = [2,50,50,50,50,50,1]  # (~10000)
    elif type_NN == 41:
        layers = [2,40,40,40,40,40,1]  # (~6400)
    elif type_NN == 42:
        layers = [2,20,20,20,20,20,1]  # (~1600)
    elif type_NN == 52:
        layers = [2,20,20,20,20,20,20,1]  # (~2000)
    elif type_NN == 3:
        layers = [2,80,40,20,20,20,1]    # l5n80s (~4000)
    elif type_NN == 4:
        layers = [2,80,40,20,20,20,20,1] # l6n80s (~4000)
    elif type_NN == 5:
        layers = [2,50,20,20,20,20,20,1] # l6n50s (~2000)  
    elif type_NN == 51:         
        layers = [2,50,20,20,20,20,20,20,1]  # l6n50s (~2500)
    elif type_NN == 912:
        layers = [2,16,16,1]     # (~300)
    elif type_NN == 913:
        layers = [2,32,32,1]     # (~1000)
    elif type_NN == 914:
        layers = [2,64,64,1]     # (~4000)
    elif type_NN == 922:
        layers = [2,16,16,16,1]     # (~600)
    elif type_NN == 923:
        layers = [2,32,32,32,1]     # (~2000)
    elif type_NN == 924:
        layers = [2,64,64,64,1]     # (~8000)
    elif type_NN == 932:
        layers = [2,16,16,16,16,1]     # (~900)
    elif type_NN == 933:
        layers = [2,32,32,32,32,1]     # (~3000)
    elif type_NN == 934:
        layers = [2,64,64,64,64,1]     # (~12000)
    elif type_NN == 942:
        layers = [2,16,16,16,16,16,1]   
    elif type_NN == 943:
        layers = [2,32,32,32,32,32,1]    # 4000
    elif type_NN == 621:
        layers = [2,10,10,1]     # (~300)
    elif type_NN == 622:
        layers = [2,20,20,1]     # (~1000)
    elif type_NN == 623:
        layers = [2,30,30,1]     # (~1000)
    elif type_NN == 624:
        layers = [2,40,40,1]     # (~1000)
    elif type_NN == 625:
        layers = [2,50,50,1]     # (~1000)
    elif type_NN == 626:
        layers = [2,60,60,1]     # (~1000)   
    elif type_NN == 627:
        layers = [2,70,70,1]     # (~1000)
    elif type_NN == 628:
        layers = [2,80,80,1]     # (~1000)
    elif type_NN == 620:
        layers = [2,100,100,1]     # (~1000)
    elif type_NN == 6212:
        layers = [2,120,120,1]     # (~1000)
    elif type_NN == 6215:
        layers = [2,150,150,1]         
    elif type_NN == 6220:
        layers = [2,200,200,1]     
    elif type_NN == 631:
        layers = [2,10,10,10,1]     # (~300)
    elif type_NN == 632:
        layers = [2,20,20,20,1]     # (~1000)
    elif type_NN == 633:
        layers = [2,30,30,30,1]     # (~1000)
    elif type_NN == 634:
        layers = [2,40,40,40,1]     # (~1000)
    elif type_NN == 635:
        layers = [2,50,50,50,1]     # (~1000)
    elif type_NN == 636:
        layers = [2,60,60,60,1]     # (~1000)   
    elif type_NN == 637:
        layers = [2,70,70,70,1]     # (~1000)
    elif type_NN == 638:
        layers = [2,80,80,80,1]     # (~1000)
    elif type_NN == 639:
        layers = [2,90,90,90,1]        
    elif type_NN == 630:
        layers = [2,100,100,100,1]     # (~20000)
    elif type_NN == 6312:
        layers = [2,120,120,120,1]   
    elif type_NN == 6320:
        layers = [2,200,200,200,1]                      
    return layers