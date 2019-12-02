#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: heqi220
"""

import os
import numpy as np
from pyDOE import lhs   # The experimental design package for python; Latin Hypercube Sampling (LHS)

## Randomize Initialization
N_i = 1
if N_i > 1:
	np.random.seed(1234)
	rand_seed = np.random.randint(0,2000,N_i)
else:
	rand_seed = np.array([111])

N_s = 10 
rand_seed_mea = 1127
print(rand_seed,rand_seed_mea)


################################
array_k = np.array([20000])
num_data = 1
array_nn  = np.array([631,632,633,634,635,636,637,638])
for i_nn in array_nn:

	directory = './Output_MPINNs_examples/ds43_pro31/MPINN_ds43_pro31_opt21m1ki200k_nn'+str(i_nn)+'/'  # test main_PINN_CAD_v3 without STD
	src_dir   = './main_MPINNs.py'          # Recall the code

	if not os.path.exists(directory):
		os.makedirs(directory)

	for ii in range(0,num_data):
		num_k = array_k[ii]

		for i_s in rand_seed:
			os.system("python {0} {1} {2} {3} \
				{4} 100 200 100 400 \
				1.0 1.0 1.0 1.0 1.0 \
				43 1 \
				{5} 923 923 1 1 1\
				21 1000 200000 0.0002 0 1.e-8 'standard' 31 \
				2 {6} 1 0.00001".format(src_dir,directory,i_s,N_s,num_k,i_nn,rand_seed_mea))

