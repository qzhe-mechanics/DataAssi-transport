#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: heqi220
"""

import os
import numpy as np
from pyDOE import lhs 

## Randomize Initialization
N_i = 1
if N_i > 1:
	np.random.seed(1234)
	rand_seed = np.random.randint(0,2000,N_i)
else:
	rand_seed = np.array([111])

N_s = 5 
rand_seed_mea = 1127
print(rand_seed,rand_seed_mea)

# Example in Sec4-2_PINN-Darcy
# array_k = np.array([16, 36, 48, 64, 80, 96])
# array_f = np.array([20, 50, 100, 200, 400])
# examples (test)
array_k = np.array([16])
array_f = np.array([20])
num_data = array_k.shape[0]

for num_f in array_f:
	directory = './Output_MPINNs_examples/ds12_pro2_Darcy/MPINN_pro2_ts5s5_n23n13_opt1_f'+str(num_f)+'/'
	src_dir   = './main_MPINNs.py'

	if not os.path.exists(directory):
	    os.makedirs(directory)

	for ii in range(0,num_data):
		num_k = array_k[ii]

		for i_s in rand_seed:
			os.system("python {0} {1} {2} {3} \
				{4} {4} {6} 64 200 \
				1.0 1.0 1.0 1.0 1.0 \
				12 1 \
				923 913 912 1 1 1\
				1 0 50000 0.0001 0 1.e-6 'standard' 2 \
				2 {5}".format(src_dir,directory,i_s,N_s,num_k,rand_seed_mea,num_f))

