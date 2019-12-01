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

# examples
# array_h = np.array([8])
# array_nn  = np.array([912])

array_h = np.array([8, 16, 32, 48, 64, 80])
array_nn  = np.array([912,913,922,923])
num_data = array_h.shape[0]

for i_nn in array_nn:
	directory = './Output_MPINNs_examples/ds12_pro32/MPINN_pro32_ts1s5_n'+str(i_nn)+'_opt1/'  # test main_PINN_CAD_v3 without STD
	src_dir   = './main_MPINNs.py'          # Recall the code

	if not os.path.exists(directory):
	    os.makedirs(directory)

	for ii in range(0,num_data):
		num_h = array_h[ii]

		for i_s in rand_seed:
			os.system("python {0} {1} {2} {3} \
				64 {4} 200 64 200 \
				1.0 1.0 1.0 1.0 1.0 \
				12 1 \
				912 {6} 912 1 1 1\
				1 0 50000 0.0001 0 1.e-6 'standard' 32 \
				2 {5}".format(src_dir,directory,i_s,N_s,num_h,rand_seed_mea,i_nn))

