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

N_s = 5
rand_seed_mea = 1127
print(rand_seed,rand_seed_mea)


# array_k = np.array([16, 36, 48, 64, 80, 96])
# array_h = np.array([16, 36, 48, 64, 80, 96])
# num_data = array_k.shape[0]
# array_nn  = np.array([923,943])
# array_c = np.array([64,128])
array_k = np.array([16])
array_h = np.array([16])
num_data = array_k.shape[0]
array_nn  = np.array([923])
array_c = np.array([64])

for i_nn in array_nn:
	for num_c in array_c:
		directory = './Output_MPINNs_examples/ds12_pro1_seq/MPINN_pro1_ts1s5_pro1_nn'+str(i_nn)+'_opt1_seq_c'+str(num_c)+'/'  # test main_PINN_CAD_v3 without STD
		src_dir   = './main_MPINNs.py' 

		if not os.path.exists(directory):
			os.makedirs(directory)

		for ii in range(0,num_data):
			num_k = array_k[ii]
			num_h = array_h[ii]

			for i_s in rand_seed:
				os.system("python {0} {1} {2} {3} \
					{4} {5} 200 {6} 1000 \
					1.0 1.0 1.0 1.0 1.0 \
					12 1 \
					{8} {8} {8} 1 1 1\
					1 0 1000000 0.001 0 1.e-6 'sequent' 1 \
					2 {7} 1 0.0008".format(src_dir,directory,i_s,N_s,num_k,num_h,num_c,rand_seed_mea,i_nn))
