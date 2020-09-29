"""
@author     : Qizhi He @ PNNL (qizhi.he@pnnl.gov)
Decription  : Customized Functions for ploting
update @ 2020.02.12
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from drawnow import drawnow, figure
from matplotlib import collections as mc, patches as mpatches, cm

######################################################################
############################# 2D Plotting ############################
######################################################################  

'''Plot the distribution of data points'''
def sub_plt_pts2D(Xm,savefig=None,visual=None):
	def draw():
		fig = plt.figure()
		plt.plot(Xm[:,0], Xm[:,1], 'ro', markersize = 1)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.xlabel('$x_1$', fontsize=16)
		plt.ylabel('$x_2$', fontsize=16)
		plt.title('$Collocation Points$', fontsize=16)
		fig.tight_layout()
		if savefig:
			path_fig_save = savefig+'map_mesh_temp2'
			fig.savefig(path_fig_save+'.png',dpi=200)
	if visual:
		drawnow(draw)
	else:
		draw()

def sub_plt_pts3D(Zm,savefig=None,visual=None):
	def draw():
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		# ax = plt.axes(projection='3d')
		# plt.plot(Zm[:,0], Zm[:,1], Zm[:,2], 'ro', markersize = 1)
		ax.scatter3D(Zm[:,0], Zm[:,1], Zm[:,2],'ro', cmap='Greens')
		# ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');
		ax.set_xlabel('$x_1$', fontsize=16)
		ax.set_ylabel('$x_2$', fontsize=16)
		ax.set_zlabel('$t$', fontsize=16)
		plt.title('$Collocation Points$', fontsize=16)
		fig.tight_layout()
		if savefig:
			path_fig_save = savefig+'map_collo'
			fig.savefig(path_fig_save+'.png',dpi=200)
	if visual:
		drawnow(draw)
	else:
		draw()

'''Surf plot of Xm and U_pred'''
def sub_plt_surf2D(lb,ub,Xm,U_pred,t,savefig=None, visual=None,plt_eps=None):
	def draw():
		nn = 200
		x = np.linspace(lb[0], ub[0], nn)
		y = np.linspace(lb[1], ub[1], nn)
		XX, YY = np.meshgrid(x,y)

		U_plot = griddata(Xm, U_pred.flatten(), (XX, YY), method='cubic')
		fig = plt.figure()

		plt.pcolor(XX, YY, U_plot, cmap='viridis')
		# plt.plot(X_k[:,0], X_k[:,1], 'ko', markersize = 1.0)
		# plt.clim(0, 250)
		plt.jet()
		plt.colorbar()
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.xlabel('$x_1$', fontsize=16)
		plt.ylabel('$x_2$', fontsize=16)
		# plt.title('$C(x_1,x_2)$', fontsize=16)
		# plt.axes().set_aspect('equal')
		fig.tight_layout()
		plt.axis('equal')
		# path_fig_save = path_fig+'map_k_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_pred'
		# # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
		# fig.savefig(path_fig_save+'.png',dpi=300)
		
		if savefig:
			path_fig_save = savefig+'_c'+'_t_'+str(t)
			fig.savefig(path_fig_save+'.png',dpi=300)
			if plt_eps:
				plt.savefig('{}.eps'.format(path_fig_save))
		fig.clf()
	if visual:
		drawnow(draw)
	else:
		draw()

def sub_plt_surf2D_wpt(lb,ub,Xm,U_pred,output=None,visual=None,plt_eps=None,points=None, cmin=None, cmax=None, title=None):
	# def draw():
	nn = 200
	x = np.linspace(lb[0], ub[0], nn)
	y = np.linspace(lb[1], ub[1], nn)
	XX, YY = np.meshgrid(x,y)

	U_plot = griddata(Xm, U_pred.flatten(), (XX, YY), method='cubic')
	# fig = plt.figure(figsize=(5, 5))
	fig = plt.figure()

	plt.pcolor(XX, YY, U_plot, cmap='viridis')
	if points is not None:
		plt.plot(points[:,0], points[:,1], 'ko', markersize = 1.0)
	
	plt.clim([cmin, cmax])
	plt.jet()
	plt.colorbar()
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.xlabel('$x_1$', fontsize=16)
	plt.ylabel('$x_2$', fontsize=16)
	if title is not None:
		plt.title(title, fontsize=14)
	# plt.axes().set_aspect('equal')
	fig.tight_layout()
	plt.axis('equal')
	
	if output is not None:
		fig.savefig(output+'.png',dpi=300)
		if plt_eps:
			plt.savefig('{}.eps'.format(output))
	fig.clf()

'''2D Voronoi Plot'''
def plotPatch(patch, value, points=None, output=None, cmin=None, cmax=None):
	fig, ax = plt.subplots(figsize=(5, 5))
	p = mc.PatchCollection(patch, cmap=cm.jet)
	p.set_array(value)
	p.set_clim([cmin, cmax])
	ax.add_collection(p)
	if points is not None:
		ax.plot(*points, 'ko', markersize=0.5)	

	ax.axis('off')
	ax.set_aspect('equal')
	ax.autoscale(tight=True)
	fig.tight_layout()
	fig.colorbar(p)
	fig.show()
	if output is not None:
	  	fig.savefig(output, dpi=300)


######################################################################
############################# 1D Plotting ############################
######################################################################  

def sub_plt_cuvlog(loss,ylabelx=None,savefig=None, visual=None,plt_eps=None):
	def draw():
		fig = plt.figure()
		plt.semilogy(loss)
		plt.xlabel('Epoch', fontsize= 20)
		plt.ylabel(ylabelx,fontsize = 20)
		# plt.xticks(fontsize=24)
		# plt.yticks(fontsize=24)
		fig.tight_layout()
		plt.axis('equal')
		if savefig:
			path_fig_save = savefig
			fig.savefig(path_fig_save+'.png',dpi=300)
			if plt_eps:
				plt.savefig('{}.eps'.format(path_fig_save))
		fig.clf()
	if visual:
		drawnow(draw)
	else:
		draw()

def sub_plt_cuvlog2(loss,Ni,ylabelx=None,savefig=None, visual=None,plt_eps=None):
	def draw():
		fig = plt.figure()
		Nx = len(loss)
		xm_ls = np.arange(0, Ni * Nx, Ni)
		plt.semilogy(xm_ls,loss)
		plt.xlabel('Epoch', fontsize = 20)
		plt.ylabel(ylabelx, fontsize = 20)
		# plt.xticks(fontsize=24)
		# plt.yticks(fontsize=24)
		fig.tight_layout()
		plt.axis('equal')
		if savefig:
			path_fig_save = savefig
			fig.savefig(path_fig_save+'.png',dpi=300)
			if plt_eps:
				plt.savefig('{}.eps'.format(path_fig_save))
		fig.clf()
	
	if visual:
		drawnow(draw)
	else:
		draw()

