from __future__ import unicode_literals
import requests
import h5py
import matplotlib.pyplot as plt
import numpy as np
#from npy_append_array import NpyAppendArray
import webbrowser
import urllib.request
from PIL import Image
from scipy import misc
from scipy.stats import binned_statistic_2d
import glob
import imageio
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm
import scipy.ndimage as sp
import sys
import os
import unicodedata
import re
from os import path
import time
from time import perf_counter as clock
from pathlib import Path
from pylab import *

#matplotlib
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons, RectangleSelector
#from matplotlib.widgets import TextBox
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
#plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.font_manager
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib import path as matplotlib_path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.offsetbox import  OffsetImage, AnnotationBbox
import matplotlib.image as image

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import ast
from scipy.interpolate import CubicSpline

print ("\n")
print ('Successfully imported all modules....')
sys.stdout.flush()
time.sleep(0.1)
print ("\n")

#Setting up cosmology
cosmo = FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.308)

#Some random formatting code
underline = '\033[4m'
end_formatting = end_format = reset = '\033[0m'

def print_cust(print_str, **kwargs):
	sleep_time = kwargs.get('sleep_time', 0.1)  # Sleep Time
	quite_val = kwargs.get('quiet_val', False)  # Quite Val
	if not quite_val:
		print ("\n")
		print (print_str)
		print ("\n")
		sys.stdout.flush()
		time.sleep(sleep_time)

#Function to check if a directory exists. If not, create the directory.
def check_directory(dir_name):
	if not os.path.exists(dir_name):
		print ('Making Directory - ', dir_name, '\n')
		sys.stdout.flush()
		time.sleep(0.1)
		os.makedirs(dir_name)
	else:
		print ('Directory - ', dir_name, 'exists \n')
		sys.stdout.flush()
		time.sleep(0.1)

#Find index for nearest element
def find_nearest_idx(array,value):
	idx = (np.abs(array-value)).argmin()
	return idx


#######################ROTATION_MATRIX_FUNCTIONS#######################
 
def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
 
def Ry(theta):
  return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
 
def Rz(theta):
  return np.matrix([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])


def rotaion_matrix(val_x, val_y, val_z, phi_deg, theta_deg):
	phi = np.radians(phi_deg)
	theta = np.radians(theta_deg)
	val_init = np.array([val_x, val_y, val_z]).T
	R = Ry(theta) * Rz(phi) * Rx(0.0)
	val_final = np.matmul(val_init, R)
	val_x_new, val_y_new, val_z_new = val_final.T
	return (val_x_new, val_y_new, val_z_new)
	
def rotaion_matrix_group(val_x_array, val_y_array, val_z_array, phi, theta):
	val_x_array_rev = np.zeros([val_x_array.shape[0], val_x_array.shape[1]])
	val_y_array_rev = np.zeros([val_x_array.shape[0], val_x_array.shape[1]])
	val_z_array_rev = np.zeros([val_x_array.shape[0], val_x_array.shape[1]])
	for i in range(val_x_array.shape[1]):
		val_x_array_rev[:,i], val_y_array_rev[:,i], val_z_array_rev[:,i] = rotaion_matrix(val_x_array[:,i], val_y_array[:,i], val_z_array[:,i], phi, theta)
	
	return (val_x_array_rev, val_y_array_rev, val_z_array_rev)

def rotaion_matrix_group_new(val_x_array, val_y_array, val_z_array, phi, theta):
	val_x_array_rev = np.zeros([val_x_array.shape[0]])
	val_y_array_rev = np.zeros([val_x_array.shape[0]])
	val_z_array_rev = np.zeros([val_x_array.shape[0]])
	#for i in range(val_x_array.shape[0]):
	val_x_array_rev[:], val_y_array_rev[:], val_z_array_rev[:] = rotaion_matrix(val_x_array[:], val_y_array[:], val_z_array[:], phi, theta)
	return (val_x_array_rev, val_y_array_rev, val_z_array_rev)

#######################ROTATION_MATRIX_FUNCTIONS#######################


snapshots_init_def = np.array([2, 3, 4, 6, 8, 11, 13, 17, 21, 25, 33, 40, 50, 59, 67, 72, 78, 84, 91, 99])
scale_factor_init_def = np.array([0.0769, 0.0833, 0.0909, 0.1, 0.1111, 0.125, 0.1429, 0.1667, 0.2, 0.25, 0.3333, 0.4, 0.5, 0.5882, 0.6667, 0.7143, 0.7692, 0.8333, 0.9091, 1])
redshift_init_def = np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.5, 1, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0])
cs_scale_factor = CubicSpline(snapshots_init_def, scale_factor_init_def)
cs_redshift = CubicSpline(snapshots_init_def, redshift_init_def)
snapshots_original = np.arange(snapshots_init_def.min(), snapshots_init_def.max()+1, 1, dtype=np.int)
scale_factor_original = cs_scale_factor(snapshots_original)
redshift_original = cs_redshift(snapshots_original)
cosmic_age_original = cosmo.age(redshift_original).value

print ("Running script: \n")
print (str(sys.argv[0]))
print ("\n")
print ("Current Working Directory: \n")
print (str(sys.argv[1]))
print ("\n")
print ("Base Working Directory: \n")
print (str(sys.argv[2]))
sys.stdout.flush()
time.sleep(0.1)
print ("\n")

#######################OBTAINING_PARAMETER_INFORMATION#######################
#Obtains parameter information from a given parameter file
#If not specified, checks the currents directory for a default paramter-
#-file named, "parameter_file.dat". If that is not available, it goes to-
#-base VYG Code directory to obtain default parameters.

parameter_file_string_current = str(sys.argv[1]) + str('/parameter_file.dat')
parameter_file_string_base = str(sys.argv[2]) + str('/basic_parameter_file.dat')

initial_guesses = {}
with open(str(parameter_file_string_base)) as f:
	for line in f:
		if '#' not in line:
			if (len(line.split())>2):
				(key, val) = line.split(':')
				key = key.replace(':', '').replace('-', '').lower()
				initial_guesses[str(key)] = ast.literal_eval(val.replace(' ', ''))
			else:
				(key, val) = line.split()
				key = key.replace(':', '').replace('-', '').lower()
				initial_guesses[str(key)] = val


if (len(sys.argv)!=4):
	print_cust("No parameter file given along command line. Searching current directory for parameter file.")
	if (os.path.isfile(parameter_file_string_current)):
		print_cust(f"Parameter file found in the current directory: {str(parameter_file_string_current)}")
		parameter_file_name_final = parameter_file_string_current
	else:
		print_cust("No parameter file (with default name - initial_parameters.dat) found in the current directory.")
		save_prompt= input("Would you like to provide the name of the parameter file? (y/n) : ")
		print ("\n")
		if (save_prompt=='y'):
			save_prompt2 = input("Please enter the name of the parameter file?: ")
			print ("\n")
			parameter_file_name_final = str(sys.argv[1]) + str("/") + str(save_prompt2)
		else:
			print ("Reverting to default parameter file: \n", str(parameter_file_string_base))
			print ("\n")
			parameter_file_name_final = parameter_file_string_base
	
else:
	parameter_file_name_final = str(sys.argv[1]) + str("/") + str(sys.argv[3])
	
print_cust(f"Executing script with data from: {str(parameter_file_name_final)}")

#######################OBTAINING_PARAMETER_INFORMATION#######################

#######################FOR_ACCESSING_THE_TNG_DATA_FROM_WEB#######################
#This key is required from accessing TNG data
#See https://www.tng-project.org/data/docs/api/ for more details

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"0088dec8086a9d73bb849abca092bad3"}

#######################FOR_ACCESSING_THE_TNG_DATA_FROM_WEB#######################

#######################FUNCTION_FOR_ESTIMATING_TEMPERATURE#######################
#Formula obtained from TNG Website
#See https://www.tng-project.org/data/docs/faq/#gen5 for details

def temperature_estimation(ie, ea):
	x_h = 0.76
	x_e = ea
	m_p = 1.6726e-24
	mu_new = (4/(1 + (3*x_h) + (4*x_h*x_e)))*m_p
	gamma = 5./3.
	k_b = 1.3807e-16
	u = ie
	temp = (gamma-1) * (u/k_b) * mu_new * 1e10
	return (temp)
	
#######################FUNCTION_FOR_ESTIMATING_TEMPERATURE#######################

#######################CUSTOM_CREATION_OF_COLORBAR_IN_SUBPLOTS_OF_MATPLOTLIB#######################
#This function is called for creating custom colorbar for subplots
#See https://stackoverflow.com/questions/23876588/matplotlib-colorbar-in-each-subplot for more details

def add_colorbar(mappable):
	last_axes = plt.gca()
	ax = mappable.axes
	fig = ax.figure
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(mappable, cax=cax)
	cbar.set_ticks(ticker.LogLocator(), update_ticks=True)
	cbar.ax.tick_params(size=0)
	#plt.sca(last_axes)
	return cbar

def add_colorbar_lin(mappable):
	last_axes = plt.gca()
	ax = mappable.axes
	fig = ax.figure
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(mappable, cax=cax)
	cbar.set_ticks(ticker.LinearLocator(), update_ticks=True)
	cbar.ax.tick_params(size=0)
	#plt.sca(last_axes)
	return cbar

#######################CUSTOM_CREATION_OF_COLORBAR_IN_SUBPLOTS_OF_MATPLOTLIB#######################

#######################GET_FUNCTION_TO_OBTAIN_DATA_FROM_TNG#######################
#This function is required for obtaining data from TNG online
#See https://www.tng-project.org/data/docs/api/ for more details

def get(path, params=None):
	# make HTTP GET request to path
	try:
		r = requests.get(path, params=params, headers=headers)
	except requests.exceptions.ConnectionError:
		try:
			time.sleep(10)
			r = requests.get(path, params=params, headers=headers)
		except Exception as e:
			print_cust(e)
	#r = requests.get(path, params=params, headers=headers)

	# raise exception if response code is not HTTP SUCCESS (200)
	r.raise_for_status()

	if r.headers['content-type'] == 'application/json':
		return r.json() # parse json responses automatically

	if 'content-disposition' in r.headers:
		filename = r.headers['content-disposition'].split("filename=")[1]
		with open(filename, 'wb') as f:
			f.write(r.content)
		return filename # return the filename string

	return r

#######################GET_FUNCTION_TO_OBTAIN_DATA_FROM_TNG#######################

#######################GET_PARTICLE_INFORMATION_FROM_TNG_DATA#######################
#This function takes a TNG specific web-url and obtains particle information in the specific cutout
#See https://www.tng-project.org/data/docs/api/ for more details

def particle_information_halo(sub_prog_url_cust):
	sub_prog_subhalo = get(sub_prog_url_cust)
	sub_prog_halo_url =	sub_prog_subhalo['related']['parent_halo']
	print (sub_prog_halo_url)
	sub_prog_halo = get(sub_prog_halo_url)
	test = sub_prog_halo['child_subhalos']['results']
	sub_prog_central_subhalo = get(test[0]['url'])
	center_subhalo_x = sub_prog_subhalo['pos_x'] - sub_prog_central_subhalo['pos_x']
	center_subhalo_y = sub_prog_subhalo['pos_y'] - sub_prog_central_subhalo['pos_y']
	center_subhalo_z = sub_prog_subhalo['pos_z'] - sub_prog_central_subhalo['pos_z']
	radius_subhalo = sub_prog_subhalo['halfmassrad']

	cutout_request = {'gas':'Coordinates,Masses,InternalEnergy,ElectronAbundance,Velocities', 'stars':'Coordinates,Masses,Velocities,GFM_StellarFormationTime', 'dm':'Coordinates,Velocities', 'bhs':'Coordinates,Masses,Velocities'}
	cutout = get(sub_prog_halo_url+"cutout.hdf5", cutout_request)
	x, y, z, dens, ie, ea, vel_gas_x, vel_gas_y, vel_gas_z, x2, y2, z2, stars, stellar_age_rev, vel_star_x, vel_star_y, vel_star_z, x3, y3, z3, x4, y4, z4, dm_mass, vel_dm_x, vel_dm_y, vel_dm_z, vyg_st_mass, fracyoung_rev = ([] for i in range(29))
	with h5py.File(cutout,'r') as f:
		catch_error_1 = f.get('PartType0')
		catch_error_2 = f.get('PartType4')
		catch_error_3 = f.get('PartType5')
		catch_error_4 = f.get('PartType1')
		scale_factor_tmp_idx = find_nearest_idx(snapshots_original, int(sub_prog_subhalo['snap']))
		scale_factor_tmp = scale_factor_original[scale_factor_tmp_idx]

		if (catch_error_1):
			x = sub_prog_subhalo['pos_x'] - f['PartType0']['Coordinates'][:,0]
			y = sub_prog_subhalo['pos_y'] - f['PartType0']['Coordinates'][:,1]
			z = sub_prog_subhalo['pos_z'] - f['PartType0']['Coordinates'][:,2]
			dens = f['PartType0']['Masses'][:]*1e10
			vel_gas = f['PartType0']['Velocities'][:] * np.sqrt(scale_factor_tmp)
			ie = f['PartType0']['InternalEnergy'][:]
			ea = f['PartType0']['ElectronAbundance'][:]
			vel_gas_x = np.zeros([vel_gas.shape[0]])
			vel_gas_y = np.zeros([vel_gas.shape[0]])
			vel_gas_z = np.zeros([vel_gas.shape[0]])
			for i in range(vel_gas.shape[0]):
				vel_gas_x[i] = sub_prog_subhalo['vel_x'] - vel_gas[i][0]
				vel_gas_y[i] = sub_prog_subhalo['vel_y'] - vel_gas[i][1]
				vel_gas_z[i] = sub_prog_subhalo['vel_z'] - vel_gas[i][2]

			mask_1 = (np.array(x)>-vel_limit) & (np.array(x)<vel_limit) & (np.array(y)>-vel_limit) & (np.array(y)<vel_limit) & (np.array(z)>-vel_limit) & (np.array(z)<vel_limit)
			x = x[mask_1]
			y = y[mask_1]
			z = z[mask_1]
			dens = dens[mask_1]
			ie = ie[mask_1]
			ea = ea[mask_1]
			vel_gas_x = vel_gas_x[mask_1]
			vel_gas_y = vel_gas_y[mask_1]
			vel_gas_z = vel_gas_z[mask_1]

		if (catch_error_2):
			x2 = sub_prog_subhalo['pos_x'] - f['PartType4']['Coordinates'][:,0]
			y2 = sub_prog_subhalo['pos_y'] - f['PartType4']['Coordinates'][:,1]
			z2 = sub_prog_subhalo['pos_z'] - f['PartType4']['Coordinates'][:,2]
			stars = f['PartType4']['Masses'][:]*1e10
			stellar_age = f['PartType4']['GFM_StellarFormationTime'][:]
			stellar_age_rev = cosmo.age((1./stellar_age)-1.).value
			#stellar_age_idx = np.searchsorted(scale_factor_original, stellar_age)
			#stellar_age_rev = cosmic_age_original[stellar_age_idx]
			vel_star = f['PartType4']['Velocities'][:] * np.sqrt(scale_factor_tmp)
			vel_star_x = np.zeros([vel_star.shape[0]])
			vel_star_y = np.zeros([vel_star.shape[0]])
			vel_star_z = np.zeros([vel_star.shape[0]])
			for i in range(vel_star.shape[0]):
				vel_star_x[i] = sub_prog_subhalo['vel_x'] - vel_star[i][0]
				vel_star_y[i] = sub_prog_subhalo['vel_y'] - vel_star[i][1]
				vel_star_z[i] = sub_prog_subhalo['vel_z'] - vel_star[i][2]
				
			vyg_st_mass = stars
			vyg_st_mass[stellar_age_rev<=(np.nanmax(cosmic_age_original)-1.)] = np.nan
			if (np.nansum(stars)):
				fracyoung = float(np.nansum(vyg_st_mass) / np.nansum(stars))
			else:
				fracyoung = 0.0
			#fracyoung_rev = list(np.full([len(stars)], fill_value=fracyoung))

			mask_2 = (np.array(x2)>-vel_limit) & (np.array(x2)<vel_limit) & (np.array(y2)>-vel_limit) & (np.array(y2)<vel_limit) & (np.array(z2)>-vel_limit) & (np.array(z2)<vel_limit)
			x2 = x2[mask_2]
			y2 = y2[mask_2]
			z2 = z2[mask_2]
			stars = stars[mask_2]
			vel_star_x = vel_star_x[mask_2]
			vel_star_y = vel_star_y[mask_2]
			vel_star_z = vel_star_z[mask_2]
			vyg_st_mass = vyg_st_mass[mask_2]
			fracyoung_rev = np.full([len(stars)], fill_value=fracyoung)
			if (len(mask_2)>1):
				fracyoung_rev = fracyoung_rev[mask_2.astype(np.int32)]
			#fracyoung_rev = fracyoung_rev[mask_2]

		if (catch_error_3):
			print ('BH Detected')
			sys.stdout.flush()
			time.sleep(0.1)
			x3 = sub_prog_subhalo['pos_x'] - f['PartType5']['Coordinates'][:,0]
			y3 = sub_prog_subhalo['pos_y'] - f['PartType5']['Coordinates'][:,1]
			z3 = sub_prog_subhalo['pos_z'] - f['PartType5']['Coordinates'][:,2]
			mask_3 = (np.array(x3)>-vel_limit) & (np.array(x3)<vel_limit) & (np.array(y3)>-vel_limit) & (np.array(y3)<vel_limit) & (np.array(z3)>-vel_limit) & (np.array(z3)<vel_limit)
			x3 = x3[mask_3]
			y3 = y3[mask_3]
			z3 = z3[mask_3]
			
		if (catch_error_4):
			x4 = sub_prog_subhalo['pos_x'] - f['PartType1']['Coordinates'][:,0]
			y4 = sub_prog_subhalo['pos_y'] - f['PartType1']['Coordinates'][:,1]
			z4 = sub_prog_subhalo['pos_z'] - f['PartType1']['Coordinates'][:,2]
			#dm_mass = f['PartType1']['Masses'][:]*1e10
			vel_dm = f['PartType1']['Velocities'][:] * np.sqrt(scale_factor_tmp)
			dm_mass = np.ones_like(x4)*1e10
			vel_dm_x = np.zeros([vel_dm.shape[0]])
			vel_dm_y = np.zeros([vel_dm.shape[0]])
			vel_dm_z = np.zeros([vel_dm.shape[0]])
			for i in range(vel_dm.shape[0]):
				vel_dm_x[i] = sub_prog_subhalo['vel_x'] - vel_dm[i][0]
				vel_dm_y[i] = sub_prog_subhalo['vel_y'] - vel_dm[i][1]
				vel_dm_z[i] = sub_prog_subhalo['vel_z'] - vel_dm[i][2]

			mask_4 = (np.array(x4)>-vel_limit) & (np.array(x4)<vel_limit) & (np.array(y4)>-vel_limit) & (np.array(y4)<vel_limit) & (np.array(z4)>-vel_limit) & (np.array(z4)<vel_limit)
			x4 = x4[mask_4]
			y4 = y4[mask_4]
			z4 = z4[mask_4]
			dm_mass = dm_mass[mask_4]
			vel_dm_x = vel_dm_x[mask_4]
			vel_dm_y = vel_dm_y[mask_4]
			vel_dm_z = vel_dm_z[mask_4]

	return (x, y, z, dens, ie, ea, vel_gas_x, vel_gas_y, vel_gas_z, x2, y2, z2, stars, vel_star_x, vel_star_y, vel_star_z, x3, y3, z3, x4, y4, z4, dm_mass, vel_dm_x, vel_dm_y, vel_dm_z, center_subhalo_x, center_subhalo_y, center_subhalo_z, radius_subhalo, stellar_age_rev, vyg_st_mass, fracyoung_rev)

#######################GET_PARTICLE_INFORMATION_FROM_TNG_DATA#######################

#######################READ_PARAMETER_FILE#######################
#Open the parameter file and read information required to run this code

d = initial_guesses
#with open("parameter_file.dat") as f:
with open(str(parameter_file_name_final)) as f:
	for line in f:
		if '#' not in line:
			if (len(line.split())>2):
				(key, val) = line.split(':')
				key = key.replace(':', '').replace('-', '').lower()
				d[str(key)] = ast.literal_eval(val.replace(' ', ''))
			else:
				(key, val) = line.split()
				key = key.replace(':', '').replace('-', '').lower()
				d[str(key)] = val

#Print the initial information given
print_cust(f"{d}")

#######################READ_PARAMETER_FILE#######################

#######################OBTAIN_PARAMETER_INFORMATION#######################
#Obtain information from file and convert it into relevant web based strings and objects

str1 = "http://www.tng-project.org/api/"
str2 = str(d['simulation'])
str3 = "/snapshots/"
str4 = str(d['snapshot'])
str5 = "/subhalos/"
str6 = str(d['galaxyid'])
str7 = "/"
total_str = str1 + str2 + str3 + str4 + str5 + str6 + str7
sub_prog_url = total_str
sub_prog = get(sub_prog_url)
colormap_new = str(d['clrmap'])
bin_number = int(d['bins_for_hist'])
window_size_init = float(d['window_size'])
xlabel_str = str(r'$\rm \Delta x$ [ckpc/h]')
ylabel_str = str(r'$\rm \Delta y$ [ckpc/h]')
size_of_font = int(d['fontsize'])
vel_limit = float(d['limiting_velocity'])

hzdict7 = {'gas_dens': 0, 'temp': 1, 'stars': 2, 'gas_vel': 3, 'star_vel': 4, 'vgas_rad': 5, 'vstar_rad': 6, 'mass_dm': 7, 'dm_vel': 8, 'stellar_age': 9, 'vyg_st_mass': 10, 'vyg_fraction': 11}
hzdict8 = {'gas_dens': 0, 'temp': 0, 'stars': 1, 'gas_vel': 0, 'star_vel': 1, 'vgas_rad': 0, 'vstar_rad': 1, 'mass_dm': 3, 'dm_vel': 3, 'stellar_age': 1, 'vyg_st_mass': 1, 'vyg_fraction': 1}


#######################OBTAIN_PARAMETER_INFORMATION#######################

#######################FIND_THE_RELATED_SUBHALO_AT_SNAPSHOT=99#######################
#This ensures that no matter which snapshot is given, the code will always retrieve-
#-information starting from 99 up to whatever staring SNAPSHOT user has specified
#It makes sure that user can provide any snapshot number and does not have to find the-
#-related halo ID at snapshot 99 from the website

while sub_prog['prog_sfid'] >= -1:
	if (sub_prog['related']['sublink_descendant'] != None):
		print (sub_prog['snap'])
		print (sub_prog['meta']['url'])
		sys.stdout.flush()
		time.sleep(0.1)
		sub_prog = get(sub_prog['related']['sublink_descendant'])
	else:
		break

print_cust(f"{sub_prog['snap']}")
print_cust(f"{sub_prog['meta']['url']}")
sub_new = (sub_prog['meta']['url'])
sub_prog_url_array = np.chararray([], itemsize=100)
sub_prog_url_array = np.append(sub_prog_url_array, sub_new.encode('utf-8'))

#######################FIND_THE_RELATED_SUBHALO_AT_SNAPSHOT=99#######################

#######################MAKING_A_DIRECTORY_RELEVANT_TO_A_SPECIFIC_HALO#######################
#Make a directory for a specific halo if it is already not created
#This will ensure that the files are systematically saved and do not have to be-
#-downloaded each time the code runs

str4_rev = str(int(sub_prog['snap']))
str6_rev = str(int(sub_prog['id']))

dir_name = str2 + "_" + str4_rev + "_" + str6_rev + "_v1"
if ('storage_directory' in d):
	str_dir_name = str(d['storage_directory']) + "/" + dir_name + "/"
else:
	str_dir_name = str(sys.argv[1]) + "/" + dir_name + "/"

check_directory(str_dir_name)

#######################MAKING_A_DIRECTORY_RELEVANT_TO_SPECIFIC_HALO#######################

#######################FIND_DATA_FROM_ALL_RELEVANT_SNAPSHOTS#######################
#Finding halo data from all the snapshots that the user has requested

snap_array = []
while sub_prog['prog_sfid'] > -1:
	print (sub_prog['snap'])
	print (sub_prog['meta']['url'])
	sys.stdout.flush()
	time.sleep(0.1)
	snap_array = np.append(snap_array, int(sub_prog['snap']))
	#while sub_prog['prog_sfid'] >= 93:
	# request the full subhalo details of the progenitor by following the sublink URL
	if (sub_prog['snap'] >= int(d['first_snapshot_number'])+1):
		sub_new = (sub_prog['related']['sublink_progenitor'])
		sys.stdout.flush()
		time.sleep(0.1)
		sub_prog_url_array = np.append(sub_prog_url_array, sub_new.encode('utf-8'))
		sub_prog = get(sub_new)
	else:
		break


#######################FIND_DATA_FROM_ALL_RELEVANT_SNAPSHOTS#######################

#######################GET_PARTICLE_DATA_AND_SAVE_IT_IN_A_FILE#######################
#Get the particle information (gas position, gas density, Temperature, velocity,-
#-star position, density and velocity) for a cutout of a specific subhalo for-
#-all the relevant snapshots requested by the user

sub_prog_url_array = np.delete(sub_prog_url_array, 0)
expanded_filename_test = np.chararray([len(sub_prog_url_array)], itemsize=1000)
linkname = np.chararray([len(sub_prog_url_array)], itemsize=1000)

for i in range(len(sub_prog_url_array)):
	expanded_filename_test[i] = (str(str_dir_name) + "_information_" + str(i) + ".hdf5")
	if (path.exists(expanded_filename_test[i])):
		print_cust(f"{expanded_filename_test[i]}: File exists")
	else:
		linkname[i] = str(sub_prog_url_array[i].decode("utf-8"))
		print (linkname[i].decode("utf-8"))
		sys.stdout.flush()
		time.sleep(0.1)
		x, y, z, dens, ie, ea, vel_gas_x, vel_gas_y, vel_gas_z, x2, y2, z2, stars, vel_star_x, vel_star_y, vel_star_z, x3, y3, z3, x4, y4, z4, dm_mass, vel_dm_x, vel_dm_y, vel_dm_z, subhalo_cen_x, subhalo_cen_y, subhalo_cen_z, subhalo_rad, st_age, stellar_mass_vyg, youngfrac_vyg = particle_information_halo(linkname[i].decode("utf-8"))
		
		vyg_counter_s = 0
		with h5py.File(expanded_filename_test[i], 'a') as hf:
			hf.create_dataset("pos_x1",  data=x)
			hf.create_dataset("pos_y1",  data=y)
			hf.create_dataset("pos_z1",  data=z)
			hf.create_dataset("pos_x2",  data=x2)
			hf.create_dataset("pos_y2",  data=y2)
			hf.create_dataset("pos_z2",  data=z2)
			hf.create_dataset("pos_x3",  data=x3)
			hf.create_dataset("pos_y3",  data=y3)
			hf.create_dataset("pos_z3",  data=z3)
			hf.create_dataset("pos_x4",  data=x4)
			hf.create_dataset("pos_y4",  data=y4)
			hf.create_dataset("pos_z4",  data=z4)
			hf.create_dataset("gas_density",  data=dens)
			hf.create_dataset("gas_ie",  data=ie)
			hf.create_dataset("gas_ea",  data=ea)
			hf.create_dataset("stellar_density",  data=stars)
			hf.create_dataset("vel_gas_x",  data=vel_gas_x)
			hf.create_dataset("vel_gas_y",  data=vel_gas_y)
			hf.create_dataset("vel_gas_z",  data=vel_gas_z)
			hf.create_dataset("vel_star_x",  data=vel_star_x)
			hf.create_dataset("vel_star_y",  data=vel_star_y)
			hf.create_dataset("vel_star_z",  data=vel_star_z)
			hf.create_dataset("dm_mass",  data=dm_mass)
			hf.create_dataset("vel_dm_x",  data=vel_dm_x)
			hf.create_dataset("vel_dm_y",  data=vel_dm_y)
			hf.create_dataset("vel_dm_z",  data=vel_dm_z)
			hf.create_dataset("center_subhalo_x",  data=subhalo_cen_x)
			hf.create_dataset("center_subhalo_y",  data=subhalo_cen_y)
			hf.create_dataset("center_subhalo_z",  data=subhalo_cen_z)
			hf.create_dataset("radius_subhalo",  data=subhalo_rad)
			hf.create_dataset("stellar_age",  data=st_age)
			hf.create_dataset("vyg_counter",  data=vyg_counter_s)
			hf.create_dataset("vyg_stellar_mass",  data=stellar_mass_vyg)
			hf.create_dataset("vyg_fraction",  data=youngfrac_vyg)


#######################GET_PARTICLE_DATA_AND_SAVE_IT_IN_A_FILE#######################

#######################LOAD_PARTICLE_INFORMATION_FROM_FILE_FOR_PREOCESSING#######################
#All particle data loaded from the file for dynamic plotting

data_x = []
subhalo_cen_x = []
data_y = []
subhalo_cen_y = []
data_z = []
subhalo_cen_z = []
data_real = []
subhalo_radius = []
stelar_mass = []
stelar_age = []
stelar_age_rev = []
vyg_counter = []
vyg_st_mass = []
vyg_fraction = []

for i in range(0, len(sub_prog_url_array)):
	with h5py.File(expanded_filename_test[i], 'r') as hf:
		vyg_counter.append(np.array([hf['vyg_counter'][()]]))
		
for i in range(0,len(sub_prog_url_array)):
	idx_cust = i
	with h5py.File(expanded_filename_test[idx_cust], 'r') as hf:
		data_x.append(np.array([np.array(hf['pos_x1'][()]), np.array(hf['pos_x2'][()]), np.array(hf['pos_x3'][()]), np.array(hf['pos_x4'][()])], dtype="object"))
		data_y.append(np.array([np.array(hf['pos_y1'][()]), np.array(hf['pos_y2'][()]), np.array(hf['pos_y3'][()]), np.array(hf['pos_y4'][()])], dtype="object"))
		data_z.append(np.array([np.array(hf['pos_z1'][()]), np.array(hf['pos_z2'][()]), np.array(hf['pos_z3'][()]), np.array(hf['pos_z4'][()])], dtype="object"))
		ie = (hf['gas_ie'][()])
		ea = (hf['gas_ea'][()])
		temp = temperature_estimation(ie, ea)
		vel_gas_x = (hf['vel_gas_x'][()])
		vel_gas_y = (hf['vel_gas_y'][()])
		vel_gas_z = (hf['vel_gas_z'][()])
		vel_star_x = (hf['vel_star_x'][()])
		vel_star_y = (hf['vel_star_y'][()])
		vel_star_z = (hf['vel_star_z'][()])
		gas_kinematics = np.sqrt(vel_gas_x**2 + vel_gas_y**2 + vel_gas_z**2)
		radial_vel_gas =  (hf['vel_gas_z'][()])
		stellar_kinematics = np.sqrt(vel_star_x**2 + vel_star_y**2 + vel_star_z**2)
		radial_vel_star = (hf['vel_star_z'][()])
		mass_dm = (hf['dm_mass'][()])
		vel_dm_x = (hf['vel_dm_x'][()])
		vel_dm_y = (hf['vel_dm_y'][()])
		vel_dm_z = (hf['vel_dm_z'][()])
		dm_kinematics = np.sqrt(vel_dm_x**2 + vel_dm_y**2 + vel_dm_z**2)
		subhalo_cen_x.append(np.array([(hf['center_subhalo_x'][()])]))
		subhalo_cen_y.append(np.array([(hf['center_subhalo_y'][()])]))
		subhalo_cen_z.append(np.array([(hf['center_subhalo_z'][()])]))
		subhalo_radius.append(np.array([(hf['radius_subhalo'][()])]))
		stelar_mass.append(np.array([(hf['stellar_density'][()])]))
		stelar_age.append(np.array([(hf['stellar_age'][()])]))
		data_real.append(np.array([(hf['gas_density'][()]), temp, (hf['stellar_density'][()]), gas_kinematics, stellar_kinematics, radial_vel_gas, radial_vel_star, mass_dm, dm_kinematics, (hf['stellar_age'][()]), (hf['vyg_stellar_mass'][()]), (hf['vyg_fraction'][()])], dtype="object"))

#######################LOAD_PARTICLE_INFORMATION_FROM_FILE_FOR_PREOCESSING#######################

#######################PLOTTING_BASE_WINDOW#######################
#Initialize certain dynamic parameters to plot the base MATPLOTLIB window

mv = 0
str_for_param_selection = 0
axis_to_use = 0
time_val_init = snap_array.min()
window_size_init = int(d['init_zoom_val'])
fig, ax = plt.subplots()
ax.cla()

rax = plt.axes(d['pos_check_button_for_showing_rings_and_bh'])
axtime = plt.axes(d['pos_slider_for_moving_through_snapshots'])
axzoom = plt.axes(d['pos_slider_for_zooming_in_out'])
axbinsize = plt.axes(d['pos_slider_for_changing_binsize_log'])
axtheta = plt.axes(d['pos_slider_for_theta_change'])
axphi = plt.axes(d['pos_slider_for_phi_change'])
rax7 = fig.add_axes(d['pos_radio_button_for_changing_parameters'])
rax9 = fig.add_axes(d['pos_radio_button_for_changing_colorbar_scale'])
movie_button_ax = plt.axes(d['pos_button_for_making_movie'])

#https://stackoverflow.com/questions/58937863/plot-average-of-scattered-values-in-2d-bins-as-a-histogram-hexplot
def plot_main(ax, x, y, z, data, statistic_val=np.nanmedian, bin_num_x=bin_number, bin_num_y=bin_number, plot_type='linear', phi=0., theta=0.):
    global im_cl
    x_turned, y_turned, z_turned = rotaion_matrix_group_new(x, y, z, phi, theta)
    x_bins = np.linspace(-window_size_init, window_size_init, int(bin_num_x))
    y_bins = np.linspace(-window_size_init, window_size_init, int(bin_num_y))
    ret = binned_statistic_2d(x_turned, y_turned, data, statistic=statistic_val, bins=[x_bins, y_bins])
    if ('log' in plot_type.lower()):
        im = ax.imshow(ret.statistic.T, origin='lower', cmap=colormap_new, norm=LogNorm(), zorder=1, extent=(-window_size_init, window_size_init, -window_size_init, window_size_init))
        im_cl = add_colorbar(im)
    else:
        im = ax.imshow(ret.statistic.T, origin='lower', cmap=colormap_new, zorder=1, extent=(-window_size_init, window_size_init, -window_size_init, window_size_init))
        im_cl = add_colorbar_lin(im)

def extra_plotting(ax, mv, subhalo_cen_x, subhalo_cen_y, subhalo_cen_z, subhalo_radius, x_bh, y_bh, z_bh, phi=0.0, theta=0.0):
    subhalo_x_turned, subhalo_y_turned, subhalo_z_turned = rotaion_matrix_group_new(subhalo_cen_x, subhalo_cen_y, subhalo_cen_z, phi, theta)
    circle2 = plt.Circle((subhalo_x_turned, subhalo_y_turned), subhalo_radius, color = str(d['vyg_ring_color']), fill=False, ls=str(d['vyg_ring_linestyle']), lw=int(d['vyg_ring_linewidth']), zorder=4)
    ax.add_artist(circle2)
    x_bh_turned, y_bh_turned, z_bh_turned = rotaion_matrix_group_new(x_bh, y_bh, z_bh, phi, theta)
    im2, = ax.plot(x_bh_turned, y_bh_turned, color=str(d['black_hole_color']), marker=str(d['black_hole_markerstyle']), ls="None", markersize=int(d['black_hole_markersize']), alpha=float(d['black_hole_plot_alpha']), zorder=2)
    return(im2, circle2)



def put_time_label(time_val, ax, window_size):
	snap_index = np.where(int(time_val)==snapshots_original)
	str_z = r'$z$=%2.2f' %float(redshift_original[snap_index])
	str_t = r'$t$=%2.2f' %float(cosmic_age_original[snap_index])
	ax.text(-window_size+10, window_size-10, str_z, fontsize = size_of_font/3, zorder=5)
	ax.text(-window_size+10, window_size-20, str_t, fontsize = size_of_font/3, zorder=5)
	ax.set_xlim(-window_size,window_size)
	ax.set_ylim(-window_size,window_size)
	ax.set_aspect('equal')
	ax.set_xlabel(xlabel_str)
	ax.set_ylabel(ylabel_str)
	plt.draw()

check = CheckButtons(rax, ('Black Hole', 'VYG Ring'), (True, True))
def bh_func(im2, circle2):
	def func(label):
		if label == 'Black Hole':
			im2.set_visible(not im2.get_visible())
		elif label == 'VYG Ring':
			circle2.set_visible(not circle2.get_visible())
		fig.canvas.draw_idle()
	check.on_clicked(func)

if (str_for_param_selection==2 or str_for_param_selection==9):
	plot_main(ax, data_x[mv][axis_to_use], data_y[mv][axis_to_use], data_z[mv][axis_to_use], data_real[mv][str_for_param_selection], statistic_val=np.nansum, plot_type='log')
else:
	plot_main(ax, data_x[mv][axis_to_use], data_y[mv][axis_to_use], data_z[mv][axis_to_use], data_real[mv][str_for_param_selection], plot_type='log')

im2, circle2 = extra_plotting(ax, mv, subhalo_cen_x[mv][vyg_counter[mv]], subhalo_cen_y[mv][vyg_counter[mv]], subhalo_cen_z[mv][vyg_counter[mv]], subhalo_radius[mv][vyg_counter[mv]], data_x[mv][2], data_y[mv][2], data_z[mv][2])
put_time_label(time_val_init, ax, window_size_init)
bh_func(im2, circle2)


#######################PLOTTING_BASE_WINDOW#######################

#######################UPDATE_THE_GUI#######################
#This function udpates the screen each time it is called, clearing out old plot and remaking it with updated values
#This is a standard matplotlib GUI based function

def update(val):
    global im_cl
    mv = int(int(np.floor(stime.val)) - snap_array.min()-1)
    str_for_param_selection = int(hzdict7[radio7.value_selected])
    axis_to_use = int(hzdict8[radio7.value_selected])
    colorbar_scaling = str(radio9.value_selected)
    binsize_updated = int(10**(float(sbinsize.val)))
    theta_updated = float(stheta.val)
    phi_updated = float(sphi.val)
    ax.cla()
    im_cl.remove()
    if (str_for_param_selection==2 or str_for_param_selection==9):
        plot_main(ax, data_x[mv][axis_to_use], data_y[mv][axis_to_use], data_z[mv][axis_to_use], data_real[mv][str_for_param_selection], plot_type=colorbar_scaling, bin_num_x=binsize_updated, bin_num_y=binsize_updated, phi=phi_updated, theta=theta_updated, statistic_val=np.nansum)
    else:
        plot_main(ax, data_x[mv][axis_to_use], data_y[mv][axis_to_use], data_z[mv][axis_to_use], data_real[mv][str_for_param_selection], plot_type=colorbar_scaling, bin_num_x=binsize_updated, bin_num_y=binsize_updated, phi=phi_updated, theta=theta_updated)

    im2, circle2 = extra_plotting(ax, mv, subhalo_cen_x[mv][vyg_counter[mv]], subhalo_cen_y[mv][vyg_counter[mv]], subhalo_cen_z[mv][vyg_counter[mv]], subhalo_radius[mv][vyg_counter[mv]], data_x[mv][2], data_y[mv][2], data_z[mv][2], phi=phi_updated, theta=theta_updated)
    put_time_label(int(stime.val), ax, float(szoom.val))
    bh_func(im2, circle2)
fig.canvas.draw_idle()

#######################UPDATE_THE_GUI#######################

#######################SLIDERS_FOR_DIFFERENT_OPTIONS#######################

# GUI Slider for moving through snapshots

stime = Slider(axtime, 'Snapshot', snap_array.min(), snap_array.max(), valinit=snap_array.min(), valfmt="%i")
stime.on_changed(update)

# GUI Slider for zooming in-out

szoom = Slider(axzoom, 'Zoom', int(d['min_zoom_allowed']), int(d['max_zoom_allowed']), valinit=int(d['init_zoom_val']), valfmt="%i")
szoom.on_changed(update)

# GUI Slider for changing binsize (in log)
init_binsize = np.log10(float(d['bins_for_hist']))
sbinsize = Slider(axbinsize, 'Binsize', float(d['min_binsize_allowed']), float(d['max_binsize_allowed']), valinit=init_binsize, valfmt="%2.2f")
sbinsize.on_changed(update)

# GUI Slider for changing theta

stheta = Slider(axtheta, 'Theta', float(d['min_theta_allowed']), float(d['max_theta_allowed']), valinit=float(d['default_theta']), valfmt="%2.2f")
stheta.on_changed(update)

# GUI Slider for changing theta

sphi = Slider(axphi, 'Phi', float(d['min_phi_allowed']), float(d['max_phi_allowed']), valinit=float(d['default_phi']), valfmt="%2.2f")
sphi.on_changed(update)

#######################SLIDERS_FOR_DIFFERENT_OPTIONS#######################


#######################RADIO_BUTTON_FOR_CHOOSING_DIFFERENT_COLORBAR_SCALE#######################

radio9 = RadioButtons(rax9, ('Linear', 'Log'), active=1)
def hzfunc9(label9):
	global im_cl
	mv = int(int(np.floor(stime.val)) - snap_array.min()-1)
	str_for_param_selection = int(hzdict7[radio7.value_selected])
	axis_to_use = int(hzdict8[radio7.value_selected])
	colorbar_scaling = str(radio9.value_selected)
	binsize_updated = int(10**(float(sbinsize.val)))
	theta_updated = float(stheta.val)
	phi_updated = float(sphi.val)
	ax.cla()
	im_cl.remove()
	if (str_for_param_selection==2 or str_for_param_selection==9):
		plot_main(ax, data_x[mv][axis_to_use], data_y[mv][axis_to_use], data_z[mv][axis_to_use], data_real[mv][str_for_param_selection], plot_type=colorbar_scaling, bin_num_x=binsize_updated, bin_num_y=binsize_updated, phi=phi_updated, theta=theta_updated, statistic_val=np.nansum)
	else:
		plot_main(ax, data_x[mv][axis_to_use], data_y[mv][axis_to_use], data_z[mv][axis_to_use], data_real[mv][str_for_param_selection], plot_type=colorbar_scaling, bin_num_x=binsize_updated, bin_num_y=binsize_updated, phi=phi_updated, theta=theta_updated)

	im2, circle2 = extra_plotting(ax, mv, subhalo_cen_x[mv][vyg_counter[mv]], subhalo_cen_y[mv][vyg_counter[mv]], subhalo_cen_z[mv][vyg_counter[mv]], subhalo_radius[mv][vyg_counter[mv]], data_x[mv][2], data_y[mv][2], data_z[mv][2], phi=phi_updated, theta=theta_updated)
	put_time_label(int(stime.val), ax, float(szoom.val))
	bh_func(im2, circle2)
radio9.on_clicked(hzfunc9)

#######################RADIO_BUTTON_FOR_CHOOSING_DIFFERENT_COLORBAR_SCALE#######################


#######################RADIO_BUTTON_FOR_CHOOSING_DIFFERENT_PARTICLE_INFORMATION#######################
# GUI Radio button for moving throught different particle information like Gas Density,-
#-Temperature, Stellar density, Gas velocity, stellar velocity, Gas radial velocity, Stellar radial velocity

radio7 = RadioButtons(rax7, ('gas_dens', 'temp', 'stars', 'gas_vel', 'star_vel', 'vgas_rad', 'vstar_rad', 'mass_dm', 'dm_vel', 'stellar_age', 'vyg_st_mass', 'vyg_fraction'))
def hzfunc7(label7):
	global im_cl
	mv = int(int(np.floor(stime.val)) - snap_array.min()-1)
	str_for_param_selection = int(hzdict7[label7])
	axis_to_use = int(hzdict8[radio7.value_selected])
	colorbar_scaling = str(radio9.value_selected)
	binsize_updated = int(10**(float(sbinsize.val)))
	theta_updated = float(stheta.val)
	phi_updated = float(sphi.val)
	ax.cla()
	im_cl.remove()
	if (str_for_param_selection==2 or str_for_param_selection==9):
		plot_main(ax, data_x[mv][axis_to_use], data_y[mv][axis_to_use], data_z[mv][axis_to_use], data_real[mv][str_for_param_selection], plot_type=colorbar_scaling, bin_num_x=binsize_updated, bin_num_y=binsize_updated, phi=phi_updated, theta=theta_updated, statistic_val=np.nansum)
	else:
		plot_main(ax, data_x[mv][axis_to_use], data_y[mv][axis_to_use], data_z[mv][axis_to_use], data_real[mv][str_for_param_selection], plot_type=colorbar_scaling, bin_num_x=binsize_updated, bin_num_y=binsize_updated, phi=phi_updated, theta=theta_updated)

	im2, circle2 = extra_plotting(ax, mv, subhalo_cen_x[mv][vyg_counter[mv]], subhalo_cen_y[mv][vyg_counter[mv]], subhalo_cen_z[mv][vyg_counter[mv]], subhalo_radius[mv][vyg_counter[mv]], data_x[mv][2], data_y[mv][2], data_z[mv][2], phi=phi_updated, theta=theta_updated)
	put_time_label(int(stime.val), ax, float(szoom.val))
	bh_func(im2, circle2)
radio7.on_clicked(hzfunc7)

#######################RADIO_BUTTON_FOR_CHOOSING_DIFFERENT_PARTICLE_INFORMATION#######################



#######################BUTTON_TO_CREATE_A_MOVIE#######################

#https://stackoverflow.com/questions/58937863/plot-average-of-scattered-values-in-2d-bins-as-a-histogram-hexplot
def plot_main_movie(ax_new, x, y, z, data, statistic_val=np.nanmedian, bin_num_x=bin_number, bin_num_y=bin_number, plot_type='linear', phi=0., theta=0.):
	global im_cl_new
	x_turned, y_turned, z_turned = rotaion_matrix_group_new(x, y, z, phi, theta)
	x_bins = np.linspace(-window_size_init, window_size_init, int(bin_num_x))
	y_bins = np.linspace(-window_size_init, window_size_init, int(bin_num_y))
	ret = binned_statistic_2d(x_turned, y_turned, data, statistic=statistic_val, bins=[x_bins, y_bins])
	if ('log' in plot_type.lower()):
		im_new = ax_new.imshow(ret.statistic.T, origin='lower', cmap=colormap_new, norm=LogNorm(), zorder=1, extent=(-window_size_init, window_size_init, -window_size_init, window_size_init))
		im_cl_new = add_colorbar(im_new)
	else:
		im_new = ax_new.imshow(ret.statistic.T, origin='lower', cmap=colormap_new, zorder=1, extent=(-window_size_init, window_size_init, -window_size_init, window_size_init))
		im_cl_new = add_colorbar_lin(im_new)
	return im_new


def ani_frame():
	global im_cl_new
	dpi = int(d['movie_dpi'])
	fig_new = plt.figure(2)
	ax_new = fig_new.add_subplot(111)
	mv_new = 0
	str_for_param_selection = int(hzdict7[radio7.value_selected])
	axis_to_use = int(hzdict8[radio7.value_selected])
	colorbar_scaling = str(radio9.value_selected)
	binsize_updated = int(10**(float(sbinsize.val)))
	theta_updated = float(stheta.val)
	phi_updated = float(sphi.val)
	movie_name_count=1
	movie_name = str('movie_') + str(radio7.value_selected) + str('_zoom_') + str(int(szoom.val)) + str('_count_') + str(movie_name_count) + str('.mp4')
	while path.exists(movie_name):
		movie_name_count+=1
		movie_name = str('movie_') + str(radio7.value_selected) + str('_zoom_') + str(int(szoom.val)) + str('_count_') + str(movie_name_count) + str('.mp4')

	if (str_for_param_selection==2 or str_for_param_selection==9):
		im_new = plot_main_movie(ax_new, data_x[mv_new][axis_to_use], data_y[mv_new][axis_to_use], data_z[mv_new][axis_to_use], data_real[mv_new][str_for_param_selection], plot_type=colorbar_scaling, bin_num_x=binsize_updated, bin_num_y=binsize_updated, phi=phi_updated, theta=theta_updated, statistic_val=np.nansum)
	else:
		im_new = plot_main_movie(ax_new, data_x[mv_new][axis_to_use], data_y[mv_new][axis_to_use], data_z[mv_new][axis_to_use], data_real[mv_new][str_for_param_selection], plot_type=colorbar_scaling, bin_num_x=binsize_updated, bin_num_y=binsize_updated, phi=phi_updated, theta=theta_updated)

	im2_new, circle2_new = extra_plotting(ax_new, mv_new, subhalo_cen_x[mv_new][vyg_counter[mv_new]], subhalo_cen_y[mv_new][vyg_counter[mv_new]], subhalo_cen_z[mv_new][vyg_counter[mv_new]], subhalo_radius[mv_new][vyg_counter[mv_new]], data_x[mv_new][2], data_y[mv_new][2], data_z[mv_new][2], phi=phi_updated, theta=theta_updated)
	put_time_label(int(stime.val), ax_new, float(szoom.val))
	bh_func(im2_new, circle2_new)
	fig_new.set_size_inches([5,5])
	title_string = str(d['simulation']) + str('_') + str(d['snapshot']) + str('_') + str(d['galaxyid'])
	fig_new.suptitle(title_string, fontsize=size_of_font/3)
	tight_layout()

	def update_img(n):
		global im_cl_new
		mv_new = int(n - snap_array.min())
		ax_new.cla()
		im_cl_new.remove()

		if (str_for_param_selection==2 or str_for_param_selection==9):
			im_new = plot_main_movie(ax_new, data_x[mv_new][axis_to_use], data_y[mv_new][axis_to_use], data_z[mv_new][axis_to_use], data_real[mv_new][str_for_param_selection], plot_type=colorbar_scaling, bin_num_x=binsize_updated, bin_num_y=binsize_updated, phi=phi_updated, theta=theta_updated, statistic_val=np.nansum)
		else:
			im_new = plot_main_movie(ax_new, data_x[mv_new][axis_to_use], data_y[mv_new][axis_to_use], data_z[mv_new][axis_to_use], data_real[mv_new][str_for_param_selection], plot_type=colorbar_scaling, bin_num_x=binsize_updated, bin_num_y=binsize_updated, phi=phi_updated, theta=theta_updated)

		im2_new, circle2_new = extra_plotting(ax_new, mv_new, subhalo_cen_x[mv_new][vyg_counter[mv_new]], subhalo_cen_y[mv_new][vyg_counter[mv_new]], subhalo_cen_z[mv_new][vyg_counter[mv_new]], subhalo_radius[mv_new][vyg_counter[mv_new]], data_x[mv_new][2], data_y[mv_new][2], data_z[mv_new][2], phi=phi_updated, theta=theta_updated)
		put_time_label(int(n), ax_new, float(szoom.val))
		bh_func(im2_new, circle2_new)
		return im_new,

	frame_list = np.arange(int(snap_array.min()), int(snap_array.max())+1, 1)
	ani = animation.FuncAnimation(fig_new,update_img,frames=frame_list, interval=int(d['movie_interval']))
	writer = animation.writers['ffmpeg'](fps=int(d['movie_fps']))
	ani.save(movie_name, writer=writer,dpi=dpi)
	plt.close(fig_new)
	return ani


movie_button = Button(movie_button_ax, 'Make Movie')
def make_movie(event):
	print_cust('Making movie...')
	ani_frame()
	print_cust('Movie saved...')
movie_button.on_clicked(make_movie)

#######################BUTTON_TO_CREATE_A_MOVIE#######################




#######################SETTING 'X' AND 'Y' DIRECTION LIMITS AND LABELS#######################


ax.set_xlabel(xlabel_str)
ax.set_ylabel(ylabel_str)
ax.set_xlim(-window_size_init,window_size_init)
ax.set_ylim(-window_size_init,window_size_init)
ax.set_aspect('equal')
title_string = str(d['simulation']) + str('_') + str(d['snapshot']) + str('_') + str(d['galaxyid'])
fig.suptitle(title_string)
plt.show()

#######################SETTING 'X' AND 'Y' DIRECTION LIMITS AND LABELS#######################


#######################DELETE_ALL_CUTOUTS#######################
#find a file ends with .txt
x = os.listdir()
for i in x:
	if i.endswith(".hdf5"):
		print_cust("deleting..., {i}")
		string_for_deleting_all_cutouts = str('rm ') + i
		print_cust(f"{string_for_deleting_all_cutouts}")
		os.system(string_for_deleting_all_cutouts)

#######################DELETE_ALL_CUTOUTS#######################


















