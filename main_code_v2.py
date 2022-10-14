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


snapshots_init_def = np.array([2, 3, 4, 6, 8, 11, 13, 17, 21, 25, 33, 40, 50, 59, 67, 72, 78, 84, 91, 99])
scale_factor_init_def = np.array([0.0769, 0.0833, 0.0909, 0.1, 0.1111, 0.125, 0.1429, 0.1667, 0.2, 0.25, 0.3333, 0.4, 0.5, 0.5882, 0.6667, 0.7143, 0.7692, 0.8333, 0.9091, 1])
redshift_init_def = np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.5, 1, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0])
cs_scale_factor = CubicSpline(snapshots_init_def, scale_factor_init_def)
cs_redshift = CubicSpline(snapshots_init_def, redshift_init_def)
snapshots_original = np.arange(snapshots_init_def.min(), snapshots_init_def.max()+1, 1, dtype=np.int)
scale_factor_original = cs_scale_factor(snapshots_original)
redshift_original = cs_redshift(snapshots_original)
cosmic_age_original = cosmo.age(redshift_original).value

print ("Running script...")
print (sys.argv[0])
print ("\n")
print ("Current Working Directory:")
print (sys.argv[1])
print ("\n")
print ("Base Working Directory:")
print (sys.argv[2])
print ("\n")
sys.stdout.flush()
time.sleep(0.1)

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

logo1=image.imread(str(sys.argv[2]) + '/nrf_logo_1.png')
logo2=image.imread(str(sys.argv[2]) + '/kasi_1.png')
logo3=image.imread(str(sys.argv[2]) + '/kasi_2.png')
addLogo1 = OffsetImage(logo1, zoom=0.13*1.3)
addLogo2 = OffsetImage(logo2, zoom=0.2*1.3)
addLogo3 = OffsetImage(logo3, zoom=0.2*1.3)

if (len(sys.argv)!=4):
	print ("No parameter file given along command line. Searching current directory for parameter file.")
	print ("\n")
	if (os.path.isfile(parameter_file_string_current)):
		print ("Parameter file found in the current directory.", str(parameter_file_string_current))
		print ("\n")
		parameter_file_name_final = parameter_file_string_current
	else:
		print ("No parameter file (with default name - initial_parameters.dat) found in the current directory.")
		print ("\n")
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
	
print ("Executing script with data from: \n", str(parameter_file_name_final))
print ("\n")
sys.stdout.flush()
time.sleep(0.1)

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
	cax1 = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(mappable, cax=cax1)
	cbar.set_ticks(ticker.LogLocator(), update_ticks=True)
	cbar.ax.tick_params(size=0)
	return cbar

def add_colorbar_lin(mappable):
	last_axes = plt.gca()
	ax = mappable.axes
	fig = ax.figure
	divider = make_axes_locatable(ax)
	cax1 = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(mappable, cax=cax1)
	cbar.set_ticks(ticker.LinearLocator(), update_ticks=True)
	cbar.ax.tick_params(size=0)
	return cbar

#######################CUSTOM_CREATION_OF_COLORBAR_IN_SUBPLOTS_OF_MATPLOTLIB#######################

#######################GET_FUNCTION_TO_OBTAIN_DATA_FROM_TNG#######################
#This function is required for obtaining data from TNG online
#See https://www.tng-project.org/data/docs/api/ for more details

def get(path, params=None):
	# make HTTP GET request to path
	r = requests.get(path, params=params, headers=headers)

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

def retrieve_particle_data(url_for_data, url_for_central_galaxy):
	sub_prog_subhalo_central_galaxy = get(url_for_central_galaxy)
	#main_data = get(url_for_data)
	#cutout_request = {'gas':'Coordinates,Masses,InternalEnergy,ElectronAbundance,Velocities', 'stars':'Coordinates,Masses,Velocities', 'dm':'Coordinates,Masses,Velocities'}

	sub_prog_subhalo_data = get(url_for_data)
	center_subhalo_x = sub_prog_subhalo_data['pos_x'] - sub_prog_subhalo_central_galaxy['pos_x']
	center_subhalo_y = sub_prog_subhalo_data['pos_y'] - sub_prog_subhalo_central_galaxy['pos_y']
	radius_subhalo = sub_prog_subhalo_data['halfmassrad']
	cutout_request = {'gas':'Coordinates,Masses,InternalEnergy,ElectronAbundance,Velocities', 'stars':'Coordinates,Masses,Velocities,GFM_StellarFormationTime', 'dm':'Coordinates,Velocities', 'bhs':'Coordinates,Masses,Velocities'}
	cutout = get(url_for_data+"cutout.hdf5", cutout_request)
	x, y, dens, ie, ea, vel_gas_x, vel_gas_y, vel_gas_z, x2, y2, stars, stellar_age_rev, vel_star_x, vel_star_y, vel_star_z, x3, y3, x4, y4, dm_mass, vel_dm_x, vel_dm_y, vel_dm_z = ([] for i in range(23))
	with h5py.File(cutout,'r') as f:
		catch_error_1 = f.get('PartType0')
		catch_error_2 = f.get('PartType4')
		catch_error_3 = f.get('PartType5')
		catch_error_4 = f.get('PartType1')
		
		if (catch_error_1):
			x = f['PartType0']['Coordinates'][:,0] - sub_prog_subhalo_central_galaxy['pos_x']
			y = f['PartType0']['Coordinates'][:,1] - sub_prog_subhalo_central_galaxy['pos_y']
			dens = f['PartType0']['Masses'][:]*1e10
			vel_gas = f['PartType0']['Velocities'][:]
			print (vel_gas.min(), vel_gas.max())
			ie = f['PartType0']['InternalEnergy'][:]
			ea = f['PartType0']['ElectronAbundance'][:]
			vel_gas_x = np.zeros([vel_gas.shape[0]])
			vel_gas_y = np.zeros([vel_gas.shape[0]])
			vel_gas_z = np.zeros([vel_gas.shape[0]])
			for i in range(vel_gas.shape[0]):
				vel_gas_x[i] = vel_gas[i][0] - sub_prog_subhalo_central_galaxy['vel_x']
				vel_gas_y[i] = vel_gas[i][1] - sub_prog_subhalo_central_galaxy['vel_y']
				vel_gas_z[i] = vel_gas[i][2] - sub_prog_subhalo_central_galaxy['vel_z']

			mask_1 = (np.array(x)>-vel_limit) & (np.array(x)<vel_limit) & (np.array(y)>-vel_limit) & (np.array(y)<vel_limit)
			x = x[mask_1]
			y = y[mask_1]
			dens = dens[mask_1]
			ie = ie[mask_1]
			ea = ea[mask_1]
			vel_gas_x = vel_gas_x[mask_1]
			vel_gas_y = vel_gas_y[mask_1]
			vel_gas_z = vel_gas_z[mask_1]

		if (catch_error_2):
			x2 = f['PartType4']['Coordinates'][:,0] - sub_prog_subhalo_central_galaxy['pos_x']
			y2 = f['PartType4']['Coordinates'][:,1] - sub_prog_subhalo_central_galaxy['pos_y']
			stars = f['PartType4']['Masses'][:]*1e10
			stellar_age = f['PartType4']['GFM_StellarFormationTime'][:]
			#print (stellar_age.min(), stellar_age.max())
			stellar_age_idx = np.searchsorted(scale_factor_original, stellar_age)
			stellar_age_rev = cosmic_age_original[stellar_age_idx]
			#stellar_age_rev = stellar_age
			#print (stellar_age_rev.min(), stellar_age_rev.max())
			vel_star = f['PartType4']['Velocities'][:]
			vel_star_x = np.zeros([vel_star.shape[0]])
			vel_star_y = np.zeros([vel_star.shape[0]])
			vel_star_z = np.zeros([vel_star.shape[0]])
			for i in range(vel_star.shape[0]):
				vel_star_x[i] = vel_star[i][0] - sub_prog_subhalo_central_galaxy['vel_x']
				vel_star_y[i] = vel_star[i][1] - sub_prog_subhalo_central_galaxy['vel_y']
				vel_star_z[i] = vel_star[i][2] - sub_prog_subhalo_central_galaxy['vel_z']
				
			mask_2 = (np.array(x2)>-vel_limit) & (np.array(x2)<vel_limit) & (np.array(y2)>-vel_limit) & (np.array(y2)<vel_limit)
			x2 = x2[mask_2]
			y2 = y2[mask_2]
			stars = stars[mask_2]
			stellar_age = stellar_age[mask_2]
			vel_star_x = vel_star_x[mask_2]
			vel_star_y = vel_star_y[mask_2]
			vel_star_z = vel_star_z[mask_2]
			
		if (catch_error_3):
			print ('BH Detected')
			sys.stdout.flush()
			time.sleep(0.1)
			x3 = f['PartType5']['Coordinates'][:,0] - sub_prog_subhalo_central_galaxy['pos_x']
			y3 = f['PartType5']['Coordinates'][:,1] - sub_prog_subhalo_central_galaxy['pos_y']
			mask_3 = (np.array(x3)>-vel_limit) & (np.array(x3)<vel_limit) & (np.array(y3)>-vel_limit) & (np.array(y3)<vel_limit)
			x3 = x3[mask_3]
			y3 = y3[mask_3]
			
		if (catch_error_4):
			x4 = f['PartType1']['Coordinates'][:,0] - sub_prog_subhalo_central_galaxy['pos_x']
			y4 = f['PartType1']['Coordinates'][:,1] - sub_prog_subhalo_central_galaxy['pos_y']
			#dm_mass = f['PartType1']['Masses'][:]*1e10
			dm_mass = np.zeros_like(x4)
			vel_dm = f['PartType1']['Velocities'][:]
			vel_dm_x = np.zeros([vel_dm.shape[0]])
			vel_dm_y = np.zeros([vel_dm.shape[0]])
			vel_dm_z = np.zeros([vel_dm.shape[0]])
			for i in range(vel_dm.shape[0]):
				vel_dm_x[i] = vel_dm[i][0] - sub_prog_subhalo_central_galaxy['vel_x']
				vel_dm_y[i] = vel_dm[i][1] - sub_prog_subhalo_central_galaxy['vel_y']
				vel_dm_z[i] = vel_dm[i][2] - sub_prog_subhalo_central_galaxy['vel_z']

			mask_4 = (np.array(x4)>-vel_limit) & (np.array(x4)<vel_limit) & (np.array(y4)>-vel_limit) & (np.array(y4)<vel_limit)
			x4 = x4[mask_4]
			y4 = y4[mask_4]
			dm_mass = dm_mass[mask_4]
			vel_dm_x = vel_dm_x[mask_4]
			vel_dm_y = vel_dm_y[mask_4]
			vel_dm_z = vel_dm_z[mask_4]

	return (x, y, dens, ie, ea, vel_gas_x, vel_gas_y, vel_gas_z, x2, y2, stars, vel_star_x, vel_star_y, vel_star_z, x3, y3, x4, y4, dm_mass, vel_dm_x, vel_dm_y, vel_dm_z, center_subhalo_x, center_subhalo_y, radius_subhalo, stellar_age_rev)


def particle_information_child_subhalos(sub_prog_url_cust):
	print ('\n')
	print ('Main VYG subhalo - ', sub_prog_url_cust)
	sys.stdout.flush()
	time.sleep(0.1)
	sub_prog_temp = get(sub_prog_url_cust)
	sub_prog_halo_link = sub_prog_temp['related']['parent_halo']
	print ('Main Parent Halo - ', sub_prog_halo_link)
	sub_prog_halo = get(sub_prog_halo_link)
	counter_for_subhalos = int(sub_prog_halo['child_subhalos']['count'])
	
	x_tot, y_tot, dens_tot, ie_tot, ea_tot, vel_gas_x_tot, vel_gas_y_tot, vel_gas_z_tot, x2_tot, y2_tot, stars_tot, vel_star_x_tot, vel_star_y_tot, vel_star_z_tot, x3_tot, y3_tot, x4_tot, y4_tot, dm_tot, vel_dm_x_tot, vel_dm_y_tot, vel_dm_z_tot, cen_subhalo_x_tot, cen_subhalo_y_tot, rad_subhalo_tot, st_age_tot = (np.array([]) for i in range(26))

	test = sub_prog_halo['child_subhalos']['results']
	print ('Sub Halo list - ')
	sys.stdout.flush()
	time.sleep(0.1)
	for i in range(int(d['child_subhalo_count'])):
		print (test[i]['url'])
		sys.stdout.flush()
		time.sleep(0.1)
		if (sub_prog_url_cust==test[i]['url']):
			counter_for_identifying_vyg_child_halo = int(i)
			
		x, y, dens, ie, ea, vel_gas_x, vel_gas_y, vel_gas_z, x2, y2, stars, vel_star_x, vel_star_y, vel_star_z, x_bh, y_bh, x_dm, y_dm, dm_mass, vel_dm_x, vel_dm_y, vel_dm_z, cen_subhalo_x, cen_subhalo_y, rad_subhalo, st_age = retrieve_particle_data(test[i]['url'], sub_prog_url_cust)

		x_tot = np.append(x_tot, x)
		y_tot = np.append(y_tot, y)
		dens_tot = np.append(dens_tot, dens)
		ie_tot = np.append(ie_tot, ie)
		ea_tot = np.append(ea_tot, ea)
		vel_gas_x_tot = np.append(vel_gas_x_tot, vel_gas_x)
		vel_gas_y_tot = np.append(vel_gas_y_tot, vel_gas_y)
		vel_gas_z_tot = np.append(vel_gas_z_tot, vel_gas_z)
		x2_tot = np.append(x2_tot, x2)
		y2_tot = np.append(y2_tot, y2)
		stars_tot = np.append(stars_tot, stars)
		vel_star_x_tot = np.append(vel_star_x_tot, vel_star_x)
		vel_star_y_tot = np.append(vel_star_y_tot, vel_star_y)
		vel_star_z_tot = np.append(vel_star_z_tot, vel_star_z)
		x3_tot = np.append(x3_tot, x_bh)
		y3_tot = np.append(y3_tot, y_bh)
		x4_tot = np.append(x4_tot, x_dm)
		y4_tot = np.append(y4_tot, y_dm)
		dm_tot = np.append(dm_tot, dm_mass)
		vel_dm_x_tot = np.append(vel_dm_x_tot, vel_dm_x)
		vel_dm_y_tot = np.append(vel_dm_y_tot, vel_dm_y)
		vel_dm_z_tot = np.append(vel_dm_z_tot, vel_dm_z)
		cen_subhalo_x_tot = np.append(cen_subhalo_x_tot, cen_subhalo_x)
		cen_subhalo_y_tot = np.append(cen_subhalo_y_tot, cen_subhalo_y)
		rad_subhalo_tot = np.append(rad_subhalo_tot, rad_subhalo)
		st_age_tot = np.append(st_age_tot, st_age)

	print ('Total subhalos - ', len(test))
	sys.stdout.flush()
	time.sleep(0.1)
	try:
		counter_for_identifying_vyg_child_halo
	except NameError:
		counter_for_identifying_vyg_child_halo = 0

	return (x_tot, y_tot, dens_tot, ie_tot, ea_tot, vel_gas_x_tot, vel_gas_y_tot, vel_gas_z_tot, x2_tot, y2_tot, stars_tot, vel_star_x_tot, vel_star_y_tot, vel_star_z_tot, x3_tot, y3_tot, x4_tot, y4_tot, dm_tot, vel_dm_x_tot, vel_dm_y_tot, vel_dm_z_tot, cen_subhalo_x_tot, cen_subhalo_y_tot, rad_subhalo_tot, st_age_tot, counter_for_identifying_vyg_child_halo)

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
print (d)
sys.stdout.flush()
time.sleep(0.1)

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

hzdict7 = {'gas_density': 0, 'temp': 1, 'stars': 2, 'gas_vel': 3, 'star_vel': 4, 'vgas_rad': 5, 'vstar_rad': 6, 'mass_dm': 7, 'dm_vel': 8, 'stellar_age': 9, 'vyg_st_mass': 10}
hzdict8 = {'gas_density': 0, 'temp': 0, 'stars': 1, 'gas_vel': 0, 'star_vel': 1, 'vgas_rad': 0, 'vstar_rad': 1, 'mass_dm': 3, 'dm_vel': 3, 'stellar_age': 1, 'vyg_st_mass': 1}



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

print (sub_prog['snap'])
print (sub_prog['meta']['url'], '\n')
sys.stdout.flush()
time.sleep(0.1)
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

dir_name = str2 + "_" + str4_rev + "_" + str6_rev + "_v2"
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
	# request the full subhalo details of the progenitor by following the sublink URL
	if (sub_prog['snap'] >= int(d['first_snapshot_number'])+1):
		sub_new = (sub_prog['related']['sublink_progenitor'])
		sys.stdout.flush()
		time.sleep(0.1)
		sub_prog_url_array = np.append(sub_prog_url_array, sub_new.encode('utf-8'))
		sub_prog = get(sub_new)
	else:
		break

print ('\n')
sys.stdout.flush()
time.sleep(0.1)
#######################FIND_DATA_FROM_ALL_RELEVANT_SNAPSHOTS#######################

#######################GET_PARTICLE_DATA_AND_SAVE_IT_IN_A_FILE#######################
#Get the particle information (gas position, gas density, Temperature, velocity,-
#-star position, density and velocity) for a cutout of a specific subhalo for-
#-all the relevant snapshots requested by the user

sub_prog_url_array = np.delete(sub_prog_url_array, 0)
filename_x = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename_y = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename_x2 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename_y2 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename_x3 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename_y3 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename_x4 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename_y4 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename1 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename2 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename3 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename4 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename5 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename6 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename7 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename8 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename9 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename10 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename11 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename12 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename13 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename14 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename15 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename16 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename17 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename19 = np.chararray([len(sub_prog_url_array)], itemsize=1000)
filename18 = np.chararray([len(sub_prog_url_array)], itemsize=1000)

linkname = np.chararray([len(sub_prog_url_array)], itemsize=1000)

for i in range(len(sub_prog_url_array)):
	filename_x[i] = (str(str_dir_name) + "pos_x_" + str(i) + ".npy")
	filename_y[i] = (str(str_dir_name) + "_y_" + str(i) + ".npy")
	filename_x2[i] = (str(str_dir_name) + "_x2_" + str(i) + ".npy")
	filename_y2[i] = (str(str_dir_name) + "_y2_" + str(i) + ".npy")
	filename_x3[i] = (str(str_dir_name) + "_x3_" + str(i) + ".npy")
	filename_y3[i] = (str(str_dir_name) + "_y3_" + str(i) + ".npy")
	filename_x4[i] = (str(str_dir_name) + "_x4_" + str(i) + ".npy")
	filename_y4[i] = (str(str_dir_name) + "_y4_" + str(i) + ".npy")
	filename1[i] = (str(str_dir_name) + "_gas_density_" + str(i) + ".npy")
	filename2[i] = (str(str_dir_name) + "_gas_ie_" + str(i) + ".npy")
	filename3[i] = (str(str_dir_name) + "_gas_ea_" + str(i) + ".npy")
	filename4[i] = (str(str_dir_name) + "_stellar_density_" + str(i) + ".npy")
	filename5[i] = (str(str_dir_name) + "_vel_gas_x_" + str(i) + ".npy")
	filename6[i] = (str(str_dir_name) + "_vel_gas_y_" + str(i) + ".npy")
	filename7[i] = (str(str_dir_name) + "_vel_gas_z_" + str(i) + ".npy")
	filename8[i] = (str(str_dir_name) + "_vel_star_x_" + str(i) + ".npy")
	filename9[i] = (str(str_dir_name) + "_vel_star_y_" + str(i) + ".npy")
	filename10[i] = (str(str_dir_name) + "_vel_star_z_" + str(i) + ".npy")
	filename11[i] = (str(str_dir_name) + "_dm_mass_" + str(i) + ".npy")
	filename12[i] = (str(str_dir_name) + "_vel_dm_x_" + str(i) + ".npy")
	filename13[i] = (str(str_dir_name) + "_vel_dm_y_" + str(i) + ".npy")
	filename14[i] = (str(str_dir_name) + "_vel_dm_z_" + str(i) + ".npy")
	filename15[i] = (str(str_dir_name) + "_center_subhalo_x_" + str(i) + ".npy")
	filename16[i] = (str(str_dir_name) + "_center_subhalo_y_" + str(i) + ".npy")
	filename17[i] = (str(str_dir_name) + "_radius_subhalo_" + str(i) + ".npy")
	filename19[i] = (str(str_dir_name) + "_stellar_age_" + str(i) + ".npy")
	filename18[i] = (str(str_dir_name) + "_vyg_counter_" + str(i) + ".npy")

	if (path.exists(filename_x[i])):
		print ("File exists")
		sys.stdout.flush()
		time.sleep(0.1)
	else:
		linkname[i] = str(sub_prog_url_array[i].decode("utf-8"))
		print (linkname[i].decode("utf-8"))
		sys.stdout.flush()
		time.sleep(0.1)
		x, y, dens, ie, ea, vel_gas_x, vel_gas_y, vel_gas_z, x2, y2, stars, vel_star_x, vel_star_y, vel_star_z, x3, y3, x4, y4, dm_mass, vel_dm_x, vel_dm_y, vel_dm_z, subhalo_cen_x, subhalo_cen_y, subhalo_rad, stars_age, vyg_counter_s = particle_information_child_subhalos(linkname[i].decode("utf-8"))
		
		np.save(filename_x[i].decode("utf-8"), x)
		np.save(filename_y[i].decode("utf-8"), y)
		np.save(filename_x2[i].decode("utf-8"), x2)
		np.save(filename_y2[i].decode("utf-8"), y2)
		np.save(filename_x3[i].decode("utf-8"), x3)
		np.save(filename_y3[i].decode("utf-8"), y3)
		np.save(filename_x4[i].decode("utf-8"), x4)
		np.save(filename_y4[i].decode("utf-8"), y4)
		np.save(filename1[i].decode("utf-8"), dens)
		np.save(filename2[i].decode("utf-8"), ie)
		np.save(filename3[i].decode("utf-8"), ea)
		np.save(filename4[i].decode("utf-8"), stars)
		np.save(filename5[i].decode("utf-8"), vel_gas_x)
		np.save(filename6[i].decode("utf-8"), vel_gas_y)
		np.save(filename7[i].decode("utf-8"), vel_gas_z)
		np.save(filename8[i].decode("utf-8"), vel_star_x)
		np.save(filename9[i].decode("utf-8"), vel_star_y)
		np.save(filename10[i].decode("utf-8"), vel_star_z)
		np.save(filename11[i].decode("utf-8"), dm_mass)
		np.save(filename12[i].decode("utf-8"), vel_dm_x)
		np.save(filename13[i].decode("utf-8"), vel_dm_y)
		np.save(filename14[i].decode("utf-8"), vel_dm_z)
		np.save(filename15[i].decode("utf-8"), subhalo_cen_x)
		np.save(filename16[i].decode("utf-8"), subhalo_cen_y)
		np.save(filename17[i].decode("utf-8"), subhalo_rad)
		np.save(filename19[i].decode("utf-8"), stars_age)
		np.save(filename18[i].decode("utf-8"), vyg_counter_s)

#######################GET_PARTICLE_DATA_AND_SAVE_IT_IN_A_FILE#######################

#######################LOAD_PARTICLE_INFORMATION_FROM_FILE_FOR_PREOCESSING#######################
#All particle data loaded from the file for dynamic plotting

data_x = []
subhalo_cen_x = []
data_y = []
subhalo_cen_y = []
data_real = []
subhalo_radius = []
stelar_mass = []
stelar_age = []
stelar_age_rev = []
vyg_counter = []
vyg_st_mass = []
str_for_param_selection = 0

for i in range(0,len(sub_prog_url_array)):
	data_x.append(np.array([(np.load(filename_x[i].decode("utf-8"), allow_pickle=True)), (np.load(filename_x2[i].decode("utf-8"), allow_pickle=True)), (np.load(filename_x3[i].decode("utf-8"), allow_pickle=True)), (np.load(filename_x4[i].decode("utf-8"), allow_pickle=True))], dtype="object"))
	data_y.append(np.array([(np.load(filename_y[i].decode("utf-8"), allow_pickle=True)), (np.load(filename_y2[i].decode("utf-8"), allow_pickle=True)), (np.load(filename_y3[i].decode("utf-8"), allow_pickle=True)), (np.load(filename_y4[i].decode("utf-8"), allow_pickle=True))], dtype="object"))

	ie = (np.load(filename2[i].decode("utf-8"), allow_pickle=True))
	ea = (np.load(filename3[i].decode("utf-8"), allow_pickle=True))
	temp = temperature_estimation(ie, ea)
	
	vel_gas_x = (np.load(filename5[i].decode("utf-8"), allow_pickle=True))
	vel_gas_y = (np.load(filename6[i].decode("utf-8"), allow_pickle=True))
	vel_gas_z = (np.load(filename7[i].decode("utf-8"), allow_pickle=True))
	vel_star_x = (np.load(filename8[i].decode("utf-8"), allow_pickle=True))
	vel_star_y = (np.load(filename9[i].decode("utf-8"), allow_pickle=True))
	vel_star_z = (np.load(filename10[i].decode("utf-8"), allow_pickle=True))
	gas_kinematics = np.sqrt(vel_gas_x**2 + vel_gas_y**2 + vel_gas_z**2)
	radial_vel_gas = (np.load(filename7[i].decode("utf-8"), allow_pickle=True))
	stellar_kinematics = np.sqrt(vel_star_x**2 + vel_star_y**2 + vel_star_z**2)
	radial_vel_star = (np.load(filename10[i].decode("utf-8"), allow_pickle=True))
	mass_dm = (np.load(filename11[i].decode("utf-8"), allow_pickle=True))
	vel_dm_x = (np.load(filename12[i].decode("utf-8"), allow_pickle=True))
	vel_dm_y = (np.load(filename13[i].decode("utf-8"), allow_pickle=True))
	vel_dm_z = (np.load(filename14[i].decode("utf-8"), allow_pickle=True))
	dm_kinematics = np.sqrt(vel_dm_x**2 + vel_dm_y**2 + vel_dm_z**2)

	subhalo_cen_x.append(np.array([(np.load(filename15[i].decode("utf-8"), allow_pickle=True))]))
	subhalo_cen_y.append(np.array([(np.load(filename16[i].decode("utf-8"), allow_pickle=True))]))
	subhalo_radius.append(np.array([(np.load(filename17[i].decode("utf-8"), allow_pickle=True))]))
	stelar_mass.append(np.array([(np.load(filename4[i].decode("utf-8"), allow_pickle=True))]))
	stelar_age.append(np.array([(np.load(filename19[i].decode("utf-8"), allow_pickle=True))]))
	vyg_counter.append(np.array([(np.load(filename18[i].decode("utf-8"), allow_pickle=True))]))
	stelar_age_rev = stelar_age[i][0] / stelar_mass[i][0]
	vyg_st_mass = stelar_mass[i][0]
	vyg_st_mass[stelar_age[i][0]>=1.] = 0

	data_real.append(np.array([(np.load(filename1[i].decode("utf-8"), allow_pickle=True)), temp, (np.load(filename4[i].decode("utf-8"), allow_pickle=True)), gas_kinematics, stellar_kinematics, radial_vel_gas, radial_vel_star, mass_dm, dm_kinematics, stelar_age_rev, vyg_st_mass], dtype="object"))


#######################LOAD_PARTICLE_INFORMATION_FROM_FILE_FOR_PREOCESSING#######################

#######################PLOTTING_BASE_WINDOW#######################
#Initialize certain dynamic parameters to plot the base MATPLOTLIB window

mv = 0
time_val_init = snap_array.min()
window_size_init = int(d['init_zoom_val'])
fig, ax = plt.subplots()
ax.cla()

ab1 = AnnotationBbox(addLogo1, (1.3, 0.8), xycoords='axes fraction', box_alignment=(1.1,-0.1))
ab2 = AnnotationBbox(addLogo2, (1.3, 0.6), xycoords='axes fraction', box_alignment=(1.1,-0.1))
ab3 = AnnotationBbox(addLogo3, (1.3, 0.4), xycoords='axes fraction', box_alignment=(1.1,-0.1))
ax.add_artist(ab1)
ax.add_artist(ab2)
ax.add_artist(ab3)


rax = plt.axes(d['pos_check_button_for_showing_rings_and_bh'])
axtime = plt.axes(d['pos_slider_for_moving_through_snapshots'])
axzoom = plt.axes(d['pos_slider_for_zooming_in_out'])
rax7 = fig.add_axes(d['pos_radio_button_for_changing_parameters'])
rax9 = fig.add_axes(d['pos_radio_button_for_changing_colorbar_scale'])
movie_button_ax = plt.axes(d['pos_button_for_making_movie'])

im = ax.hist2d(data_x[mv][0],data_y[mv][0],weights=data_real[mv][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, norm=LogNorm(), zorder=1)
im_cl = add_colorbar(im[3])

def extra_plotting(ax, mv):
	circle2 = plt.Circle((subhalo_cen_x[mv][0][vyg_counter[mv]], subhalo_cen_y[mv][0][vyg_counter[mv]]), subhalo_radius[mv][0][vyg_counter[mv]], color = str(d['vyg_ring_color']), fill=False, ls=str(d['vyg_ring_linestyle']), lw=int(d['vyg_ring_linewidth']), zorder=4)
	ax.add_artist(circle2)
	im2, = ax.plot(data_x[mv][2], data_y[mv][2], color=str(d['black_hole_color']), marker=str(d['black_hole_markerstyle']), ls="None", markersize=int(d['black_hole_markersize']), alpha=float(d['black_hole_plot_alpha']), zorder=2)
	circle_final = []
	for i in range(len(subhalo_cen_x[mv][0])):
		circle1 = plt.Circle((subhalo_cen_x[mv][0][i], subhalo_cen_y[mv][0][i]), subhalo_radius[mv][0][i], color = str(d['subhalo_ring_color']), fill=False, ls=str(d['subhalo_ring_linestyle']), lw=int(d['subhalo_ring_linewidth']), zorder=3)
		ax.add_artist(circle1)
		circle_final.append(circle1)
	return(im2, circle_final)

im2, circle_final = extra_plotting(ax, mv)

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

put_time_label(time_val_init, ax, window_size_init)

def add_label(ax):
	ab1 = AnnotationBbox(addLogo1, (1.3, 0.8), xycoords='axes fraction', box_alignment=(1.1,-0.1))
	ab2 = AnnotationBbox(addLogo2, (1.3, 0.6), xycoords='axes fraction', box_alignment=(1.1,-0.1))
	ab3 = AnnotationBbox(addLogo3, (1.3, 0.4), xycoords='axes fraction', box_alignment=(1.1,-0.1))
	ax.add_artist(ab1)
	ax.add_artist(ab2)
	ax.add_artist(ab3)

add_label(ax)

check = CheckButtons(rax, ('Black Hole', 'Subhalo Rings'), (True, True))
def func(label):
	if label == 'Black Hole':
		im2.set_visible(not im2.get_visible())
	elif label == 'Subhalo Rings':
		for i in range(len(circle_final)):
			circle_final[i].set_visible(not circle_final[i].get_visible())
	fig.canvas.draw_idle()
check.on_clicked(func)


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
	ax.cla()
	im_cl.remove()
	if (colorbar_scaling=='Linear'):
		im = ax.hist2d(data_x[mv][axis_to_use],data_y[mv][axis_to_use],weights=data_real[mv][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, zorder=1, density=False)
		im_cl = add_colorbar_lin(im[3])
	else:
		if (np.any(data_real[mv][str_for_param_selection])<=0.):
			print ('Found negative values in data. Plotting on Linear Scale')
			sys.stdout.flush()
			time.sleep(0.1)
			im = ax.hist2d(data_x[mv][axis_to_use],data_y[mv][axis_to_use],weights=data_real[mv][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, zorder=1, density=False)
			im_cl = add_colorbar_lin(im[3])
		else:
			im = ax.hist2d(data_x[mv][axis_to_use],data_y[mv][axis_to_use],weights=data_real[mv][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, norm=LogNorm(), zorder=1, density=False)
			im_cl = add_colorbar(im[3])

	im2, circle_final = extra_plotting(ax, mv)
	put_time_label(int(stime.val), ax, float(szoom.val))
	add_label(ax)

	def func(label):
		if label == 'Black Hole':
			im2.set_visible(not im2.get_visible())
		elif label == 'Subhalo Rings':
			for i in range(len(circle_final)):
				circle_final[i].set_visible(not circle_final[i].get_visible())
		fig.canvas.draw_idle()
	check.on_clicked(func)
fig.canvas.draw_idle()



#######################UPDATE_THE_GUI#######################

#######################SLIDER_FOR_MOVING_THROUGH_SNAPSHOTS#######################

# GUI Slider for moving through snapshots

stime = Slider(axtime, 'Snapshot', snap_array.min(), snap_array.max(), valinit=snap_array.min(), valfmt="%i")
stime.on_changed(update)

# GUI Slider for zooming in-out

szoom = Slider(axzoom, 'Zoom', int(d['min_zoom_allowed']), int(d['max_zoom_allowed']), valinit=int(d['init_zoom_val']), valfmt="%i")
szoom.on_changed(update)

#######################SLIDER_FOR_MOVING_THROUGH_SNAPSHOTS#######################


#######################RADIO_BUTTON_FOR_CHOOSING_DIFFERENT_COLORBAR_SCALE#######################

radio9 = RadioButtons(rax9, ('Linear', 'Log'), active=1)
def hzfunc9(label9):
	global im_cl
	mv = int(int(np.floor(stime.val)) - snap_array.min()-1)
	str_for_param_selection = int(hzdict7[radio7.value_selected])
	axis_to_use = int(hzdict8[radio7.value_selected])
	colorbar_scaling = str(radio9.value_selected)
	ax.cla()
	im_cl.remove()
	if (colorbar_scaling=='Linear'):
		im = ax.hist2d(data_x[mv][axis_to_use],data_y[mv][axis_to_use],weights=data_real[mv][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, zorder=1)
		im_cl = add_colorbar_lin(im[3])
	else:
		if (np.any(data_real[mv][str_for_param_selection])<=0.):
			print ('Found negative values in data. Plotting on Linear Scale')
			sys.stdout.flush()
			time.sleep(0.1)
			im = ax.hist2d(data_x[mv][axis_to_use],data_y[mv][axis_to_use],weights=data_real[mv][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, zorder=1)
			im_cl = add_colorbar_lin(im[3])
		else:
			im = ax.hist2d(data_x[mv][axis_to_use],data_y[mv][axis_to_use],weights=data_real[mv][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, norm=LogNorm(), zorder=1)
			im_cl = add_colorbar(im[3])

	im2, circle_final = extra_plotting(ax, mv)
	put_time_label(int(stime.val), ax, float(szoom.val))
	add_label(ax)
	
	def func(label):
		if label == 'Black Hole':
			im2.set_visible(not im2.get_visible())
		elif label == 'Subhalo Rings':
			for i in range(len(circle_final)):
				circle_final[i].set_visible(not circle_final[i].get_visible())
		fig.canvas.draw_idle()
	check.on_clicked(func)
radio9.on_clicked(hzfunc9)

#######################RADIO_BUTTON_FOR_CHOOSING_DIFFERENT_COLORBAR_SCALE#######################



#######################RADIO_BUTTON_FOR_CHOOSING_DIFFERENT_PARTICLE_INFORMATION#######################
# GUI Radio button for moving throught different particle information like Gas Density,-
#-Temperature, Stellar density, Gas velocity, stellar velocity, Gas radial velocity, Stellar radial velocity

radio7 = RadioButtons(rax7, ('gas_density', 'temp', 'stars', 'gas_vel', 'star_vel', 'vgas_rad', 'vstar_rad', 'mass_dm', 'dm_vel', 'stellar_age', 'vyg_st_mass'), active=0)
def hzfunc7(label7):
	global im_cl
	mv = int(int(np.floor(stime.val)) - snap_array.min()-1)
	str_for_param_selection = int(hzdict7[label7])
	axis_to_use = int(hzdict8[radio7.value_selected])
	colorbar_scaling = str(radio9.value_selected)
	ax.cla()
	im_cl.remove()
	if (colorbar_scaling=='Linear'):
		im = ax.hist2d(data_x[mv][axis_to_use],data_y[mv][axis_to_use],weights=data_real[mv][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, zorder=1)
		im_cl = add_colorbar_lin(im[3])
	else:
		if (np.any(data_real[mv][str_for_param_selection])<=0.):
			print ('Found negative values in data. Plotting on Linear Scale')
			sys.stdout.flush()
			time.sleep(0.1)
			im = ax.hist2d(data_x[mv][axis_to_use],data_y[mv][axis_to_use],weights=data_real[mv][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, zorder=1)
			im_cl = add_colorbar_lin(im[3])
		else:
			im = ax.hist2d(data_x[mv][axis_to_use],data_y[mv][axis_to_use],weights=data_real[mv][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, norm=LogNorm(), zorder=1)
			im_cl = add_colorbar(im[3])

	im2, circle_final = extra_plotting(ax, mv)
	put_time_label(int(stime.val), ax, float(szoom.val))
	add_label(ax)
	
	def func(label):
		if label == 'Black Hole':
			im2.set_visible(not im2.get_visible())
		elif label == 'Subhalo Rings':
			for i in range(len(circle_final)):
				circle_final[i].set_visible(not circle_final[i].get_visible())
		fig.canvas.draw_idle()
	check.on_clicked(func)
radio7.on_clicked(hzfunc7)

#######################RADIO_BUTTON_FOR_CHOOSING_DIFFERENT_PARTICLE_INFORMATION#######################


#######################BUTTON_TO_CREATE_A_MOVIE#######################


def ani_frame():
	global im_cl_new
	dpi = int(d['movie_dpi'])
	fig_new = plt.figure(2)
	ax_new = fig_new.add_subplot(111)
	mv = 0
	str_for_param_selection = int(hzdict7[radio7.value_selected])
	axis_to_use = int(hzdict8[radio7.value_selected])
	colorbar_scaling = str(radio9.value_selected)

	movie_name_count=1
	movie_name = str('movie_') + str(radio7.value_selected) + str('_zoom_') + str(int(szoom.val)) + str('_count_') + str(movie_name_count) + str('.mp4')
	while path.exists(movie_name):
		movie_name_count+=1
		movie_name = str('movie_') + str(radio7.value_selected) + str('_zoom_') + str(int(szoom.val)) + str('_count_') + str(movie_name_count) + str('.mp4')

	if (colorbar_scaling=='Linear'):
		im_new = ax_new.hist2d(data_x[mv][axis_to_use],data_y[mv][axis_to_use],weights=data_real[mv][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, zorder=1)
		im_cl_new = add_colorbar_lin(im_new[3])
	else:
		if (np.any(data_real[mv][str_for_param_selection])<=0.):
			print ('Found negative values in data. Plotting on Linear Scale')
			sys.stdout.flush()
			time.sleep(0.1)
			im_new = ax_new.hist2d(data_x[mv][axis_to_use],data_y[mv][axis_to_use],weights=data_real[mv][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, zorder=1)
			im_cl_new = add_colorbar_lin(im_new[3])
		else:
			im_new = ax_new.hist2d(data_x[mv][axis_to_use],data_y[mv][axis_to_use],weights=data_real[mv][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, norm=LogNorm(), zorder=1)
			im_cl_new = add_colorbar(im_new[3])

	im2_new, circle_final_new = extra_plotting(ax_new, mv)
	put_time_label(int(stime.val), ax_new, float(szoom.val))
	fig_new.set_size_inches([5,5])
	title_string = str(d['simulation']) + str('_') + str(d['snapshot']) + str('_') + str(d['galaxyid'])
	fig_new.suptitle(title_string, fontsize=size_of_font/3)
	tight_layout()

	def update_img(n):
		global im_cl_new
		mv_new = int(n - snap_array.min())
		ax_new.cla()
		im_cl_new.remove()

		if (colorbar_scaling=='Linear'):
			im_new = ax_new.hist2d(data_x[mv_new][axis_to_use],data_y[mv_new][axis_to_use],weights=data_real[mv_new][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, zorder=1)
			im_cl_new = add_colorbar_lin(im_new[3])
		else:
			if (np.any(data_real[mv][str_for_param_selection])<=0.):
				print ('Found negative values in data. Plotting on Linear Scale')
				sys.stdout.flush()
				time.sleep(0.1)
				im_new = ax_new.hist2d(data_x[mv_new][axis_to_use],data_y[mv_new][axis_to_use],weights=data_real[mv_new][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, zorder=1)
				im_cl_new = add_colorbar_lin(im_new[3])
			else:
				im_new = ax_new.hist2d(data_x[mv_new][axis_to_use],data_y[mv_new][axis_to_use],weights=data_real[mv_new][str_for_param_selection],bins=[bin_number,bin_number], cmap=colormap_new, norm=LogNorm(), zorder=1)
				im_cl_new = add_colorbar(im_new[3])

		im2_new, circle_final_new = extra_plotting(ax_new, mv_new)
		put_time_label(int(n), ax_new, float(szoom.val))
		return im_new,

	frame_list = np.arange(int(snap_array.min()), int(snap_array.max())+1, 1)
	ani = animation.FuncAnimation(fig_new,update_img,frames=frame_list, interval=int(d['movie_interval']))
	writer = animation.writers['ffmpeg'](fps=int(d['movie_fps']))
	ani.save(movie_name, writer=writer,dpi=dpi)
	plt.close(fig_new)
	return ani


movie_button = Button(movie_button_ax, 'Make Movie')
def make_movie(event):
	print ('Making movie...')
	sys.stdout.flush()
	time.sleep(0.1)
	ani_frame()
	print ('Movie saved...')
	sys.stdout.flush()
	time.sleep(0.1)
movie_button.on_clicked(make_movie)

#######################BUTTON_TO_CREATE_A_MOVIE#######################




#######################SETTING 'X' AND 'Y' DIRECTION LIMITS AND LABELS#######################


ax.set_xlabel(xlabel_str)
ax.set_ylabel(ylabel_str)
ax.set_xlim(-window_size_init,window_size_init)
ax.set_ylim(-window_size_init,window_size_init)
ax.set_aspect('equal')
title_string = str(d['simulation']) + str('_') + str(d['snapshot']) + str('_') + str(d['galaxyid'])
plt.title(title_string)
plt.show()

#######################SETTING 'X' AND 'Y' DIRECTION LIMITS AND LABELS#######################


#######################DELETE_ALL_CUTOUTS#######################
#find a file ends with .txt
x = os.listdir()
for i in x:
	if i.endswith(".hdf5"):
		print('deleting...', i)
		string_for_deleting_all_cutouts = str('rm ') + i
		print (string_for_deleting_all_cutouts)
		os.system(string_for_deleting_all_cutouts)

#######################DELETE_ALL_CUTOUTS#######################



















