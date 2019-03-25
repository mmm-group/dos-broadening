# DOS processing from CP2K
# Author: Razak Elmaslmane
# Email: arem502@york.ac.uk

# Outputs a dat or pickle file with the following headers:
# Energy, Total DoS

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle, os, time

# Place all pdos files to process in the following directory:
dir = str('inputs')

"""
parameter
------------------------------------------------------------------------------------------------------------------------------
smearing : float

Gaussian smearing value

"""
smearing = 0.05

"""
parameter
------------------------------------------------------------------------------------------------------------------------------
method : string

Method to use for smearing the density of states
Choose from: shift, linalg, add or conv 
Most reliable: linalg and add
Fastest: shift
Choose debug to compare them all
"""
# Default: linalg
method = "linalg"

"""
parameter
------------------------------------------------------------------------------------------------------------------------------
grid_spacing : float

Energy grid spacing

For  shift: Use 0.0005 or smaller
For   conv: Use 0.0001 or smaller - does not work properly
For linalg: Use 0.01
For    add: Use 0.01
"""
# Default: 0.01
grid_spacing = 0.01

"""
parameters
------------------------------------------------------------------------------------------------------------------------------
minCalc : float
maxCalc : float

Energy range minimum and maximum (in eV), zero is fermi energy
Final grid returned is slightly smaller than requested value in shift - use a slightly larger grid for this method
"""
# Default: -12
minCalc = -12

# Default: 12
maxCalc =  12

"""
parameters
------------------------------------------------------------------------------------------------------------------------------
plot : bool
savePickle : bool
saveTxt : bool

Boolean parameters to show a plot, save a pickle file or save a text file
"""
# Should the program show a plot?
# Default: False
plot = False

# Save for loading in pickle later?
# Default: True
savePickle = True

# Save text file?
# Default: True
saveTxt = False

#=======================================================
#                Ignore these params:                  
#=======================================================

# Number used to convert from (a.u.) to eV
# set to 1 for a.u. results
conversion2ev = 27.2113838565563

#=======================================================
#                   Useful functions                  
#=======================================================

def gaus_func(x, mu, sigma):
	"""
	Gaussian function calculator

	Parameters
	------------- 
	
	x : list
	energy grid, floating point numbers
	
	mu : float
	centre of unsmeared dos value

	sigma : float 
	smearing value of DOS
	
	Returns
	--------------
	list
	A list of values corresponding to a gaussian with unit area, centred around mu

	"""
	return np.exp(-np.power((x - mu)/sigma, 2) *0.5 )*(1/(sigma*np.sqrt(2*np.pi)))

def dos_convolution(energies, dos_total, nGridpoints, smearing):
	start = time.time()
	# energy grid:
	energyGrid = np.linspace(min_energy, max_energy, nGridpoints)

	# Final dos:
	final_dos = np.zeros(nGridpoints)

	# Make a gaussian:
	gaussian = gaus_func(energyGrid, 0, smearing)

	max_gaus = max(gaussian)

	for index, item in enumerate(energies):
		idx = (np.abs(energyGrid - item)).argmin()
		final_dos[idx] = dos_total[index]*max_gaus*2

	final_dos = signal.fftconvolve(gaussian, final_dos, mode="same")*0.92*0.5
	
	# Print time:
	end = time.time()
	print(f"Time elapsed (s),    convolution method: {end-start:.5f}")

	return energyGrid, final_dos

def dos_gaussian_addition(energies, dos_total, nGridpoints, smearing):
	"""
	Addition approach to smearing eigenvalues - should yield exactly the same answer as linalg

	Advantages:
	+ Very accurate
	+ Output grids are consistent

	Disadvantages:
	- Slower than shift method

	Parameters
	------------- 
	
	energies : list
	list of eigenvalues, floating point numbers
	
	dos_total : list
	DOS weightings

	nGridPoints : float 
	Number of grid points to perform this method on
	
	smearing : float
	Smearing value
	
	Returns
	--------------

	list, list 
	A list of energies and smeared eigenvalues

	"""
	# Time elapsed calculation:
	start = time.time()

	energyGrid = np.linspace(min_energy, max_energy, nGridpoints)
	# Final dos:
	final_dos = np.zeros(nGridpoints)

	for index, item in enumerate(energies):
		final_dos += gaus_func(energyGrid, item, smearing)*dos_total[index]
	
	# Time elapsed calculation:
	end = time.time()
	print(f"Time elapsed (s),       addition method: {end-start:.5f}")

	return energyGrid, final_dos

def dos_linalg(energies, dos_total, nGridpoints, smearing):
	"""
	Matrix approach to smearing eigenvalues - should yield exactly the same answer as add

	Advantages:
	+ Very accurate
	+ Output grids are consistent

	Disadvantages:
	- Slower than shift method

	Parameters
	------------- 
	
	energies : list
	list of eigenvalues, floating point numbers
	
	dos_total : list
	DoS weightings

	nGridPoints : float 
	Number of grid points to perform this method on
	
	smearing : float
	Smearing value
	
	Returns
	--------------
	list, list 
	A list of energies and smeared eigenvalues

	"""

	# Time elapsed calculation:
	start = time.time()

	# Create energy values grid:
	energyGrid = np.linspace(min_energy, max_energy, nGridpoints)

	# Creates the mu part of the gaussian (in exponent)
	x = np.ones((len(energyGrid), len(energies))) * (energies)
	# Creates the x  part of the gaussian (in exponent)
	mu = np.tensordot(energyGrid, np.ones(len(energies)), axes=0)
	
	# -(x-mu)^2/2\sigma^2
	final_dos = -np.square((x - mu))/(2*smearing*smearing)
	# exp() of matrix elements
	final_dos = np.exp(final_dos)
	# Multiply by weightings of each gaussian and scale to get correct area
	final_dos = np.matmul(final_dos, dos_total)/(smearing*np.sqrt(2*np.pi))
	
	# Time elapsed calculation:
	end = time.time()
	print(f"Time elapsed (s), linear algebra method: {end-start:.5f}")

	return energyGrid, final_dos

def dos_gaussian_shift(energies, dos_total, nGridpoints, smearing):
	"""
	Produces a single gaussian function then shifts the gaussian around the grid

	Advantages:
	+ Very fast compared to other methods

	Disadvantages:
	- Produces an edge effect, energy range should be larger than required
	- Very reliable, but not as accurate as addition method as mean needs to be on energy grid
	- Due to edge effect, grids produced will vary in size
	- Grids can be made consistent but edge effect will be shown in data

	Parameters
	------------- 
	energies : list
	list of eigenvalues, floating point numbers
	
	dos_total : list
	Density of states weightings

	nGridPoints : float 
	Number of grid points to perform this method on
	
	smearing : float
	Smearing value
	
	Returns
	--------------
	list, list 
	A list of energies and smeared eigenvalues

	"""
	# Start time for function:
	start = time.time()

	# Create grid for energy values:
	energyGrid = np.linspace(min_energy, max_energy, nGridpoints)
	# Final dos using np:
	final_dos = np.zeros(nGridpoints)
	# Define gaussian function:
	func = gaus_func(energyGrid, 0, smearing)
	# Find max index of gaussian:
	maximum = func.argmax()

	# Move gaussian around grid until mean of gaussian is nearest to the DOS value
	for index, item in enumerate(energies):
		maximum = func.argmax()
		idx = (np.abs(energyGrid - item)).argmin()
		final_dos += np.roll(func, idx-maximum)*dos_total[index]

	# Remove 3% of grid due to edge effects:
	n = int(0.03*func.size)
	final_dos = final_dos[n:-n]
	energyGrid = energyGrid[n:-n]

	# finish timing:
	end = time.time()
	print(f"Time elapsed (s),          shift method: {end-start:.5f}")

	return energyGrid, final_dos

#=======================================================
#                   Start of code                  
#=======================================================

# Reads a list of the files in the above directory:
file_list = np.array(sorted(os.listdir('./' + dir)))

print (f"Files in input folder: {file_list.size}")

for index, item in enumerate(file_list): 
	print (f"File {index+1}: {item}") 


for file in file_list:

	# Opens the file, counts the number of lines:
	with open (f"./{dir}/{file}", "r") as file_handle:
		raw_data = file_handle.readlines()

	print (f"-"*100)
	print (f"Starting file: {file}")
	print (f"Number of lines: {len(raw_data)}")
	
	# Read the first line:
	line = raw_data.pop(0).split()
	fermi_energy = float(line[-2])

	print (f"Fermi energy = {fermi_energy:.5f}" )

	# Gets table headers, removes unnecessary split values
	table_headers = raw_data.pop(0).split()
	table_headers.pop(1)
	table_headers.pop(3)

	# Line count:
	num_lines = len(raw_data)

	# Creates arrays based on line count and headers:
	energies = [None]*num_lines
	occupations = [None]*num_lines
	dos_total = [None]*num_lines
	num_projections = len(table_headers) - 3

	# Hold projection data:
	projections = [[None for x in range(num_projections)] for y in range(num_lines)]

	# Arranges data:
	for index, item in enumerate(raw_data):
		item = item.split()
		energies[index] = (float(item[1]) - fermi_energy)*conversion2ev
		occupations[index] = float(item[2])
		projections[index] = [float(i) for i in item[3:]]
		dos_total[index] = sum(projections[index])
	
	# Get min and max energies from data:
	min_energy = min(energies)
	max_energy = max(energies)
 	
 	# Print to screen energy data:
	text1 = f"Energy minimum in file is {min_energy:10.5f} eV below fermi energy, only doing up from {minCalc:10.5f} eV"
	text2 = f"Energy minimum in file is {min_energy:10.5f} eV below fermi energy, within selected range"
	print (text1) if min_energy < minCalc else print (text2)
	
	text1 = f"Energy maximum in file is {max_energy:10.5f} eV above fermi energy, only doing up to   {maxCalc:10.5f} eV"
	text2 = f"Energy maximum in file is {max_energy:10.5f} eV above fermi energy, within selected range"
	print (text1) if max_energy > maxCalc else print (text2)
	
	# Set appropriate min and max energy values:
	max_energy = max_energy if max_energy < maxCalc else maxCalc
	min_energy = min_energy if min_energy > minCalc else minCalc

	# Grid points corresponding to final dos energy:
	nGridpoints_convolution = int( (max_energy - min_energy) / grid_spacing )
	nGridpoints_addition = int( (max_energy - min_energy) / grid_spacing )

	# Remove indicies which indicies are too big/too small:
	count = 0
	for index, item in enumerate(energies):
		if item < min_energy or item > max_energy:
			count += 1
			energies.pop(index)
			occupations.pop(index)
			dos_total.pop(index)

	# Prints no data points removed:
	print(f"Number of items removed that were out of defined range: {count}")
	
	# grid and dos parameters:
	grid = 0
	final_dos = 0

	# fork in code, depends on which method you chose
	# Debug option, calculate all methods and plot them:
	if method == "debug":
		grid,  final_dos  = dos_linalg(energies, dos_total, nGridpoints_addition, smearing)
		grid1, final_dos1 = dos_convolution(energies, dos_total, nGridpoints_convolution, smearing)
		grid2, final_dos2 = dos_gaussian_shift(energies, dos_total, nGridpoints_addition, smearing)
		grid3, final_dos3 = dos_gaussian_addition(energies, dos_total, nGridpoints_addition, smearing)

		plt.figure()
		plt.plot(grid,  final_dos,  label="linear algebra")
		plt.plot(grid1, final_dos1, label="convolution")
		plt.plot(grid2, final_dos2, label="shift method")
		plt.plot(grid3, final_dos3, label="addition method")

		plt.legend()
		plt.show()

	# Shift method
	elif method == "shift":
		grid, final_dos = dos_gaussian_shift(energies, dos_total, nGridpoints_addition, smearing)
	
	# Linear algebra method
	elif method == "linalg":
		grid, final_dos = dos_linalg(energies, dos_total, nGridpoints_addition, smearing)

	# Addition method, same as old DoS script:
	elif method == "add":
		grid, final_dos = dos_gaussian_addition(energies, dos_total, nGridpoints_addition, smearing)
	
	# Convolution method, doesn't work very well:
	elif method == "conv":
		grid, final_dos = dos_convolution(energies, dos_total, nGridpoints_convolution, smearing)

	# Goes here if you've done something wrong:
	else:
		print("Method does not exist, please choose from: shift, linalg, add or conv.")
		print("Conv has numerical issues, don't use it unless you know what you're doing.")
		print("For debug option, all methods will be used and plotted.")

	if plot:
		fontsize = 14
		trans = 0.9
		plt.rc('text', usetex=True)
		plt.rc('font', family='Times')
		plt.rc('xtick', labelsize=fontsize)
		plt.rc('ytick', labelsize=fontsize)

		plt.plot(grid, final_dos, alpha=trans, label=r'Total DOS')

		plt.xlabel('Energy (e.V.)', fontsize=fontsize)
		plt.ylabel('Density of States (arb. units)', fontsize=fontsize)

		plt.xlim(-10,10)
		plt.grid(False)
		plt.tick_params(which='both', direction='in', right='on', top='on', bottom='on')
		plt.legend(fontsize=fontsize-1)

		plt.show()

	if not os.path.isdir("./outputs"): os.mkdir('./outputs')

	if saveTxt: np.savetxt('./outputs/' + file.replace('pdos', 'dat'), np.c_[grid, final_dos])
	if savePickle: pickle.dump(np.c_[grid, final_dos], open('./outputs/' + file.replace('pdos', 'pickle'), 'wb' ), protocol=pickle.HIGHEST_PROTOCOL )