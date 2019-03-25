# Plotting processed DOS from pdos.py script
# Only works for files with spin
# Author: Razak Elmaslmane
# Email: arem502@york.ac.uk

import numpy as np
import matplotlib.pyplot as plt
import pickle, os

# Location of processed pickle files
plot_dir  = str("./outputs")

# Should the program show a plot in this run:
should_plot = True

# Should the program save figs:
should_save = False

def main():
	# Get project and file names:
	project_name, alpha_file_list, beta_file_list = get_filenames()

	"""
	The following bit loads in all alpha and beta files into two lists:

	Structure of these lists is as follows:
	1. The first index is the file, e.g. alpha_spins[0] is file 0 for spin alpha
	2. Inside this index is a numpy array, where the first column is energy, second is total DOS
		e.g. alpha_spins[0][:,0] and alpha_spins[0][:,1] are the energy and total DOS for file 0
	"""
	alpha_spins = spin_loader(alpha_file_list, project_name, "ALPHA")
	beta_spins  = spin_loader(beta_file_list, project_name, "BETA")

	# --------------------- The plotting bit  ---------------------

	if (should_plot):
		# Font size on figure:
		fontsize = 12

		# Transparency:
		trans = 0.9

		# Pre-amble to render figure text using latex:
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		# Set tick labels to equal font size:
		plt.rc('xtick', labelsize=fontsize)
		plt.rc('ytick', labelsize=fontsize)

		#Grain boundary gb:
		fig = plt.figure(figsize=(6,5))

		# x-axis limits:
		plt.xlim(-7.5,10)

		# y-axis limits:
		#plt.ylim(-ylim,ylim)

		# Don't show a grid on figure:
		plt.grid(False)

		# Kind 1:
		# Plot energy verses alpha and beta spins, set beta spin to negative y:
		plt.plot(alpha_spins[0][:,0], alpha_spins[0][:,1], color='k', linewidth=1.5, alpha=trans, label=r'Kind 1 total DOS')
		plt.plot(beta_spins[0][:,0], beta_spins[0][:,1] * -1, color='k', linewidth=1.5, alpha=trans)

		# Kind 2:
		# Plot energy verses alpha and beta spins, set beta spin to negative y:
		plt.plot(alpha_spins[1][:,0], alpha_spins[1][:,1], '-.', color='k', linewidth=1.5, alpha=trans, label=r'Kind 2 total DOS')
		plt.plot(beta_spins[1][:,0], beta_spins[1][:,1] * -1, '-.', color='k', linewidth=1.5, alpha=trans)

		# Axis labels:
		plt.xlabel('Energy (e.V.)', fontsize=fontsize)
		plt.ylabel('Density of States (arb. units)', fontsize=fontsize)

		# Show legend, no box:
		plt.legend(fontsize=fontsize-1, frameon=False)

		# Show minor tick marks, set direction in:
		plt.minorticks_on()
		plt.tick_params(axis='both',direction='in',which='both', right='on', top='on', labelsize=fontsize)

		if should_save:
			# Make a directory for output graphs and save figure there
			if not os.path.isdir("./graphs"): os.mkdir('./graphs')
			plt.savefig("graphs/graph.pdf", bbox_inches='tight')

		plt.show()


def get_filenames():
	# Get files in directory
	ls = sorted(os.listdir(plot_dir))
	# Seperate file name strings by _
	temp = [item.split("_") for item in ls]
	# If its got ALPHA in the name it's an alpha spin:
	alpha_file_list = [item[1] for item in temp if "ALPHA" in item[0]]
	# If its got BETA in the name it's an alpha spin:
	beta_file_list  = [item[1] for item in temp  if "BETA"  in item[0]]
	# Get project name:
	project_name = temp[0][0].split("-")[0]

	# Prints project name from files:
	print("-"*50)
	print ( f"Project name: {project_name}" )
	print("-"*50)

	return project_name, alpha_file_list, beta_file_list

def spin_loader(spin_file_list, project_name, spin_channel):
	name_preamble = f"{project_name}-SPIN_END"
	spin_dos = [None]*len(spin_file_list)

	for index, item in enumerate(spin_file_list):
		name = name_preamble.replace("SPIN", spin_channel)
		name = name.replace("END", item)
		loaded_dos = pickle.load(open( f"{plot_dir}/{name}", 'rb'))
		spin_dos[index] = pickle.load(open( f"{plot_dir}/{name}", 'rb'))
		print( f"{spin_channel.lower()} file {index} = {name}" )
	print("-"*50)
	return spin_dos

if __name__ == '__main__':
    main()