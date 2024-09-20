
#%% ---------------------------------------- NON-SPIKING CLOSED REGION OF ATTRACTION PINN EXAMPLE MAIN SCRIPT ----------------------------------------

# This file serves to implement the main code necessary to integrate the Yuan-Li PDE for a dynamical system with a closed ROA using the Non-Spiking Physics Informed Neural Network (PINN) framework.


#%% ---------------------------------------- IMPORT LIBRARIES ----------------------------------------

# Import standard libraries.
import os
import sys
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# Ensure that the utilities folder for this project is on the system path.
sys.path.append( r'./ann/utilities' )
sys.path.append( r'./ann/utilities' )

# Import custom libraries.
from pinn_options_class import pinn_options_class as pinn_options_class
from hyperparameters_class import hyperparameters_class as hyperparameters_class
from problem_specifications_class import problem_specifications_class as problem_specifications_class
from pinn_class import pinn_class as pinn_class

#%% ---------------------------------------- SETUP ENVIRONMENT PROPERTIES ----------------------------------------

# Clear the terminal.
os.system( 'cls' if os.name == 'nt' else 'clear' )

# Set matplotlib options.
plt.rcParams.update( { 'figure.max_open_warning': 0 } )                     # Disable maximum open figure warning.

# Set the random seeds.
random.seed( 0 )
np.random.seed( 0 )
torch.manual_seed( 0 )

# Set the computational device.
device = 'cuda:9' if torch.cuda.is_available(  ) else 'cpu'
# device = 'cpu'

# Define the save options.
save_path = r'./ann/simple_pendulum/save'                                                                    # [-] Relative path to the directory in which to save network data, figures, etc.
load_path = r'./ann/simple_pendulum/load'                                                                    # [-] Relative path to the directory from which to load network data.


#%% ---------------------------------------- IMPORT NUMERICAL DATA ----------------------------------------

# Load the numerical data.
Xs_numerical = np.loadtxt( load_path + '/Xs_numerical.csv', delimiter = ',', dtype = np.float32 )
Ys_numerical = np.loadtxt( load_path + '/Ys_numerical.csv', delimiter = ',', dtype = np.float32 )
Zs_numerical = np.loadtxt( load_path + '/Zs_numerical.csv', delimiter = ',', dtype = np.float32 )


#%% ---------------------------------------- IMPORT GRID SEARCH NETWORK ----------------------------------------

# Load the pinn options for the grid search network.
pinn_options_grid = pinn_options_class(  )
pinn_options_grid = pinn_options_grid.load( load_path = load_path, file_name = r'pinn_options_grid.pkl' )

# Load the hyperparameters for the grid search network.
hyperparameters_grid = hyperparameters_class(  )
hyperparameters_grid = hyperparameters_grid.load( load_path = load_path, file_name = r'hyperparameters_grid.pkl' )

# Load the problem specifications for the grid search network.
problem_specifications_grid = problem_specifications_class(  )
problem_specifications_grid = problem_specifications_grid.load( load_path = load_path, file_name = r'problem_specifications_grid.pkl' )

# Load the pinn class for the grid search network.
pinn_grid = pinn_class( pinn_options_grid, hyperparameters_grid, problem_specifications_grid )
pinn_grid = pinn_grid.load( load_path, 'pinn_after_training_grid.pkl' )

# Load the final level set for the grid search network.
Xs_grid = np.loadtxt( load_path + '/Xs_grid.csv', delimiter = ',', dtype = np.float32 )
Ys_grid = np.loadtxt( load_path + '/Ys_grid.csv', delimiter = ',', dtype = np.float32 )
Zs_grid = np.loadtxt( load_path + '/Zs_grid.csv', delimiter = ',', dtype = np.float32 )

#%% ---------------------------------------- IMPORT BO NETWORK ----------------------------------------

# Load the pinn options for the BO network.
pinn_options_BO = pinn_options_class(  )
pinn_options_BO = pinn_options_BO.load( load_path = load_path, file_name = r'pinn_options_BO.pkl' )

# Load the hyperparameters for the BO network.
hyperparameters_BO = hyperparameters_class(  )
hyperparameters_BO = hyperparameters_BO.load( load_path = load_path, file_name = r'hyperparameters_BO.pkl' )

# Load the problem specifications for the BO network.
problem_specifications_BO = problem_specifications_class(  )
problem_specifications_BO = problem_specifications_BO.load( load_path = load_path, file_name = r'problem_specifications_BO.pkl' )

# Load the pinn class for the BO network.
pinn_BO = pinn_class( pinn_options_BO, hyperparameters_BO, problem_specifications_BO )
pinn_BO = pinn_BO.load( load_path, 'pinn_after_training_BO.pkl' )

# Load the final level set for the grid search network.
Xs_BO = np.loadtxt( load_path + '/Xs_BO.csv', delimiter = ',', dtype = np.float32 )
Ys_BO = np.loadtxt( load_path + '/Ys_BO.csv', delimiter = ',', dtype = np.float32 )
Zs_BO = np.loadtxt( load_path + '/Zs_BO.csv', delimiter = ',', dtype = np.float32 )


#%% ---------------------------------------- PLOT THE NETWORK RESULTS ----------------------------------------

# fig_final_prediction, ax_final_prediction = pinn_grid.plot_network_final_prediction( pinn_grid.network.plotting_data, pinn_grid.domain, pinn_grid.network, projection_dimensions = None, projection_values = None, level = torch.tensor( 0, dtype = torch.float32, device = pinn_grid.pinn_options.device ), fig = None, save_directory = save_path, as_surface = True, as_stream = True, as_contour = True, show_plot = False )

# fig_final_prediction, ax_final_prediction = pinn_BO.plot_network_final_prediction( pinn_BO.network.plotting_data, pinn_BO.domain, pinn_BO.network, projection_dimensions = None, projection_values = None, level = torch.tensor( 0, dtype = torch.float32, device = pinn_BO.pinn_options.device ), fig = None, save_directory = save_path, as_surface = True, as_stream = True, as_contour = True, show_plot = False )


# Plot the flow field.
fig, ax = pinn_grid.plot_flow_field( plotting_data = pinn_grid.network.plotting_data, flow_functions = pinn_grid.flow_functions, projection_dimensions = torch.tensor( [ 0 ], dtype = torch.uint8, device = device ), projection_values = torch.tensor( [ pinn_grid.problem_specifications.temporal_domain[ -1 ] ], dtype = torch.float32, device = pinn_grid.pinn_options.device ), fig = None, input_labels = None, title_string = 'Flow Field', save_directory = save_path, as_surface = False, as_stream = True, as_contour = False, show_plot = False )

# Plot the ROA estimates.
contour_numerical = ax[ 0 ].contour( Xs_numerical, Ys_numerical, Zs_numerical, levels = [ 0 ], colors = 'black', linewidths = 2.5, alpha = 0.75, linestyles = 'solid' )
contour_grid = ax[ 0 ].contour( Xs_grid, Ys_grid, Zs_grid, levels = [ 0 ], colors = 'red', linewidths = 2.5, alpha = 0.75, linestyles = 'solid' )
contour_BO = ax[ 0 ].contour( Xs_BO, Ys_BO, Zs_BO, levels = [ 0 ], colors = 'blue', linewidths = 2.5, alpha = 0.75, linestyles = 'solid' )
plt.legend( [ contour_numerical.legend_elements(  )[ 0 ][ 0 ], contour_grid.legend_elements(  )[ 0 ][ 0 ], contour_BO.legend_elements(  )[ 0 ][ 0 ] ], [ 'Numerical', 'Grid Search', 'Bayesian' ], loc = 'lower center', ncols = 3, fontsize = 11 )

# Update the plot labels.
plt.xlabel( r'State Variable 1, $\theta$ [rad]', fontsize = 14 )
plt.ylabel( r'State Variable 2, $\omega$ [rad/s]', fontsize = 14 )
plt.title( 'Simple Pendulum ROA Estimates', fontsize = 14, fontweight = 'bold' )

# Save the figure.
plt.savefig( save_path + '/' + f'Figure_{plt.gcf(  ).number}.png' )
