####################################################################################### 
# THIS SOURCE CODE IS PROPERTY OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. 
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND 
# CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION 
# ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN 
# LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE,
# CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL 
# PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641. 
####################################################################################### 


#%% ---------------------------------------- YUAN-LI PDE CLOSED REGION OF ATTRACTION PINN EXAMPLE MAIN SCRIPT ----------------------------------------

# This file serves to implement the main code necessary to integrate the Yuan-Li PDE using the Physics Informed Neural Network (PINN) framework given a closed region of attraction (ROA).


#%% ---------------------------------------- IMPORT LIBRARIES ----------------------------------------

# Import standard libraries.
import os
import sys
import torch
import matplotlib.pyplot as plt
import math

# Ensure that the utilities folder for this project is on the system path.
sys.path.append( r'./Closed_ROA/Utilities' )

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
torch.manual_seed( 0 )

# Set the computational device.
device = 'cuda:1' if torch.cuda.is_available(  ) else 'cpu'


#%% ---------------------------------------- DEFINE PINN OPTIONS ----------------------------------------

# The pinn options are parameters that have no impact on the pde initial-boundary value problem being solved or the neural network that is being trained to solve it.
# Instead, the pinn option parameters are those that define the tasks the user would like performed and adjust quality-of-life factors, such as where and how often to save, print, and plot relevant network data before, during, and after the training process.

# Define the save options.
save_path = r'./Closed_ROA/Save'                                                                    # [-] Relative path to the directory in which to save network data, figures, etc.
save_frequency = torch.tensor( 10, dtype = torch.int16, device = device )                           # [#] Number of epochs after which to save intermediate networks during training. e.g., 1 = Save after every training epoch, 10 = Save after every ten training epochs, 100 = Save after every hundred training epochs.
save_flag = True                                                                                    # [T/F] Flag that determines whether to save networks during and after training, as well as training and network analysis plots.

# Define the load options.
load_path = r'./Closed_ROA/Load'                                                                    # [-] Relative path to the directory from which to load network data.
load_flag = True                                                                                    # [T/F] Flag that determines whether to load network data from the given load directory before training.

# Define the training options.
train_flag = False                                                                                  # [T/F] Flag that determines whether to train the network after creation or loading.

# Define the printing options.
batch_print_frequency = torch.tensor( 10, dtype = torch.int16, device = device )                    # [#] Number of batches after which to print training information (during an epoch that has been selected for printing).
epoch_print_frequency = torch.tensor( 10, dtype = torch.int16, device = device )                    # [#] Number of epochs after which to print training information
print_flag = True

# Define the plotting options.
num_plotting_samples = torch.tensor( 20, dtype = torch.int16, device = device )                     # [#] Number of sample points to use per dimension when plotting network results.
plot_flag = True                                                                                    # [T/F] Flag that determines whether training and network analysis plots are created.

# Define the verbosity setting.
verbose_flag = True                                                                                 # [T/F] Flag that determines whether to print more or less information when printing.

# Define the newton parameters.
newton_tolerance = torch.tensor( 1e-6, dtype = torch.float32, device = device )                     # [-] Convergence tolerance for the Newton's root finding method.
newton_max_iterations = torch.tensor( 1e2, dtype = torch.int32, device = device )                   # [#] Maximum number of Newton's method steps to perform.

# Define the exploration parameters.
exploration_volume_percentage = torch.tensor( 1e-2, dtype = torch.float32, device = device )        # [%] The level set method step size represented as a percentage of the domain volume.  This parameter conveniently scales the step size of the level set method as the dimension of the problem is adjusted. # This works for both initial and final times.
num_exploration_points = torch.tensor( 10, dtype = torch.int16, device = device )                   # [#] Number of exploration points to generate at each level set method step.
unique_volume_percentage = torch.tensor( 1e-4, dtype = torch.float32, device = device )             # [%] The tolerance used to determine whether level set points are unique as a percentage of the domain volume.  This parameter conveniently scales the unique tolerance of the level set points as the dimension of the problem is adjusted.

# Define the classification parameters.
classification_noise_percentage = torch.tensor( 1e-5, dtype = torch.float32, device = device )      # [%] The classification point noise magnitude represented as a percentage of the domain volume.  This parameter conveniently scales the noise magnitude of the classification points as the dimension of the problem is adjusted.

# Create the pinn options object.
pinn_options = pinn_options_class( save_path, save_frequency, save_flag, load_path, load_flag, train_flag, batch_print_frequency, epoch_print_frequency, print_flag, num_plotting_samples, newton_tolerance, newton_max_iterations, exploration_volume_percentage, num_exploration_points, unique_volume_percentage, classification_noise_percentage, plot_flag, device, verbose_flag )

# Save the pinn options.
pinn_options.save( save_path, r'pinn_options.pkl' )


#%% ---------------------------------------- DEFINE PROBLEM SPECIFICATIONS ----------------------------------------

# The problem specification are parameters that define the pde initial-boundary problem that the user would like to solve.
# Such parameters include the pde residual, its initial-boundary conditions, and the domain of interest.

# Define the number of inputs and outputs.
num_inputs = torch.tensor( 3, dtype = torch.uint8, device = device )                                                                            # [#] Number of network inputs.  For the Yuan-Li PDE, this is the same as the number of spatiotemporal state variables associated with the underlying dynamical system.
num_outputs = torch.tensor( 1, dtype = torch.uint8, device = device )                                                                           # [#] Number of network outputs.  For the Yuan-Li PDE, this is always one.

# Define the temporal and spatial domains.
domain_type = 'cartesian'                                                                                                                       # [-] The type of domain (cartesian, spherical, etc.).  Only cartesian domains are currently supported.
temporal_domain = torch.tensor( [ 0, 30 ], dtype = torch.float32, device = device )                                                             # [-] Temporal domain of the underlying dynamical system.
spatial_domain = torch.tensor( [ [ -1, 4 ], [ -1, 4 ] ], dtype = torch.float32, device = device ).T                                             # [-] Spatial domain of the underlying dynamical system.

# Define the initial condition parameters.
R0 = torch.tensor( 1, dtype = torch.float32, device = device )                                                                                  # [-] Initial condition radius.
A0 = torch.tensor( 2, dtype = torch.float32, device = device )                                                                                  # [-] Initial condition amplitude.
S0 = torch.tensor( 20, dtype = torch.float32, device = device )                                                                                 # [-] Initial condition slope.
P0_shift = torch.tensor( [ math.pi/2, math.pi/2 ], dtype = torch.float32, device = device )                                                     # [-] Initial condition input offset.
z0_shift = -A0/2                                                                                                                                # [-] Initial condition output offset.

# Define the flow functions.
flow_function1 = lambda s: torch.unsqueeze( -torch.sin( s[ :, 1 ] )*( -0.1*torch.cos( s[ :, 1 ] ) - torch.cos( s[ :, 2 ] ) ), dim = 1 )         # [-] Flow function associated with the first state of the underlying dynamical system.
flow_function2 = lambda s: torch.unsqueeze( -torch.sin( s[ :, 2 ] )*( torch.cos( s[ :, 1 ] ) - 0.1*torch.cos( s[ :, 2 ] ) ), dim = 1 )          # [-] Flow function associated with the second state of the underlying dynamical system.
flow_functions = [ flow_function1, flow_function2 ]                                                                                             # [-] Flow functions associated with the underlying dynamical system.

# Define the residual function.
residual_function = lambda s, dphidt, dphidx1, dphidx2: dphidt - torch.minimum( torch.zeros( size = ( s.shape[ 0 ], 1 ), dtype = torch.float32, device = device ), dphidx1*flow_functions[ 0 ]( s ) + dphidx2*flow_functions[ 1 ]( s ) )                    # [-] Residual function associated with the Yuan-Li PDE.

# Define the residual code.
residual_code = [ None, torch.tensor( [ 0 ], dtype = torch.uint8, device = device ), torch.tensor( [ 1 ], dtype = torch.uint8, device = device ), torch.tensor( [ 2 ], dtype = torch.uint8, device = device ) ]                                             # [-] Residual code.  This list specifies which derivatives with respect to the network inputs are required for the residual function inputs.

# Define the temporal code.
temporal_code = [ torch.tensor( [ 0 ], dtype = torch.uint8, device = device ) ]                                                                                                                                                                             # [-] Temporal code.  Determines how to compute the temporal derivative of the network output.      

# Define the initial-boundary condition functions.
f_ic = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1: ] - P0_shift, 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift                # [-] Initial condition function.
f_bc_1 = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1: ] - P0_shift, 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift              # [-] Boundary condition function.
f_bc_2 = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1: ] - P0_shift, 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift              # [-] Boundary condition function.

# Define the initial-boundary condition information.
ibc_types = [ 'dirichlet', 'dirichlet', 'dirichlet' ]                                                                                           # [-] Initial-Boundary condition types (e.g., dirichlet, neumann, etc.).
ibc_dimensions = torch.tensor( [ 0, 1, 2 ], dtype = torch.uint8, device = device )                                                              # [-] Dimensions associated with each initial-boundary condition.
ibc_condition_functions = [ f_ic, f_bc_1, f_bc_2 ]                                                                                              # [-] List of initial-boundary conditions.
ibc_placements = [ 'lower', 'lower', 'lower' ]                                                                                                  # [Lower/Upper] Initial-Boundary condition placement.

# Define the PDE name and type.
pde_name = 'Yuan-Li PDE: Closed ROA'                                                                                                            # [-] PDE name.
pde_type = 'First Order'                                                                                                                        # [-] PDE type.

# Create the problem specifications object.
problem_specifications = problem_specifications_class( num_inputs, num_outputs, temporal_domain, spatial_domain, domain_type, residual_function, residual_code, temporal_code, flow_functions, ibc_types, ibc_dimensions, ibc_condition_functions, ibc_placements, pde_name, pde_type, save_path, load_path )

# Save the problem specifications.
problem_specifications.save( save_path, r'problem_specifications.pkl' )


#%% ---------------------------------------- DEFINE HYPER-PARAMETERS ----------------------------------------

# The hyper-parameters are those that do not affect the problem that is being solved but impact how that problem is being solved, typically by adjusting the underlying neural architecture under consideration or the techniques used to train this network.
# Examples of several hyper-parameters include the number of network hidden layers, along with their widths and activation functions, as well as the optimizer learning rate and training data quantity.

# Store the network parameters.
activation_function = 'tanh'                                                                # [-] Activation function (e.g., tanh, sigmoid, etc.)
num_hidden_layers = torch.tensor( 3, dtype = torch.uint8, device = device )                 # [#] Number of hidden layers.
hidden_layer_widths = torch.tensor( 50, dtype = torch.uint8, device = device )              # [#] Hidden layer widths.

# This set works for variational loss integration order 1.
num_training_data = torch.tensor( int( 100e3 ), dtype = torch.int32, device = device )      # [#] Number of training data points.
num_testing_data = torch.tensor( int( 20e3 ), dtype = torch.int32, device = device )        # [#] Number of testing data points.

# Define the percent of training and testing data that should be sampled from the initial condition, the boundary condition, and the interior of the domain.
p_initial = torch.tensor( 0.25, dtype = torch.float16, device = device )                    # [%] Percentage of training and testing data associated with the initial condition.
p_boundary = torch.tensor( 0.25, dtype = torch.float16, device = device )                   # [%] Percentage of training and testing data associated with the boundary condition.
p_residual = torch.tensor( 0.5, dtype = torch.float16, device = device )                    # [%] Percentage of training and testing data associated with the residual.

# Define the number of training epochs.
num_epochs = torch.tensor( int( 5e3 ), dtype = torch.int32, device = device )               # [#] Number of training epochs to perform.

# Define the residual batch size.
residual_batch_size = torch.tensor( int( 10e3 ), dtype = torch.int32, device = device )     # [#] Training batch size. # This works for variational loss integration order 1.

# Store the optimizer parameters.
learning_rate = torch.tensor( 5e-3, dtype = torch.float32, device = device )                # [-] Learning rate.

# Define the element computation option.
element_computation_option = 'precompute'                                                   # [string] Determines whether to precompute the finite elements associated with the variational loss (costs more memory) or to dynamically generate these elements during training (costs more time per epoch) (e.g., 'precompute, 'dynamic', etc.).

# Define the element type.
element_type = 'rectangular'                                                                # [string] Finite element type associated with the variational loss (e.g., rectangular, spherical, etc.).  Only rectangular elements are currently supported.

# Define the element volume percentage.
element_volume_percent = torch.tensor( 0.01, dtype = torch.float32, device = device )       # [%] The finite element volume size associated with the variational loss represented as a percentage of the domain volume.  

# Define the integration order.
integration_order = torch.tensor( 1, dtype = torch.uint8, device = device )                 # [#] Gauss-Legendre integration order.

# Store the loss coefficients.
c_IC = torch.tensor( 1.0, dtype = torch.float32, device = device )                          # [-] Initial condition loss weight.
c_BC = torch.tensor( 1.0, dtype = torch.float32, device = device )                          # [-] Boundary condition loss weight.
c_residual = torch.tensor( 1.0, dtype = torch.float32, device = device )                    # [-] Residual loss weight.
c_variational = torch.tensor( 1.0, dtype = torch.float32, device = device )                 # [-] Variational loss weight.
c_monotonicity = torch.tensor( 10.0, dtype = torch.float32, device = device )               # [-] Monotonicity loss weight.

# Create the hyper-parameters object.
hyperparameters = hyperparameters_class( activation_function, num_hidden_layers, hidden_layer_widths, num_training_data, num_testing_data, p_initial, p_boundary, p_residual, num_epochs, residual_batch_size, learning_rate, integration_order, element_volume_percent, element_type, element_computation_option, c_IC, c_BC, c_residual, c_variational, c_monotonicity, save_path, load_path )

# Save the hyperparameters.
hyperparameters.save( save_path, r'hyperparameters.pkl' )


#%% ---------------------------------------- CREATE THE NEURAL NETWORK ----------------------------------------

# Create the pinn object.
pinn = pinn_class( pinn_options, hyperparameters, problem_specifications )

# Load the pinn object.
pinn = pinn.load( load_path, 'pinn_after_training.pkl' )

# Set the training flag object.
pinn.pinn_options.train_flag = train_flag

# Save the network before training.
pinn.save( save_path, 'pinn_before_training.pkl' )


#%% ---------------------------------------- TRAIN THE NEURAL NETWORK ----------------------------------------

# Train the neural network.
pinn.network = pinn.train( pinn.network, pinn_options.train_flag )

# Save the network after training.
pinn.save( save_path, 'pinn_after_training.pkl' )


#%% ---------------------------------------- COMPUTE CLASSIFICATION LOSS ----------------------------------------

# Compute the classification loss.
classification_loss = pinn.compute_classification_loss( pde = pinn.pde, network = pinn.network, classification_data = None, num_spatial_dimensions = pinn.domain.num_spatial_dimensions, domain = pinn.domain, plot_time = pinn.domain.temporal_domain[ 1, : ], level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), level_set_guesses = None, num_guesses = torch.tensor( 1e2, dtype = torch.int64, device = pinn.pinn_options.device ), newton_tolerance = newton_tolerance, newton_max_iterations = newton_max_iterations, exploration_radius = pinn.network.exploration_radius_spatial, num_exploration_points = num_exploration_points, unique_tolerance = pinn.network.unique_tolerance_spatial, classification_noise_magnitude = pinn.network.classification_noise_magnitude_spatial, domain_subset_type = 'spatial', tspan = torch.tensor( [ 0, 10 ], dtype = torch.float32, device = device ), dt = torch.tensor( 1e-3, dtype = torch.float32, device = device ) )


#%% ---------------------------------------- PLOT THE NEURAL NETWORK RESULTS ----------------------------------------

# Plot the network domain.
figs_domain, axes_domain = pinn.plot_domain( pinn.pde, projection_dimensions = None, projection_values = None, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), fig = None, domain_type = 'spatiotemporal', save_directory = save_path, as_surface = True, as_stream = True, as_contour = True, show_plot = False )

# Plot the initial-boundary conditions.
figs, axes = pinn.plot_initial_boundary_condition( 20*torch.ones( pinn.domain.spatiotemporal_domain.shape[ -1 ], dtype = torch.int16, device = pinn.pinn_options.device ), pinn.pde, projection_dimensions = None, projection_values = None, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), fig = None, save_directory = save_path, as_surface = True, as_stream = True, as_contour = True, show_plot = False )

# Plot the network training data.
figs_training_data, axes_training_data = pinn.plot_training_data( pinn.network, projection_dimensions = None, projection_values = None, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), fig = None, plot_type1 = 'all', plot_type2 = 'all', save_directory = save_path, as_surface = True, as_stream = True, as_contour = True, show_plot = False )

# Plot the network testing data.
figs_testing_data, axes_testing_data = pinn.plot_testing_data( pinn.network, projection_dimensions = None, projection_values = None, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), fig = None, plot_type1 = 'all', plot_type2 = 'all', save_directory = save_path, as_surface = True, as_stream = True, as_contour = True, show_plot = False )

# Plot the network plotting data.
fig_plotting_data, ax_plotting_data = pinn.plot_plotting_data( pinn.network, projection_dimensions = None, projection_values = None, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), fig = None, save_directory = save_path, as_surface = True, as_stream = True, as_contour = True, show_plot = False )

# Plot the network prediction.
fig_prediction, ax_prediction = pinn.plot_network_predictions( pinn.network.plotting_data, pinn.network, projection_dimensions = None, projection_values = None, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), fig = None, save_directory = save_path, as_surface = True, as_stream = True, as_contour = True, show_plot = False )

# Plot the network initial condition prediction.
fig_initial_prediction, ax_initial_prediction = pinn.plot_network_initial_prediction( pinn.network.plotting_data, pinn.domain, pinn.network, projection_dimensions = None, projection_values = None, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), fig = None, save_directory = save_path, as_surface = True, as_stream = True, as_contour = True, show_plot = False )

# Plot the network final condition prediction.
fig_final_prediction, ax_final_prediction = pinn.plot_network_final_prediction( pinn.network.plotting_data, pinn.domain, pinn.network, projection_dimensions = None, projection_values = None, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), fig = None, save_directory = save_path, as_surface = True, as_stream = True, as_contour = True, show_plot = False )

# Plot the network training results.
fig_training, ax_training = pinn.plot_training_results( pinn.network, save_directory = save_path, show_plot = False )

# Plot the flow field.
fig_flow_field, ax_flow_field = pinn.plot_flow_field( pinn.network.plotting_data, pinn.flow_functions, projection_dimensions = torch.tensor( [ 0 ], dtype = torch.uint8, device = device ), projection_values = torch.tensor( [ temporal_domain[ -1 ] ], dtype = torch.float32, device = device ), level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), fig = None, input_labels = None, title_string = 'Flow Field', save_directory = save_path, as_surface = True, as_stream = True, as_contour = True, show_plot = False )

# Plot the ROA boundary.
fig_roa, ax_roa = pinn.plot_roa_boundary( pinn.network.plotting_data, pinn.domain, pinn.network, pinn.flow_functions, projection_dimensions = torch.tensor( [ 0 ], dtype = torch.uint8, device = device ), projection_values = torch.tensor( [ temporal_domain[ -1 ] ], dtype = torch.float32, device = device ), level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), fig = None, input_labels = None, title_string = 'ROA Boundary Prediction', save_directory = save_path, show_plot = False )

# Plot the initial level set estimate.
pinn.network.exploration_radius_spatial = pinn.compute_exploration_radius( exploration_volume_percentage, pinn.domain, 'spatial' )
pinn.network.exploration_radius_spatiotemporal = pinn.compute_exploration_radius( exploration_volume_percentage, pinn.domain, 'spatiotemporal' )
pinn.network.unique_tolerance_spatial = pinn.compute_exploration_radius( unique_volume_percentage, pinn.domain, 'spatial' )
pinn.network.unique_tolerance_spatiotemporal = pinn.compute_exploration_radius( unique_volume_percentage, pinn.domain, 'spatiotemporal' )
fig_initial_level_set, ax_initial_level_set = pinn.plot_network_initial_level_set( domain = pinn.domain, network = pinn.network, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), level_set_guess = None, num_guesses = torch.tensor( 100, dtype = torch.int64, device = pinn.pinn_options.device ), newton_tolerance = newton_tolerance, newton_max_iterations = newton_max_iterations, exploration_radius = pinn.network.exploration_radius_spatial, num_exploration_points = num_exploration_points, unique_tolerance = pinn.network.unique_tolerance_spatial, projection_dimensions = None, projection_values = None, fig = fig_initial_prediction, dimension_labels = pinn.domain.dimension_labels, save_directory = save_path, as_surface = False, as_stream = False, as_contour = False, show_plot = False )

# Plot the final level set estimate.
fig_final_level_set, ax_final_level_set = pinn.plot_network_final_level_set( domain = pinn.domain, network = pinn.network, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), level_set_guess = None, num_guesses = torch.tensor( 1e3, dtype = torch.int64, device = pinn.pinn_options.device ), newton_tolerance = newton_tolerance, newton_max_iterations = newton_max_iterations, exploration_radius = pinn.network.exploration_radius_spatial, num_exploration_points = num_exploration_points, unique_tolerance = pinn.network.unique_tolerance_spatial, projection_dimensions = None, projection_values = None, fig = fig_final_prediction, dimension_labels = pinn.domain.dimension_labels, save_directory = save_path, as_surface = False, as_stream = False, as_contour = False, show_plot = False )

