####################################################################################### 
# THIS SOURCE CODE IS PROPERTY OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. 
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND 
# CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION 
# ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN 
# LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE,
# CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL 
# PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641. 
####################################################################################### 


#%% ---------------------------------------- NON-SPIKING SIMPLE PENDULUM PINN EXAMPLE MAIN SCRIPT ----------------------------------------

# This file serves to implement the main code necessary to integrate the Yuan-Li PDE for a simple pendulum system using the Non-Spiking Physics Informed Neural Network (PINN) framework.


#%% ---------------------------------------- IMPORT LIBRARIES ----------------------------------------

# Import standard libraries.
import os
import sys
import math
import time
import torch
import random
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# Ensure that the utilities folder for this project is on the system path.
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


#%% ---------------------------------------- DEFINE CONFIGURATION ----------------------------------------

# # Define the base configuration.
# BASE_CONFIG = {
#     'classification_parameters': {
#         'num_noisy_samples_per_level_set_point': int( 5 ),
#         'noise_percentage': float( 1e-3 ),
#         'dt': float( 1e-2 ),
#         'tfinal': float( 10 ),
#     },
#     'exploration_parameters': {
#         'volume_percentage': float( 1e-2 ),
#         'num_points': int( 50 ),
#         'unique_percentage': float( 1e-4 ),
#     },
#     'hyperparameters': {
#         'activation_function': 'sigmoid',
#         'c_IC': float( 1.0 ),
#         'c_BC': float( 1.0 ),
#         'c_residual': float( 1e-4 ),
#         'c_residual_gradient': float( 0 ),
#         'c_variational': float( 1e-4 ),
#         'c_monotonicity': float( 100 ),
#         'hidden_layer_widths': int( 175 ),
#         'num_epochs': int( 400 ),
#         'num_hidden_layers': int( 5 ),
#         'num_training_data': int( 100e3 ),
#         'num_testing_data': int( 20e3 ),
#         'learning_rate': float( 0.005 ),
#     },
#     'newton_parameters': {
#         'tolerance': float( 1e-4 ),
#         'max_iterations': int( 1e2 ),
#     },
#     'paths': {
#         'save_path': r'./ann/simple_pendulum/save',
#         'load_path': r'./ann/simple_pendulum/load',
#     },
#     'plotting_parameters': {
#         'num_plotting_samples': int( 20 ),
#         'plot_flag': bool( False ),
#     },
#     'printing_parameters': {
#         'batch_print_frequency': int( 10 ),
#         'epoch_print_frequency': int( 10 ),
#         'print_flag': bool( True ),
#     },
#     'runtime': {
#         'device': 'cuda:9' if torch.cuda.is_available(  ) else 'cpu',
#         'load_flag': bool( False ),
#         'seed': int( 0 ),
#         'train_flag': bool( True ),
#         'verbose_flag': bool( True ),
#     },
#     'saving_parameters': {
#         'save_flag': bool( True ),
#         'save_frequency': int( 10 ),
#     }
# }


# Coarse Grid Search Best. (Closed Boundary Conditions)
BASE_CONFIG = {
    'classification_parameters': {
        'num_noisy_samples_per_level_set_point': int( 5 ),
        'noise_percentage': float( 1e-3 ),
        'dt': float( 1e-2),
        'tfinal': float( 10 ),
    },
    'exploration_parameters': {
        'volume_percentage': float( 1e-2 ),
        'num_points': int( 50 ),
        'unique_percentage': float( 1e-4 ),
    },
    'hyperparameters': {
        'activation_function': 'sigmoid',
        'c_IC': float( 1.0 ),
        'c_BC': float( 1.0 ),
        'c_residual': float( 1e-4 ),
        'c_residual_gradient': float( 0 ),
        'c_variational': float( 1e-5 ),
        'c_monotonicity': float( 100 ),
        'hidden_layer_widths': int( 175 ),
        'num_epochs': int( 400 ),
        'num_hidden_layers': int( 7 ),
        'num_training_data': int( 100e3 ),
        'num_testing_data': int( 20e3 ),
        'learning_rate': float( 5e-4 ),
    },
    'newton_parameters': {
        'tolerance': float( 1e-4 ),
        'max_iterations': int( 1e2 ),
    },
    'paths': {
        'save_path': r'./ann/simple_pendulum/save',
        'load_path': r'./ann/simple_pendulum/load',
    },
    'plotting_parameters': {
        'num_plotting_samples': int( 20 ),
        'plot_flag': bool( False ),
    },
    'printing_parameters': {
        'batch_print_frequency': int( 10 ),
        'epoch_print_frequency': int( 10 ),
        'print_flag': bool( True ),
    },
    'runtime': {
        'device': 'cuda:8' if torch.cuda.is_available(  ) else 'cpu',
        'seed': int( 0 ),
        'load_flag': bool( False ),
        'train_flag': bool( True ),
        'verbose_flag': bool( True ),
    },
    'saving_parameters': {
        'save_flag': bool( True ),
        'save_frequency': int( 10 ),
    }
}


# # Coarse grid search worst. (Closed Boundary Conditions)
# BASE_CONFIG = {
#     'classification_parameters': {
#         'num_noisy_samples_per_level_set_point': int( 5 ),
#         'noise_percentage': float( 1e-3 ),
#         'dt': float( 1e-2),
#         'tfinal': float( 10 ),
#     },
#     'exploration_parameters': {
#         'volume_percentage': float( 1e-2 ),
#         'num_points': int( 50 ),
#         'unique_percentage': float( 1e-4 ),
#     },
#     'hyperparameters': {
#         'activation_function': 'sigmoid',
#         'c_IC': float( 1.0 ),
#         'c_BC': float( 1.0 ),
#         'c_residual': float( 1e-5 ),
#         'c_residual_gradient': float( 0 ),
#         'c_variational': float( 1e-5 ),
#         'c_monotonicity': float( 100 ),
#         'hidden_layer_widths': int( 50 ),
#         'num_epochs': int( 400 ),
#         'num_hidden_layers': int( 3 ),
#         'num_training_data': int( 100e3 ),
#         'num_testing_data': int( 20e3 ),
#         'learning_rate': float( 5e-4 ),
#     },
#     'newton_parameters': {
#         'tolerance': float( 1e-4 ),
#         'max_iterations': int( 1e2 ),
#     },
#     'paths': {
#         'save_path': r'./ann/simple_pendulum/save',
#         'load_path': r'./ann/simple_pendulum/load',
#     },
#     'plotting_parameters': {
#         'num_plotting_samples': int( 20 ),
#         'plot_flag': bool( True ),
#     },
#     'printing_parameters': {
#         'batch_print_frequency': int( 10 ),
#         'epoch_print_frequency': int( 10 ),
#         'print_flag': bool( True ),
#     },
#     'runtime': {
#         'device': 'cuda:9' if torch.cuda.is_available(  ) else 'cpu',
#         'seed': int( 0 ),
#         'load_flag': bool( False ),
#         'train_flag': bool( True ),
#         'verbose_flag': bool( True ),
#     },
#     'saving_parameters': {
#         'save_flag': bool( True ),
#         'save_frequency': int( 10 ),
#     }
# }


# # Refined grid search best. (Closed Boundary Conditions)
# BASE_CONFIG = {
#     'classification_parameters': {
#         'num_noisy_samples_per_level_set_point': int( 5 ),
#         'noise_percentage': float( 1e-3 ),
#         'dt': float( 1e-2),
#         'tfinal': float( 10 ),
#     },
#     'exploration_parameters': {
#         'volume_percentage': float( 1e-2 ),
#         'num_points': int( 50 ),
#         'unique_percentage': float( 1e-4 ),
#     },
#     'hyperparameters': {
#         'activation_function': 'sigmoid',
#         'c_IC': float( 1.0 ),
#         'c_BC': float( 1.0 ),
#         'c_residual': float( 5e-7 ),
#         'c_residual_gradient': float( 0 ),
#         'c_variational': float( 5e-7 ),
#         'c_monotonicity': float( 100 ),
#         'hidden_layer_widths': int( 175 ),
#         'num_epochs': int( 400 ),
#         'num_hidden_layers': int( 5 ),
#         'num_training_data': int( 100e3 ),
#         'num_testing_data': int( 20e3 ),
#         'learning_rate': float( 5e-3 ),
#     },
#     'newton_parameters': {
#         'tolerance': float( 1e-4 ),
#         'max_iterations': int( 1e2 ),
#     },
#     'paths': {
#         'save_path': r'./ann/simple_pendulum/save',
#         'load_path': r'./ann/simple_pendulum/load',
#     },
#     'plotting_parameters': {
#         'num_plotting_samples': int( 20 ),
#         'plot_flag': bool( False ),
#     },
#     'printing_parameters': {
#         'batch_print_frequency': int( 10 ),
#         'epoch_print_frequency': int( 10 ),
#         'print_flag': bool( True ),
#     },
#     'runtime': {
#         'device': 'cuda:6' if torch.cuda.is_available(  ) else 'cpu',
#         'seed': int( 0 ),
#         'load_flag': bool( False ),
#         'train_flag': bool( True ),
#         'verbose_flag': bool( True ),
#     },
#     'saving_parameters': {
#         'save_flag': bool( True ),
#         'save_frequency': int( 10 ),
#     }
# }


# # Refined grid search worst. (Closed Boundary Conditions)
# BASE_CONFIG = {
#     'classification_parameters': {
#         'num_noisy_samples_per_level_set_point': int( 5 ),
#         'noise_percentage': float( 1e-3 ),
#         'dt': float( 1e-2),
#         'tfinal': float( 10 ),
#     },
#     'exploration_parameters': {
#         'volume_percentage': float( 1e-2 ),
#         'num_points': int( 50 ),
#         'unique_percentage': float( 1e-4 ),
#     },
#     'hyperparameters': {
#         'activation_function': 'sigmoid',
#         'c_IC': float( 1.0 ),
#         'c_BC': float( 1.0 ),
#         'c_residual': float( 1e-7 ),
#         'c_residual_gradient': float( 0 ),
#         'c_variational': float( 5e-2 ),
#         'c_monotonicity': float( 100 ),
#         'hidden_layer_widths': int( 175 ),
#         'num_epochs': int( 400 ),
#         'num_hidden_layers': int( 5 ),
#         'num_training_data': int( 100e3 ),
#         'num_testing_data': int( 20e3 ),
#         'learning_rate': float( 5e-3 ),
#     },
#     'newton_parameters': {
#         'tolerance': float( 1e-4 ),
#         'max_iterations': int( 1e2 ),
#     },
#     'paths': {
#         'save_path': r'./ann/simple_pendulum/save',
#         'load_path': r'./ann/simple_pendulum/load',
#     },
#     'plotting_parameters': {
#         'num_plotting_samples': int( 20 ),
#         'plot_flag': bool( True ),
#     },
#     'printing_parameters': {
#         'batch_print_frequency': int( 10 ),
#         'epoch_print_frequency': int( 10 ),
#         'print_flag': bool( True ),
#     },
#     'runtime': {
#         'device': 'cuda:9' if torch.cuda.is_available(  ) else 'cpu',
#         'seed': int( 0 ),
#         'load_flag': bool( False ),
#         'train_flag': bool( True ),
#         'verbose_flag': bool( True ),
#     },
#     'saving_parameters': {
#         'save_flag': bool( True ),
#         'save_frequency': int( 10 ),
#     }
# }


# # Coarse grid search best. (Open Boundary Conditions)
# BASE_CONFIG = {
#     'classification_parameters': {
#         'num_noisy_samples_per_level_set_point': int( 5 ),
#         'noise_percentage': float( 1e-3 ),
#         'dt': float( 1e-2),
#         'tfinal': float( 10 ),
#     },
#     'exploration_parameters': {
#         'volume_percentage': float( 1e-2 ),
#         'num_points': int( 50 ),
#         'unique_percentage': float( 1e-4 ),
#     },
#     'hyperparameters': {
#         'activation_function': 'sigmoid',
#         'c_IC': float( 1.0 ),
#         'c_BC': float( 1.0 ),
#         'c_residual': float( 1e-3 ),
#         'c_residual_gradient': float( 0 ),
#         'c_variational': float( 1e-5 ),
#         'c_monotonicity': float( 100 ),
#         'hidden_layer_widths': int( 500 ),
#         'num_epochs': int( 400 ),
#         'num_hidden_layers': int( 7 ),
#         'num_training_data': int( 100e3 ),
#         'num_testing_data': int( 20e3 ),
#         'learning_rate': float( 5e-4 ),
#     },
#     'newton_parameters': {
#         'tolerance': float( 1e-4 ),
#         'max_iterations': int( 1e2 ),
#     },
#     'paths': {
#         'save_path': r'./ann/simple_pendulum/save',
#         'load_path': r'./ann/simple_pendulum/load',
#     },
#     'plotting_parameters': {
#         'num_plotting_samples': int( 20 ),
#         # 'plot_flag': bool( False ),
#         'plot_flag': bool( True ),
#     },
#     'printing_parameters': {
#         'batch_print_frequency': int( 10 ),
#         'epoch_print_frequency': int( 10 ),
#         'print_flag': bool( True ),
#     },
#     'runtime': {
#         'device': 'cuda:9' if torch.cuda.is_available(  ) else 'cpu',
#         'seed': int( 0 ),
#         'load_flag': bool( False ),
#         'train_flag': bool( True ),
#         'verbose_flag': bool( True ),
#     },
#     'saving_parameters': {
#         'save_flag': bool( True ),
#         'save_frequency': int( 10 ),
#     }
# }


#%% ---------------------------------------- CONFIGURATION EVALUATION ----------------------------------------

# Implement a function to evaluate the simple pendulum roa.
def eval_simple_pendulum( config: dict = BASE_CONFIG ) -> int:

    # Print out a message saying that we are beginning Deep ROA Trial.
    print( '\n' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( 'DEEP ROA TRIAL...' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( '\n' )

    # Retrieve the starting time.
    start_time = time.time(  )

    # Print out a message saying that we are setting up.
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( 'SETTING UP...' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( '\n' )

    # Retrieve the starting time for setting up.
    start_time_setup = time.time(  )

    # Create a copy of the default configuration.
    new_config = deepcopy( BASE_CONFIG )

    # Determine whether to update the configuration.
    if config:                  # If there is a configuration to use...

        # Update the hyperparmaeters of the default configuration to match the user provided configuration.
            # # new_config[ 'hyperparameters' ].update( config )
        new_config[ 'hyperparameters' ].update( config[ 'hyperparameters' ] )

    # Make a copy of the new configuration.
    config = deepcopy( new_config )

    # Set the random seeds.
    np.random.seed( config[ 'runtime' ]['seed' ] )
    random.seed( config[ 'runtime' ][ 'seed' ] )
    torch.manual_seed( config[ 'runtime' ][ 'seed' ] )

    # Set the computational device.
    device = torch.device( config[ 'runtime' ][ 'device' ] )

    # Retrieve the start time.
    start_time = time.time(  )


    #%% ---------------------------------------- DEFINE PINN OPTIONS ----------------------------------------

    # The pinn options are parameters that have no impact on the pde initial-boundary value problem being solved or the neural network that is being trained to solve it.
    # Instead, the pinn option parameters are those that define the tasks the user would like performed and adjust quality-of-life factors, such as where and how often to save, print, and plot relevant network data before, during, and after the training process.

    # Print out a message saying that we are setting up the pinn options.
    print( 'Settig up PINN options...' )

    # Retrieve the pinn options setup starting time.
    start_time_pinn_options = time.time(  )

    # Define the save options.
    save_path = str( config[ 'paths' ][ 'save_path' ] )                                                                                                             # [-] Relative path to the directory in which to save network data, figures, etc.
    save_frequency = torch.tensor( int( config[ 'saving_parameters' ][ 'save_frequency' ] ), dtype = torch.int16, device = device )                                 # [#] Number of epochs after which to save intermediate networks during training. e.g., 1 = Save after every training epoch, 10 = Save after every ten training epochs, 100 = Save after every hundred training epochs.
    save_flag = bool( config[ 'saving_parameters' ][ 'save_flag' ] )                                                                                                # [T/F] Flag that determines whether to save networks during and after training, as well as training and network analysis plots.

    # Define the load options.
    load_path = str( config[ 'paths' ][ 'load_path' ] )                                                                                                             # [-] Relative path to the directory from which to load network data.
    load_flag = bool( config[ 'runtime' ][ 'load_flag' ] )                                                                                                          # [T/F] Flag that determines whether to load network data from the given load directory before training.

    # Define the training options.
    train_flag = bool( config[ 'runtime' ][ 'train_flag' ] )                                                                                                        # [T/F] Flag that determines whether to train the network after creation or loading.

    # Define the printing options.
    batch_print_frequency = torch.tensor( float( config[ 'printing_parameters' ][ 'batch_print_frequency' ] ), dtype = torch.float32, device = device )             # [%] Percent of batches after which to print training information (during an epoch that has been selected for printing).
    epoch_print_frequency = torch.tensor( float( config[ 'printing_parameters' ][ 'epoch_print_frequency' ] ), dtype = torch.float32, device = device )             # [%] Percent of epochs after which to print training information.
    print_flag = bool( config[ 'printing_parameters' ][ 'print_flag' ] )                                                                                            # [T/F] Flag that determines whether to print more or less information when printing.

    # Define the plotting options.
    num_plotting_samples = torch.tensor( int( config[ 'plotting_parameters' ][ 'num_plotting_samples' ] ), dtype = torch.int16, device = device )                   # [#] Number of sample points to use per dimension when plotting network results.
    plot_flag = bool( config[ 'plotting_parameters' ][ 'plot_flag' ] )                                                                                              # [T/F] Flag that determines whether training and network analysis plots are created.

    # Define the verbosity setting.
    verbose_flag = bool( config[ 'runtime' ][ 'verbose_flag' ] )                                                                                                                                # [T/F] Flag that determines whether to print more or less information when printing.

    # Define the newton parameters (used for level set generation).
    newton_tolerance = torch.tensor( float( config[ 'newton_parameters' ][ 'tolerance' ] ), dtype = torch.float32, device = device )                                                            # [-] Convergence tolerance for the Newton's root finding method.
    newton_max_iterations = torch.tensor( int( config[ 'newton_parameters' ][ 'max_iterations' ] ), dtype = torch.int32, device = device )                                                      # [#] Maximum number of Newton's method steps to perform.

    # Define the exploration parameters (used for level set generation).
    exploration_volume_percentage = torch.tensor( float( config[ 'exploration_parameters' ][ 'volume_percentage' ] ), dtype = torch.float32, device = device )                                  # [%] The level set method step size represented as a percentage of the domain volume.  This parameter conveniently scales the step size of the level set method as the dimension of the problem is adjusted. # This works for both initial and final times.
    num_exploration_points = torch.tensor( int( config[ 'exploration_parameters' ][ 'num_points' ] ), dtype = torch.int16, device = device )                                                    # [#] Number of exploration points to generate at each level set method step.
    unique_volume_percentage = torch.tensor( float( config[ 'exploration_parameters' ][ 'unique_percentage' ] ), dtype = torch.float32, device = device )                                       # [%] The tolerance used to determine whether level set points are unique as a percentage of the domain volume.  This parameter conveniently scales the unique tolerance of the level set points as the dimension of the problem is adjusted.

    # Define the classification parameters.
    num_noisy_samples_per_level_set_point = torch.tensor( int( config[ 'classification_parameters' ][ 'num_noisy_samples_per_level_set_point' ] ), dtype = torch.int16, device = device )       # [#] Number of noisy samples per level set point.
    classification_noise_percentage = torch.tensor( float( config[ 'classification_parameters' ][ 'noise_percentage' ] ), dtype = torch.float32, device = device )                              # [%] The classification point noise magnitude represented as a percentage of the domain volume.  This parameter conveniently scales the noise magnitude of the classification points as the dimension of the problem is adjusted.
    classification_dt = torch.tensor( float( config[ 'classification_parameters' ][ 'dt' ] ), dtype = torch.float32, device = device )                                                          # [s] The classification simulation timestep used to forecast classification points.
    classification_tfinal = torch.tensor( float( config[ 'classification_parameters' ][ 'tfinal' ] ), dtype = torch.float32, device = device )                                                  # [s] The classification simulation duration used to forecast classification points.

    # Create the pinn options object.
    pinn_options = pinn_options_class( save_path, save_frequency, save_flag, load_path, load_flag, train_flag, batch_print_frequency, epoch_print_frequency, print_flag, num_plotting_samples, newton_tolerance, newton_max_iterations, exploration_volume_percentage, num_exploration_points, unique_volume_percentage, classification_noise_percentage, num_noisy_samples_per_level_set_point, classification_dt, classification_tfinal, plot_flag, device, verbose_flag )

    # Save the pinn options.
    pinn_options.save( save_path, r'pinn_options.pkl' )

    # Retrieve the pinn options setup end time.
    end_time_pinn_options = time.time(  )

    # Compute the pinn options duration.
    pinn_options_duration = end_time_pinn_options - start_time_pinn_options

    # Print out a message saying that we are done setting up pinn options.
    print( f'Setting up PINN options... Done. Duration = {pinn_options_duration}s = {pinn_options_duration/60}min = {pinn_options_duration/3600}hr' )
    print( '\n' )


    #%% ---------------------------------------- DEFINE PROBLEM SPECIFICATIONS ----------------------------------------

    # The problem specification are parameters that define the pde initial-boundary problem that the user would like to solve.
    # Such parameters include the pde residual, its initial-boundary conditions, and the domain of interest.

    # Print out a message saying that we are setting up problem specifications.
    print( f'Setting up problem specifications...' )

    # Retrieve the pinn options setup starting time.
    start_time_problem_specifications = time.time(  )

    # Define the number of inputs and outputs.
    num_inputs = torch.tensor( 3, dtype = torch.uint8, device = device )                                                                            # [#] Number of network inputs.  For the Yuan-Li PDE, this is the same as the number of spatiotemporal state variables associated with the underlying dynamical system.
    num_outputs = torch.tensor( 1, dtype = torch.uint8, device = device )                                                                           # [#] Number of network outputs.  For the Yuan-Li PDE, this is always one.

    # Define the temporal and spatial domains.
    domain_type = 'cartesian'                                                                                                                       # [-] The type of domain (cartesian, spherical, etc.).  Only cartesian domains are currently supported.
    temporal_domain = torch.tensor( [ 0, 10 ], dtype = torch.float32, device = device )                                                             # [-] Temporal domain of the underlying dynamical system.
    spatial_domain = torch.tensor( [ [ -2*math.pi, 2*math.pi ], [ -6*math.pi, 6*math.pi ] ], dtype = torch.float32, device = device ).T             # [-] Spatial domain of the underlying dynamical system.

    # Define the initial condition parameters.                                                                                                      # [-] Initial condition radius.
    R0 = torch.tensor( 2.5, dtype = torch.float32, device = device )                                                                                # [-] Initial condition radius.
    A0 = torch.tensor( 2.0, dtype = torch.float32, device = device )                                                                                # [-] Initial condition amplitude.
    S0 = torch.tensor( 20.0, dtype = torch.float32, device = device )                                                                               # [-] Initial condition slope.
    P0_shift = torch.tensor( [ 0, 0 ], dtype = torch.float32, device = device )                                                                     # [-] Initial condition input offset.
    z0_shift = -A0/2                                                                                                                                # [-] Initial condition output offset.

    # Define the flow functions.
    flow_function1 = lambda s: torch.unsqueeze( s[ :, 2 ], dim = 1 )                                                                                # [-] Flow function associated with the first state of the underlying dynamical system.
    flow_function2 = lambda s: torch.unsqueeze( -1.417322835*s[ :, 2 ] + -73.575*torch.sin( s[ :, 1 ] ), dim = 1 )                                  # [-] Flow function associated with the second state of the underlying dynamical system.
    flow_functions = [ flow_function1, flow_function2 ]                                                                                             # [-] Flow functions associated with the underlying dynamical system.

    # Define the residual function.
    residual_function = lambda s, dphidt, dphidx1, dphidx2: dphidt - torch.minimum( torch.zeros( size = ( s.shape[ 0 ], 1 ), dtype = torch.float32, device = device ), dphidx1*flow_functions[ 0 ]( s ) + dphidx2*flow_functions[ 1 ]( s ) )                    # [-] Residual function associated with the Yuan-Li PDE.

    # Define the residual code.
    residual_code = [ None, torch.tensor( [ 0 ], dtype = torch.uint8, device = device ), torch.tensor( [ 1 ], dtype = torch.uint8, device = device ), torch.tensor( [ 2 ], dtype = torch.uint8, device = device ) ]                                             # [-] Residual code.  This list specifies which derivatives with respect to the network inputs are required for the residual function inputs.

    # Define the temporal code.
    temporal_code = [ torch.tensor( [ 0 ], dtype = torch.uint8, device = device ) ]                                                                                                                                                                             # [-] Temporal code.  Determines how to compute the temporal derivative of the network output.      

    # Define the initial-boundary condition functions.
    f_ic = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1: ] - P0_shift, 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift              # [-] Initial condition function.
    f_bc_1 = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1: ] - P0_shift, 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift            # [-] Boundary condition function.
    f_bc_2 = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1: ] - P0_shift, 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift            # [-] Boundary condition function.
    f_bc_3 = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1: ] - P0_shift, 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift            # [-] Boundary condition function.
    f_bc_4 = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1: ] - P0_shift, 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift            # [-] Boundary condition function.

    # f_ic = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1: ] - P0_shift, 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift                # [-] Initial condition function.
    # f_bc_1 = lambda s: torch.zeros( ( s.shape[ 0 ], 1 ), dtype = torch.float32, device = device )                                                   # [-] Boundary condition function 1.
    # f_bc_2 = lambda s: torch.zeros( ( s.shape[ 0 ], 1 ), dtype = torch.float32, device = device )                                                   # [-] Boundary condition function 2.
    # f_bc_3 = lambda s: torch.zeros( ( s.shape[ 0 ], 1 ), dtype = torch.float32, device = device )                                                   # [-] Boundary condition function 3.
    # f_bc_4 = lambda s: torch.zeros( ( s.shape[ 0 ], 1 ), dtype = torch.float32, device = device )                                                   # [-] Boundary condition function 4.

    # Define the initial-boundary condition information.
    ibc_types = [ 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet' ]                                                               # [-] Initial-Boundary condition types (e.g., dirichlet, neumann, etc.).
    ibc_dimensions = torch.tensor( [ 0, 1, 1, 2, 2 ], dtype = torch.uint8, device = device )
    ibc_condition_functions = [ f_ic, f_bc_1, f_bc_2, f_bc_3, f_bc_4 ]
    ibc_placements = [ 'lower', 'lower', 'upper', 'lower', 'upper' ]  

    # ibc_types = [ 'dirichlet', 'yuan-li', 'yuan-li', 'yuan-li', 'yuan-li' ]
    # ibc_dimensions = torch.tensor( [ 0, 1, 1, 2, 2 ], dtype = torch.uint8, device = device )
    # ibc_condition_functions = [ f_ic, f_bc_1, f_bc_2, f_bc_3, f_bc_4 ]
    # ibc_placements = [ 'lower', 'lower', 'upper', 'lower', 'upper' ]  

    # Define the PDE name and type.
    pde_name = 'Yuan-Li PDE: Simple Pendulum'                                                                                                       # [-] PDE name.
    pde_type = 'First Order'                                                                                                                        # [-] PDE type.

    # Create the problem specifications object.
    problem_specifications = problem_specifications_class( num_inputs, num_outputs, temporal_domain, spatial_domain, domain_type, residual_function, residual_code, temporal_code, flow_functions, ibc_types, ibc_dimensions, ibc_condition_functions, ibc_placements, pde_name, pde_type, save_path, load_path )

    # Save the problem specifications.
    problem_specifications.save( save_path, r'problem_specifications.pkl' )

    # Retrieve the pinn options setup end time.
    end_time_problem_specifications = time.time(  )

    # Compute the pinn options duration.
    problem_specifications_duration = end_time_problem_specifications - start_time_problem_specifications

    # Print out a message saying that we are done setting up problem specifications.
    print( f'Setting up problem specifications... Done. Duration = {problem_specifications_duration}s = {problem_specifications_duration/60}min = {problem_specifications_duration/3600}hr' )
    print( '\n' )


    #%% ---------------------------------------- DEFINE HYPERPARAMETERS ----------------------------------------

    # The hyperparameters are those that do not affect the problem that is being solved but impact how that problem is being solved, typically by adjusting the underlying neural architecture under consideration or the techniques used to train this network.
    # Examples of several hyperparameters include the number of network hidden layers, along with their widths and activation functions, as well as the optimizer learning rate and training data quantity.

    # Print out a message saying that we are setting up hyperparameters.
    print( f'Setting up hyperparameters...' )

    # Retrieve the hyperparameter setup start time.
    start_time_hyperparameters = time.time(  )

    # Store the network parameters.
    activation_function = str( config[ 'hyperparameters' ][ 'activation_function' ] )                                                                   # [-] Activation function (e.g., tanh, sigmoid, etc.)
    num_hidden_layers = torch.tensor( int( config[ 'hyperparameters' ][ 'num_hidden_layers' ] ), dtype = torch.uint8, device = device )                 # [#] Number of hidden layers.
    hidden_layer_widths = torch.tensor( int( config[ 'hyperparameters' ][ 'hidden_layer_widths' ] ), dtype = torch.int16, device = device )             # [#] Hidden layer widths.

    # This set works for variational loss integration order 1.
    num_training_data = torch.tensor( int( config[ 'hyperparameters' ][ 'num_training_data' ] ), dtype = torch.int32, device = device )                 # [#] Number of training data points.
    num_testing_data = torch.tensor( int( config[ 'hyperparameters' ][ 'num_testing_data' ] ), dtype = torch.int32, device = device )                   # [#] Number of testing data points.

    # Define the percent of training and testing data that should be sampled from the initial condition, the boundary condition, and the interior of the domain.
    p_initial = torch.tensor( 0.25, dtype = torch.float16, device = device )                                                                            # [%] Percentage of training and testing data associated with the initial condition.
    p_boundary = torch.tensor( 0.25, dtype = torch.float16, device = device )                                                                           # [%] Percentage of training and testing data associated with the boundary condition.
    p_residual = torch.tensor( 0.5, dtype = torch.float16, device = device )                                                                            # [%] Percentage of training and testing data associated with the residual.

    # Define the number of training epochs.
    num_epochs = torch.tensor( int( config[ 'hyperparameters' ][ 'num_epochs' ] ), dtype = torch.int32, device = device )                               # [#] Number of training epochs to perform.

    # Define the residual batch size.
    residual_batch_size = torch.tensor( int( 10e3 ), dtype = torch.int32, device = device )                                                             # [#] Training batch size. # This works for variational loss integration order 1.

    # Store the optimizer parameters.
    learning_rate = torch.tensor( float( config[ 'hyperparameters' ][ 'learning_rate' ] ), dtype = torch.float32, device = device )                     # [-] Learning rate.

    # Define the element computation option.
    element_computation_option = 'precompute'                                                                                                           # [string] Determines whether to precompute the finite elements associated with the variational loss (costs more memory) or to dynamically generate these elements during training (costs more time per epoch) (e.g., 'precompute, 'dynamic', etc.).

    # Define the element type.
    element_type = 'rectangular'                                                                                                                        # [string] Finite element type associated with the variational loss (e.g., rectangular, spherical, etc.).  Only rectangular elements are currently supported.

    # Define the element volume percentage.
    element_volume_percent = torch.tensor( 0.01, dtype = torch.float32, device = device )                                                               # [%] The finite element volume size associated with the variational loss represented as a percentage of the domain volume.  

    # Define the integration order.
    integration_order = torch.tensor( 1, dtype = torch.uint8, device = device )                                                                         # [#] Gauss-Legendre integration order.

    # Store the loss coefficients.
    c_IC = torch.tensor( float( config[ 'hyperparameters' ][ 'c_IC' ] ), dtype = torch.float32, device = device )                                       # [-] Initial condition loss weight.
    c_BC = torch.tensor( float( config[ 'hyperparameters' ][ 'c_BC' ] ), dtype = torch.float32, device = device )                                       # [-] Boundary condition loss weight.
    c_residual = torch.tensor( float( config[ 'hyperparameters' ][ 'c_residual' ] ), dtype = torch.float32, device = device )                           # [-] Residual loss weight.
    c_residual_gradient = torch.tensor( float( config[ 'hyperparameters' ][ 'c_residual_gradient' ] ), dtype = torch.float32, device = device )         # [-] Residual gradient loss weight.
    c_variational = torch.tensor( float( config[ 'hyperparameters' ][ 'c_variational' ] ), dtype = torch.float32, device = device )                     # [-] Variational loss weight.
    c_monotonicity = torch.tensor( float( config[ 'hyperparameters' ][ 'c_monotonicity' ] ), dtype = torch.float32, device = device )                   # [-] Monotonicity loss weight.

    # Create the hyperparameters object.
    hyperparameters = hyperparameters_class( activation_function, num_hidden_layers, hidden_layer_widths, num_training_data, num_testing_data, p_initial, p_boundary, p_residual, num_epochs, residual_batch_size, learning_rate, integration_order, element_volume_percent, element_type, element_computation_option, c_IC, c_BC, c_residual, c_residual_gradient, c_variational, c_monotonicity, save_path, load_path )

    # Save the hyperparameters.
    hyperparameters.save( save_path, r'hyperparameters.pkl' )

    # Retrieve the end time.
    end_time_hyperparameters = time.time(  )

    # Compute the hyperparameter setup duration.
    hyperparameters_duration = end_time_hyperparameters - start_time_hyperparameters

    # Print out a message saying that we are done setting up hyperparameters.
    print( f'Setting up hyperparameters... Done. Duration = {hyperparameters_duration}s = {hyperparameters_duration/60}min = {hyperparameters_duration/3600}hr' )
    print( '\n' )


    #%% ---------------------------------------- CREATE THE NEURAL NETWORK ----------------------------------------

    # Print out a message saying that we are setting up the neural network.
    print( 'Setting up neural network...' )

    # Retrieve the neural network setup start time.
    start_time_network = time.time(  )

    # Create the pinn object.
    pinn = pinn_class( pinn_options, hyperparameters, problem_specifications )

    # Load the pinn object.
    pinn = pinn.load( load_path, 'pinn_after_training.pkl' )

    # Set the training flag object.
    pinn.pinn_options.train_flag = train_flag

    # Save the network before training.
    pinn.save( save_path, 'pinn_before_training.pkl' )

    # Retrieve the end time.
    end_time_network = time.time(  )

    # Compute the duration.
    network_duration = end_time_network - start_time_network

    # Print out a message saying that we are done setting up the neural network.
    print( f'Setting up neural network... Done. Duration = {network_duration}s = {network_duration/60}min = {network_duration/3600}hr' )
    print( '\n' )

    # Retrieve the setup end time.
    end_time_setup = time.time(  )

    # Compute the duration.
    setup_duration = end_time_setup - start_time_setup

    # Print out a message saying we are done setting up.
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( f'SETTING UP... DONE. Duration = {setup_duration}s = {setup_duration/60}min = {setup_duration/3600}hr' )
    print( '------------------------------------------------------------------------------------------------------------------------' )


    #%% ---------------------------------------- TRAIN THE NEURAL NETWORK ----------------------------------------

    # Train the neural network.
    pinn.network = pinn.train( pinn.network, pinn_options.train_flag )

    # Save the network after training.
    pinn.save( save_path, 'pinn_after_training.pkl' )


    #%% ---------------------------------------- COMPUTE CLASSIFICATION LOSS ----------------------------------------

    # Print out a message saying that we are computing the classification loss.
    print( '\n' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( 'COMPUTING CLASSIFICATION LOSS...' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( '\n' )

    # Retrieve the starting classification time.
    start_time_classification = time.time(  )

    # Ensure that the number of noisy samples per level set point are correct.
    pinn.pinn_options.num_noisy_samples_per_level_set_point = num_noisy_samples_per_level_set_point

    # Compute the classification loss.
    classification_loss, num_classification_points = pinn.compute_classification_loss( pde = pinn.pde, network = pinn.network, classification_data = None, num_spatial_dimensions = pinn.domain.num_spatial_dimensions, domain = pinn.domain, plot_time = pinn.domain.temporal_domain[ 1, : ], level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), level_set_guesses = None, num_guesses = torch.tensor( int( 1e2 ), dtype = torch.int64, device = pinn.pinn_options.device ), newton_tolerance = newton_tolerance, newton_max_iterations = newton_max_iterations, exploration_radius = pinn.network.exploration_radius_spatial, num_exploration_points = num_exploration_points, unique_tolerance = pinn.network.unique_tolerance_spatial, classification_noise_magnitude = pinn.network.classification_noise_magnitude_spatial, num_noisy_samples_per_level_set_point = pinn.pinn_options.num_noisy_samples_per_level_set_point, domain_subset_type = 'spatial', tspan = torch.tensor( [ 0, classification_tfinal.item(  ) ], dtype = classification_tfinal.dtype, device = classification_tfinal.device ), dt = classification_dt )

    # Print the classification loss.
    print( f'# of Classification Points: {num_classification_points}' )
    print( f'Classification Loss: {classification_loss}' )

    # Retrieve the ending classification time.
    end_time_classification = time.time(  )

    # Compute the classification duration.
    classification_duration = end_time_classification - start_time_classification

    # Print out a message saying that we are computing the classification loss.
    print( '\n' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( f'COMPUTING CLASSIFICATION LOSS... DONE. Duration = {classification_duration}s = {classification_duration/60}min = {classification_duration/3600}hr' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( '\n' )


    #%% ---------------------------------------- PLOT THE NEURAL NETWORK RESULTS ----------------------------------------

    # Determine whether to plot the results.
    if pinn_options.plot_flag:              # If we want to plot the results...

        # Print out a message saying that we are plotting the results.
        print( '------------------------------------------------------------------------------------------------------------------------' )
        print( 'PLOTTING RESULTS...' )
        print( '------------------------------------------------------------------------------------------------------------------------' )
        print( '\n' )

        # Retrieve the start time.
        start_time_plotting = time.time(  )

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
        fig_final_level_set, ax_final_level_set = pinn.plot_network_final_level_set( domain = pinn.domain, network = pinn.network, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), level_set_guess = None, num_guesses = torch.tensor( int( 1e3 ), dtype = torch.int64, device = pinn.pinn_options.device ), newton_tolerance = newton_tolerance, newton_max_iterations = newton_max_iterations, exploration_radius = pinn.network.exploration_radius_spatial, num_exploration_points = num_exploration_points, unique_tolerance = pinn.network.unique_tolerance_spatial, projection_dimensions = None, projection_values = None, fig = fig_final_prediction, dimension_labels = pinn.domain.dimension_labels, save_directory = save_path, as_surface = False, as_stream = False, as_contour = False, show_plot = False )

        # Plot the classification data.
        fig_classification, ax_classification = pinn.plot_network_classifications( network = pinn.network, fig = fig_roa, dimension_labels = pinn.domain.dimension_labels, save_directory = save_path, show_plot = False )

        # Retrieve the end time.
        end_time_plotting = time.time(  )

        # Compute the plotting duration.
        plotting_duration = end_time_plotting - start_time_plotting

        # Print out a complete message.
        print( '------------------------------------------------------------------------------------------------------------------------' )
        print( f'PLOTTING RESULTS... DONE. Duration = {plotting_duration}s = {plotting_duration/60}min = {plotting_duration/3600}hr' )
        print( '------------------------------------------------------------------------------------------------------------------------' )
        print( '\n' )

    # Retrieve the end time.
    end_time = time.time(  )

    # Compute the deep roa trial duration.
    deep_roa_duration = end_time - start_time

    # Print out a message stating that the deep roa trial is complete.
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( f'DEEP ROA TRIAL... DONE. Duration = {deep_roa_duration}s = {deep_roa_duration/60}min = {deep_roa_duration/3600}hr' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( '\n' )

    # Return the classification loss.
    return classification_loss.cpu(  ).detach(  ).numpy(  ).item(  )


# Define behavior when running as main.
if __name__ == "__main__":

    # Compute the classification loss of the simple pendulum example.
    loss = eval_simple_pendulum(  )


