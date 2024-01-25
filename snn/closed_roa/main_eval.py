####################################################################################### 
# THIS SOURCE CODE IS PROPERTY OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. 
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND 
# CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION 
# ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN 
# LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE,
# CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL 
# PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641. 
####################################################################################### 

# Import standard libraries.
import numpy as np
import random
import sys
import torch
import math

# Ensure that the utilities folder for this project is on the system path.
sys.path.append( r'./snn/utilities' )

# Import custom libraries.
from pinn_options_class import pinn_options_class as pinn_options_class
from hyperparameters_class import hyperparameters_class as hyperparameters_class
from problem_specifications_class import problem_specifications_class as problem_specifications_class
from pinn_class import pinn_class as pinn_class


BASE_CONFIG = {
    'classification_parameters': {
        'num_noisy_samples_per_level_set_point': int( 5 ),
        'noise_percentage': float( 1e-3 ),
        'dt': float( 1e-3 ),
        'tfinal': float( 10 ),
    },
    'exploration_parameters': {
        'volume_percentage': float( 1e-2 ),
        'num_points': int( 50 ),
        'unique_percentage': float( 1e-4 ),
    },
    'hyperparameters': {
        'activation_function': 'tanh',
        'c_IC': float( 1.0 ),
        'c_BC': float( 1.0 ),
        'c_residual': float( 0 ),
        'c_variational': float( 0 ),
        'c_monotonicity': float( 0 ),
        'hidden_layer_widths': int( 5e2 ),
        'num_epochs': int( 1000 ),
        'num_hidden_layers': int( 4 ),
        'num_training_data': int( 100e3 ),
        'num_testing_data': int( 20e3 ),
        'learning_rate': float( 0.005 ),
        "neuron_threshold": float( 0.5 ),
        "neuron_current_decay": float( 1.0 ),
        "neuron_voltage_decay": float( 1.0 ),
        "neuron_persistent_state": bool( False ),
        "neuron_requires_grad": bool( False ),
        "synapse_gain": float( 3.0 ),
        'num_timesteps': int( 10 ),
    },
    'newton_parameters': {
        'tolerance': float( 1e-6 ),
        'max_iterations': int( 1e2 ),
    },
    'paths': {
        'save_path': r'./ann/closed_roa/save',
        'load_path': r'./ann/closed_roa/load',
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
        'device': 'cuda:0' if torch.cuda.is_available(  ) else 'cpu',
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


def eval_closed_roa(config: dict = BASE_CONFIG) -> int:
    """
    Evaluate the closed region of attraction (ROA) for a given configuration.

    Args:
        config (dict): Configuration parameters for the evaluation.

    Returns:
        int: The result of the evaluation.

    """

    # Set the random seeds.
    np.random.seed(config["runtime"]["seed"])
    random.seed(config["runtime"]["seed"])
    torch.manual_seed(config["runtime"]["seed"])

    # Set the computational device.
    device = torch.device(config["runtime"]["device"])

    # ----------------------------------------
    # DEFINE PINN OPTIONS
    # ----------------------------------------

    # The pinn options are parameters that have no impact on the pde
    # initial-boundary value problem being solved or the neural network that is
    # being trained to solve it. Instead, the pinn option parameters are those
    # that define the tasks the user would like performed and adjust
    # quality-of-life factors, such as where and how often to save, print, and
    # plot relevant network data before, during, and after the training process.

    # Define the save options.
    save_path = str(config["paths"]["save_path"])                                # [-] Relative path to the directory in which to save network data, figures, etc.
    save_frequency = torch.tensor(int(config['saving_parameters']['save_frequency']), dtype = torch.int16, device = device )               # [#] Number of epochs after which to save intermediate networks during training. e.g., 1 = Save after every training epoch, 10 = Save after every ten training epochs, 100 = Save after every hundred training epochs.
    save_flag = bool(config['saving_parameters']['save_flag'])                                                                                # [T/F] Flag that determines whether to save networks during and after training, as well as training and network analysis plots.

    # Define the load options.
    load_path = str( config[ 'paths' ][ 'load_path' ] )                       # [-] Relative path to the directory from which to load network data.
    # load_flag = True                                                                              # [T/F] Flag that determines whether to load network data from the given load directory before training.
    load_flag = bool( config[ 'runtime' ][ 'load_flag' ] )                                                                               # [T/F] Flag that determines whether to load network data from the given load directory before training.

    # Define the training options.
    train_flag = train_flag = bool( config[ 'runtime' ][ 'train_flag' ] )                                                                              # [T/F] Flag that determines whether to train the network after creation or loading.
    # train_flag = False                                                                            # [T/F] Flag that determines whether to train the network after creation or loading.

    # Define the printing options.
    batch_print_frequency = torch.tensor( int( config[ 'printing_parameters' ][ 'batch_print_frequency' ] ), dtype = torch.int16, device = device )        # [#] Number of batches after which to print training information (during an epoch that has been selected for printing).
    epoch_print_frequency = torch.tensor( config[ 'printing_parameters' ][ 'epoch_print_frequency' ], dtype = torch.int16, device = device )        # [#] Number of epochs after which to print training information
    print_flag = bool( config[ 'printing_parameters' ][ 'print_flag' ] )

    # Define the plotting options.
    num_plotting_samples = torch.tensor( int( config[ 'plotting_parameters' ][ 'num_plotting_samples' ] ), dtype = torch.int16, device = device )         # [#] Number of sample points to use per dimension when plotting network results.
    plot_flag = bool( config[ 'plotting_parameters' ][ 'plot_flag' ] )                                                                                # [T/F] Flag that determines whether training and network analysis plots are created.

    # Define the verbosity setting.
    verbose_flag = bool( config[ 'runtime' ][ 'verbose_flag' ] )                                                                              # [T/F] Flag that determines whether to print more or less information when printing.

    # Define the newton parameters (used for level set generation).
    newton_tolerance = torch.tensor(float( config[ 'newton_parameters' ][ 'tolerance' ] ), dtype = torch.float32, device = device )                     # [-] Convergence tolerance for the Newton's root finding method.
    newton_max_iterations = torch.tensor( int( config[ 'newton_parameters' ][ 'max_iterations' ] ), dtype = torch.int32, device = device )                   # [#] Maximum number of Newton's method steps to perform.

    # Define the exploration parameters (used for level set generation).
    exploration_volume_percentage = torch.tensor( float( config[ 'exploration_parameters' ][ 'volume_percentage' ] ), dtype = torch.float32, device = device )        # [%] The level set method step size represented as a percentage of the domain volume.  This parameter conveniently scales the step size of the level set method as the dimension of the problem is adjusted. # This works for both initial and final times.
    num_exploration_points = torch.tensor( int( config[ 'exploration_parameters' ][ 'num_points' ] ), dtype = torch.int16, device = device )                   # [#] Number of exploration points to generate at each level set method step.
    unique_volume_percentage = torch.tensor( float( config[ 'exploration_parameters' ][ 'unique_percentage' ] ), dtype = torch.float32, device = device )             # [%] The tolerance used to determine whether level set points are unique as a percentage of the domain volume.  This parameter conveniently scales the unique tolerance of the level set points as the dimension of the problem is adjusted.

    # Define the classification parameters.
    num_noisy_samples_per_level_set_point = torch.tensor( int( config[ 'classification_parameters' ][ 'num_noisy_samples_per_level_set_point' ] ), dtype = torch.int16, device = device )   # [#] Number of noisy samples per level set point.
    classification_noise_percentage = torch.tensor( float( config[ 'classification_parameters' ][ 'noise_percentage' ] ), dtype = torch.float32, device = device )      # [%] The classification point noise magnitude represented as a percentage of the domain volume.  This parameter conveniently scales the noise magnitude of the classification points as the dimension of the problem is adjusted.
    classification_dt = torch.tensor( float( config[ 'classification_parameters' ][ 'dt' ] ), dtype = torch.float32, device = device )                    # [s] The classification simulation timestep used to forecast classification points.
    classification_tfinal = torch.tensor( float( config[ 'classification_parameters' ][ 'tfinal' ] ), dtype = torch.float32, device = device )                  # [s] The classification simulation duration used to forecast classification points.

    # Create the pinn options object.
    pinn_options = pinn_options_class( save_path, save_frequency, save_flag, load_path, load_flag, train_flag, batch_print_frequency, epoch_print_frequency, print_flag, num_plotting_samples, newton_tolerance, newton_max_iterations, exploration_volume_percentage, num_exploration_points, unique_volume_percentage, classification_noise_percentage, num_noisy_samples_per_level_set_point, classification_dt, classification_tfinal, plot_flag, device, verbose_flag )

    # Save the pinn options.
    pinn_options.save( save_path, r'pinn_options.pkl' )


    #%% ---------------------------------------- DEFINE PROBLEM SPECIFICATIONS ----------------------------------------

    # The problem specification are parameters that define the pde initial-boundary problem that the user would like to solve.
    # Such parameters include the pde residual, its initial-boundary conditions, and the domain of interest.

    # Define the number of inputs and outputs.
    num_inputs = torch.tensor( 3, dtype = torch.uint8, device = device )                                        # [#] Number of network inputs.  For the Yuan-Li PDE, this is the same as the number of spatiotemporal state variables associated with the underlying dynamical system.
    num_outputs = torch.tensor( 1, dtype = torch.uint8, device = device )                                       # [#] Number of network outputs.  For the Yuan-Li PDE, this is always one.

    # Define the temporal and spatial domains.
    domain_type = 'cartesian'                                                                                   # [-] The type of domain (cartesian, spherical, etc.).  Only cartesian domains are currently supported.
    temporal_domain = torch.tensor( [ 0, 30 ], dtype = torch.float32, device = device )                         # [-] Temporal domain of the underlying dynamical system.
    spatial_domain = torch.tensor( [ [ -1, 4 ], [ -1, 4 ] ], dtype = torch.float32, device = device ).T         # [-] Spatial domain of the underlying dynamical system.

    # Define the initial condition parameters.
    R0 = torch.tensor( 1.0, dtype = torch.float32, device = device )                                            # [-] Initial condition radius.
    A0 = torch.tensor( 2.0, dtype = torch.float32, device = device )                                            # [-] Initial condition amplitude.
    S0 = torch.tensor( 20.0, dtype = torch.float32, device = device )                                           # [-] Initial condition slope.
    P0_shift = torch.tensor( [ math.pi/2, math.pi/2 ], dtype = torch.float32, device = device )                 # [-] Initial condition input offset.
    z0_shift = -A0/2                                                                                            # [-] Initial condition output offset.

    # Define the flow functions.
    flow_function1 = lambda s: torch.unsqueeze( torch.unsqueeze( -torch.sin( s[ :, 1, -1 ] )*( -0.1*torch.cos( s[ :, 1, -1 ] ) - torch.cos( s[ :, 2, -1 ] ) ), dim = 1 ), dim = 2 )     # [-] Flow function associated with the first state of the underlying dynamical system.
    flow_function2 = lambda s: torch.unsqueeze( torch.unsqueeze( -torch.sin( s[ :, 2, -1 ] )*( torch.cos( s[ :, 1, -1 ] ) - 0.1*torch.cos( s[ :, 2, -1 ] ) ), dim = 1 ), dim = 2 )      # [-] Flow function associated with the second state of the underlying dynamical system.
    flow_functions = [ flow_function1, flow_function2 ]                                                                                                                                 # [-] Flow functions associated with the underlying dynamical system.

    # Define the residual function.
    residual_function = lambda s, dphidt, dphidx1, dphidx2: dphidt - torch.minimum( torch.zeros( size = ( s.shape[ 0 ], 1, num_timesteps ), dtype = torch.float32, device = device ), dphidx1*flow_functions[ 0 ]( s ) + dphidx2*flow_functions[ 1 ]( s ) )         # [-] Residual function associated with the Yuan-Li PDE.

    # Define the residual code.
    residual_code = [ None, torch.tensor( [ 0 ], dtype = torch.uint8, device = device ), torch.tensor( [ 1 ], dtype = torch.uint8, device = device ), torch.tensor( [ 2 ], dtype = torch.uint8, device = device ) ]                                                 # [-] Residual code.  This list specifies which derivatives with respect to the network inputs are required for the residual function inputs.

    # Define the temporal code.
    temporal_code = [ torch.tensor( [ 0 ], dtype = torch.uint8, device = device ) ]                                                                                                                                                                                 # [-] Temporal code.  Determines how to compute the temporal derivative of the network output.      

    # Define the initial-boundary condition functions.
    # f_ic = lambda s: 0.32*torch.norm( s[ :, 1:, -1 ], 2, dim = 1, keepdim = True )**2 - 0.96*torch.norm( s[ :, 1:, -1 ], 2, dim = 1, keepdim = True ) - 0.28               # [-] Initial condition function.
    # f_bc_1 = lambda s: 0.32*torch.norm( s[ :, 1:, -1 ], 2, dim = 1, keepdim = True )**2 - 0.96*torch.norm( s[ :, 1:, -1 ], 2, dim = 1, keepdim = True ) - 0.28              # [-] Boundary condition function.
    # f_bc_2 = lambda s: 0.32*torch.norm( s[ :, 1:, -1 ], 2, dim = 1, keepdim = True )**2 - 0.96*torch.norm( s[ :, 1:, -1 ], 2, dim = 1, keepdim = True ) - 0.28              # [-] Boundary condition function.
    # f_bc_3 = lambda s: 0.32*torch.norm( s[ :, 1:, -1 ], 2, dim = 1, keepdim = True )**2 - 0.96*torch.norm( s[ :, 1:, -1 ], 2, dim = 1, keepdim = True ) - 0.28              # [-] Boundary condition function.
    # f_bc_4 = lambda s: 0.32*torch.norm( s[ :, 1:, -1 ], 2, dim = 1, keepdim = True )**2 - 0.96*torch.norm( s[ :, 1:, -1 ], 2, dim = 1, keepdim = True ) - 0.28              # [-] Boundary condition function.

    # f_ic = lambda s: torch.unsqueeze( 0.32*s[ :, 1, -1 ]**2 - 0.96*s[ :, 1, -1 ] - 0.28, dim = 1 )               # [-] Initial condition function.
    # f_bc_1 = lambda s: torch.unsqueeze( 0.32*s[ :, 1, -1 ]**2 - 0.96*s[ :, 1, -1 ] - 0.28, dim = 1 )              # [-] Boundary condition function.
    # f_bc_2 = lambda s: torch.unsqueeze( 0.32*s[ :, 1, -1 ]**2 - 0.96*s[ :, 1, -1 ] - 0.28, dim = 1 )              # [-] Boundary condition function.
    # f_bc_3 = lambda s: torch.unsqueeze( 0.32*s[ :, 1, -1 ]**2 - 0.96*s[ :, 1, -1 ] - 0.28, dim = 1 )              # [-] Boundary condition function.
    # f_bc_4 = lambda s: torch.unsqueeze( 0.32*s[ :, 1, -1 ]**2 - 0.96*s[ :, 1, -1 ] - 0.28, dim = 1 )              # [-] Boundary condition function.

    # f_ic = lambda s: torch.unsqueeze( ( 2/5 )*s[ :, 1, -1 ] - ( 3/5 ), dim = 1 )               # [-] Initial condition function.
    # f_bc_1 = lambda s: torch.unsqueeze( ( 2/5 )*s[ :, 1, -1 ] - ( 3/5 ), dim = 1 )              # [-] Boundary condition function.
    # f_bc_2 = lambda s: torch.unsqueeze( ( 2/5 )*s[ :, 1, -1 ] - ( 3/5 ), dim = 1 )              # [-] Boundary condition function.
    # f_bc_3 = lambda s: torch.unsqueeze( ( 2/5 )*s[ :, 1, -1 ] - ( 3/5 ), dim = 1 )              # [-] Boundary condition function.
    # f_bc_4 = lambda s: torch.unsqueeze( ( 2/5 )*s[ :, 1, -1 ] - ( 3/5 ), dim = 1 )              # [-] Boundary condition function.

    # f_ic = lambda s: torch.unsqueeze( ( 2/5 )*s[ :, 2, -1 ] - ( 3/5 ), dim = 1 )               # [-] Initial condition function.
    # f_bc_1 = lambda s: torch.unsqueeze( ( 2/5 )*s[ :, 2, -1 ] - ( 3/5 ), dim = 1 )              # [-] Boundary condition function.
    # f_bc_2 = lambda s: torch.unsqueeze( ( 2/5 )*s[ :, 2, -1 ] - ( 3/5 ), dim = 1 )              # [-] Boundary condition function.
    # f_bc_3 = lambda s: torch.unsqueeze( ( 2/5 )*s[ :, 2, -1 ] - ( 3/5 ), dim = 1 )              # [-] Boundary condition function.
    # f_bc_4 = lambda s: torch.unsqueeze( ( 2/5 )*s[ :, 2, -1 ] - ( 3/5 ), dim = 1 )              # [-] Boundary condition function.

    f_ic = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1:, -1 ] - P0_shift, 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift                # [-] Initial condition function.
    f_bc_1 = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1:, -1 ] - P0_shift, 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift              # [-] Boundary condition function.
    f_bc_2 = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1:, -1 ] - P0_shift, 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift              # [-] Boundary condition function.
    f_bc_3 = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1:, -1 ] - P0_shift, 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift              # [-] Boundary condition function.
    f_bc_4 = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1:, -1 ] - P0_shift, 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift              # [-] Boundary condition function.

    # f_ic = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1:, : ] - torch.unsqueeze( torch.unsqueeze( P0_shift, dim = 0 ), dim = 2 ), 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift                # [-] Initial condition function.
    # f_bc_1 = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1:, : ] - torch.unsqueeze( torch.unsqueeze( P0_shift, dim = 0 ), dim = 2 ), 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift              # [-] Boundary condition function.
    # f_bc_2 = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1:, : ] - torch.unsqueeze( torch.unsqueeze( P0_shift, dim = 0 ), dim = 2 ), 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift              # [-] Boundary condition function.
    # f_bc_3 = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1:, : ] - torch.unsqueeze( torch.unsqueeze( P0_shift, dim = 0 ), dim = 2 ), 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift              # [-] Boundary condition function.
    # f_bc_4 = lambda s: A0/( 1 + torch.exp( -S0*( torch.norm( s[ :, 1:, : ] - torch.unsqueeze( torch.unsqueeze( P0_shift, dim = 0 ), dim = 2 ), 2, dim = 1, keepdim = True ) - R0 ) ) ) + z0_shift              # [-] Boundary condition function.

    # Define the initial-boundary condition information.
    ibc_types = [ 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet' ]                                                                     # [-] Initial-Boundary condition types (e.g., dirichlet, neumann, etc.).
    ibc_dimensions = torch.tensor( [ 0, 1, 1, 2, 2 ], dtype = torch.uint8, device = device )                                                            # [-] Dimensions associated with each initial-boundary condition.
    ibc_condition_functions = [ f_ic, f_bc_1, f_bc_2, f_bc_3, f_bc_4 ]                                                                                  # [-] List of initial-boundary conditions.
    ibc_placements = [ 'lower', 'lower', 'upper', 'lower', 'upper' ]                                                                                    # [Lower/Upper] Initial-Boundary condition placement.

    # Define the PDE name and type.
    pde_name = 'Yuan-Li PDE: Closed ROA'                                                                                                                # [-] PDE name.
    pde_type = 'First Order'                                                                                                                            # [-] PDE type.

    # Create the problem specifications object.
    problem_specifications = problem_specifications_class( num_inputs, num_outputs, temporal_domain, spatial_domain, domain_type, residual_function, residual_code, temporal_code, flow_functions, ibc_types, ibc_dimensions, ibc_condition_functions, ibc_placements, pde_name, pde_type, save_path, load_path )

    # Save the problem specifications.
    problem_specifications.save( save_path, r'problem_specifications.pkl' )


    #%% ---------------------------------------- DEFINE HYPER-PARAMETERS ----------------------------------------

    # The hyper-parameters are those that do not affect the problem that is being solved but impact how that problem is being solved, typically by adjusting the underlying neural architecture under consideration or the techniques used to train this network.
    # Examples of several hyper-parameters include the number of network hidden layers, along with their widths and activation functions, as well as the optimizer learning rate and training data quantity.

    # Set the neuron & synapse parameters.
    # neuron_parameters = { 'threshold' : torch.tensor( 1.0, dtype = torch.float32, device = device ), 'current_decay' : torch.tensor( 1.0, dtype = torch.float32, device = device ), 'voltage_decay' : torch.tensor( 1.0, dtype = torch.float32, device = device ), 'persistent_state': False, 'requires_grad' : False }
    # synapse_parameters = { 'gain' : torch.tensor( 1.0, dtype = torch.float32, device = device ) }

    # neuron_parameters = { 'threshold' : torch.tensor( 0.5, dtype = torch.float32, device = device ), 'current_decay' : torch.tensor( 1.0, dtype = torch.float32, device = device ), 'voltage_decay' : torch.tensor( 1.0, dtype = torch.float32, device = device ), 'persistent_state': False, 'requires_grad' : False }
    # synapse_parameters = { 'gain' : torch.tensor( 3.0, dtype = torch.float32, device = device ) }

    neuron_parameters = {
        'threshold': torch.tensor( float(config['hyperparameters']['neuron_threshold']), dtype = torch.float32, device = device ),
        'current_decay': torch.tensor( float(config['hyperparameters']['neuron_current_decay']), dtype = torch.float32, device = device ),
        'voltage_decay': torch.tensor( float(config['hyperparameters']['neuron_voltage_decay']), dtype = torch.float32, device = device ),
        'persistent_state': bool(config['hyperparameters']['neuron_persistent_state']),
        'requires_grad': bool(config['hyperparameters']['neuron_requires_grad'])
    }
    synapse_parameters = {
        'gain': torch.tensor( float(config['hyperparameters']['synapse_gain']), dtype = torch.float32, device = device )
    }

    # Define the number of timesteps for which each input is presented to the network.
    num_timesteps = torch.tensor( int(config['hyperparameters']['num_timesteps']), dtype = torch.int16, device = device )                                 # [#] Number of timesteps for which each input is presented to the network.
    # num_timesteps = torch.tensor( 10, dtype = torch.int16, device = device )                                 # [#] Number of timesteps for which each input is presented to the network.

    # Store the network parameters.
    activation_function = str( config[ 'hyperparameters' ][ 'activation_function' ] )                                                                             # [-] Activation function (e.g., tanh, sigmoid, etc.)

    # num_hidden_layers = torch.tensor( 3, dtype = torch.uint8, device = device )                             # [#] Number of hidden layers.
    num_hidden_layers = torch.tensor( int( config[ 'hyperparameters' ][ 'num_hidden_layers' ] ), dtype = torch.uint8, device = device )                             # [#] Number of hidden layers.
    # num_hidden_layers = torch.tensor( 5, dtype = torch.uint8, device = device )                             # [#] Number of hidden layers.

    # hidden_layer_widths = torch.tensor( int( 5e1 ), dtype = torch.int16, device = device )                # [#] Hidden layer widths.
    # hidden_layer_widths = torch.tensor( int( 1e2 ), dtype = torch.int16, device = device )                # [#] Hidden layer widths.

    # # This captured the initial & boundary condition fairly well.
    hidden_layer_widths = torch.tensor( int( config[ 'hyperparameters' ][ 'hidden_layer_widths' ] ), dtype = torch.int16, device = device )                  # [#] Hidden layer widths.

    # hidden_layer_widths = torch.tensor( int( 1e3 ), dtype = torch.int16, device = device )                  # [#] Hidden layer widths.
    # hidden_layer_widths = torch.tensor( int( 2e3 ), dtype = torch.int16, device = device )                  # [#] Hidden layer widths.
    # hidden_layer_widths = torch.tensor( int( 5e3 ), dtype = torch.int16, device = device )                  # [#] Hidden layer widths.
    # hidden_layer_widths = torch.tensor( int( 1e4 ), dtype = torch.int16, device = device )                  # [#] Hidden layer widths.

    # Set the quantity of training and testing data.
    num_training_data = torch.tensor( int( config[ 'hyperparameters' ][ 'num_training_data' ] ), dtype = torch.int32, device = device )                  # [#] Number of training data points.
    num_testing_data = torch.tensor( int( config[ 'hyperparameters' ][ 'num_testing_data' ] ), dtype = torch.int32, device = device )                    # [#] Number of testing data points.

    # Define the percent of training and testing data that should be sampled from the initial condition, the boundary condition, and the interior of the domain.
    p_initial = torch.tensor( 0.25, dtype = torch.float16, device = device )                                # [%] Percentage of training and testing data associated with the initial condition.
    p_boundary = torch.tensor( 0.25, dtype = torch.float16, device = device )                               # [%] Percentage of training and testing data associated with the boundary condition.
    p_residual = torch.tensor( 0.5, dtype = torch.float16, device = device )                                # [%] Percentage of training and testing data associated with the residual.

    # Define the number of training epochs.
    # num_epochs = torch.tensor( int( 1e2 ), dtype = torch.int32, device = device )                           # [#] Number of training epochs to perform.
    num_epochs = torch.tensor( int( config[ 'hyperparameters' ][ 'num_epochs' ] ), dtype = torch.int32, device = device )                           # [#] Number of training epochs to perform.
    # num_epochs = torch.tensor( int( 5e3 ), dtype = torch.int32, device = device )                           # [#] Number of training epochs to perform.

    # Define the residual batch size.
    residual_batch_size = torch.tensor( int( 10e3 ), dtype = torch.int32, device = device )                 # [#] Training batch size. # This works for variational loss integration order 1.

    # Store the optimizer parameters.
    learning_rate = torch.tensor( float( config[ 'hyperparameters' ][ 'learning_rate' ] ), dtype = torch.float32, device = device )                            # [-] Learning rate.

    # Define the element computation option.
    element_computation_option = 'precompute'                                                               # [string] Determines whether to precompute the finite elements associated with the variational loss (costs more memory) or to dynamically generate these elements during training (costs more time per epoch) (e.g., 'precompute, 'dynamic', etc.).
    # element_computation_option = 'dynamic'

    # Define the element type.
    element_type = 'rectangular'                                                                            # [string] Finite element type associated with the variational loss (e.g., rectangular, spherical, etc.).  Only rectangular elements are currently supported.

    # Define the element volume percentage.
    element_volume_percent = torch.tensor( 0.01, dtype = torch.float32, device = device )                   # [%] The finite element volume size associated with the variational loss represented as a percentage of the domain volume.  

    # Define the integration order.
    integration_order = torch.tensor( 1, dtype = torch.uint8, device = device )                             # [#] Gauss-Legendre integration order.

    # Store the loss coefficients.
    # c_IC = torch.tensor( 1.0, dtype = torch.float32, device = device )                      # [-] Initial condition loss weight.
    # c_BC = torch.tensor( 1.0, dtype = torch.float32, device = device )                      # [-] Boundary condition loss weight.
    # c_residual = torch.tensor( 1.0, dtype = torch.float32, device = device )                # [-] Residual loss weight.
    # c_variational = torch.tensor( 1.0, dtype = torch.float32, device = device )             # [-] Variational loss weight.
    # c_monotonicity = torch.tensor( 10.0, dtype = torch.float32, device = device )           # [-] Monotonicity loss weight.

    # Store the loss coefficients.
    c_IC = torch.tensor( float( config[ 'hyperparameters' ][ 'c_IC' ] ), dtype = torch.float32, device = device )                          # [-] Initial condition loss weight.
    c_BC = torch.tensor( float( config[ 'hyperparameters' ][ 'c_BC' ] ), dtype = torch.float32, device = device )                          # [-] Boundary condition loss weight.
    c_residual = torch.tensor( float( config[ 'hyperparameters' ][ 'c_residual' ] ), dtype = torch.float32, device = device )                    # [-] Residual loss weight.
    c_variational = torch.tensor( float( config[ 'hyperparameters' ][ 'c_variational' ] ), dtype = torch.float32, device = device )                 # [-] Variational loss weight.
    c_monotonicity = torch.tensor( float( config[ 'hyperparameters' ][ 'c_monotonicity' ] ), dtype = torch.float32, device = device )               # [-] Monotonicity loss weight.

    # Create the hyper-parameters object.
    hyperparameters = hyperparameters_class( neuron_parameters, synapse_parameters, num_timesteps, activation_function, num_hidden_layers, hidden_layer_widths, num_training_data, num_testing_data, p_initial, p_boundary, p_residual, num_epochs, residual_batch_size, learning_rate, integration_order, element_volume_percent, element_type, element_computation_option, c_IC, c_BC, c_residual, c_variational, c_monotonicity, save_path, load_path )

    # Save the hyperparameters.
    hyperparameters.save( save_path, r'hyperparameters.pkl' )


    #%% ---------------------------------------- CREATE THE NEURAL NETWORK ----------------------------------------

    # Create the pinn object.
    pinn = pinn_class( pinn_options, hyperparameters, problem_specifications )

    # Load the pinn object.
    pinn = pinn.load( load_path, 'pinn_after_training.pkl' )

    # Set the training flag object.
    pinn.pinn_options.train_flag = train_flag

    # Ensure that the user specified exploration radii are used (not those that were loaded).
    pinn.network.exploration_radius_spatial = pinn.compute_exploration_radius( exploration_volume_percentage, pinn.domain, 'spatial' )
    pinn.network.exploration_radius_spatiotemporal = pinn.compute_exploration_radius( exploration_volume_percentage, pinn.domain, 'spatiotemporal' )

    # Ensure that the user specified unique tolerances are used (not those that were loaded).
    pinn.network.unique_tolerance_spatial = pinn.compute_exploration_radius( unique_volume_percentage, pinn.domain, 'spatial' )
    pinn.network.unique_tolerance_spatiotemporal = pinn.compute_exploration_radius( unique_volume_percentage, pinn.domain, 'spatiotemporal' )

    # Ensure that the user specified classification noise parameters are used (not those that were loaded).
    pinn.pinn_options.classification_noise_percentage = classification_noise_percentage
    pinn.network.classification_noise_magnitude_spatial = pinn.compute_classification_noise_magnitude( pinn.pinn_options.classification_noise_percentage, pinn.domain, 'spatial' )
    pinn.network.classification_noise_magnitude_spatiotemporal = pinn.compute_classification_noise_magnitude( pinn.pinn_options.classification_noise_percentage, pinn.domain, 'spatiotemporal' )

    # Save the network before training.
    pinn.save( save_path, 'pinn_before_training.pkl' )


    #%% ---------------------------------------- TRAIN THE NEURAL NETWORK ----------------------------------------

    # Train the neural network.
    pinn.network = pinn.train( pinn.network, pinn_options.train_flag )

    # Save the network after training.
    pinn.save( save_path, 'pinn_after_training.pkl' )


    #%% ---------------------------------------- COMPUTE CLASSIFICATION LOSS ----------------------------------------

    pinn.pinn_options.num_noisy_samples_per_level_set_point = num_noisy_samples_per_level_set_point

    # Print out a classification loss starting message.
    print( '\n' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( 'CLASSIFICATION LOSS' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( 'Computing classification loss...' )

    # Compute the classification loss.
    classification_loss, num_classification_points = pinn.compute_classification_loss( pde = pinn.pde, network = pinn.network, classification_data = None, num_spatial_dimensions = pinn.domain.num_spatial_dimensions, domain = pinn.domain, plot_time = pinn.domain.temporal_domain[ 1, : ], level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), level_set_guesses = None, num_guesses = torch.tensor( int( 1e2 ), dtype = torch.int64, device = pinn.pinn_options.device ), newton_tolerance = newton_tolerance, newton_max_iterations = newton_max_iterations, exploration_radius = pinn.network.exploration_radius_spatial, num_exploration_points = num_exploration_points, unique_tolerance = pinn.network.unique_tolerance_spatial, classification_noise_magnitude = pinn.network.classification_noise_magnitude_spatial, num_noisy_samples_per_level_set_point = pinn.pinn_options.num_noisy_samples_per_level_set_point, domain_subset_type = 'spatial', tspan = torch.tensor( [ 0, classification_tfinal.item(  ) ], dtype = classification_tfinal.dtype, device = classification_tfinal.device ), dt = classification_dt )

    # Print the classification loss.
    print( '\n' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( f'# OF CLASSIFICATION POINTS: {num_classification_points}' )
    print( f'CLASSIFICATION LOSS: {classification_loss}' )
    print( '------------------------------------------------------------------------------------------------------------------------' )


    #%% ---------------------------------------- PLOT THE NEURAL NETWORK RESULTS ----------------------------------------

    # Plot the network domain.
    figs_domain, axes_domain = pinn.plot_domain( pinn.pde, projection_dimensions = None, projection_values = None, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), fig = None, domain_type = 'spatiotemporal', save_directory = save_path, as_surface = True, as_stream = True, as_contour = True, show_plot = False )

    # Plot the initial-boundary conditions.
    figs, axes = pinn.plot_initial_boundary_condition( 20*torch.ones( pinn.domain.spatiotemporal_domain.shape[ -1 ], dtype = torch.int16, device = pinn.pinn_options.device ), pinn.hyperparameters.num_timesteps, pinn.pde, projection_dimensions = None, projection_values = None, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), fig = None, save_directory = save_path, as_surface = True, as_stream = True, as_contour = True, show_plot = False )

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
    fig_initial_level_set, ax_initial_level_set = pinn.plot_network_initial_level_set( domain = pinn.domain, network = pinn.network, num_timesteps = pinn.hyperparameters.num_timesteps, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), level_set_guess = None, num_guesses = torch.tensor( int( 1e2 ), dtype = torch.int64, device = pinn.pinn_options.device ), newton_tolerance = newton_tolerance, newton_max_iterations = newton_max_iterations, exploration_radius = pinn.network.exploration_radius_spatial, num_exploration_points = num_exploration_points, unique_tolerance = pinn.network.unique_tolerance_spatial, projection_dimensions = None, projection_values = None, fig = fig_initial_prediction, dimension_labels = pinn.domain.dimension_labels, save_directory = save_path, as_surface = False, as_stream = False, as_contour = False, show_plot = False )

    # Plot the final level set estimate.
    fig_final_level_set, ax_final_level_set = pinn.plot_network_final_level_set( domain = pinn.domain, network = pinn.network, num_timesteps = pinn.hyperparameters.num_timesteps, level = torch.tensor( 0, dtype = torch.float32, device = pinn.pinn_options.device ), level_set_guess = None, num_guesses = torch.tensor( int( 1e2 ), dtype = torch.int64, device = pinn.pinn_options.device ), newton_tolerance = newton_tolerance, newton_max_iterations = newton_max_iterations, exploration_radius = pinn.network.exploration_radius_spatial, num_exploration_points = num_exploration_points, unique_tolerance = pinn.network.unique_tolerance_spatial, projection_dimensions = None, projection_values = None, fig = fig_final_prediction, dimension_labels = pinn.domain.dimension_labels, save_directory = save_path, as_surface = False, as_stream = False, as_contour = False, show_plot = False )

    # Print out a complete message.
    print( '\n' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( 'COMPLETE' )
    print( '------------------------------------------------------------------------------------------------------------------------' )
    print( '\n' )

    return classification_loss.cpu(  ).detach(  ).numpy(  ).item(  )

if __name__ == "__main__":
    loss = eval_closed_roa()
    print(f"Loss: {loss}")
