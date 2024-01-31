#%%---------------------------------------- CLOSED ROA HYPERPARAMETER GRID SEARCH ----------------------------------------

# This file performs a grid search over the hyperparameter space of the deep roa network for the closed roa problem.


#%% ---------------------------------------- IMPORT LIBRARIES ----------------------------------------

# Import standard libraries.
import os
import sys
import time
import datetime
import itertools
import pickle as pkl
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import List

# Edit the system path to include the working directory.
sys.path.append(r'./ann/closed_roa')

# Import custom libraries.
from main_eval import BASE_CONFIG, eval_closed_roa


#%% ---------------------------------------- GRID SEARCH SETUP ----------------------------------------

# Define the number of times to repeat each configuration in the grid.
NUM_REPEATS = 1                 # [#] Number of times each configuration in the grid search is run.

# Define the save directory for all grid search runs.
SAVE_DIR = r'./ann/closed_roa/save'

# Define the folder in which to save results for this particular grid search.
SEARCH_ID = 'run1_coarse_grid'
# SEARCH_ID = 'run2_fine_grid'


#%% ---------------------------------------- DEFINE GRID SEARCH SPACE ----------------------------------------

# # Define the search space. SHAY'S SPACE
# SEARCH_SPACE = {
#     'c_IC': [ float( 17 ), float( 22.1 ), float( 27 ) ],
#     'c_BC': [ float( 27 ), float( 31.1 ), float( 36 ) ],
#     'c_residual':     [ float( 64 ), float( 69.1 ), float( 74 ) ],
#     'c_variational':  [ float( 35 ), float( 39.1 ), float( 43 ) ],
#     'c_monotonicity': [ float( 75 ), float( 80.1 ), float( 85 ) ],
#     'hidden_layer_widths': [ int( 125 ), int( 150 ), int( 175 ) ],
#     'num_hidden_layers':   [ int( 3 ), int( 4 ), int( 5 ) ],
#     'learning_rate':       [ float( 0.01 ), float( 0.005 ), float( 0.001 ) ],
# }

# # Define the search space. Run 1 - Coarse grid.
# SEARCH_SPACE = {
#     'c_IC': [ float( 1.0 ) ],
#     'c_BC': [ float( 1.0 ) ],
#     'c_residual':     [ float( 1e-2 ), float( 1 ), float( 1e2 ) ],
#     'c_variational':  [ float( 1e-2 ), float( 1 ), float( 1e2 ) ],
#     'c_monotonicity': [ float( 1e3 ) ],
#     'hidden_layer_widths': [ int( 50 ), int( 175 ), int( 500 ) ],
#     'num_hidden_layers':   [ int( 3 ), int( 5 ), int( 7 ) ],
#     'learning_rate':       [ float( 5e-4 ), float( 5e-3 ), float( 5e-2 ) ],
# }

# Define the search space. Run 2 - Refined grid.
SEARCH_SPACE = {
    'c_IC': [ float( 1.0 ) ],
    'c_BC': [ float( 1.0 ) ],
    'c_residual':     [ float( 1e-3 ), float( 5e-3 ), float( 1e-2 ), float( 5e-2 ), float( 1e-1 ), float( 5e-1 ), float( 1 ), float( 5 ), float( 1e1 ), float( 5e1 ), float( 1e2 ), float( 5e2 ), float( 1e3 ) ],
    'c_variational':  [ float( 1e-3 ), float( 5e-3 ), float( 1e-2 ), float( 5e-2 ), float( 1e-1 ), float( 5e-1 ), float( 1 ), float( 5 ), float( 1e1 ), float( 5e1 ), float( 1e2 ), float( 5e2 ), float( 1e3 ) ],
    'c_monotonicity': [ float( 1e3 ) ],
    'hidden_layer_widths': [ int( 175 ) ],
    'num_hidden_layers':   [ int( 5 ) ],
    'learning_rate':       [ float( 0.005 ) ],
}

# # Define the search space. SINGLE TEST PARAMETERS.
# SEARCH_SPACE = {
#     'c_IC': [ float( 1.0 ) ],
#     'c_BC': [ float( 1.0 ) ],
#     'c_residual':     [ float( 1 ) ],
#     'c_variational':  [ float( 1 ) ],
#     'c_monotonicity': [ float( 1e3 ) ],
#     'hidden_layer_widths': [ int( 175 ) ],
#     'num_hidden_layers':   [ int( 5 ) ],
#     'learning_rate':       [ float( 0.005 ) ],
# }


# Implement the main function.
def main( base_config = BASE_CONFIG, num_repeats = NUM_REPEATS, search_id = SEARCH_ID, save_dir = SAVE_DIR, search_space = SEARCH_SPACE ):

    # -------------------- SETUP SAVE PATHS --------------------

    # Create a copy of the default configuration.
    base_config = base_config.copy(  )
    
    # Update the save path of the default condfiguration.
    base_config[ 'paths' ][ 'save_path' ] = os.path.join( save_dir, search_id + '/' )

    # Create a directory to save files.
    os.makedirs( base_config[ 'paths' ][ 'save_path' ], exist_ok = True )

    # Perform the Cartesian product of the search space to generate a list of configurations.
    parameter_configs = list( itertools.product( *search_space.values(  ) ) )

    # parameter_configs = itertools.product( *search_space.values(  ) )
    # parameter_configs = list( parameter_configs )

    # Convert the configuration list into a dictionary.
    named_parameter_configs: List[ dict ] = [ dict( zip( search_space.keys(  ), config ) ) for config in parameter_configs ]

    # Retrieve the save directory.
    save_dir = base_config[ 'paths' ][ 'save_path' ]

    # Define a path to a log file.
    std_out_path = os.path.join( save_dir, 'std_out.txt' )

    # Define the configuration save path.
    configs_save_path = os.path.join( save_dir, 'configs.pkl' )


    # -------------------- INITIALIZE GRID SEARCH --------------------

    # Write the configurations to the configuration file.
    with open( configs_save_path, 'wb' ) as file:                               # With the configurations file open...

        # Write the configurations to the configurations file.
        pkl.dump( named_parameter_configs, file )

    # Create paths to files to store the configuration losses.
    avg_config_losses_save_path = os.path.join( save_dir, 'avg_config_losses.pkl' )
    config_losses_save_path = os.path.join( save_dir, 'config_losses.pkl' )
    
    # Initialize list to store the loss associated with each configuration.
    avg_config_losses: List[ float ] = [  ]                                     # [%] Average configuration.  If NUM_REPEATS = 1, then this will equal config_losses.
    config_losses: List[ float ] = [  ]                                         # [%] Configuration loss.

    # Preallocate a variable to store the best loss.
    best_loss = float( 'inf' )

    # -------------------- PERFORM GRID SEARCH --------------------

    # Perform the grid search.
    with open( std_out_path, 'w' ) as std_out:                                  # With the log file open...
        
        # Get the process id.
        process_id = os.getpid(  )

        # State that we are initializing the grid search.
        message = f'-------------------- GRID SEARCH - Timestamp: {datetime.datetime.now(  )} - Process ID: {process_id} --------------------'
        print( message )
        std_out.write( message + '\n' )

        # Compute the loss associated with each configuration in the search space.
        for idx, config in enumerate( named_parameter_configs ):                # Iterate through each of the configurations in the search space...
            
            # Retrieve the pinn options setup end time.
            start_time_problem_specifications = time.time(  )

            # Initialize a variable to store the losses.
            losses = [  ]

            # Compute the loss associated with this configuration for the specified number of repetitions.
            for repeat in range( num_repeats ):                                 # Iterate through each of the repetitions.
                
                # Initialize this configuration by creating a copy of the defaut configuration.
                eval_config = deepcopy( base_config )

                # Update the hyperparameters of this configuration.
                eval_config[ 'hyperparameters' ].update( config )

                # Update the seed of this configuration to match the repeat number.
                eval_config[ 'runtime' ][ 'seed' ] = repeat

                # Update the save path of this configuration.
                eval_config[ 'paths' ][ 'save_path' ] = os.path.join( base_config[ 'paths' ][ 'save_path' ], 'individual_configs/', SEARCH_ID + '_config' + str( idx ) + '_repeat' + str( repeat ) + '/' )

                # Ensure that the updated configuration is valid.
                if len( eval_config ) != len( base_config ):                    # If this configuration does not have the same number of entries as the default configuration...

                    # Throw an error.
                    raise ValueError( "Invalid configuration\n\n" + str( eval_config ) + "\n\n" + str( base_config ) )

                # Create the save path.
                os.makedirs( eval_config[ 'paths' ][ 'save_path' ], exist_ok = True )

                # Evaluate the loss of this configuration.
                loss = eval_closed_roa( eval_config )

                # Append this loss to the loss array.
                losses.append( loss )

            # Compute the average loss for this configuration.
            mean_loss = sum( losses ) / len( losses )

            # Append the average loss for this configuration to the list of average losses.
            avg_config_losses.append( mean_loss )

            # Append the losses for this configuration to this list of losses (note that there will be NUM_REPEATS losses).
            config_losses.append( losses )

            # Determine whether to update the best loss tracker.
            if mean_loss < best_loss:                                                   # If the average loss for this configuration is less than that which was previuosly recorded...

                # Update the best loss.
                best_loss = mean_loss

            # Retrieve the pinn options setup end time.
            end_time_problem_specifications = time.time(  )

            # Compute the pinn options duration.
            problem_specifications_duration = end_time_problem_specifications - start_time_problem_specifications

            # Print a status update.
            message = f'Config {idx + 1} / {len( named_parameter_configs )} ({100*( idx + 1 )/len( named_parameter_configs )} %) -  Mean Loss: {mean_loss} (Best Loss: {best_loss}) - Duration = {problem_specifications_duration}s = {problem_specifications_duration/60}min = {problem_specifications_duration/3600}hr - Time Stamp: {datetime.datetime.now(  )}'
            print( message )
            std_out.write( message + '\n' )

            # Write the average configuration losses to the appropriate file.
            with open( avg_config_losses_save_path, 'wb' ) as losses_writer:            # With the average configuration losses file open...

                # Write the average configuration losses to the appropriate file.
                pkl.dump( avg_config_losses, losses_writer )

            # Write the configuration losses to the appropriate file.
            with open( config_losses_save_path, 'wb') as avg_losses_writer:             # With the configuration losses file open...

                # Write the configuration lsoses to the appropriate file.
                pkl.dump( config_losses, avg_losses_writer )

            # Plot the average configuration losses.
            plt.plot( sorted( avg_config_losses ) )

            # Save the average configuration losses plot.
            plt.savefig( os.path.join( save_dir, 'loss_plot.png' ) )

            # Close the average configuration losses plot.
            plt.close(  )

        # Print a status update.
        message = f'-----------------------------------------------------------------------------------------------'
        print( message )
        std_out.write( message + '\n' )


# Define behavior when running as main.
if __name__ == '__main__':                                                          # If this file was run as main (as opposed to being called by another file)...
    
    # Run the main function.
    main(  )
