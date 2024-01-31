#%% ---------------------------------------- CONFIGURATION ANALYSIS ----------------------------------------

# This script analzes the results of a DROA grid search by analyzing searched configurations and their associated losses.


#%% ---------------------------------------- IMPORT LIBRARIES ----------------------------------------

# Import standard libraries.
import dill as pkl
import numpy as np

# Import custom libraries.


#%% ---------------------------------------- READ IN GRID SEARCH CONFIGURATIONS ----------------------------------------

# # Define the config, config losses, and average config losses paths.
# configs_path = r'ann/simple_pendulum/save/run1_coarse_grid_repeat1/configs.pkl'
# losses_path = r'ann/simple_pendulum/save/run1_coarse_grid_repeat1/config_losses.pkl'
# avg_losses_path = r'ann/simple_pendulum/save/run1_coarse_grid_repeat1/avg_config_losses.pkl'

# configs_path = r'ann/simple_pendulum/save/run1_coarse_grid_repeat2/configs.pkl'
# losses_path = r'ann/simple_pendulum/save/run1_coarse_grid_repeat2/config_losses.pkl'
# avg_losses_path = r'ann/simple_pendulum/save/run1_coarse_grid_repeat2/avg_config_losses.pkl'

# # This set appears to be incomplete.
# configs_path = r'ann/simple_pendulum/save/run1_coarse_grid_repeat3/configs.pkl'
# losses_path = r'ann/simple_pendulum/save/run1_coarse_grid_repeat3/config_losses.pkl'
# avg_losses_path = r'ann/simple_pendulum/save/run1_coarse_grid_repeat3/avg_config_losses.pkl'

# configs_path = r'ann/simple_pendulum/save/run2_coarse_grid_repeat1/configs.pkl'
# losses_path = r'ann/simple_pendulum/save/run2_coarse_grid_repeat1/config_losses.pkl'
# avg_losses_path = r'ann/simple_pendulum/save/run2_coarse_grid_repeat1/avg_config_losses.pkl'

# configs_path = r'ann/simple_pendulum/save/run2_coarse_grid_repeat2/configs.pkl'
# losses_path = r'ann/simple_pendulum/save/run2_coarse_grid_repeat2/config_losses.pkl'
# avg_losses_path = r'ann/simple_pendulum/save/run2_coarse_grid_repeat2/avg_config_losses.pkl'

# configs_path = r'ann/simple_pendulum/save/run1_coarse_grid_repeat1/configs.pkl'
# losses_path = r'ann/simple_pendulum/save/run1_coarse_grid_repeat1/config_losses.pkl'
# avg_losses_path = r'ann/simple_pendulum/save/run1_coarse_grid_repeat1/avg_config_losses.pkl'

# configs_path = r'ann/simple_pendulum/save/run2_fine_grid_repeat1/configs.pkl'
# losses_path = r'ann/simple_pendulum/save/run2_fine_grid_repeat1/config_losses.pkl'
# avg_losses_path = r'ann/simple_pendulum/save/run2_fine_grid_repeat1/avg_config_losses.pkl'

configs_path = r'ann/simple_pendulum/save/run6_larger_coarse_grid_open_BCs/configs.pkl'
losses_path = r'ann/simple_pendulum/save/run6_larger_coarse_grid_open_BCs/config_losses.pkl'
avg_losses_path = r'ann/simple_pendulum/save/run6_larger_coarse_grid_open_BCs/avg_config_losses.pkl'

# Open the config file.
with open( configs_path, 'rb' ) as file:                   # With the config file open...

    # Load the config.
    configs = pkl.load( file )

# Open the config losses file.
with open( losses_path, 'rb' ) as file:                   # With the config losses file open...

    # Load the config losses.
    losses = pkl.load( file )

# Open the average config losses file.
with open( avg_losses_path, 'rb' ) as file:              # With the average config losses file open...

    # Load the average config losses data.
    avg_losses = pkl.load( file )


#%% ---------------------------------------- ANALYZE GRID SEARCH CONFIGURATIONS ----------------------------------------

# Compute the number of completed configurations.
num_configs = len( configs )
num_completed_configs = len( avg_losses )
percent_configs_completed = 100*( num_completed_configs/num_configs )

# Convert the losses to np arrays.
losses = np.array( losses )
avg_losses = np.array( avg_losses )

# Compute the minimum loss.
min_loss = np.min( losses )
min_loss_index = np.argmin( losses )
min_config = configs[ min_loss_index ]

# Compute the minimum average loss.
min_avg_loss = np.min( avg_losses )
min_avg_loss_index = np.argmin( avg_losses )
min_avg_config = configs[ min_avg_loss_index ]

# Compute the maximum loss.
max_loss = np.max( losses )
max_loss_index = np.argmax( losses )
max_config = configs[ max_loss_index ]

# Compute the maximum average loss.
max_avg_loss = np.max( avg_losses )
max_avg_loss_index = np.argmax( avg_losses )
max_avg_config = configs[ max_avg_loss_index ]

# Compute the loss range.
loss_range = max_loss - min_loss


#%% ---------------------------------------- PRINT GRID SEARCH INFORMATION ----------------------------------------

# Print out the relavent information.
print( '\n' )
print( '-------------------- PATH INFORMATION --------------------' )
print( f'Configs Path: {configs_path}' )
print( f'Losses Path: {losses_path}' )
print( f'Average Losses Path: {avg_losses_path}' )
print( '----------------------------------------------------------' )

print( '\n' )
print( '-------------------- MINIMUM LOSS INFORMATION --------------------' )
print( f'Minimum Config (Config #{min_loss_index}): {min_config}' )
print( f'c_IC: {min_config["c_IC"]}' )
print( f'c_BC: {min_config["c_BC"]}' )
print( f'c_residual: {min_config["c_residual"]}' )
print( f'c_variational: {min_config["c_variational"]}' )
print( f'c_monotonicity: {min_config["c_monotonicity"]}' )
print( f'hidden_layer_widths: {min_config["hidden_layer_widths"]}' )
print( f'num_hidden_layers: {min_config["num_hidden_layers"]}' )
print( f'learning_rate: {min_config["learning_rate"]}' )
print( f'Minimum Loss (Config #{min_loss_index}): {min_loss} [%]' )
print( '------------------------------------------------------------------' )

# print( '\n' )
# print( '-------------------- MINIMUM AVERAGE LOSS INFORMATION --------------------' )
# print( f'Minimum Config (Config #{min_avg_loss_index}): {min_avg_config}' )
# print( f'c_IC: {min_avg_config["c_IC"]}' )
# print( f'c_BC: {min_avg_config["c_BC"]}' )
# print( f'c_residual: {min_avg_config["c_residual"]}' )
# print( f'c_variational: {min_avg_config["c_variational"]}' )
# print( f'c_monotonicity: {min_avg_config["c_monotonicity"]}' )
# print( f'hidden_layer_widths: {min_avg_config["hidden_layer_widths"]}' )
# print( f'num_hidden_layers: {min_avg_config["num_hidden_layers"]}' )
# print( f'learning_rate: {min_avg_config["learning_rate"]}' )
# print( f'Minimum Loss (Config #{min_avg_loss_index}): {min_avg_loss} [%]' )
# print( '--------------------------------------------------------------------------' )

print( '\n' )
print( '-------------------- MAXIMUM LOSS INFORMATION --------------------' )
print( f'Maximum Config (Config #{max_loss_index}): {max_config}' )
print( f'c_IC: {max_config["c_IC"]}' )
print( f'c_BC: {max_config["c_BC"]}' )
print( f'c_residual: {max_config["c_residual"]}' )
print( f'c_variational: {max_config["c_variational"]}' )
print( f'c_monotonicity: {max_config["c_monotonicity"]}' )
print( f'hidden_layer_widths: {max_config["hidden_layer_widths"]}' )
print( f'num_hidden_layers: {max_config["num_hidden_layers"]}' )
print( f'learning_rate: {max_config["learning_rate"]}' )
print( f'Maximum Loss (Config #{max_loss_index}): {max_loss} [%]' )
print( '------------------------------------------------------------------' )

# print( '\n' )
# print( '-------------------- MAXIMUM AVERAGE LOSS INFORMATION --------------------' )
# print( f'Maximum Config (Config #{max_avg_loss_index}): {max_avg_config}' )
# print( f'c_IC: {max_avg_config["c_IC"]}' )
# print( f'c_BC: {max_avg_config["c_BC"]}' )
# print( f'c_residual: {max_avg_config["c_residual"]}' )
# print( f'c_variational: {max_avg_config["c_variational"]}' )
# print( f'c_monotonicity: {max_avg_config["c_monotonicity"]}' )
# print( f'hidden_layer_widths: {max_avg_config["hidden_layer_widths"]}' )
# print( f'num_hidden_layers: {max_avg_config["num_hidden_layers"]}' )
# print( f'learning_rate: {max_avg_config["learning_rate"]}' )
# print( f'Maximum Loss (Config #{max_avg_loss_index}): {max_avg_loss} [%]' )
# print( '--------------------------------------------------------------------------' )

print( '\n' )
print( '-------------------- SUMMARY INFORMATION --------------------' )
print( f'# of Configs: {num_completed_configs} / {num_configs} ({percent_configs_completed} [%])' )
print( f'Minimum Loss: {min_loss} [%] (Avg. {min_avg_loss} [%])' )
print( f'Maximum Loss: {max_loss} [%] (Avg. {max_avg_loss} [%])' )
print( f'Loss Range: {loss_range} [%]' )
print( '-------------------------------------------------------------' )

