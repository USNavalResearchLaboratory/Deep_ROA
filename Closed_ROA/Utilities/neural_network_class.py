####################################################################################### 
# THIS SOURCE CODE IS PROPERTY OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. 
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND 
# CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION 
# ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN 
# LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE,
# CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL 
# PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641. 
####################################################################################### 


#%% ------------------------------------------------------------ NEURAL NETWORK CLASS ------------------------------------------------------------

# This file implements a neural network class for the purpose of integrating PDEs.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch
import matplotlib.pyplot as plt
import warnings
import time

# Import custom libraries.
from tensor_utilities_class import tensor_utilities_class as tensor_utilities_class
from classification_utilities_class import classification_utilities_class as classification_utilities_class
from plotting_utilities_class import plotting_utilities_class as plotting_utilities_class


#%% ------------------------------------------------------------ NEURAL NETWORK CLASS ------------------------------------------------------------

# Implement the neural network class.
class neural_network_class( torch.nn.Module ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self, layers, activation_string, learning_rate, residual_batch_size, num_epochs, residual_function, residual_code, temporal_code, training_data, testing_data, plotting_data, dimension_labels, element_computation_option, batch_print_frequency, epoch_print_frequency, c_IC, c_BC, c_residual, c_variational, c_monotonicity, newton_tolerance, newton_max_iterations, exploration_radius_spatial, exploration_radius_spatiotemporal, num_exploration_points, unique_tolerance_spatial, unique_tolerance_spatiotemporal, classification_noise_magnitude_spatial, classification_noise_magnitude_spatiotemporal, device = 'cpu', verbose_flag = False ):

        # Construct the parent class.
        super( neural_network_class, self ).__init__(  )

        # Create an instance of the tensor utilities class.
        self.tensor_utilities = tensor_utilities_class(  )

        # Create an instance of the classification utilities class.
        self.classification_utilities = classification_utilities_class(  )

        # Create an instance of the plotting utilities class.
        self.plotting_utilities = plotting_utilities_class(  )

        # Store the device.
        self.device = device

        # Store the verbosity setting.
        self.verbose_flag = verbose_flag

        # Store the dimension labels.
        self.dimension_labels = dimension_labels

        # Store the element computation option.
        self.element_computation_option = element_computation_option


        #%% -------------------- Store Hyperparameters --------------------

        # Store the network structure data.
        self.layers = self.validate_layers( layers )
        self.num_inputs = self.layers[ 0 ]
        self.num_outputs = self.layers[ -1 ]
        self.num_hidden_layers = self.compute_num_hidden_layers( self.layers )
        self.num_weights_biases = self.compute_num_weights_biases( self.layers )
        self.activation_string = self.validate_activation_string( activation_string )

        # Store the training parameters.
        self.learning_rate = learning_rate
        self.residual_batch_size = residual_batch_size
        self.num_epochs = num_epochs


        #%% -------------------- PDE INFORMATION --------------------

        # Store the residual information.
        self.residual_function = self.validate_residual_function( residual_function )
        self.residual_code = self.validate_residual_code( residual_code )
        self.temporal_code = self.validate_temporal_code( temporal_code )
        self.derivative_required_for_residual = self.residual_code2derivative_requirements( self.residual_code )
        self.derivative_required_for_temporal_gradient = self.temporal_code2derivative_requirements( self.temporal_code )
        self.num_residual_inputs = self.compute_num_residual_required_derivatives( self.residual_code )

        # Ensure that the residual function and residual code are compatible.
        assert self.is_residual_function_code_compatible( self.residual_function, num_residual_function_inputs = self.num_residual_inputs )


        #%% -------------------- Store Training, Testing, & Plotting Data --------------------

        # Store the training data.
        self.training_data = training_data

        # Store the testing data.
        self.testing_data = testing_data

        # Store the plotting data.
        self.plotting_data = plotting_data


        #%% -------------------- Store Printing Options --------------------

        # Store the batch and epoch print frequencies.
        self.batch_print_frequency = batch_print_frequency
        self.epoch_print_frequency = epoch_print_frequency


        #%% -------------------- Store Loss Parameters --------------------

        # Store the loss parameters.
        self.c_IC = c_IC
        self.c_BC = c_BC
        self.c_residual = c_residual
        self.c_variational = c_variational
        self.c_monotonicity = c_monotonicity
        

        #%% -------------------- Store the Level Set Parameters --------------------

        # Store the newton parameters.
        self.newton_tolerance = newton_tolerance
        self.newton_max_iterations = newton_max_iterations
        
        # Store the exploration parameters.
        self.exploration_radius_spatial = exploration_radius_spatial
        self.exploration_radius_spatiotemporal = exploration_radius_spatiotemporal
        self.num_exploration_points = num_exploration_points
        self.unique_tolerance_spatial = unique_tolerance_spatial
        self.unique_tolerance_spatiotemporal = unique_tolerance_spatiotemporal

        # Store the classification loss parameters.
        self.classification_noise_magnitude_spatial = classification_noise_magnitude_spatial
        self.classification_noise_magnitude_spatiotemporal = classification_noise_magnitude_spatiotemporal


        #%% -------------------- Initialization Functions --------------------

        # Compute the number of batches.
        self.num_batches, self.initial_condition_batch_size, self.boundary_condition_batch_size = self.initialize_batch_info( self.training_data.initial_condition_data[ 0 ].num_data_points, self.training_data.boundary_condition_data[ 0 ].num_data_points, self.training_data.num_residual_points, self.residual_batch_size )

        # Store the activation function choice.
        self.activation = self.initialize_activation_function( self.activation_string )

        # Initialize the forward stack.
        self.forward_stack = self.initialize_forward_stack( self.layers, self.activation )

        # Initialize the optimizer.
        self.optimizer = torch.optim.Adam( self.parameters(  ), lr = self.learning_rate, betas = ( 0.9, 0.999 ), eps = 1e-08, weight_decay = 0 )

        # Initialize the training losses, testing epochs, and testing losses.
        self.training_losses, self.testing_epochs, self.testing_losses = self.initialize_training_testing_losses( self.num_epochs )

        # Initialize the classification loss function.
        self.classification_loss_function = torch.nn.BCELoss(  )

        # Initialize the classification loss.
        self.set_classification_loss( torch.empty( 1, dtype = torch.float32, device = self.device ) )



    #%% ------------------------------------------------------------ PREPROCESS FUNCTIONS ------------------------------------------------------------

    # Implement a function to preprocess the layers.
    def preprocess_layers( self, layers = None ):

        # Determine whether to set the layers to be the stored value.
        if layers is None:              # If layers were not provided...

            # Set the layers to be the stored value.
            layers = self.layers

        # Return the layers.
        return layers

    
    # Implement a function to preprocess the residual code.
    def preprocess_residual_code( self, residual_code = None ):

        # Determine whether to set the residual code to be the stored value.
        if residual_code is None:       # If the residual code was not provided...

            # Set the residual code to be the stored value.
            residual_code = self.residual_code

        # Return the residual code.
        return residual_code


    # Implement a function to preprocess the temporal code.
    def preprocess_temporal_code( self, temporal_code = None ):

        # Determine whether to set the temporal code to be the stored value.
        if temporal_code is None:       # If the temporal code was not provided...

            # Set the temporal code to be the stored value.
            temporal_code = self.temporal_code

        # Return the temporal code.
        return temporal_code


    # Implement a function to preprocess the number of inputs.
    def preprocess_num_inputs( self, num_inputs = None ):

        # Determine whether to use the stored number of inputs.
        if num_inputs is None:                  # If the number of inputs was not provided...

            # Set the number of inputs to be the stored value.
            num_inputs = self.num_inputs

        # Return the number of inputs.
        return num_inputs


    # Implement a function to preprocess the number of epochs.
    def preprocess_num_epochs( self, num_epochs = None ):

        # Determine whether to use the stored number of epochs.
        if num_epochs is None:                  # If the number of epochs is None...

            # Set the number of epochs to be the stored value.
            num_epochs = self.num_epochs

        # Return the number of epochs.
        return num_epochs


    # Implement a function to preprocess the number of batches.
    def preprocess_num_batches( self, num_batches = None ):

        # Determine whether to use the stored number of batches.
        if num_batches is None:                     # If the number of batches is not provided...

            # Set the number of batches to be the stored value.
            num_batches = self.num_batches

        # Return the number of batches.
        return num_batches

    
    # Implement a function to preprocess the batch number.
    def preprocess_batch_number( self, batch_number = None ):

        # Determine whether to set the batch number to a default value.
        if batch_number is None:                # If the batch number was not provided...

            # Set the batch number to be zero.
            batch_number = torch.tensor( 0, dtype = torch.int64, device = self.device )

        # Return the batch number.
        return batch_number


    # Implement a function to preprocess the activation string.
    def preprocess_activation_string( self, activation_string = None ):

        # Determine whether to use the stored activation string.
        if activation_string is None:           # If the activation string was not provided...

            # Set the activation string to be the stored value.
            activation_string = self.activation_string

        # Return the activation string.
        return activation_string


    # Implement a function to preprocess the training data.
    def preprocess_training_data( self, training_data = None ):

        # Determine whether to use the stored training data.
        if training_data is None:               # If the training data was not provided...

            # Set the training data.
            training_data = self.training_data

        # Return the training data.
        return training_data


    # Implement a function to preprocess the testing data.
    def preprocess_testing_data( self, testing_data = None ):

        # Determine whether to use the stored testing data.
        if testing_data is None:                # If the testing data was not provided...

            # Set the testing data.
            testing_data = self.testing_data

        # Return the testing data.
        return testing_data


    # Implement a function to preprocess plotting data.
    def preprocess_plotting_data( self, plotting_data = None ):

        # Determine whether to use the stored plotting data.
        if plotting_data is None:               # If the plotting data was not provided...

            # Set the plotting data.
            plotting_data = self.plotting_data
        
        # Return the plotting data.
        return plotting_data


    # Implement a function to preprocess plot times.
    def preprocess_plot_times( self, plot_times = None ):

        # Determine whether to set the default plot time.
        if plot_times is None:                                  # If no plot times were provided...

            # Set the plot times to be a zero tensor.
            plot_times = torch.tensor( 0, dtype = torch.float32, device = self.device )

        # Return the plot time.
        return plot_times


    # Implement a function to preprocess the number of initial condition points.
    def preprocess_num_initial_condition_points( self, num_initial_condition_points = None ):

        # Determine whether to use the stored number of initial condition points.
        if num_initial_condition_points is None:                   # If the number of initial condition points was not provided...

            # Set the number of initial condition points to be the stored value.
            num_initial_condition_points = self.training_data.initial_condition_data[ 0 ].num_data_points

        # Return the number of initial condition points.
        return num_initial_condition_points


    # Implement a function to preprocess the number of boundary condition points.
    def preprocess_num_boundary_condition_points( self, num_boundary_condition_points = None ):

        # Determine whether to use the stored number of boundary condition points.
        if num_boundary_condition_points is None:                   # If the number of boundary condition points was not provided...

            # Set the number of boundary condition points to be the stored value.
            num_boundary_condition_points = self.training_data.boundary_condition_data[ 0 ].num_data_points

        # Return the number of boundary condition points.
        return num_boundary_condition_points


    # Implement a function to preprocess the number of residual points.
    def preprocess_num_residual_points( self, num_residual_points = None ):

        # Determine whether to use the stored number of residual points.
        if num_residual_points is None:             # If the number of residual points was not provided...

            # Set the number of residual points to be the stored value.
            num_residual_points = self.training_data.num_residual_points

        # Return the number of residual points.
        return num_residual_points


    # Implement a function to preprocess the residual batch size.
    def preprocess_residual_batch_size( self, residual_batch_size = None ):

        # Determine whether to use the stored residual batch size.
        if residual_batch_size is None:             # If the residual batch size was not provided...

            # Set the residual batch size to be the stored value.
            residual_batch_size = self.residual_batch_size

        # Return the residual batch size.
        return residual_batch_size


    # Implement a function to preprocess residual input batch data.
    def preprocess_residual_input_data_batch( self, residual_input_data_batch = None ):

        # Determine whether to use the stored residual input data batch.
        if residual_input_data_batch is None:                # If the input data batch was not provided...

            # Use the stored residual input data batch.
            residual_input_data_batch = self.training_data.residual_data.residual_input_data_batch

        # Return the residual input batch data.
        return residual_input_data_batch


    # Implement a function to preprocess the integration points batch.
    def preprocess_integration_points_batch( self, xs_integration_points_batch = None ):

        # Determine whether to use the stored integration points batch.
        if xs_integration_points_batch is None:             # If the integration points batch was not provided...

            # Set the integration points batch to be the stored value.
            xs_integration_points_batch = self.training_data.variational_data.xs_integration_points_batch

        # Return the integration points batch.
        return xs_integration_points_batch

    
    # Implement a function to preprocess the basis values batch.
    def preprocess_basis_values_batch( self, G_basis_values_batch = None ):

        # Determine whether to use the stored basis values batch.
        if G_basis_values_batch is None:             # If the basis values batch was not provided...

            # Set the basis values batch to be the stored value.
            G_basis_values_batch = self.training_data.variational_data.G_basis_values_batch

        # Return the basis values batch.
        return G_basis_values_batch


    # Implement a function to preprocess the integration weights batch.
    def preprocess_integration_weights_batch( self, W_integration_weights_batch = None ):

        # Determine whether to use the stored integration weights batch.
        if W_integration_weights_batch is None:             # If the integration weights batch was not provided...

            # Set the integration weights batch to be the stored value.
            W_integration_weights_batch = self.training_data.variational_data.W_integration_weights_batch

        # Return the integration weights batch.
        return W_integration_weights_batch


    # Implement a function to preprocess the jacobian batch.
    def preprocess_jacobian_batch( self, sigma_jacobian_batch = None ):

        # Determine whether to use the stored jacobian batch.
        if sigma_jacobian_batch is None:             # If the jacobian batch was not provided...

            # Set the jacobian batch to be the stored value.
            sigma_jacobian_batch = self.training_data.variational_data.sigma_jacobian_batch

        # Return the jacobian batch.
        return sigma_jacobian_batch


    # Implement a function to preprocess initial condition data.
    def preprocess_initial_condition_data( self, initial_condition_data = None ):

        # Determine whether to use the stored initial condition data.
        if initial_condition_data is None:                          # If the initial condition data was not provided...

            # Set the initial condition data to be the stored value.
            initial_condition_data = self.training_data.initial_condition_data

        # Return the initial condition data.
        return initial_condition_data


    # Implement a function to preprocess boundary condition data.
    def preprocess_boundary_condition_data( self, boundary_condition_data = None ):

        # Determine whether to use the stored boundary condition data.
        if boundary_condition_data is None:                     # If the boundary condition data was not provided...

            # Set the boundary condition data to be the stored value.
            boundary_condition_data = self.training_data.boundary_condition_data

        # Return the boundary condition data.
        return boundary_condition_data

    
    # Implement a function to preprocess residual data.
    def preprocess_residual_data( self, residual_data = None ):

        # Determine whether to use the stored residual data.
        if residual_data is None:                           # If the residual data was not provided...

            # Set the residual data to be the stored value.
            residual_data = self.training_data.residual_data

        # Return the residual data.
        return residual_data


    # Implement a function to preprocess variational data.
    def preprocess_variational_data( self, variational_data = None ):

        # Determine whether to use the stored variational data.
        if variational_data is None:                    # If the variational data was not provided...

            # Set the variational data to be the stored value.
            variational_data = self.training_data.variational_data
        
        # Return the variational data.
        return variational_data


    # Implement a function to preprocess the epoch print frequency.
    def preprocess_epoch_print_frequency( self, epoch_print_frequency = None ):

        # Determine whether to use the stored epoch print frequency.
        if epoch_print_frequency is None:               # If the epoch print frequency was not provided...

            # Set the epoch print frequency to be the stored value.
            epoch_print_frequency = self.epoch_print_frequency

        # Return the epoch print frequency.
        return epoch_print_frequency


    # Implement a function to preprocess the batch print frequency.
    def preprocess_batch_print_frequency( self, batch_print_frequency = None ):

        # Determine whether to use the stored batch print frequency.
        if batch_print_frequency is None:               # If the batch print frequency was not provided...

            # Set the batch print frequency to be the stored value.
            batch_print_frequency = self.batch_print_frequency

        # Return the batch print frequency.
        return batch_print_frequency


    # Implement a function to preprocess the verbose flag.
    def preprocess_verbose_flag( self, verbose_flag = None ):

        # Determine whether to use the stored verbose flag.
        if verbose_flag is None:                        # If the verbose flag was not provided...

            # Set the verbose flag to be the stored value.
            verbose_flag = self.verbose_flag

        # Return the verbose flag.
        return verbose_flag


    # Implement a function to preprocess the derivative required for residual value.
    def preprocess_derivative_required_for_residual( self, derivative_required_for_residual = None ):

        # Determine whether to use the stored derivative required for residual value.
        if derivative_required_for_residual is None:                # If the derivative required for residual value was not provided...

            # Set the derivative required for residual to be the stored value.
            derivative_required_for_residual = self.derivative_required_for_residual

        # Return the derivative required for residual.
        return derivative_required_for_residual


    # Implement a function to preprocess the derivative required for temporal gradient.
    def preprocess_derivative_required_for_temporal_gradient( self, derivative_required_for_temporal_gradient = None ):

        # Determine whether to use the stored derivative required for temporal gradient.
        if derivative_required_for_temporal_gradient is None:                # If the derivative required for temporal gradient was not provided...

            # Set the derivative required for temporal gradient to be the stored value.
            derivative_required_for_temporal_gradient = self.derivative_required_for_temporal_gradient

        # Return the derivative required for temporal gradient.
        return derivative_required_for_temporal_gradient


    # Implement a function to preprocess the dimension labels.
    def preprocess_dimension_labels( self, dimension_labels = None ):

        # Determine whether to use the stored dimension labels.
        if dimension_labels is None:                # If the dimension labels were not provided...

            # Set the dimension labels to be the stored value.
            dimension_labels = self.dimension_labels

        # Return the dimension labels.
        return dimension_labels


    # Implement a function to preprocess training epochs.
    def preprocess_training_epochs( self, training_epochs = None ):

        # Determine whether to use the stored training epochs
        if training_epochs is None:                 # If the training epochs were not provided...

            # Set the training epochs to be the stored value.
            training_epochs = self.training_epochs
        
        # Return the training epochs.
        return training_epochs


    # Implement a function to preprocess training losses.
    def preprocess_training_losses( self, training_losses = None ):

        # Determine whether to use the stored training losses.
        if training_losses is None:                 # If the training losses were not provided...

            # Set the training losses to be the stored value.
            training_losses = self.training_losses

        # Return the training losses.
        return training_losses


    # Implement a function to preprocess testing epochs.
    def preprocess_testing_epochs( self, testing_epochs = None ):

        # Determine whether to use the stored testing epochs.
        if testing_epochs is None:              # If the testing epochs were not provided...

            # Set the testing epochs to be the stored value.
            testing_epochs = self.testing_epochs

        # Return the testing epochs.
        return testing_epochs


    # Implement a function to preprocess testing losses.
    def preprocess_testing_losses( self, testing_losses = None ):

        # Determine whether to use hte stored testing losses.
        if testing_losses is None:              # If the testing losses were not provided...

            # Set the testing losses to be the stored value.
            testing_losses = self.testing_losses

        # Return the testing losses.
        return testing_losses


    # Implement a function to preprocess the level.
    def preprocess_level( self, level = None ):

        # Determine whether to use the default level.
        if level is None:           # If the level was not provided...

            # Set the level to be zero.
            level = torch.tensor( 0, dtype = torch.float32, device = self.device )

        # Return the level.
        return level


    # Implement a function to preprocess the number of dimensions.
    def preprocess_num_dimensions( self, num_dimensions = None ):

        # Determine whether to use the default number of inputs.
        if num_dimensions is None:          # If the number of dimensions was not provided...

            # Set the number of dimensions to be the number of inputs.
            num_dimensions = self.num_inputs

        # Return the number of dimensions.
        return num_dimensions


    # Implement a function to preprocess the level set guess.
    def preprocess_level_set_guess( self, level_set_guess = None, num_dimensions = None ):

        # Preprocess the number of dimensions.
        num_dimensions = self.preprocess_num_dimensions( num_dimensions )

        # Determine whether to use the default level set guess.
        if level_set_guess is None:             # If the level set guess was not provided...

            # Set the level set guess to be zero.
            level_set_guess = torch.zeros( ( 1, num_dimensions ), dtype = torch.float32, device = self.device )

        # Return the level set guess.
        return level_set_guess

    
    # Implement a function to preprocess the newton tolerance.
    def preprocess_newton_tolerance( self, newton_tolerance = None ):

        # Determine whether to use the default network tolerance.
        if newton_tolerance is None:            # If the newton tolerance was not provided...

            # Set the newton tolerance to be the stored value.
            newton_tolerance = self.newton_tolerance

        # Return the newton tolerance.
        return newton_tolerance

    
    # Implement a function to preprocess the maximum number of newton iterations.
    def preprocess_newton_max_iterations( self, newton_max_iterations = None ):

        # Determine whether to use the default maximum number of newton iterations.
        if newton_max_iterations is None:           # If the maximum number of newton iterations were not provided...

            # Set the newton maximum iterations to be the stored value.
            newton_max_iterations = self.newton_max_iterations

        # Return the maximum number of newton iterations.
        return newton_max_iterations


    # Implement a function to preprocess the spatial exploration radius.
    def preprocess_exploration_radius_spatial( self, exploration_radius_spatial = None ):

        # Determine whether to use the stored spatial exploration radius.
        if exploration_radius_spatial is None:              # If the spatial exploration radius was not provided...

            # Set the exploration radius to be the stored value.
            exploration_radius_spatial = self.exploration_radius_spatial

        # Return the spatial exploration radius.
        return exploration_radius_spatial


    # Implement a function to preprocess the spatiotemporal exploration radius.
    def preprocess_exploration_radius_spatiotemporal( self, exploration_radius_spatiotemporal = None ):

        # Determine whether to use the stored spatiotemporal exploration radius.
        if exploration_radius_spatiotemporal is None:              # If the spatiotemporal exploration radius was not provided...

            # Set the spatiotemporal exploration radius to be the stored value.
            exploration_radius_spatiotemporal = self.exploration_radius_spatiotemporal

        # Return the spatiotemporal exploration radius.
        return exploration_radius_spatiotemporal


    # Implement a function to preprocess the exploration radius.
    def preprocess_exploration_radius( self, exploration_radius = None, domain_subset_type = 'spatiotemporal' ):

        # Determine how to preprocess the exploration radius.
        if domain_subset_type.lower(  ) == 'spatial':               # If the domain subset type is spatial...

            # Preprocess the exploration radius.
            exploration_radius = self.preprocess_exploration_radius_spatial( exploration_radius )

        elif domain_subset_type.lower(  ) == 'spatiotemporal':      # If the domain subset type is spatiotemporal...

            # Preprocess the exploration radius.
            exploration_radius = self.preprocess_exploration_radius_spatiotemporal( exploration_radius )

        else:                                                       # Otherwise... ( i.e., the domain subset is not recognized... )

            # Throw an error.
            raise ValueError( f'Invalid domain subset type: {domain_subset_type}' )

        # Return the exploration radius.
        return exploration_radius


    # Implement a function to preprocess the number of exploration points.
    def preprocess_num_exploration_points( self, num_exploration_points = None ):

        # Determine whether to use the stored number of exploration points.
        if num_exploration_points is None:          # If the number of exploration points was not provided...

            # Set the number of exploration points to be the stored value.
            num_exploration_points = self.num_exploration_points

        # Return the number of exploration points.
        return num_exploration_points


    # Implement a function to preprocess the spatial unique tolerance.
    def preprocess_unique_tolerance_spatial( self, unique_tolerance_spatial = None ):

        # Determine whether to use the stored spatial unique tolerance.
        if unique_tolerance_spatial is None:                # If the spatial unique tolerance was not provided...

            # Set the spatial unique tolerance to be the stored value.
            unique_tolerance_spatial = self.unique_tolerance_spatial

        # Return the spatial unique tolerance.
        return unique_tolerance_spatial


    # Implement a function to preprocess the spatiotemporal unique tolerance.
    def preprocess_unique_tolerance_spatiotemporal( self, unique_tolerance_spatiotemporal = None ):

        # Determine whether to use the stored spatiotemporal unique tolerance.
        if unique_tolerance_spatiotemporal is None:                                     # If the spatiotemporal unique tolerance was not provided...

            # Set the spatiotemporal unique tolerance to be the stored value.
            unique_tolerance_spatiotemporal = self.unique_tolerance_spatiotemporal

        # Return the spatiotemporal unique tolerance.
        return unique_tolerance_spatiotemporal


    # Implement a function to preprocess the unique tolerance.
    def preprocess_unique_tolerance( self, unique_tolerance = None, domain_subset_type = 'spatiotemporal' ):

        # Determine how to preprocess the unique tolerance.
        if domain_subset_type.lower(  ) == 'spatiotemporal':                                                # If the domain subset type is spatiotemporal...

            # Preprocess the spatial unique tolerance.
            unique_tolerance = self.preprocess_unique_tolerance_spatial( unique_tolerance )

        elif domain_subset_type.lower(  ) == 'spatial':                                                     # If the domain subset type is spatial...

            # Preprocess the spatiotemporal unique tolerance.
            unique_tolerance = self.preprocess_unique_tolerance_spatiotemporal( unique_tolerance )

        else:                                                                                               # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid domain subset type: {domain_subset_type}' )

        # Return the unique tolerance.
        return unique_tolerance


    # Implement a function to preprocess the spatial classification noise magnitude.
    def preprocess_classification_noise_magnitude_spatial( self, classification_noise_magnitude_spatial ):

        # Determine whether to use the stored spatial classification noise magnitude.
        if classification_noise_magnitude_spatial is None:              # If the spatial classification noise magnitude was not provided...

            # Set the spatial classification noise magnitude to be the stored value.
            classification_noise_magnitude_spatial = self.classification_noise_magnitude_spatial

        # Return the spatial classification noise magnitude.
        return classification_noise_magnitude_spatial


    # Implement a function to preprocess the spatiotemporal classification noise magnitude.
    def preprocess_classification_noise_magnitude_spatiotemporal( self, classification_noise_magnitude_spatiotemporal ):

        # Determine whether to use the stored spatiotemporal classification noise magnitude.
        if classification_noise_magnitude_spatiotemporal is None:              # If the spatiotemporal classification noise magnitude was not provided...

            # Set the spatiotemporal classification noise magnitude to be the stored value.
            classification_noise_magnitude_spatiotemporal = self.classification_noise_magnitude_spatiotemporal

        # Return the spatiotemporal classification noise magnitude.
        return classification_noise_magnitude_spatiotemporal


    # Implement a function to preprocess the classification noise magnitude.
    def preprocess_classification_noise_magnitude( self, classification_noise_magnitude = None, domain_subset_type = 'spatiotemporal' ):

        # Determine how to preprocess the classification noise magnitude.
        if domain_subset_type.lower(  ) == 'spatial':                                                # If the domain subset type is spatial...

            # Preprocess the spatial classification noise magnitude.
            classification_noise_magnitude = self.preprocess_classification_noise_magnitude_spatial( classification_noise_magnitude )

        elif domain_subset_type.lower(  ) == 'spatiotemporal':                                                     # If the domain subset type is spatiotemporal...

            # Preprocess the spatiotemporal classification noise magnitude.
            classification_noise_magnitude = self.preprocess_classification_noise_magnitude_spatiotemporal( classification_noise_magnitude )

        else:                                                                                               # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid domain subset type: {domain_subset_type}' )

        # Return the classification noise magnitude.
        return classification_noise_magnitude


    # Implement a function to preprocess the level set guess.
    def preprocess_level_set_guesses( self, level_set_guesses, num_guesses, domain, domain_subset_type = 'spatiotemporal' ):

        # Determine whether to use the default level set guesses.
        if level_set_guesses is None:               # If the level set guesses were not provided...

            # Generate level set guesses.
            level_set_guesses = domain.sample_domain( num_guesses, domain_subset_type )

        # Return the level set guesses.
        return level_set_guesses


    # Implement a function to preprocess the number of level set guesses.
    def preprocess_num_guesses( self, num_guesses ):

        # Determine whether to use the default number of level set guesses.
        if num_guesses is None:                 # If the number of level set guesses was not provided.

            # Set the number of level set guesses to be the default value.
            num_guesses = torch.tensor( int( 1e2 ), dtype = torch.int64, device = self.device )

        # Return the number of guesses.
        return num_guesses


    # Implement a function to preprocess the classification data.
    def preprocess_classification_data( self, classification_data, num_spatial_dimensions, domain, plot_time, level, level_set_guesses, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, classification_noise_magnitude, domain_subset_type ):

        # Determine whether to generate the classification data at which to compute the classification loss.
        if classification_data is None:                 # If the classification data was not provided...

            # Determine whether to generate the classification data.
            if ( num_spatial_dimensions is not None ) and ( domain is not None ) and ( plot_time is not None ):

                # Generate the classification data.
                classification_data = self.generate_classification_data( num_spatial_dimensions, domain, plot_time, level, level_set_guesses, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, classification_noise_magnitude, domain_subset_type )

            else:

                # Throw an error.
                raise ValueError( 'Either classification data or num_spatial_dimensions, domain, and plot_time must not be None.' )

        # Return the classification data.
        return classification_data


    # Implement a function to preprocess the temporal integration span.
    def preprocess_temporal_integration_span( self, tspan ):

        # Preprocess the temporal integration span.
        if tspan is None:                           # If the temporal integration span was not provided...

            # Set the default temporal integration span.
            tspan = torch.tensor( [ 0, 1 ], dtype = torch.float32, device = self.device )

        # Return the temporal integration span.
        return tspan


    # Implement a function to preprocess the temporal step size.
    def preprocess_temporal_step_size( self, dt ):

        # Preprocess the temporal step size.
        if dt is None:                              # If the temporal step size was not provided...

            # Set the default temporal step size.
            dt = torch.tensor( 1e-3, dtype = torch.float32, device = self.device )

        # Return the temporal step size.
        return dt


    #%% ------------------------------------------------------------ IS VALID FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate the network layers.
    def is_layers_valid( self, layers = None ):

        # Determine whether the provided layers variable is valid.
        if torch.is_tensor( layers ) and ( layers.dim(  ) == 1 ) and ( layers.numel(  ) >= 2 ):             # If the layers variable 

            # Set the valid flag to true.
            valid_flag = True

        elif isinstance( layers, list ) and ( len( layers ) >= 2 ):

            # Set the valid flag to true.
            valid_flag = True

        else:                                                                                               # Otherwise... (i.e., the layers variable is invalid...)

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the residual function.
    def is_residual_function_valid( self, residual_function = None ):

        # Determine whether the residual function is valid.
        if callable( residual_function ):                       # If the residual function is callable...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                   # Otherwise... (i.e., the residual function is not valid...)

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the residual code.
    def is_residual_code_valid( self, residual_code = None ):

        # Determine whether the residual code is valid.
        if residual_code is None:                                   # If the residual code is None...

            # Set the valid flag to true.
            valid_flag = True

        elif torch.is_tensor( residual_code ):                      # If the residual code is itself a tensor...

            # Set the valid flag to true.
            valid_flag = True

        elif isinstance( residual_code, list ):                     # If the  residual code is a list...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                       # Otherwise... (i.e., the residual code is not recognized...)

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the temporal code.
    def is_temporal_code_valid( self, temporal_code = None ):

        # Determine whether the temporal code is valid.
        if temporal_code is None:                                   # If the temporal code is None...

            # Set the valid flag to true.
            valid_flag = True

        elif torch.is_tensor( temporal_code ):                      # If the temporal code is itself a tensor...

            # Set the valid flag to true.
            valid_flag = True

        elif isinstance( temporal_code, list ):                     # If the  temporal code is a list...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                       # Otherwise... (i.e., the temporal code is not recognized...)

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the activation string.
    def is_activation_string_valid( self, activation_string = None ):

        # Determine whether the activation string is valid.
        if ( activation_string.lower(  ) == 'tanh' ) or ( activation_string.lower(  ) == 'sigmoid' ) or ( activation_string.lower(  ) == 'relu' ):                  # If the activation string is tanh, sigmoid, or relu...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                                                                                                                       # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the compatibility of the residual function and residual code.
    def is_residual_function_code_compatible( self, residual_function = None, residual_code = None, num_residual_function_inputs = None ):

        # Determine whether to use the stored residual function.
        if residual_function is None:               # If the residual function was not provided...

            # Use the stored residual function.
            residual_function = self.residual_function

        # Determine whether to use the stored number of residual function inputs.
        if ( num_residual_function_inputs is None ) and ( residual_code is None ):              # If neither the residual code nor the number of residual function inputs was not provided...

            # Use the stored number of residual function inputs.
            num_residual_function_inputs = self.num_residual_function_inputs

        elif ( num_residual_function_inputs is None ) and ( residual_code is not None ):        # If the number of residual function was not provided but a residual code was provided...

            # Set the number of residual function inputs to be that associated with the provided residual code.
            num_residual_function_inputs = self.compute_num_residual_required_derivatives( residual_code )

        # Determine whether the inputs to this function are compatible.
        elif ( residual_code is not None ) and ( num_residual_function_inputs is not None ):    # If neither the residual code nor the number of residual function inputs is None...

            # Throw an error.
            raise ValueError( f'In order to test residual function and residual code compatibility, specify either the residual code or the number of residual function inputs associated with this residual code, not both.' )

        # Determine whether the residual function and the residual code are compatible.
        if ( residual_function is None ) and ( residual_code is None ):                         # If both the residual function and residual code are None...

            # Set the valid flag to true.
            valid_flag = True

        elif ( residual_function.__code__.co_argcount == num_residual_function_inputs ):        # If the residual function has the same number of inputs as the number of inputs associated with the residual code...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                                                   # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the compatibility of plot times and dimension labels.
    def is_plot_times_dimension_labels_compatible( self, plot_times, dimension_labels ):

        # Retrieve the number of temporal variables from the dimension labels.
        num_temporal_variables = self.dimension_labels2num_temporal_dimensions( dimension_labels )

        # Ensure that the number of plot times is the same as the number of temporal variables.
        valid_flag = plot_times.numel(  ) == num_temporal_variables

        # Return the valid flag.
        return valid_flag


    # Validate the compatibility of the projection dimensions and projection values.
    def is_projection_dimensions_values_compatible( self, projection_dimensions, projection_values ):

        # Ensure that the projection dimensions are either both none or both tensors.
        if ( projection_dimensions is None ) and ( projection_values is None ):                     # If both the projection dimensions and values are None...

            # Set the valid flag to true.
            valid_flag = True

        elif ( torch.is_tensor( projection_dimensions ) and projection_dimensions.numel(  ) != 0 ) and ( torch.is_tensor( projection_values ) and projection_values.numel(  ) != 0 ):

            # Determine whether the number of projection dimensions and projection values is equal.
            valid_flag = projection_dimensions.numel(  ) == projection_values.numel(  )

        else:                                                                                                                                                                                       # Otherwise... (i.e., the projection dimensions and values are not both None nor non-empty tensors...)

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    #%% ------------------------------------------------------------ SET FUNCTIONS ------------------------------------------------------------

    # Implement a function to set the network layers.
    def set_layers( self, layers, set_flag = True ):

        # Determine whether to set the layers.
        if set_flag:                    # If we want to set the layers...

            # Set the layers.
            self.layers = layers


    # Implement a function to set the residual function.
    def set_residual_function( self, residual_function, set_flag = True ):

        # Determine whether to set the residual function.
        if set_flag:                    # If we want to set the residual function...

            # Set the residual function.
            self.residual_function = residual_function


    # Implement a function to set the residual code.
    def set_residual_code( self, residual_code, set_flag = True ):

        # Determine whether to set the residual code.
        if set_flag:                    # If we want to set the residual code...

            # Set the residual code.
            self.residual_code = residual_code


    # Implement a function to set the temporal code.
    def set_temporal_code( self, temporal_code, set_flag = True ):

        # Determine whether to set the temporal code.
        if set_flag:                    # If we want to set the temporal code...

            # Set the temporal code.
            self.temporal_code = temporal_code
            

    # Implement a function to set the activation string.
    def set_activation_string( self, activation_string, set_flag = True ):

        # Determine whether to set the activation string.
        if set_flag:                    # If we want to set the activation string...

            # Set the activation string.
            self.activation_string = activation_string

    
    # Implement a function to set the number of hidden layers.
    def set_num_hidden_layers( self, num_hidden_layers, set_flag = True ):

        # Determine whether to set the number of hidden layers.
        if set_flag:                    # If we want to set the number of hidden layers...

            # Set the number of hidden layers.
            self.num_hidden_layers = num_hidden_layers


    # Implement a function to set the number of weights and biases.
    def set_num_weights_biases( self, num_weights_biases, set_flag = True ):

        # Determine whether to set the number of weights and biases.
        if set_flag:                    # If we want to set the number of weights and biases...

            # Set the number of weights and biases.
            self.num_weights_biases = num_weights_biases


    # Implement a function to set the number of residual inputs.
    def set_num_residual_inputs( self, num_residual_inputs, set_flag = True ):

        # Determine whether to set the number of residual inputs.
        if set_flag:                    # If we want to set the number of residual inputs...

            # Set the number of residual inputs.
            self.num_residual_inputs = num_residual_inputs


    # Implement a function to set the number of temporal derivatives.
    def set_num_temporal_derivatives( self, num_temporal_derivatives, set_flag = True ):

        # Determine whether to set the number of temporal derivatives.
        if set_flag:                    # If we want to set the number of temporal derivatives...

            # Set the number of temporal derivatives.
            self.num_temporal_derivatives = num_temporal_derivatives


    # Implement a function to set the training data.
    def set_training_data( self, training_data, set_flag = True ):

        # Determine whether to set the training data.
        if set_flag:                        # If we want to set the training data...

            # Set the training data.
            self.training_data = training_data


    # Implement a function to set the training epochs.
    def set_training_epochs( self, training_epochs, set_flag = True ):

        # Determine whether to set the training epochs.
        if set_flag:                        # If we want to set the training epochs...

            # Set the training epochs.
            self.training_epochs = training_epochs


    # Implement a function to set the training losses.
    def set_training_losses( self, training_losses, set_flag = True ):

        # Determine whether to set the training losses.
        if set_flag:                            # If we want to set the training losses...

            # Set the training losses.
            self.training_losses = training_losses


    # Implement a function to set the testing epochs.
    def set_testing_epochs( self, testing_epochs, set_flag = True ):

        # Determine whether to set the testing epochs.
        if set_flag:                        # If we want to set the testing epochs...

            # Set the testing epochs.
            self.testing_epochs = testing_epochs


    # Implement a function to set the testing losses.
    def set_testing_losses( self, testing_losses, set_flag = True ):

        # Determine whether to set the testing losses.
        if set_flag:                        # If we want to set the testing losses...

            # Set the testing losses.
            self.testing_losses = testing_losses

    
    # Implement a function to set the classification loss.
    def set_classification_loss( self, classification_loss, set_flag = True ):

        # Determine whether to set the classification loss.
        if set_flag:                        # If we want to set the classification loss...

            # Set the classification loss.
            self.classification_loss = classification_loss


    # Implement a function to set the number of batches.
    def set_num_batches( self, num_batches, set_flag = True ):

        # Determine whether to set the batch information.
        if set_flag:                        # If we want to set the batch information...

            # Set the number of batches.
            self.num_batches = num_batches


    # Implement a function to set the initial batch size.
    def set_initial_condition_batch_size( self, initial_condition_batch_size, set_flag = True ):

        # Determine whether to set the initial batch size.
        if set_flag:                        # If we want to set the initial batch size...

            # Set the initial batch size.
            self.initial_condition_batch_size = initial_condition_batch_size


    # Implement a function to set the boundary batch size.
    def set_boundary_condition_batch_size( self, boundary_condition_batch_size, set_flag = True ):

        # Determine whether to set the boundary batch size.
        if set_flag:                        # If we want to set the boundary batch size...
            
            # Set the boundary batch size.
            self.boundary_condition_batch_size = boundary_condition_batch_size


    #%% ------------------------------------------------------------ VALIDATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate the network layers.
    def validate_layers( self, layers, set_flag = False ):

        # Determine how to set the layers.
        if not self.is_layers_valid( layers ):                  # If the layers variable is not valid...

            # Throw an error.
            raise ValueError( f'Invalid layers variable: {layers}' )

        # Set the layers (as required).
        self.set_layers( layers, set_flag )

        # Return the layers variable.
        return layers


    # Implement a function to validate the residual function.
    def validate_residual_function( self, residual_function, set_flag = False ):

        # Determine how to set the residual function.
        if not self.is_residual_function_valid( residual_function ):            # If the residual function is not valid...

            # Throw an error.
            raise ValueError( f'Invalid residual function: {residual_function}' )

        # Set the residual function (as required).
        self.set_residual_function( residual_function, set_flag )

        # Return the residual function.
        return residual_function


    # Implement a function to validate the residual code.
    def validate_residual_code( self, residual_code, set_flag = False ):

        # Determine how to set the residual code.
        if not self.is_residual_code_valid( residual_code ):            # If the residual code is not valid...

            # Throw an error.
            raise ValueError( f'Invalid residual code: {residual_code}' )

        # Set the residual code (as required).
        self.set_residual_code( residual_code, set_flag )

        # Return the residual code.
        return residual_code


    # Implement a function to validate the temporal code.
    def validate_temporal_code( self, temporal_code, set_flag = False ):

        # Determine how to set the temporal code.
        if not self.is_temporal_code_valid( temporal_code ):            # If the temporal code is not valid...

            # Throw an error.
            raise ValueError( f'Invalid temporal code: {temporal_code}' )

        # Set the temporal code (as required).
        self.set_temporal_code( temporal_code, set_flag )

        # Return the temporal code.
        return temporal_code


    # Implement a function to validate the activation string.
    def validate_activation_string( self, activation_string, set_flag = False ):

        # Determine how to set the activation string.
        if not self.is_activation_string_valid( activation_string ):            # If the activation string is not valid...

            # Throw an error.
            raise ValueError( f'Invalid activation string: {activation_string}' )

        # Set the activation string (as required).
        self.set_activation_string( activation_string, set_flag )

        # Return the activation string.
        return activation_string


    #%% ------------------------------------------------------------ SETUP FUNCTIONS ------------------------------------------------------------

    # Implement a function to setup for batch info initialization.
    def setup_batch_info_initialization( self, num_initial_condition_points = None, num_boundary_condition_points = None, num_residual_points = None, residual_batch_size = None ):

        # Preprocess the number of initial condition points.
        num_initial_condition_points = self.preprocess_num_initial_condition_points( num_initial_condition_points )

        # Preprocess the number of boundary condition points.
        num_boundary_condition_points = self.preprocess_num_boundary_condition_points( num_boundary_condition_points )

        # Preprocess the number of residual points.
        num_residual_points = self.preprocess_num_residual_points( num_residual_points )

        # Preprocess the residual batch size.
        residual_batch_size = self.preprocess_residual_batch_size( residual_batch_size )

        # Return the batch info initialization information.
        return num_initial_condition_points, num_boundary_condition_points, num_residual_points, residual_batch_size


    # Implement a function to setup the variation calculation.
    def setup_variation( self, xs_integration_points_batch = None, G_basis_values_batch = None, W_integration_weights_batch = None, sigma_jacobian_batch = None ):

        # Preprocess the integration points batch.
        xs_integration_points_batch = self.preprocess_integration_points_batch( xs_integration_points_batch )

        # Preprocess the basis values batch.
        G_basis_values_batch = self.preprocess_basis_values_batch( G_basis_values_batch )

        # Preprocess the integration weights batch.
        W_integration_weights_batch = self.preprocess_integration_weights_batch( W_integration_weights_batch )

        # Preprocess the jacobian batch.
        sigma_jacobian_batch = self.preprocess_jacobian_batch( sigma_jacobian_batch )

        # Return the variation information.
        return xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch


    # Implement a function to setup for loss computation.
    def setup_loss_computation( self, initial_condition_data = None, boundary_condition_data = None, residual_data = None, variational_data = None ):

        # Preprocess the initial condition data.
        initial_condition_data = self.preprocess_initial_condition_data( initial_condition_data )

        # Preprocess the boundary condition data.
        boundary_condition_data = self.preprocess_boundary_condition_data( boundary_condition_data )

        # Preprocess the residual data.
        residual_data = self.preprocess_residual_data( residual_data )

        # Preprocess the variational data.
        variational_data = self.preprocess_variational_data( variational_data )

        # Return the data necessary for loss computation.
        return initial_condition_data, boundary_condition_data, residual_data, variational_data


    # Implement a function to setup for a training epoch.
    def setup_training_epoch( self, training_data = None, num_batches = None ):

        # Preprocess the training data.
        training_data = self.preprocess_training_data( training_data )

        # Preprocess the number of batches.
        num_batches = self.preprocess_num_batches( num_batches )

        # Return the information required to perform a training epoch.
        return training_data, num_batches


    # Implement a function to setup for training.
    def setup_training( self, training_data = None, testing_data = None, num_batches = None, num_epochs = None, epoch_print_frequency = None, verbose_flag = None ):

        # Preprocess the training data.
        training_data = self.preprocess_training_data( training_data )

        # Preprocess the testing data.
        testing_data = self.preprocess_testing_data( testing_data )

        # Preprocess the number of batches.
        num_batches = self.preprocess_num_batches( num_batches )

        # Preprocess the number of epochs.
        num_epochs = self.preprocess_num_epochs( num_epochs )

        # Preprocess the epoch print frequency.
        epoch_print_frequency = self.preprocess_epoch_print_frequency( epoch_print_frequency )

        # Preprocess the verbose flag.
        verbose_flag = self.preprocess_verbose_flag( verbose_flag )

        # Return the data necessary for training.
        return training_data, testing_data, num_batches, num_epochs, epoch_print_frequency, verbose_flag


    # Implement a function to setup for batch data staging.
    def setup_batch_data_staging( self, training_data = None, batch_number = None ):

        # Preprocess the training data.
        training_data = self.preprocess_training_data( training_data )

        # Preprocess the batch number.
        batch_number = self.preprocess_batch_number( batch_number )

        # Return the information required for staging batch data.
        return training_data, batch_number


    # Implement a function to setup for derivative requirements computation.
    def setup_derivative_requirements( self, num_inputs = None, residual_code = None ):

        # Preprocess the number of inputs.
        num_inputs = self.preprocess_num_inputs( num_inputs )

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Preallocate a boolean tensor to store the required derivatives.
        derivative_required_for_residual = torch.empty( num_inputs, dtype = torch.bool, device = self.device )

        # Return the information necessary for computing derivative requirements.
        return num_inputs, residual_code, derivative_required_for_residual


    # Implement a function to setup for derivative gradient enabling.
    def setup_gradient_enabling( self, num_inputs = None, derivative_required_for_residual = None ):

        # Preprocess the number of inputs.
        num_inputs = self.preprocess_num_inputs( num_inputs )

        # Preprocess the derivative required for residual value.
        derivative_required_for_residual = self.preprocess_derivative_required_for_residual( derivative_required_for_residual )

        # Return the information required for enabling gradients for derivative computations.
        return num_inputs, derivative_required_for_residual


    # Implement a function to setup for plotting network predictions.
    def setup_network_predictions( self, plotting_data = None, dimension_labels = None ):

        # Preprocess the plotting data.
        plotting_data = self.preprocess_plotting_data( plotting_data )

        # Preprocess the dimension labels.
        dimension_labels = self.preprocess_dimension_labels( dimension_labels )

        # Return the plotting data and dimension labels.
        return plotting_data, dimension_labels


    # Implement a function to setup for network predictions at a specific time.
    def setup_network_predictions_at_time( self, plotting_data = None, dimension_labels = None, plot_times = None ):

        # Preprocess the plotting data.
        plotting_data = self.preprocess_plotting_data( plotting_data )

        # Preprocess the dimension labels.
        dimension_labels = self.preprocess_dimension_labels( dimension_labels )

        # Preprocess the plot times.
        plot_times = self.preprocess_plot_times( plot_times )

        # Return the information required for plotting network predictions at a specific time.
        return plotting_data, dimension_labels, plot_times


    # Implement a function to setup for plotting the training / testing losses.
    def setup_training_testing_losses( self, training_losses = None, testing_epochs = None, testing_losses = None ):

        # Preprocess the training losses.
        training_losses = self.preprocess_training_losses( training_losses )
        
        # Preprocess the testing epochs.
        testing_epochs = self.preprocess_testing_epochs( testing_epochs )

        # Preprocess the testing losses.
        testing_losses = self.preprocess_testing_losses( testing_losses )

        # Return the training / testing losses.
        return training_losses, testing_epochs, testing_losses


    # Implement a function to setup for level set computation and plotting.
    def setup_level_set( self, num_spatial_dimensions, level = None, level_set_guess = None, newton_tolerance = None, newton_max_iterations = None, exploration_radius = None, num_exploration_points = None, unique_tolerance = None, domain_subset_type = 'spatiotemporal', at_time_flag = False ):

        # Preprocess the level.
        level = self.preprocess_level( level )

        # Determine how to preprocess the level set guess.
        if at_time_flag:                # If we are computing / plotting a level set at a given time...
            
            # Preprocess the level set guess using the spatial dimensions.
            level_set_guess = self.preprocess_level_set_guess( level_set_guess, num_spatial_dimensions )

        else:                           # Otherwise... ( i.e., we are computing / plotting a level set at any time... )

            # Preprocess the level set guess using all of the dimensions.
            level_set_guess = self.preprocess_level_set_guess( level_set_guess )

        # Preprocess the newton tolerance.
        newton_tolerance = self.preprocess_newton_tolerance( newton_tolerance )

        # Preprocess the maximum number of newton iterations.
        newton_max_iterations = self.preprocess_newton_max_iterations( newton_max_iterations )

        # Preprocess the exploration radius.
        exploration_radius = self.preprocess_exploration_radius( exploration_radius, domain_subset_type )

        # Preprocess the number of exploration points.
        num_exploration_points = self.preprocess_num_exploration_points( num_exploration_points )

        # Preprocess the unique tolerance.
        unique_tolerance = self.preprocess_unique_tolerance( unique_tolerance, domain_subset_type )   

        # Return the level set plotting information.
        return level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance


    # Implement a function to setup for the classification loss.
    def setup_classification_loss( self, classification_data = None, level_set_guesses = None, classification_noise_magnitude = None, unique_tolerance = None, num_exploration_points = None, exploration_radius = None, newton_max_iterations = None, newton_tolerance = None, num_guesses = None, level = None, plot_time = None, tspan = None, dt = None, num_spatial_dimensions = None, domain = None, domain_subset_type = 'spatial' ):

        # Preprocess the integration time step size.
        dt = self.preprocess_temporal_step_size( dt )

        # Preprocess the integration temporal span.
        tspan = self.preprocess_temporal_integration_span( tspan )

        # Preprocess the plot time.
        plot_time = self.preprocess_plot_times( plot_time )

        # Preprocess the level set value.
        level = self.preprocess_level( level )

        # Preprocess the number of level set guesses.
        num_guesses = self.preprocess_num_guesses( num_guesses )

        # Preprocess the newton tolerance.
        newton_tolerance = self.preprocess_newton_tolerance( newton_tolerance )

        # Preprocess the newton maximum number of iterations.
        newton_max_iterations = self.preprocess_newton_max_iterations( newton_max_iterations )

        # Preprocess the exploration radius.
        exploration_radius = self.preprocess_exploration_radius( exploration_radius, domain_subset_type )

        # Preprocess the number of exploration points.
        num_exploration_points = self.preprocess_num_exploration_points( num_exploration_points )

        # Preprocess the unique tolerance.
        unique_tolerance = self.preprocess_unique_tolerance( unique_tolerance, domain_subset_type )

        # Preprocess the classification noise magnitude.
        classification_noise_magnitude = self.preprocess_classification_noise_magnitude( classification_noise_magnitude, domain_subset_type )

        # Preprocess the level set guesses.
        level_set_guesses = self.preprocess_level_set_guesses( level_set_guesses, num_guesses, domain, domain_subset_type )

        # Preprocess the classification data.
        classification_data = self.preprocess_classification_data( classification_data, num_spatial_dimensions, domain, plot_time, level, level_set_guesses, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, classification_noise_magnitude, domain_subset_type )

        # Return the classification loss setup data.
        return classification_data, level_set_guesses, classification_noise_magnitude, unique_tolerance, num_exploration_points, exploration_radius, newton_max_iterations, newton_tolerance, num_guesses, level, plot_time, tspan, dt


    #%% ------------------------------------------------------------ COMPUTE FUNCTIONS ------------------------------------------------------------

    # Implement a function to compute the number of hidden layers.
    def compute_num_hidden_layers( self, layers = None, set_flag = False ):

        # Preprocess the layers.
        layers = self.preprocess_layers( layers )

        # Determine how to compute the number of hidden layers.
        if torch.is_tensor( layers ):               # If the layers variable is a tensor...

            # Compute the number of hidden layers based on the number of elements in the layers tensor.
            num_hidden_layers = torch.tensor( layers.numel(  ) - 2, dtype = torch.uint8, device = self.device )

        elif isinstance( layers, list ):            # If the layers variable is a list...

            # Compute the number of hidden layers based on the length of the layers list.
            num_hidden_layers = torch.tensor( len( layers ) - 2, dtype = torch.uint8, device = self.device )

        else:                                       # Otherwise... ( i.e., the layers variable is invalid... )

            # Throw an error.
            raise ValueError( f'Invalid layers variable: {layers}' )

        # Set the number of hidden layers (as required).
        self.set_num_hidden_layers( num_hidden_layers, set_flag )

        # Return the number of hidden layers.
        return num_hidden_layers


    # Implement a function to compute the number of weights and biases.
    def compute_num_weights_biases( self, layers = None, set_flag = False ):

        # Preprocess the layers.
        layers = self.preprocess_layers( layers )

        # Determine how to compute the number of hidden layers.
        if torch.is_tensor( layers ):               # If the layers variable is a tensor...

            # Compute the number of weights and biases.
            num_weights_biases = torch.tensor( layers.numel(  ) + 1, dtype = torch.uint8, device = self.device )

        elif isinstance( layers, list ):            # If the layers variable is a list...

            # Compute the number of weights and biases.
            num_weights_biases = torch.tensor( len( layers ) + 1, dtype = torch.uint8, device = self.device )

        else:                                       # Otherwise... (i.e., the layers variable is invalid...)

            # Throw an error.
            raise ValueError( f'Invalid layers variable: {layers}' )

        # Set the number of weights and biases (as required).
        self.set_num_weights_biases( num_weights_biases, set_flag )

        # Return the number of weights and biases.
        return num_weights_biases


    # Implement a function to compute the number of required derivatives.
    def compute_num_required_derivatives( self, derivative_code ):

        # Determine the number of residual inputs from the residual code.
        if derivative_code is None:                                   # If the residual code is None...

            # Set the number of residual inputs to None.
            num_required_derivatives = None

        elif torch.is_tensor( derivative_code ):                      # If the residual code is itself a tensor...

            # Set the number of residual inputs to be one.
            num_required_derivatives = torch.tensor( 1, dtype = torch.uint8, device = self.device )

        elif isinstance( derivative_code, list ):                     # If the  residual code is a list...

            # Set the number of residual inputs.
            num_required_derivatives = torch.tensor( len( derivative_code ), dtype = torch.uint8, device = self.device )

        else:                                                       # Otherwise... (i.e., the residual code is not recognized...)

            # Throw an error.
            raise ValueError( f'Invalid residual code: {derivative_code}' )

        # Return the number of residual inputs.
        return num_required_derivatives


    # Implement a function to compute the number of derivatives required for residual computation.
    def compute_num_residual_required_derivatives( self, residual_code = None, set_flag = False ):

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Compute the number of required derivatives.
        num_residual_inputs = self.compute_num_required_derivatives( residual_code )

        # Set the number of residual inputs (as required).
        self.set_num_residual_inputs( num_residual_inputs, set_flag )

        # Return the number of residual inputs.
        return num_residual_inputs

    
    # Implement a function to compute the number of derivatives required for temporal gradient computation.
    def compute_num_temporal_required_derivatives( self, temporal_code = None, set_flag = False ):

        # Preprocess the temporal code.
        temporal_code = self.preprocess_temporal_code( temporal_code )

        # Compute the number of required derivatives.
        num_temporal_derivatives = self.compute_num_required_derivatives( temporal_code )

        # Set the number of temporal derivatives.
        self.set_num_temporal_derivatives( num_temporal_derivatives, set_flag )

        # Return the number of temporal derivatives.
        return num_temporal_derivatives


    # Implement a function to compute the residual.
    def compute_residual( self, residual_data = None, derivative_required_for_residual = None, residual_code = None ):

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Preprocess the derivative required fo residual.
        derivative_required_for_residual = self.preprocess_derivative_required_for_residual( derivative_required_for_residual )

        # Preprocess the residual data.
        residual_data = self.preprocess_residual_data( residual_data )

        # Compute the residual.
        residual = self.residual( residual_data.input_data_batch, derivative_required_for_residual, residual_code )

        # Return the residual.
        return residual


    # Implement a function to compute the variation.
    def compute_variation( self, variational_data = None, derivative_required_for_residual = None, residual_code = None ):

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Preprocess the derivative required for residual.
        derivative_required_for_residual = self.preprocess_derivative_required_for_residual( derivative_required_for_residual )

        # Preprocess the variational data.
        variational_data = self.preprocess_variational_data( variational_data )

        # Compute the variation.
        variation = self.variation( variational_data.xs_integration_points_batch, variational_data.G_basis_values_batch, variational_data.W_integration_weights_batch, variational_data.sigma_jacobian_batch, derivative_required_for_residual, residual_code )

        # Return the variation.
        return variation


    # Implement a function to compute the temporal gradient.
    def compute_temporal_derivative( self, residual_data = None, derivative_required_for_temporal_gradient = None, temporal_code = None ):

        # Preprocess the temporal code.
        temporal_code = self.preprocess_temporal_code( temporal_code )

        # Preprocess the derivative required for temporal code.
        derivative_required_for_temporal_gradient = self.preprocess_derivative_required_for_temporal_gradient( derivative_required_for_temporal_gradient )

        # Preprocess the residual data.
        residual_data = self.preprocess_residual_data( residual_data )

        # Compute the temporal derivative associated with this batch of residual data.
        temporal_derivative = self.temporal_derivative( residual_data.input_data_batch, derivative_required_for_temporal_gradient, temporal_code )

        # Return the temporal derivative.
        return temporal_derivative


    #%% ------------------------------------------------------------ INITIALIZATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to empty the training and testing losses.
    def initialize_training_testing_losses( self, num_epochs = None, set_flag = False ):

        # Preprocess the number of epochs.
        num_epochs = self.preprocess_num_epochs( num_epochs )

        # Empty training losses.
        training_losses = torch.empty( num_epochs, dtype = torch.float32, device = self.device )

        # Empty testing losses.
        testing_epochs = torch.empty( num_epochs, dtype = torch.int64, device = self.device )
        testing_losses = torch.empty( num_epochs, dtype = torch.float32, device = self.device )

        # Set the training and testing losses (as required).
        self.set_training_losses( training_losses, set_flag )
        self.set_testing_epochs( testing_epochs, set_flag )
        self.set_testing_losses( testing_losses, set_flag )

        # Return the training losses.
        return training_losses, testing_epochs, testing_losses

    
    # Implement a function to initialize the activation function.
    def initialize_activation_function( self, activation_string = None ):

        # Preprocess the activation string.
        activation_string = self.preprocess_activation_string( activation_string )

        # Determine how to initialize the activation function.
        if activation_string.lower(  ) == 'tanh':                          # If the activation function is set to tanh()...

            # Set the activation function to be hyperbolic tangent.
            activation = torch.nn.Tanh(  )

        elif activation_string.lower(  ) == 'relu':                        # If the activation function is set to relu()....

            # Set the activation function to be a ReLU function.
            activation = torch.nn.ReLU(  )

        elif activation_string.lower(  ) == 'sigmoid':                     # If the activation function is set to sigmoid()...

            # Set the activation function to be a sigmoid function.
            activation = torch.nn.Sigmoid(  )

        else:                                                                # Otherwise...       

            # Throw an error.
            raise ValueError( f'Invalid activation function: {activation_string}' )

        # Return the activation function.
        return activation


    # Implement a function to compute the number of batches.
    def initialize_batch_info( self, num_initial_condition_points = None, num_boundary_condition_points = None, num_residual_points = None, residual_batch_size = None, set_flag = False ):

        # Setup for batch info initialization.
        num_initial_condition_points, num_boundary_condition_points, num_residual_points, residual_batch_size = self.setup_batch_info_initialization( num_initial_condition_points = None, num_boundary_condition_points = None, num_residual_points = None, residual_batch_size = None )

        # Compute the number of batches.
        num_batches = torch.round( num_residual_points/residual_batch_size ).to( torch.int32 )

        # Determine whether to set the number of batches to one.
        if num_batches == 0:               # If the number of batches is zero...

            # Set the number of batches to one.
            num_batches = torch.tensor( 1, dtype = torch.int32, device = self.device )

            # Throw a warning.
            warnings.warn( 'The residual batch size is greater than the number of residual points.  Setting number of batches to one.' )

        # Compute the initial condition batch size.
        initial_condition_batch_size = ( num_initial_condition_points/num_batches ).to( torch.int32 )

        # Compute the boundary condition batch size.
        boundary_condition_batch_size = ( num_boundary_condition_points/num_batches ).to( torch.int32 )

        # Set the number of batches (as required).
        self.set_num_batches( num_batches, set_flag )

        # Set the initial and boundary batch sizes (as required).
        self.set_initial_condition_batch_size( initial_condition_batch_size, set_flag )
        self.set_boundary_condition_batch_size( boundary_condition_batch_size, set_flag )

        # Return the number of batches.
        return num_batches, initial_condition_batch_size, boundary_condition_batch_size


    # Implement a function to initialize the forward stack.
    def initialize_forward_stack( self, layers = None, activation = None ):

        # Preprocess the network layers.
        layers = self.preprocess_layers( layers )

        # Compute the number of hidden layers.
        num_hidden_layers = self.compute_num_hidden_layers( layers )

        # Initialize the forward stack.
        forward_stack = torch.nn.Sequential(  )

        # Create each of the layers in the forward stack.
        for k in range( num_hidden_layers ):                                                   # Iterate through each of the hidden layers...

            # Create the linear step for this layer.
            forward_stack.append( torch.nn.Linear( layers[ k ], layers[ k + 1 ] ) )

            # Create the activation step for this layer.
            forward_stack.append( activation )

        # Create the final linear layer.
        forward_stack.append( torch.nn.Linear( layers[ -2 ], layers[ -1 ] ) )

        # Return the forward stack.
        return forward_stack


    #%% ------------------------------------------------------------ NETWORK DEFINITION FUNCTIONS ------------------------------------------------------------

    # Implement the network forward function.
    def forward( self, network_input ):

        # Compute the network output associated with these network inputs.
        network_output = self.forward_stack( network_input )

        # Return the network output.
        return network_output


    # Implement a function to predict the network output when the network input is represented by a tuple containing one tensor per input channel.
    def predict( self, network_input_tuple ):

        # Concatenate the network input tuples into a single network input.
        network_input = torch.cat( network_input_tuple, dim = 1 )

        # Compute the network output.
        network_output = self.forward( network_input )

        # Return the network output.
        return network_output


    # Implement a function to predict the network output when the network input is represented as a grid tensor.
    def predict_over_grid( self, network_input_grid ):

        # Convert the network input grid into a network input.
        network_input = self.tensor_utilities.flatten_grid( network_input_grid )

        # Generate the network outputs associated with these network inputs.
        network_output = self.forward( network_input )

        # Convert the network output into a network output grid.
        network_output_grid = self.tensor_utilities.expand_grid( network_output, network_input_grid.shape[ :-1 ] )

        # Return the network output grid.
        return network_output_grid


    # Implement a function to compute the network residual.
    def residual( self, residual_input_data_batch = None, derivative_required_for_residual = None, residual_code = None ):

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Preprocess the derivative required for residual.
        derivative_required_for_residual = self.preprocess_derivative_required_for_residual( derivative_required_for_residual )

        # Preprocess the residual input data batch.
        residual_input_data_batch = self.preprocess_residual_input_data_batch( residual_input_data_batch )

        # Split the network input into its constituent dimensions.
        residual_input_data_batch_tuple = residual_input_data_batch.split( 1, dim = 1 )

        # Enable gradients with respect to the inputs as required for residual calculation.
        residual_input_data_batch_tuple = self.enable_residual_gradients( residual_input_data_batch_tuple, derivative_required_for_residual )

        # Compute the network prediction.
        network_output = self.predict( residual_input_data_batch_tuple )

        # Compute the residual function inputs.
        residual_function_inputs = self.compute_residual_function_inputs( residual_input_data_batch_tuple, network_output, residual_code )

        # Compute the residual output.
        residual = self.residual_function( *residual_function_inputs )

        # Return the residual.
        return residual


    # Implement a function to compute the network variation.
    def variation( self, xs_integration_points_batch = None, G_basis_values_batch = None, W_integration_weights_batch = None, sigma_jacobian_batch = None, derivative_required_for_residual = None, residual_code = None ):

        # xs_integration_points_batch = ( # elements, # points per element, # dimensions )
        # G_basis_values_batch = ( # elements, # basis functions, # points per element )
        # W_integration_weights_batch = ( # elements, # basis functions, # points per element )
        # sigma_jacobian_batch = ( # elements, # basis functions )

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Preprocess the derivative required for residual.
        derivative_required_for_residual = self.preprocess_derivative_required_for_residual( derivative_required_for_residual )

        # Setup for the variation calculation.
        xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch = self.setup_variation( xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch )

        # Infer the number of dimensions, number of basis functions, and the number of points per element from the given variational data.
        num_dimensions = xs_integration_points_batch.shape[ 2 ]
        num_basis_functions = G_basis_values_batch.shape[ 1 ]
        num_points_per_element = G_basis_values_batch.shape[ 2 ]

        # Compute the residual at the integration points ( Note that the integration points must be reshaped before being passed to the residual function. )
        residual = self.residual( torch.reshape( xs_integration_points_batch, ( -1, num_dimensions ) ), derivative_required_for_residual, residual_code )

        # Construct the residual tensor by reshaping the residual.
        R_residual_batch = torch.repeat_interleave( torch.reshape( residual, ( -1, 1, num_points_per_element ) ), repeats = num_basis_functions, axis = 1 )         # [nc x nb x ne]

        # Compute the variation for each element and each basis function.
        variation = sigma_jacobian_batch*torch.sum( W_integration_weights_batch*G_basis_values_batch*R_residual_batch, axis = -1 )           # [nc x nb]

        # Return the variation.
        return variation


    # Implement a function to compute the network temporal derivative.
    def temporal_derivative( self, residual_input_data_batch = None, derivative_required_for_temporal_gradient = None, temporal_code = None ):

        # Preprocess the temporal code.
        temporal_code = self.preprocess_temporal_code( temporal_code )

        # Preprocess the derivative required for temporal code.
        derivative_required_for_temporal_gradient = self.preprocess_derivative_required_for_temporal_gradient( derivative_required_for_temporal_gradient )

        # Preprocess the residual input data batch.
        residual_input_data_batch = self.preprocess_residual_input_data_batch( residual_input_data_batch )

        # Split the network input into its constituent dimensions.
        residual_input_data_batch_tuple = residual_input_data_batch.split( 1, dim = 1 )

        # Enable gradients with respect to the inputs as required for residual calculation.
        residual_input_data_batch_tuple = self.enable_residual_gradients( residual_input_data_batch_tuple, derivative_required_for_temporal_gradient )

        # Compute the network prediction.
        network_output = self.predict( residual_input_data_batch_tuple )

        # Compute the temporal derivative.
        temporal_derivative = self.compute_temporal_derivative_direct( residual_input_data_batch_tuple, network_output, temporal_code )

        # Return the temporal derivative.
        return temporal_derivative


    # Implement a function to classify network inputs as stable or unstable.
    def classify( self, classification_data ):

        # Compute the network stability estimates
        phis = self.forward( classification_data )

        # Compute the network classifications.
        classifications = phis < 0                                      # [T/F] Classifications. True if point is stable, False if point is not stable.

        # Return the classifications.
        return classifications


    #%% ------------------------------------------------------------ LOSS FUNCTIONS ------------------------------------------------------------

    # Implement a function to compute the classification loss.
    def compute_classification_loss( self, pde, classification_data = None, num_spatial_dimensions = None, domain = None, plot_time = None, level = None, level_set_guesses = None, num_guesses = None, newton_tolerance = None, newton_max_iterations = None, exploration_radius = None, num_exploration_points = None, unique_tolerance = None, classification_noise_magnitude = None, domain_subset_type = 'spatial', tspan = None, dt = None ):

        # Setup the data for computing the classification loss.
        classification_data, level_set_guesses, classification_noise_magnitude, unique_tolerance, num_exploration_points, exploration_radius, newton_max_iterations, newton_tolerance, num_guesses, level, plot_time, tspan, dt = self.setup_classification_loss( classification_data, level_set_guesses, classification_noise_magnitude, unique_tolerance, num_exploration_points, exploration_radius, newton_max_iterations, newton_tolerance, num_guesses, level, plot_time, tspan, dt, num_spatial_dimensions, domain, domain_subset_type )

        # Move the classification data along their associated flow lines.
        classification_data_forecast, _ = pde.integrate_flow( pde.flow_function, tspan = tspan, x0 = classification_data, dt = dt )

        # Compute the network classifications associated with the classification data.
        network_classifications = self.classify( classification_data )

        # Compute the network classifications associated with the classification data.
        actual_classifications = self.classify( classification_data_forecast )

        # Determine whether any of the forecasted data is out-of-bounds.
        in_bounds_flags = domain.contains( classification_data_forecast[ :, 1: ], domain_type = 'spatial' )

        # Update the actual classifications to set all points that went out of bounds to False.
        actual_classifications[ ~in_bounds_flags ] = False

        # Compute the classification loss.
        classification_loss = self.classification_loss_function( network_classifications.type( torch.float32 ), actual_classifications.type( torch.float32 ) )

        # Return the classification loss.
        return classification_loss


    # Implement a function to compute the initial-boundary condition loss.
    def initial_boundary_condition_loss( self, initial_boundary_condition_data ):

        # Retrieve the number of initial-boundary conditions.
        num_initial_boundary_conditions = torch.tensor( len( initial_boundary_condition_data ), dtype = torch.uint8, device = self.device )

        # Initialize the initial-boundary condition loss.
        initial_boundary_condition_loss = torch.tensor( 0.0, dtype = torch.float32, device = self.device )

        # Compute the loss associated with each initial-boundary condition.
        for k1 in range( num_initial_boundary_conditions ):                       # Iterate through each of the initial-boundary conditions...

            # Retrieve the maximum derivative order associated with the initial-boundary condition output data.
            max_derivative_order = torch.max( initial_boundary_condition_data[ k1 ].output_derivative_order )

            # Split the network input into its constituent dimensions.
            input_data_batch_tuple = initial_boundary_condition_data[ k1 ].input_data_batch.split( 1, dim = 1 )

            # Set the specified input tensor dimension to track gradients.
            input_data_batch_tuple[ initial_boundary_condition_data[ k1 ].dimension.item(  ) ].requires_grad = True

            # Compute the network prediction for the input data associated with this initial-boundary condition.
            network_output = self.predict( input_data_batch_tuple )

            # Compute any necessary derivatives of the network prediction with respect to the dimension of interest.
            network_derivative = self.compute_nth_derivative( network_output, input_data_batch_tuple[ initial_boundary_condition_data[ k1 ].dimension.item(  ) ], max_derivative_order )

            # Compute the loss associated with this initial-boundary condition.
            for k2 in range( initial_boundary_condition_data[ k1 ].num_output_sources ):                                      # Iterate through each of the output sources for this initial-boundary condition...
                
                # Retrieve the targets associated with this output source.
                targets = initial_boundary_condition_data[ k1 ].output_data_batch[ k2 ]

                # Retrieve the values associated with this output source.
                values = network_derivative[ initial_boundary_condition_data[ k1 ].output_derivative_order[ k2 ].item(  ) ]

                # Compute the loss associated with this output source of this initial-boundary condition.
                initial_boundary_condition_loss += torch.nn.functional.mse_loss( values, targets )

        # Return the loss.
        return initial_boundary_condition_loss


    # Implement a function to compute the initial condition loss.
    def initial_condition_loss( self, initial_condition_data = None ):

        # Preprocess the initial condition data.
        initial_condition_data = self.preprocess_initial_condition_data( initial_condition_data )

        # Compute the initial condition loss.
        initial_condition_loss = self.initial_boundary_condition_loss( initial_condition_data )

        # Return the initial condition loss.
        return initial_condition_loss


    # Implement a function to compute the boundary condition loss.
    def boundary_condition_loss( self, boundary_condition_data = None ):

        # Preprocess the boundary condition data.
        boundary_condition_data = self.preprocess_initial_condition_data( boundary_condition_data )

        # Compute the boundary condition loss.
        boundary_condition_loss = self.initial_boundary_condition_loss( boundary_condition_data )

        # Return the boundary condition loss.
        return boundary_condition_loss


    # Implement a function to compute the residual loss.
    def residual_loss( self, residual_data = None, derivative_required_for_residual = None, residual_code = None ):

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Preprocess the derivative required for residual.
        derivative_required_for_residual = self.preprocess_derivative_required_for_residual( derivative_required_for_residual )

        # Preprocess the residual data.
        residual_data = self.preprocess_residual_data( residual_data )

        # Compute the network residual.
        residual = self.compute_residual( residual_data, derivative_required_for_residual, residual_code )

        # Compute the residual loss.
        residual_loss = torch.nn.functional.mse_loss( residual, torch.zeros_like( residual ) )

        # Return the residual loss.
        return residual_loss


    # Implement a function to compute the variational loss.
    def variational_loss( self, variational_data = None, derivative_required_for_residual = None, residual_code = None ):

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Preprocess the derivative required for residual.
        derivative_required_for_residual = self.preprocess_derivative_required_for_residual( derivative_required_for_residual )

        # Preprocess the variational data.
        variational_data = self.preprocess_variational_data( variational_data )

        # Compute the network variation.
        variation = self.compute_variation( variational_data, derivative_required_for_residual, residual_code )

        # Compute the variational loss.
        variational_loss = torch.nn.functional.mse_loss( variation, torch.zeros_like( variation ) )

        # Return the variational loss.
        return variational_loss


    # Implement a function to compute the monotonicity loss.
    def monotonicity_loss( self, residual_data = None, derivative_required_for_temporal_gradient = None, temporal_code = None ):

        # Preprocess the temporal code.
        temporal_code = self.preprocess_temporal_code( temporal_code )

        # Preprocess the derivative required for the temporal gradient.
        derivative_required_for_temporal_gradient = self.preprocess_derivative_required_for_temporal_gradient( derivative_required_for_temporal_gradient )

        # Preprocess the residual data.
        residual_data = self.preprocess_residual_data( residual_data )

        # Compute the temporal derivative.
        temporal_derivative = self.compute_temporal_derivative( residual_data, derivative_required_for_temporal_gradient, temporal_code )

        # Rectify the temporal derivative.
        rectified_temporal_derivative = torch.maximum( temporal_derivative, torch.zeros_like( temporal_derivative ) )

        # Compute the monotonicity loss.
        monotonicity_loss = torch.nn.functional.mse_loss( rectified_temporal_derivative, torch.zeros_like( rectified_temporal_derivative ) )

        # Return the monotonicity loss.
        return monotonicity_loss


    # Implement the loss function.
    def loss( self, initial_condition_data = None, boundary_condition_data = None, residual_data = None, variational_data = None, derivative_required_for_residual = None, residual_code = None, derivative_required_for_temporal_gradient = None, temporal_code = None ):

        # Preprocess the temporal code.
        temporal_code = self.preprocess_temporal_code( temporal_code )

        # Preprocess the derivative required for temporal gradients.
        derivative_required_for_temporal_gradient = self.preprocess_derivative_required_for_temporal_gradient( derivative_required_for_temporal_gradient )

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Preprocess the derivative required for residual.
        derivative_required_for_residual = self.preprocess_derivative_required_for_residual( derivative_required_for_residual )

        # Setup for the loss computation.
        initial_condition_data, boundary_condition_data, residual_data, variational_data = self.setup_loss_computation( initial_condition_data, boundary_condition_data, residual_data, variational_data )

        # Compute the initial condition loss.
        loss_ic = self.initial_condition_loss( initial_condition_data )

        # Compute the boundary condition loss.
        loss_bc = self.boundary_condition_loss( boundary_condition_data )

        # Compute the residual loss.
        loss_residual = self.residual_loss( residual_data, derivative_required_for_residual, residual_code )

        # Compute the variational loss.
        loss_variational = self.variational_loss( variational_data, derivative_required_for_residual, residual_code )

        # Compute the monotonicity loss.
        loss_monotonicity = self.monotonicity_loss( residual_data, derivative_required_for_temporal_gradient, temporal_code )

        # Compute the complete loss.
        loss = self.c_IC*loss_ic + self.c_BC*loss_bc + self.c_residual*loss_residual + self.c_variational*loss_variational + self.c_monotonicity*loss_monotonicity

        # Return the loss.
        return loss


    #%% ------------------------------------------------------------ TRAINING FUNCTIONS ------------------------------------------------------------

    # Implement a function to perform the steps necessary to train on a single batch.
    def train_batch( self, training_data = None, derivative_required_for_residual = None, residual_code = None ):

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Preprocess the derivative required for residual.
        derivative_required_for_residual = self.preprocess_derivative_required_for_residual( derivative_required_for_residual )

        # Preprocess the training data.
        training_data = self.preprocess_training_data( training_data )

        # Retrieve the starting time.
        start_time = torch.tensor( time.time(  ), dtype = torch.float64, device = self.device ) 

        # Zero out the optimizer gradients.
        self.optimizer.zero_grad(  )

        # Compute the loss associated with this batch element.
        batch_loss = self.loss( training_data.initial_condition_data, training_data.boundary_condition_data, training_data.residual_data, training_data.variational_data, derivative_required_for_residual, residual_code )

        # Compute the gradients associated with the loss.
        batch_loss.backward(  )

        # Perform an optimizer step on this batch.
        self.optimizer.step(  )

        # Detach the batch loss.
        batch_loss = batch_loss.detach(  )

        # Retrieve the end time.
        end_time = torch.tensor( time.time(  ), dtype = torch.float64, device = self.device ) 

        # Compute the duration.
        batch_duration = end_time - start_time

        # Return the batch loss.
        return batch_loss, batch_duration


    # Implement a function to perform the steps necessary to train for an epoch.
    def train_epoch( self, training_data = None, num_batches = None, derivative_required_for_residual = None, residual_code = None, print_batch_info_flag = True ):

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Preprocess the derivative required for residual.
        derivative_required_for_residual = self.preprocess_derivative_required_for_residual( derivative_required_for_residual )

        # Setup for the training epoch.
        training_data, num_batches = self.setup_training_epoch( training_data, num_batches )

        # Retrieve the epoch start time.
        start_time = torch.tensor( time.time(  ), dtype = torch.float64, device = self.device ) 

        # Shuffle the training data.
        training_data = self.shuffle_training_data( training_data )

        # Initialize the percent completion.
        percent_complete = torch.tensor( 0.0, dtype = torch.float32, device = self.device )

        # Train over all of the batches.
        for k in range( num_batches ):                  # Iterate through each of the batches...

            # Stage the data associated with this batch.
            training_data = self.stage_batch_data( training_data, k )

            # Determine whether to generate a batch of elements.
            if self.element_computation_option.lower(  ) in ( 'dynamic', 'during', 'during training', 'during_training', 'duringtraining' ):

                # Generate a batch of elements.
                training_data = self.generate_element_batch( training_data, replace_flag = False, batch_option = 'compute', batch_number = torch.tensor( 0, dtype = torch.int64, device = self.device ), batch_size = training_data.variational_data.batch_size )

            # Train over this batch.
            batch_loss, batch_duration = self.train_batch( training_data, derivative_required_for_residual, residual_code )

            # Determine whether to delete the batch of elements.
            if self.element_computation_option.lower(  ) in ( 'dynamic', 'during', 'during training', 'during_training', 'duringtraining' ):

                # Delete the batch of elements.
                training_data = self.delete_elements( None, training_data, batch_option = 'delete', batch_number = torch.tensor( 0, dtype = torch.int64, device = self.device ), batch_size = training_data.variational_data.batch_size )

            # Print the batch status (if required).
            percent_complete = self.print_batch_status( k, batch_loss, percent_complete, batch_duration, print_batch_info_flag )

        # Retrieve the end time.
        end_time = torch.tensor( time.time(  ), dtype = torch.float64, device = self.device ) 

        # Compute the epoch duration.
        epoch_duration = end_time - start_time

        # Return the final batch loss.
        return batch_loss, epoch_duration


    # Implement a function to perform the steps to fully train a network for the specified number of epochs.
    def train( self, pde, training_data = None, testing_data = None, num_batches = None, num_epochs = None, derivative_required_for_residual = None, residual_code = None, epoch_print_frequency = None, verbose_flag = None, set_flag = False ):

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Preprocess the derivative required for residual.
        derivative_required_for_residual = self.preprocess_derivative_required_for_residual( derivative_required_for_residual )

        # Setup the data necessary to perform training.
        training_data, testing_data, num_batches, num_epochs, epoch_print_frequency, verbose_flag = self.setup_training( training_data, testing_data, num_batches, num_epochs, epoch_print_frequency, verbose_flag )

        # Print the starting training status.
        self.print_starting_training_status( verbose_flag )

        # Initialize the training start time.
        start_time = torch.tensor( time.time(  ), dtype = torch.float64, device = self.device ) 

        # Initialize loop counter variables.
        k1 = torch.tensor( 0, dtype = torch.int64, device = self.device )
        k2 = torch.tensor( 0, dtype = torch.int64, device = self.device )

        # Initialize the early stopping flag to false.
        stop_early_flag = False

        # Initialize the percent completion.
        old_percent_complete = torch.tensor( 0.0, dtype = torch.float32, device = self.device )

        # Empty the training and testing losses.
        training_losses, testing_epochs, testing_losses = self.initialize_training_testing_losses( num_epochs )

        # Train the network for the specified number of epochs.
        while ( not stop_early_flag ) and ( k1 < num_epochs ) :                                             # While we have not met early stopping criteria and have not yet performed all of the training epochs...

            # Compute the percent completion.
            new_percent_complete = self.compute_percent_completion( k1, num_epochs )

            # Determine whether to print information from this epoch.
            if ( ( new_percent_complete - old_percent_complete ) >= epoch_print_frequency ):                # If this is an epoch for which we would like to test the network...

                # Set the test network flag to true.
                test_network_flag = True

                # Set the old percent completion to be the new percent completion.
                old_percent_complete = new_percent_complete

            else:                                                                                           # Otherwise... (If we do not want to test the network on this epoch...)

                # Set the test network flag to false.
                test_network_flag = False

            # Print starting epoch information (if required).
            self.print_starting_epoch_status( k1, old_percent_complete, test_network_flag )

            # Perform a training epoch.
            training_losses[ k1 ], epoch_duration = self.train_epoch( training_data, num_batches, derivative_required_for_residual, residual_code, test_network_flag )

            # Determine whether to perform a testing epoch.
            if test_network_flag:                                                                           # Determine whether to test the network...

                # Store this testing epoch.
                testing_epochs[ k2 ] = k1 + 1

                # Perform a testing epoch.
                testing_losses[ k2 ] = self.test_epoch( testing_data, derivative_required_for_residual, residual_code )

            # Print ending epoch information.
            self.print_ending_epoch_status( training_losses[ k1 ], testing_losses[ k2 ], epoch_duration, test_network_flag )
            
            # Check the early stopping criteria.
            stop_early_flag = self.check_early_stop_criteria(  )

            # Advance the first loop counter.
            k1 += 1

            # Determine whether to advance the second loop counter.
            if test_network_flag:                                       # If the test network flag is true...

                # Advance the second loop counter.
                k2 += 1

        # Remove the extra training and testing losses.
        training_losses = training_losses[ :k1 ]
        testing_epochs = testing_epochs[ :k2 ]
        testing_losses = testing_losses[ :k2 ]

        # Create a tensor of training epochs.
        training_epochs = torch.arange( k1.item(  ) ) + 1

        # Initialize the training end time.
        end_time = torch.tensor( time.time(  ), dtype = torch.float64, device = self.device ) 

        # Compute the training duration.
        training_duration = end_time - start_time

        # Print the ending training status.
        self.print_ending_training_status( training_duration, verbose_flag )

        # # Compute the network's classification loss.
        # classification_loss = self.classification_loss( pde )

        # classification_loss = self.compute_classification_loss( pde, classification_data, num_spatial_dimensions, domain, plot_time, level, level_set_guesses, num_guesses, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, classification_noise_magnitude, domain_subset_type, tspan, dt )

        # Set the training and testing epochs and losses (as required).
        self.set_training_epochs( training_epochs, set_flag )
        self.set_training_losses( training_epochs, set_flag )
        self.set_testing_epochs( testing_epochs, set_flag )
        self.set_testing_losses( testing_losses, set_flag )

        # # Set the classification loss (as required).
        # self.set_classification_loss( classification_loss, set_flag )

        # classification_loss = self.classification_loss

        # Return the training and testing losses.
        return training_epochs, training_losses, testing_epochs, testing_losses
        # return training_epochs, training_losses, testing_epochs, testing_losses, classification_loss


    #%% ------------------------------------------------------------ TESTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to evaluate the network over the test data.
    def test_epoch( self, testing_data = None, derivative_required_for_residual = None, residual_code = None ):

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Preprocess the derivative required for residual.
        derivative_required_for_residual = self.preprocess_derivative_required_for_residual( derivative_required_for_residual )

        # Preprocess the testing data.
        testing_data = self.preprocess_testing_data( testing_data )

        # Compute the loss associated with this batch element.
        test_loss = self.loss( testing_data.initial_condition_data, testing_data.boundary_condition_data, testing_data.residual_data, testing_data.variational_data, derivative_required_for_residual, residual_code )

        # Detach the test loss.
        test_loss = test_loss.detach(  )

        # Return the testing loss.
        return test_loss


    #%% ------------------------------------------------------------ DERIVATIVE FUNCTIONS ------------------------------------------------------------

    # Implement a function to compute the derivative of a tensor with respect to a different tensor.
    def compute_derivative( self, numerator, denominator ):

        # Compute the derivative of tensor num with respect to tensor den.
        numerator.backward( gradient = torch.ones_like( numerator ), create_graph = True, inputs = denominator )

        # Clone the derivative so that the gradient calculation can be zeroed out.
        derivative = torch.clone( denominator.grad )

        # Zero out the gradient.
        denominator.grad = torch.zeros_like( denominator.grad )

        # Return the gradient.
        return derivative


    # Implement a function to compute the nth derivative of a tensor with respect to a different tensor.
    def compute_nth_derivative( self, numerator, denominators, num_derivatives = None ):

        # Determine how to interpret the input arguments.
        if torch.is_tensor( denominators ) and ( num_derivatives is not None ) and ( num_derivatives >= 0 ) :               # If the input tensor is a tensor and num_derivatives is not None...

            # Convert the input tensor to a list of num_derivatives repetitions.
            denominators = [ denominators ]*num_derivatives

        elif isinstance( denominators, list ):                                                                              # If the input tensor is a list...

            # Determine the number of derivatives to perform.
            num_derivatives = len( denominators )

        else:                                                                                                               # Otherwise...

            # Throw an error.
            raise ValueError( f'Incompatible derivative tensor and derivative order.' )

        # Initialize a list to store the derivatives.
        derivatives = [ numerator ]

        # Compute each of the requested derivatives.
        for k in range( num_derivatives ):                                                        # Iterate through each derivative...

            # Compute this derivative.
            derivatives.append( self.compute_derivative( derivatives[ k ], denominators[ k ] ) )

        # Return the list of derivatives.
        return derivatives


    # Implement a function to determine which derivatives are required for the derivative computation.
    def derivative_code2derivative_requirements( self, derivative_code, num_inputs = None ):

        # Preprocess the number of inputs.
        num_inputs = self.preprocess_num_inputs( num_inputs )

        # Preallocate a boolean tensor to store the required derivatives.
        derivative_requirements = torch.empty( num_inputs, dtype = torch.bool, device = self.device )

        # Determine how to compute the derivative requirements from the derivative code.
        if derivative_code is None:                                 # If the derivative code is None...

            # Set the derivatives required for the derivative to be None (this indicates that no derivatives are required for the derivative calculation)
            derivative_requirements = None

        elif isinstance( derivative_code, list ):                   # If the derivative code is a list...

            # Determine how to compute the derivative requirements from the derivative code when the derivative code is a list.
            if not derivative_code:                                 # If the derivative code is an empty list...

                # Set the derivatives required for the derivative to be None (this indicates that no derivatives are required for the derivative calculation)
                derivative_requirements = None

            else:                                                   # Otherwise... (i.e., the derivative code list is non-empty...)

                # Retrieve the number of elements of the derivative code list.
                num_derivative_code_elements = len( derivative_code )

                # Determine how to compute the derivative requirements from the derivative code when the derivative code is a non-empty list.
                if all( ( ( derivative_code[ k ] is None ) or ( torch.is_tensor( derivative_code[ k ] ) ) ) for k in range( num_derivative_code_elements ) ):                                  # If all of the list entries are valid...

                    # Determine how to compute the derivative requirements from the derivative code when the derivative code is a list of valid values.
                    if all( ( derivative_code[ k ] is None ) for k in range( num_derivative_code_elements ) ):                                  # If the list contains all none values...

                        # Set the derivatives required for the derivative to be None (this indicates that no derivatives are required for the derivative calculation)
                        derivative_requirements = None

                    else:                                                                                                                       # Otherwise... ( i.e., there are at least some tensor values in the list... )
                        
                        # Determine how to compute the derivative requirements from the derivative code when not all of the derivative code list entries are none.
                        if any( ( derivative_code[ k ] is None ) for k in range( num_derivative_code_elements ) ):                              # If the list contains any none values...

                            # Remove the none values.
                            derivative_code = [ derivative_code[ k ] for k in range( num_derivative_code_elements ) if derivative_code[ k ] is not None ]

                            # Retrieve the new number of derivative code elements.
                            num_derivative_code_elements = len( derivative_code )

                        # Determine how to compute the derivative requirements from the derivative code when not all of the derivative code list entries are none.
                        if any( ( derivative_code[ k ].numel(  ) == 0 ) for k in range( num_derivative_code_elements ) ):                       # If the list contains any empty tensors...

                            # Remove the empty tensors.
                            derivative_code = [ derivative_code[ k ] for k in range( num_derivative_code_elements ) if derivative_code[ k ].numel(  ) != 0 ]

                            # Retrieve the new number of derivative code elements.
                            num_derivative_code_elements = len( derivative_code )

                        # Create an empty tensor that will store the flattened derivative code tensors.
                        derivative_code_tensor = torch.tensor( [  ], dtype = torch.uint8, device = self.device )

                        # Retrieve the unique entries of the tensors in the derivative code lists.
                        for k in range( num_derivative_code_elements ):                                 # Iterate through each of the derivative code elements...                     

                            # Concatenate the data contained in this derivative tensor with the previously existing data.
                            derivative_code_tensor = torch.cat( ( derivative_code_tensor, derivative_code[ k ].ravel(  ) ), dim = 0 )
                            
                        # Retrieve the unique entries of the derivative code tensor.
                        unique_derivative_code = derivative_code_tensor.unique(  )

                else:                                               # Otherwise...

                    # Throw an error.
                    raise ValueError( f'Invalid derivative code: {derivative_code}' )

        elif torch.is_tensor( derivative_code ):                  # If the  derivative code is a tensor...

            # Retrieve the unique entries of the derivative code tensor.
            unique_derivative_code = derivative_code.unique(  )

        else:                                                   # Otherwise... (i.e., the derivative code is not recognized...)    

            # Throw an error.
            raise ValueError( f'Invalid derivative code: {derivative_code}' )

        # Determine whether we still need to compute the derivatives required for the derivative.
        if derivative_requirements is not None: 

            # Determine which dimension derivatives are required based on the derivative code.
            for k in range( num_inputs ):                           # Iterate through each of the unique derivatives...

                if any( unique_derivative_code == k ):                    # If this input is in the unique derivative code...

                    # Indicate that a derivative with respect to this input dimension is required for the derivative calculation.
                    derivative_requirements[ k ] = True

                else:                                                   # Otherwise... (If this input is not in the unique derivative code...)  

                    # Indicate that a derivative with respect to this input dimension is not required for the derivative calculation.
                    derivative_requirements[ k ] = False

        # Return the derivatives requirements.
        return derivative_requirements


    # Implement a function to determine which derivatives are required for the residual computation.
    def residual_code2derivative_requirements( self, residual_code = None, num_inputs = None ):

        # Preprocess the number of inputs.
        num_inputs = self.preprocess_num_inputs( num_inputs )

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Compute the derivatives required given this derivative code.
        derivative_required_for_residual = self.derivative_code2derivative_requirements( residual_code, num_inputs )

        # Return the derivatives required to compute the residual.
        return derivative_required_for_residual


    # Implement a function to determine which derivatives are required for the temporal computation.
    def temporal_code2derivative_requirements( self, temporal_code = None, num_inputs = None ):

        # Preprocess the number of inputs.
        num_inputs = self.preprocess_num_inputs( num_inputs )

        # Preprocess the temporal code.
        temporal_code = self.preprocess_temporal_code( temporal_code )

        # Compute the derivatives required given this derivative code.
        derivative_required_for_temporal_gradient = self.derivative_code2derivative_requirements( temporal_code, num_inputs )

        # Return the derivatives required to compute the temporal.
        return derivative_required_for_temporal_gradient


    # Implement a function to enable gradient calculations with respect to each input dimension that is required for tensor calculation.
    def enable_tensor_gradients( self, input_data_batch_tuple, derivative_requirements ):

        # Compute the number of inputs.
        num_inputs = torch.tensor( derivative_requirements.numel(  ), dtype = torch.uint8, device = self.device )

        # Enable gradient calculations for each of the required input dimensions.
        for k in range( num_inputs ):                               # Iterate through each of the inputs...

            # Determine whether to enable gradient calculations for this input.
            if derivative_requirements[ k ]:                        # If a derivative with respect to this input is required for the tensor calculation...

                # Enable gradient calculations with respect to this input.
                input_data_batch_tuple[ k ].requires_grad = True

        # Return the input data batch tuple.
        return input_data_batch_tuple


    # Implement a function to enable gradient calculations with respect to each input dimension that is required for residual calculation.
    def enable_residual_gradients( self, input_data_batch_tuple, derivative_required_for_residual = None ):

        # Preprocess the derivative required for residual.
        derivative_required_for_residual = self.preprocess_derivative_required_for_residual( derivative_required_for_residual )

        # Enable the appropriate gradients for the residual computation.
        input_data_batch_tuple = self.enable_tensor_gradients( input_data_batch_tuple, derivative_required_for_residual )

        # Return the input data batch tuple.
        return input_data_batch_tuple


    # Implement a function to enable gradient calculations with respect to each input dimension that is required for temporal calculation.
    def enable_temporal_gradients( self, input_data_batch_tuple, derivative_required_for_temporal_gradient = None ):

        # Preprocess the derivative required for temporal.
        derivative_required_for_temporal_gradient = self.preprocess_derivative_required_for_temporal_gradient( derivative_required_for_temporal_gradient )

        # Enable the appropriate gradients for the temporal computation.
        input_data_batch_tuple = self.enable_tensor_gradients( input_data_batch_tuple, derivative_required_for_temporal_gradient )

        # Return the input data batch tuple.
        return input_data_batch_tuple


    # Implement a function to compute the necessary network derivatives.
    def compute_network_derivatives( self, network_input_tuple, network_output, derivative_code ):

        # Compute the number of required derivatives based on the provided derivative code.
        num_required_derivatives = self.compute_num_required_derivatives( derivative_code )

        # Determine whether there are any network derivatives.
        if num_required_derivatives is None:                                             # If there are no derivatives to compute...

            # Set the network derivatives to None.
            network_derivatives = None

        else:                                                                       # Otherwise... (i.e., if the number of network derivatives is not none...)

            # Determine whether to embed the derivative code tensor in a list.
            if torch.is_tensor( derivative_code ):                                    # If the derivative code is itself a tensor...

                # Embed the derivative code tensor in a list.
                derivative_code = [ derivative_code ]

            # Initialize an empty list to store the network derivatives.
            network_derivatives = [  ]

            # Compute each of the network derivatives.
            for k1 in range( num_required_derivatives ):             # Iterate through each of the network derivatives...

                # Determine whether there are derivatives to compute associated with this network derivative.
                if derivative_code[ k1 ] is not None:             # If this derivative code list entry is not None...

                    # Initialize this network derivative to the network output.
                    network_derivative = network_output

                    # Compute any necessary derivatives for this network derivative.
                    for k2 in range( derivative_code[ k1 ].numel(  ) ):                                   # Iterate through each of the derivatives that are required for this residual input...

                        # Compute the derivatives associated with this residual input.
                        network_derivative = self.compute_derivative( network_derivative, network_input_tuple[ derivative_code[ k1 ][ k2 ] ] )

                else:                                           # Otherwise... ( i.e., this derivative code list entry is None... )

                    # Set the network derivative to be the network inputs.
                    network_derivative = torch.cat( network_input_tuple, dim = 1 )

                # Append this network derivative to the list of network derivatives.
                network_derivatives.append( network_derivative )

        # Return the network derivatives.
        return network_derivatives


    # Implement a function to compute the residual function inputs.
    def compute_residual_function_inputs( self, network_input_tuple, network_output, residual_code = None ):

        # Preprocess the residual code.
        residual_code = self.preprocess_residual_code( residual_code )

        # Compute the residual function inputs.
        residual_function_inputs = self.compute_network_derivatives( network_input_tuple, network_output, residual_code )

        # Return the residual function inputs.
        return residual_function_inputs


    # Implement a function to compute the temporal derivative.
    def compute_temporal_derivative_direct( self, network_input_tuple, network_output, temporal_code = None ):

        # Preprocess the temporal code.
        temporal_code = self.preprocess_temporal_code( temporal_code )

        # Compute the temporal derivative.
        temporal_derivative = self.compute_network_derivatives( network_input_tuple, network_output, temporal_code )

        # Return the temporal derivative.
        return temporal_derivative[ 0 ]
        

    #%% ------------------------------------------------------------ SHUFFLE FUNCTIONS ------------------------------------------------------------

    # Implement a function to shuffle training data.
    def shuffle_training_data( self, training_data = None, shuffle_indexes = None, set_flag = False ):

        # Preprocess the training data.
        training_data = self.preprocess_training_data( training_data )

        # Shuffle the training data.
        training_data.initial_condition_data, training_data.boundary_condition_data, training_data.residual_data, training_data.variational_data = training_data.shuffle_all_data( training_data.initial_condition_data, training_data.boundary_condition_data, training_data.residual_data, training_data.variational_data, shuffle_indexes )

        # Set the training data (as required).
        self.set_training_data( training_data, set_flag )

        # Return the training data.
        return training_data
        

    #%% ------------------------------------------------------------ BATCH FUNCTIONS ------------------------------------------------------------

    # Implement a function to retrieve the batch data from the training data.
    def stage_batch_data( self, training_data = None, batch_number = None, set_flag = False ):

        # Setup for batch data staging.
        training_data, batch_number = self.setup_batch_data_staging( training_data, batch_number )

        # Retrieve the initial, boundary, and residual batch data from the training data.
        training_data.initial_condition_data, training_data.boundary_condition_data, training_data.residual_data, training_data.variational_data = training_data.compute_all_batch_data( training_data.initial_condition_data, training_data.boundary_condition_data, training_data.residual_data, training_data.variational_data, batch_number, training_data.initial_condition_batch_size, training_data.boundary_condition_batch_size, training_data.residual_batch_size, training_data.variational_batch_size )

        # Set the training data (as required.)
        self.set_training_data( training_data, set_flag )

        # Return the initial, boundary, and residual batch data from the training data.
        return training_data


    #%% ------------------------------------------------------------ EARLY STOPPING FUNCTIONS ------------------------------------------------------------

    # Implement a function to compute the percent completion with an epoch or entire training.
    def compute_percent_completion( self, iterator, total_num_iterators ):

        # Compute the percent completion.
        percent_complete = torch.round( 100*( ( iterator + 1 )/total_num_iterators ), decimals = 2 )

        # Return the percent completion.
        return percent_complete


    # Implement a function to check the early stopping criteria.
    def check_early_stop_criteria( self ):

        # Set the early stop flag to false.
        b_early_stop = False

        # Return the early stop false.
        return b_early_stop


    #%% ------------------------------------------------------------ UTILITY FUNCTIONS ------------------------------------------------------------

    # Implement a function to determine the number of temporal variables associated with a list of dimension labels.
    def dimension_labels2num_temporal_dimensions( self, dimension_labels ):

        # Retrieve the number of dimension labels.
        num_dimension_labels = torch.tensor( len( dimension_labels ), dtype = torch.uint8, device = self.device )

        # Initialize the number of temporal dimensions to zero.
        num_temporal_dimensions = torch.tensor( 0, dtype = torch.uint8, device = self.device )

        # Compute the number of temporal dimensions.
        for k in range( num_dimension_labels ):                     # Iterate through each of the dimension labels...

            # Determine whether this dimension label is temporal.
            if dimension_labels[ k ][ 0 ].lower(  ) == 't':         # If this dimension label is a temporal dimension...

                # Advance the number of temporal dimensions counter.
                num_temporal_dimensions += 1

        # Return the number of temporal dimensions.
        return num_temporal_dimensions


    # Implement a function to convert the dimension labels to temporal indexes.
    def dimension_labels2temporal_indexes( self, dimension_labels ):

        # Retrieve the number of dimension labels.
        num_dimension_labels = torch.tensor( len( dimension_labels ), dtype = torch.uint8, device = self.device )

        # Compute the number of temporal indexes.
        num_temporal_dimensions = self.dimension_labels2num_temporal_dimensions( dimension_labels )

        # Preallocate a tensor to store the temporal indexes.
        temporal_indexes = torch.empty( num_temporal_dimensions, dtype = torch.uint8, device = self.device )

        # Initialize a temporal index counter.
        k2 = torch.tensor( 0, dtype = torch.int64, device = self.device )

        # Retrieve the temporal indexes.
        for k1 in range( num_dimension_labels ):                    # Iterate through each of the dimension labels...

            # Determine whether this dimension label is associated with a temporal variable.
            if dimension_labels[ k1 ][ 0 ].lower(  ) == 't':        # If this dimension label is a temporal dimension...

                # Store this index.
                temporal_indexes[ k2 ] = k1

                # Advance the temporal index counter.
                k2 += 1

        # Return the temporal indexes.
        return temporal_indexes


    # Implement a function to generate the network prediction plot title.
    def generate_network_prediction_title( self, projection_dimensions = None, projection_values = None ):

        # Determine how to generate the network prediction title.
        if ( projection_dimensions is None ) and ( projection_values is None ):                                                                                                                     # If both the projection dimensions and values are None...

            # Set the network prediction title.
            title_string = f'Network Prediction'

        elif ( torch.is_tensor( projection_dimensions ) and ( projection_dimensions.numel(  ) != 0 ) ) and ( torch.is_tensor( projection_values ) and ( projection_values.numel(  ) != 0 ) ):       # If both th  projection dimensions and values are non-empty tensors...

            # Set the network prediction title.
            title_string = f'Network Prediction: {projection_dimensions.tolist(  )} Dims @ {projection_values.tolist(  )} Values'

        else:                                                                                                                                                                                       # Otherwise... (i.e., the projection dimensions and values are invalid...)

            # Throw an error.
            raise ValueError( f'Projection dimensions {projection_dimensions} and projection values {projection_values} are invalid.' )

        # Return the title string.
        return title_string


    #%% ------------------------------------------------------------ ELEMENT FUNCTIONS ------------------------------------------------------------

    # Implement a function to generate training data elements.
    def generate_elements( self, xs_element_centers, training_data = None, replace_flag = False, batch_option = 'keep', batch_number = None, batch_size = None, set_flag = False ):

        # Preprocess the training data.
        training_data = self.preprocess_training_data( training_data )

        # Generate the elements.
        training_data.variational_data = training_data.generate_elements( xs_element_centers, training_data.variational_data, replace_flag, batch_option, batch_number, batch_size )

        # Set the training data (as required).
        self.set_training_data( training_data, set_flag )

        # Return the training data.
        return training_data


    # Implement a function to delete training data elements.
    def delete_elements( self, indexes, training_data = None, batch_option = 'keep', batch_number = None, batch_size = None, set_flag = False ):

        # Preprocess the training data.
        training_data = self.preprocess_training_data( training_data )

        # Delete the elements.
        training_data.variational_data = training_data.delete_elements( indexes, training_data.variational_data, batch_option, batch_number, batch_size )

        # Set the training data (as required).
        self.set_training_data( training_data, set_flag )

        # Return the training data.
        return training_data


    # Implement a function to generate a batch of elements.
    def generate_element_batch( self, training_data = None, replace_flag = False, batch_option = 'draw', batch_number = None, batch_size = None, set_flag = False ):

        # Preprocess the training data.
        training_data = self.preprocess_training_data( training_data )

        # Retrieve the element centers from the residual training data batch.
        xs_element_centers = training_data.residual_data.input_data_batch

        # Generate the batch elements.
        training_data = self.generate_elements( xs_element_centers, training_data, replace_flag, batch_option, batch_number, batch_size )

        # Set the training data (as required).
        self.set_training_data( training_data, set_flag )

        # Return the training data.
        return training_data


    #%% ------------------------------------------------------------ LEVEL SET FUNCTIONS ------------------------------------------------------------

    # Implement the network's level function.
    def level_function( self, spatiotemporal_point, domain ):

        # Determine how to compute the network's value.
        if domain.contains( spatiotemporal_point, 'spatiotemporal' ):                # If the given point is in the domain...

            # Compute the network prediction at this point.
            level_value = self.forward( spatiotemporal_point )

        else:                                   # Otherwise... ( i.e., the given point is not in the domain... )

            # Set the network prediction to be some constant.
            level_value = torch.linalg.norm( spatiotemporal_point, ord = 2, keepdim = True )**2

        # Return the level value.
        return level_value


    # Implement the network's level function at a specific time.
    def level_function_at_time( self, spatial_points, time, domain ):

        # Retrieve the number of spatial points.
        num_spatial_points = spatial_points.shape[ 0 ]

        # Determine whether each of the spatial points are within the given domain.
        masks = torch.unsqueeze( domain.contains( spatial_points, 'spatial' ), dim = 1 )

        # Compute the network prediction at each point.
        level_values_network = self.forward( torch.cat( ( torch.repeat_interleave( torch.unsqueeze( time, dim = 0 ), num_spatial_points, dim = 0 ), spatial_points ), dim = 1 ) )

        # Compute the distance to each point.
        level_values_distance = torch.linalg.norm( spatial_points, ord = 2, dim = 1, keepdim = True )**2

        # Determine whether to use the network prediction or the distance based on whether the point is or is not in the given domain.
        level_values = level_values_network*masks + level_values_distance*(~masks)

        # Return the level values.
        return level_values


    # Implement a function to generate a level set of the network.
    def generate_level_set( self, num_spatial_dimensions, domain, level_function = None, level = None, level_set_guess = None, newton_tolerance = None, newton_max_iterations = None, exploration_radius = None, num_exploration_points = None, unique_tolerance = None, domain_subset_type = 'spatiotemporal' ):

        # Setup for the level set generation.
        level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance = self.setup_level_set( num_spatial_dimensions, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, domain_subset_type )

        # Determine whether to use the default level function.
        if level_function is None:                  # If the level function was not provided...

            # Set the level function to be the default value.
            level_function = lambda s: self.level_function( s, domain )

        # Compute the level set points.
        level_set_points = self.tensor_utilities.generate_level_set( level_function, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance )

        # Return the level set points.
        return level_set_points


    # Implement a function to generate a level set of the network at a specified time.
    def generate_level_set_at_time( self, num_spatial_dimensions, domain, plot_time, level = None, level_set_guess = None, newton_tolerance = None, newton_max_iterations = None, exploration_radius = None, num_exploration_points = None, unique_tolerance = None, domain_subset_type = 'spatial' ):

        # Setup for the level set generation.
        level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance = self.setup_level_set( num_spatial_dimensions, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, domain_subset_type, at_time_flag = True )

        # Define the level function.
        level_function = lambda x: self.level_function_at_time( x, plot_time, domain )

        # Compute the level set points.
        level_set_points = self.generate_level_set( num_spatial_dimensions, domain, level_function, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, domain_subset_type )

        # Concatenate the plot time onto each of the level set points.
        level_set_points = torch.cat( ( plot_time*torch.ones( ( level_set_points.shape[ 0 ], 1 ), dtype = torch.float32, device = self.device ), level_set_points ), dim = 1 )

        # Return the level set points.
        return level_set_points


    # Implement a function to generate classification data.
    def generate_classification_data( self, num_spatial_dimensions, domain, plot_time, level = None, level_set_guess = None, newton_tolerance = None, newton_max_iterations = None, exploration_radius = None, num_exploration_points = None, unique_tolerance = None, classification_noise_magnitude = None, domain_subset_type = 'spatial' ):

        # Setup for the classification data generation.
        level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance = self.setup_level_set( num_spatial_dimensions, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, domain_subset_type, at_time_flag = True )

        # Preprocess the classification noise magnitude.
        classification_noise_magnitude = self.preprocess_classification_noise_magnitude( classification_noise_magnitude, domain_subset_type )

        # Define the level function.
        level_function = lambda x: self.level_function_at_time( x, plot_time, domain )

        # Generate the level set points at the given time.        
        level_set_points_noisy = self.tensor_utilities.generate_noisy_level_set( level_function, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, classification_noise_magnitude )

        # Concatenate the plot time onto each of the level set points.
        level_set_points_noisy = torch.cat( ( plot_time*torch.ones( ( level_set_points_noisy.shape[ 0 ], 1 ), dtype = torch.float32, device = self.device ), level_set_points_noisy ), dim = 1 )

        # Return the noisy level set points.
        return level_set_points_noisy


    #%% ------------------------------------------------------------ PRINTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to print out network summary information.
    def print_summary( self, num_dashes = 20, decoration_flag = True ):

        # Print a header.
        self.printing_utilities.print_header( 'NETWORK SUMMARY', num_dashes, decoration_flag )

        # Print general information.
        print( 'General Information' )
        print( f'Device: {self.device}' )
        print( f'Verbose: {self.verbose_flag}' )
        print( f'Dimension Labels: {self.dimension_labels}' )

        # Print network structure.
        print( 'Network Structure' )
        print( f'# of Inputs: {self.num_inputs}' )
        print( f'# of Outputs: {self.num_outputs}' )
        print( f'# of Hidden Layers: {self.num_hidden_layers}' )
        print( f'Layers: {self.layers}' )
        print( f'# of Weights / Biases: {self.num_weights_biases}' )
        print( f'Activation Function: {self.activation}' )
        print( f'Forward Stack: {self.forward_stack}' )

        # Print residual information.
        print( 'Residual Information' )
        print( f'Residual Function: {self.residual_function}' )
        print( f'Residual Code: {self.residual_code}' )
        print( f'Derivative Required for Residual: {self.derivative_required_for_residual}' )
        print( f'# of Residual Inputs: {self.num_residual_inputs}' )

        # Print the training, testing, and plotting data.
        print( 'Training / Testing / Plotting Information' )
        print( f'Training Data: {self.training_data}' )
        print( f'Testing Data: {self.testing_data}' )
        print( f'Plotting Data: {self.plotting_data}' )

        # Print the loss parameters.
        print( 'Loss Information' )
        print( f'Initial Condition Loss Coefficient: {self.c_IC}' )
        print( f'Boundary Condition Loss Coefficient: {self.c_BC}' )
        print( f'Residual Loss Coefficient: {self.c_residual}' )

        # Print the epoch and batch information.
        print( 'Epoch / Batch Information' )
        print( f'# of Epochs: {self.num_epochs}' )
        print( f'# of Batches: {self.num_batches}' )
        print( f'Initial Batch Size: {self.initial_condition_batch_size}' )
        print( f'Boundary Batch Size: {self.boundary_condition_batch_size}' )
        print( f'Residual Batch Size: {self.residual_batch_size}' )

        # Print the learning information.
        print( 'Learning Information' )
        print( f'Learning Rate: {self.learning_rate}' )
        print( f'Optimizer: {self.optimizer}' )

        # Print the training and testing losses.
        print( 'Training / Testing Losses' )
        print( f'Training Losses: {self.training_losses}' )
        print( f'Testing Losses: {self.testing_losses}' )

        # Print the batch and epoch print frequency.
        print( 'Print Information' )
        print( f'Batch Print Frequency: {self.batch_print_frequency}' )
        print( f'Epoch Print Frequency: {self.epoch_print_frequency}' )

        # Print a footer.
        self.printing_utilities.print_footer( num_dashes, decoration_flag )


    # Implement a function to print out information during a training batch.
    def print_batch_info( self, batch_number, loss, percent_complete, batch_duration, num_batches = None, verbose_flag = True ):

        # Preprocess the number of batches.
        num_batches = self.preprocess_num_batches( num_batches )

        # Preprocess the verbose flag.
        verbose_flag = self.preprocess_verbose_flag( verbose_flag )

        # Determine whether to print.
        if verbose_flag:                    # If we want to be verbose_flag...

            # Print out the loss and percent complete information for this batch.
            print( f'Batch {batch_number + 1:>2.0f}/{num_batches.item(  ):>2.0f} ({percent_complete.item(  ):>6.2f}%): Training Loss = {loss.item(  ):>10.8f}, Duration = {batch_duration.item(  ):>3.8f}s = {batch_duration.item(  )/60:>3.8f}min = {batch_duration.item(  )/3600:>3.8f}hr' )

    
    # Implement a function to print out the status of a batch during training (if required).
    def print_batch_status( self, batch_number, batch_loss, old_percent_complete, batch_duration, print_flag, num_batches = None, batch_print_frequency = None ):

        # Preprocess the number of batches.
        num_batches = self.preprocess_num_batches( num_batches )

        # Preprocess the batch print frequency.
        batch_print_frequency = self.preprocess_batch_print_frequency( batch_print_frequency )

        # Determine whether we want to print information from any of these batches.
        if print_flag:                                                                          # If we want to print information from any of these batches...

            # Compute our percent completion of this epoch.
            new_percent_complete = self.compute_percent_completion( batch_number, num_batches )

            # Determine whether to print out batch information.
            if ( ( new_percent_complete - old_percent_complete ) >= batch_print_frequency ):                                 # If this is a batch whose information we would like to print...

                # Print out batch information.
                self.print_batch_info( batch_number, batch_loss, new_percent_complete, batch_duration )

                # Set the old percent complete to be the new percent complete.
                old_percent_complete = new_percent_complete

        # Return the old percent completion.
        return old_percent_complete


    # Implement a function to print out information at the beginning of a training epoch.
    def print_starting_epoch_info( self, epoch_number, percent_complete, num_epochs = None, verbose_flag = True ):

        # Preprocess the number of epochs.
        num_epochs = self.preprocess_num_epochs( num_epochs )

        # Preprocess the verbose flag.
        verbose_flag = self.preprocess_verbose_flag( verbose_flag )

        # Determine whether to print.
        if verbose_flag:                    # If we want to be verbose_flag...

            # Print out starting epoch information.
            print( f'\n------------------------------------------------------------------------------------------------------------------------' )
            print( f'Epoch {(epoch_number.item(  ) + 1):>4.0f}/{num_epochs.item(  ):>4.0f} ({percent_complete.item(  ):>6.2f}%)' )
            print( f'------------------------------------------------------------------------------------------------------------------------' )


    # Implement a function to print out information at the beginning of a training epoch (if required).
    def print_starting_epoch_status( self, epoch_number, percent_complete, print_flag ):

        # Determine whether to print the starting information for this epoch.
        if print_flag:                                          # If we want to print information for this epoch...
            
            # Print the starting information for this epoch.
            self.print_starting_epoch_info( epoch_number, percent_complete )


    # Implement a function to print out information at the end of a training epoch.
    def print_ending_epoch_info( self, training_loss, testing_loss, epoch_duration, verbose_flag = True ):

        # Preprocess the verbose flag.
        verbose_flag = self.preprocess_verbose_flag( verbose_flag )

        # Determine whether to print.
        if verbose_flag:                    # If we want to be verbose_flag...

            # Print out starting epoch information.
            print( f'------------------------------------------------------------------------------------------------------------------------' )
            print( f'Training Loss = {training_loss.item(  ):>10.8f}, Testing Loss = {testing_loss.item(  ):>10.8f}, Duration = {epoch_duration.item(  ):>3.8f}s = {epoch_duration.item(  )/60:>3.8f}min = {epoch_duration.item(  )/3600:>3.8f}hr' )
            print( f'------------------------------------------------------------------------------------------------------------------------\n' )


    # Implement a function to print out information at the end of a training epoch (if required).
    def print_ending_epoch_status( self, training_loss, testing_loss, epoch_duration, print_flag ):

        # Determine whether to print the ending information for this epoch.
        if print_flag:                                          # If we want to print information for this epoch...
            
            # Print the ending information for this epoch.
            self.print_ending_epoch_info( training_loss, testing_loss, epoch_duration )


    # Implement a function to print starting training information.
    def print_starting_training_info( self, verbose_flag = True ):

        # Preprocess the verbose flag.
        verbose_flag = self.preprocess_verbose_flag( verbose_flag )

        # Determine whether to print.
        if verbose_flag:                    # If we want to be verbose_flag...

            # Print out starting training information.
            print( f'\n------------------------------------------------------------------------------------------------------------------------' )
            print( f'STARTING TRAINING' )
            print( f'------------------------------------------------------------------------------------------------------------------------' )


    # Implement a function to print starting training status.
    def print_starting_training_status( self, print_flag ):

        # Determine whether to print the starting information for this epoch.
        if print_flag:                                          # If we want to print information for this training session...

            # Print starting training info.
            self.print_starting_training_info(  )


    # Implement a function to print ending training info.
    def print_ending_training_info( self, duration, verbose_flag = True ):

        # Preprocess the verbose flag.
        verbose_flag = self.preprocess_verbose_flag( verbose_flag )

        # Determine whether to print.
        if verbose_flag:                    # If we want to be verbose_flag...

            # Print out ending training information.
            print( f'\n------------------------------------------------------------------------------------------------------------------------' )
            print( f'ENDING TRAINING: Duration = {duration.item(  ):>3.8f}s = {duration.item(  )/60:>3.8f}min = {duration.item(  )/3600:>3.8f}hr' )
            print( f'------------------------------------------------------------------------------------------------------------------------' )


    # Implement a function to print ending training status.
    def print_ending_training_status( self, duration, print_flag ):

        # Determine whether to print the ending information for this epoch.
        if print_flag:                                          # If we want to print information for this training session...

            # Print ending training info.
            self.print_ending_training_info( duration )


    #%% ------------------------------------------------------------ PLOTTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to plot the training data.
    def plot_training_data( self, training_data = None, projection_dimensions = None, projection_values = None, level = 0, fig = None, plot_type1 = 'all', plot_type2 = 'all', save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Preprocess the training data.
        training_data = self.preprocess_training_data( training_data )

        # Plot the training data.
        figs, axes = training_data.plot( training_data.initial_condition_data, training_data.boundary_condition_data, training_data.residual_data, projection_dimensions, projection_values, level, fig, plot_type1, plot_type2, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figures and axes.
        return figs, axes


    # Implement a function to plot the testing data.
    def plot_testing_data( self, testing_data = None, projection_dimensions = None, projection_values = None, level = 0, fig = None, plot_type1 = 'all', plot_type2 = 'all', save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Preprocess the testing data.
        testing_data = self.preprocess_testing_data( testing_data )

        # Plot the testing data.
        figs, axes = testing_data.plot( testing_data.initial_condition_data, testing_data.boundary_condition_data, testing_data.residual_data, projection_dimensions, projection_values, level, fig, plot_type1, plot_type2, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figures and axes.
        return figs, axes


    # Implement a function to plot the plotting data.
    def plot_plotting_data( self, plotting_data = None, projection_dimensions = None, projection_values = None, level = 0, dimension_labels = None, fig = None, save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Preprocess the plotting data.
        plotting_data = self.preprocess_plotting_data( plotting_data )

        # Preprocess the dimension labels.
        dimension_labels = self.preprocess_dimension_labels( dimension_labels )

        # Generate the input labels.
        input_labels = self.plotting_utilities.dimension_labels2axis_labels( dimension_labels )

        # Define the title string.
        title_string = 'Neural Network Plotting Data'

        # Plot the plotting data.
        fig, ax = self.plotting_utilities.plot( plotting_data, [  ], projection_dimensions, projection_values, level, fig, input_labels, title_string, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot the network predictions.
    def plot_network_predictions( self, plotting_data = None, projection_dimensions = None, projection_values = None, level = 0, dimension_labels = None, fig = None, save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Setup for network predictions.
        plotting_data, dimension_labels = self.setup_network_predictions( plotting_data, dimension_labels )

        # Generate the network output grid.
        network_output_grid = self.predict_over_grid( plotting_data )
        
        # Generate the input labels.
        input_labels = self.plotting_utilities.dimension_labels2axis_labels( dimension_labels )

        # Generate the title string.
        title_string = self.generate_network_prediction_title( projection_dimensions, projection_values )

        # Plot the network prediction.
        fig, ax = self.plotting_utilities.plot( plotting_data, network_output_grid, projection_dimensions, projection_values, level, fig, input_labels, title_string, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figure.
        return fig, ax


    # Implement a function to plot the network prediction at a specific time.
    def plot_network_prediction_at_time( self, plotting_data = None, plot_times = None, projection_dimensions = None, projection_values = None, level = 0, dimension_labels = None, fig = None, save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Setup for plotting the network predictions at a specific time.
        plotting_data, dimension_labels, plot_times = self.setup_network_predictions_at_time( plotting_data, dimension_labels, plot_times )

        # Ensure that the plot times and dimension labels are compatible..
        assert self.is_plot_times_dimension_labels_compatible( plot_times, dimension_labels )

        # Ensure that the projection dimensions and projection values are compatible.
        assert self.is_projection_dimensions_values_compatible( projection_dimensions, projection_values )

        # Retrieve the indexes associated with the temporal variables.
        temporal_indexes = self.dimension_labels2temporal_indexes( dimension_labels )

        # Substitute the plot time into the network input grid.
        plotting_data = self.tensor_utilities.substitute_values_into_grid( plotting_data, plot_times, temporal_indexes )

        # Ensure that the temporal indexes are included in the projection dimensions.
        if projection_dimensions is None:                   # If the projection dimensions is None...

            # Set the projection dimensions and values.
            projection_dimensions = temporal_indexes
            projection_values = plot_times

        else:                                               # Otherwise... ( i.e., the projection dimensions and values are non-empty tensors... )

            # Concatenate the temporal indexes and values to the projection dimensions and values.
            projection_dimensions = torch.cat( ( projection_dimensions, temporal_indexes )  )
            projection_values = torch.cat( ( projection_values, plot_times ) )

        # Plot the network predictions.
        fig, ax = self.plot_network_predictions( plotting_data, projection_dimensions, projection_values, level, dimension_labels, fig, save_directory, as_surface, as_stream, as_contour, show_plot )
        
        # Return the figure.
        return fig, ax


    # Implement a function to plot the training results.
    def plot_training_results( self, training_losses = None, testing_epochs = None, testing_losses = None, save_directory = r'.', show_plot = False ):

        # Setup for plotting training results
        training_losses, testing_epochs, testing_losses = self.setup_training_testing_losses( training_losses, testing_epochs, testing_losses )

        # Convert the given tensors to numpy arrays.
        training_losses = self.plotting_utilities.plot_process( torch.squeeze( training_losses ) )
        testing_epochs = self.plotting_utilities.plot_process( torch.squeeze( testing_epochs ) )
        testing_losses = self.plotting_utilities.plot_process( torch.squeeze( testing_losses ) )

        # Create a plot to store the training and testing losses.
        fig = plt.figure(  ); plt.xlabel( 'Epoch [#]' ), plt.ylabel( 'Loss [-]' ), plt.title('Training & Testing Losses vs Epoch Number')
        plt.plot( training_losses, '*-', label = 'Training Losses' )
        plt.plot( testing_epochs, testing_losses, '*-', label = 'Testing Losses' )
        plt.legend(  )

        # Retrieve the current axes.
        ax = plt.gca(  )

        # Save the figure.
        plt.savefig( save_directory + r'/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Determine whether to show the plot.
        if show_plot:               # If we want to show the plot...

            # Show the figure.
            plt.show( block = False )

        # Return the figure.
        return fig, ax


    # Implement a function to plot a level set of the network.
    def plot_network_level_set( self, num_spatial_dimensions, level = None, level_set_guess = None, newton_tolerance = None, newton_max_iterations = None, exploration_radius = None, num_exploration_points = None, unique_tolerance = None, domain_subset_type = 'spatiotemporal', projection_dimensions = None, projection_values = None, fig = None, dimension_labels = None, save_directory = r'.', as_surface = False, as_stream = False, as_contour = False, show_plot = False ):

        # Setup for plotting the network level set.
        level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance = self.setup_level_set( num_spatial_dimensions, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, domain_subset_type )

        # Generate the input labels.
        input_labels = self.plotting_utilities.dimension_labels2axis_labels( dimension_labels )

        # Generate the title string.
        title_string = f'Network Level Set: Level = {level}'

        # Set the network level set function.
        level_function = lambda s: self.forward( s )

        # Generate the level set points associated with the network.
        level_set_points = self.generate_level_set( num_spatial_dimensions, level_function, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, domain_subset_type )

        # Compute the values associated with the level set points.
        level_set_values = self.forward( level_set_points )

        level_set_values = torch.zeros_like( level_set_points )

        # Plot the level set.
        figs, axes = self.plotting_utilities.plot( level_set_points, level_set_values, projection_dimensions, projection_values, level, fig, input_labels, title_string, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figures and axes.
        return figs, axes


    # Implement a function to plot a network level set at a specific time.
    def plot_network_level_set_at_time( self, num_spatial_dimensions, domain, plot_times = None, level = None, level_set_guess = None, newton_tolerance = None, newton_max_iterations = None, exploration_radius = None, num_exploration_points = None, unique_tolerance = None, domain_subset_type = 'spatial', projection_dimensions = None, projection_values = None, fig = None, dimension_labels = None, save_directory = r'.', as_surface = False, as_stream = False, as_contour = False, show_plot = False ):

        # Preprocess the plot times.
        plot_times = self.preprocess_plot_times( plot_times )

        # Setup for plotting the network level set.
        level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance = self.setup_level_set( num_spatial_dimensions, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, domain_subset_type, at_time_flag = True )

        # Ensure that the plot times and dimension labels are compatible..
        assert self.is_plot_times_dimension_labels_compatible( plot_times, dimension_labels )

        # Ensure that the projection dimensions and projection values are compatible.
        assert self.is_projection_dimensions_values_compatible( projection_dimensions, projection_values )

        # Generate the input labels.
        input_labels = self.plotting_utilities.dimension_labels2axis_labels( dimension_labels )

        # Generate the title string.
        title_string = f'Network Level Set: Level = {level}, Time = {plot_times}'

        # Generate the level set points associated with the network.
        level_set_points = self.generate_level_set_at_time( num_spatial_dimensions, domain, plot_times, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, domain_subset_type )
        # level_set_points = self.generate_classification_data( num_spatial_dimensions, domain, plot_times, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, self.classification_noise_magnitude_spatial, domain_subset_type )

        # Compute the values associated with the level set points.
        level_set_values = self.forward( level_set_points )
        level_set_values = torch.zeros_like( level_set_values )

        # Set the project dimensions and values to be None.
        projection_dimensions = None
        projection_values = None

        # Plot the level set.
        figs, axes = self.plotting_utilities.plot( level_set_points[ :, 1: ], level_set_values, projection_dimensions, projection_values, level, fig, input_labels, title_string, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figures and axes.
        return figs, axes
