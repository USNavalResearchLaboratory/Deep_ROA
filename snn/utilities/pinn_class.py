<<<<<<< HEAD
####################################################################################### 
# THIS SOURCE CODE IS PROPERTY OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. 
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND 
# CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION 
# ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN 
# LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE,
# CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL 
# PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641. 
####################################################################################### 


=======
>>>>>>> 55162c78e9fb0c13d60ea20df5463b1e4d4f30fb
#%% ------------------------------------------------------------ PINN CLASS ------------------------------------------------------------

# This file implements a class for storing and managing pinn information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch
import math

# Import custom libraries.
from domain_class import domain_class as domain_class
from initial_boundary_condition_class import initial_boundary_condition_class as initial_boundary_condition_class
from pde_class import pde_class as pde_class
from ibc_data_class import ibc_data_class as ibc_data_class
from residual_data_class import residual_data_class as residual_data_class
from variational_data_class import variational_data_class as variational_data_class
from pinn_data_manager_class import pinn_data_manager_class as pinn_data_manager_class
from neural_network_class import neural_network_class as neural_network_class
from pinn_options_class import pinn_options_class as pinn_options_class
from hyperparameters_class import hyperparameters_class as hyperparameters_class
from problem_specifications_class import problem_specifications_class as problem_specifications_class
from save_load_utilities_class import save_load_utilities_class as save_load_utilities_class
from tensor_utilities_class import tensor_utilities_class as tensor_utilities_class
from plotting_utilities_class import plotting_utilities_class as plotting_utilities_class
from printing_utilities_class import printing_utilities_class as printing_utilities_class


#%% ------------------------------------------------------------ PINN CLASS ------------------------------------------------------------

# Implement the pinn class.
class pinn_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self, pinn_options, hyperparameters, problem_specifications ):

        # Create an instance of the plotting utilities class.
        self.plotting_utilities = plotting_utilities_class(  )

        # Create an instance of the printing utilities class.
        self.printing_utilities = printing_utilities_class(  )

        # Create an instance of the tensor utilities class.
        self.tensor_utilities = tensor_utilities_class(  )

        # Create an instance of the save-load utilities class.
        self.save_load_utilities = save_load_utilities_class(  )

        # Store the pinn options, hyper-parameters, and problem specifications.
        self.pinn_options = pinn_options
        self.hyperparameters = hyperparameters
        self.problem_specifications = problem_specifications

        # Create the domain object.
        self.domain = domain_class( self.problem_specifications.temporal_domain, self.problem_specifications.spatial_domain, self.problem_specifications.domain_type, self.pinn_options.device )

        # Create the initial-boundary conditions.
        self.initial_boundary_conditions = self.create_initial_boundary_conditions( self.domain.dimension_labels, self.problem_specifications.ibc_types, self.problem_specifications.ibc_dimensions, self.problem_specifications.ibc_condition_functions, self.problem_specifications.ibc_placements )

        # Create the flow functions.
        self.flow_functions = self.problem_specifications.flow_functions

        # Create the pde object.
        # self.pde = pde_class( self.problem_specifications.pde_name, self.problem_specifications.pde_type, self.domain, self.initial_boundary_conditions, self.problem_specifications.residual_function, self.problem_specifications.residual_code, self.pinn_options.device )
        self.pde = pde_class( self.problem_specifications.pde_name, self.problem_specifications.pde_type, self.domain, self.initial_boundary_conditions, self.problem_specifications.residual_function, self.problem_specifications.residual_code, self.problem_specifications.flow_functions, self.pinn_options.device )

        # Create the training data.
        training_data = self.generate_training_testing_data( domain = self.domain, initial_boundary_conditions = self.initial_boundary_conditions, residual_batch_size = self.hyperparameters.residual_batch_size, element_volume_percent = self.hyperparameters.element_volume_percent, element_type = self.hyperparameters.element_type, element_computation_option = self.hyperparameters.element_computation_option, integration_order = self.hyperparameters.integration_order, application = 'training' )

        # Create the testing data.
        testing_data = self.generate_training_testing_data( domain = self.domain, initial_boundary_conditions = self.initial_boundary_conditions, residual_batch_size = self.hyperparameters.residual_batch_size, element_volume_percent = self.hyperparameters.element_volume_percent, element_type = self.hyperparameters.element_type, element_computation_option = self.hyperparameters.element_computation_option, integration_order = self.hyperparameters.integration_order, application = 'testing' )

        # Create the plotting data.
        plotting_data = self.generate_plotting_data( num_plotting_samples = self.pinn_options.num_plotting_samples, domain = self.domain )

        # Create the network layers.
        layers = self.generate_layers( self.problem_specifications.num_inputs, self.problem_specifications.num_outputs, self.hyperparameters.num_hidden_layers, self.hyperparameters.hidden_layer_widths )

        # Compute the exploration radius.
        exploration_radius_spatial = self.compute_exploration_radius( self.pinn_options.exploration_volume_percentage, self.domain, 'spatial' )
        exploration_radius_spatiotemporal = self.compute_exploration_radius( self.pinn_options.exploration_volume_percentage, self.domain, 'spatiotemporal' )

        # Compute the unique tolerance.
        unique_tolerance_spatial = self.compute_unique_tolerance( self.pinn_options.unique_volume_percentage, self.domain, 'spatial' )
        unique_tolerance_spatiotemporal = self.compute_unique_tolerance( self.pinn_options.unique_volume_percentage, self.domain, 'spatiotemporal' )

        # Compute the classification noise magnitude.
        classification_noise_magnitude_spatial = self.compute_classification_noise_magnitude( self.pinn_options.classification_noise_percentage, self.domain, 'spatial' )
        classification_noise_magnitude_spatiotemporal = self.compute_classification_noise_magnitude( self.pinn_options.classification_noise_percentage, self.domain, 'spatiotemporal' )

        # Create the network object.
        self.network = neural_network_class( self.hyperparameters.neuron_parameters, self.hyperparameters.synapse_parameters, layers, self.hyperparameters.activation_function, self.hyperparameters.learning_rate, self.hyperparameters.residual_batch_size, self.hyperparameters.num_epochs, self.problem_specifications.residual_function, self.problem_specifications.residual_code, self.problem_specifications.temporal_code, training_data, testing_data, plotting_data, self.domain.dimension_labels, self.hyperparameters.element_computation_option, self.pinn_options.batch_print_frequency, self.pinn_options.epoch_print_frequency, self.hyperparameters.c_IC, self.hyperparameters.c_BC, self.hyperparameters.c_residual, self.hyperparameters.c_variational, self.hyperparameters.c_monotonicity, self.pinn_options.newton_tolerance, self.pinn_options.newton_max_iterations, exploration_radius_spatial, exploration_radius_spatiotemporal, self.pinn_options.num_exploration_points, unique_tolerance_spatial, unique_tolerance_spatiotemporal, classification_noise_magnitude_spatial, classification_noise_magnitude_spatiotemporal, self.pinn_options.device, self.pinn_options.verbose_flag ).to( device = self.pinn_options.device )


    #%% ------------------------------------------------------------ PREPROCESS FUNCTIONS ------------------------------------------------------------

    # Implement a function to preprocess the number of inputs.
    def preprocess_num_inputs( self, num_inputs = None ):

        # Determine whether to use the stored number of inputs.
        if num_inputs is None:              # If no number of inputs was provided...

            # Set the number of inputs to be the stored value.
            num_inputs = self.problem_specifications.num_inputs

        # Return the number of inputs.
        return num_inputs


    # Implement a function to preprocess the number of outputs.
    def preprocess_num_outputs( self, num_outputs = None ):

        # Determine whether to use the stored number of outputs.
        if num_outputs is None:              # If no number of outputs was provided...

            # Set the number of outputs to be the stored value.
            num_outputs = self.problem_specifications.num_outputs

        # Return the number of outputs.
        return num_outputs


    # Implement a function to preprocess the domain.
    def preprocess_domain( self, domain = None ):

        # Determine whether to use the stored domain.
        if domain is None:          # If the domain was not provided...

            # Set the domain to be the stored value.
            domain = self.domain

        # Return the domain.
        return domain


    # Implement a function to preprocess the initial-boundary conditions.
    def preprocess_initial_boundary_conditions( self, initial_boundary_conditions = None ):

        # Determine whether to use the stored initial-boundary conditions.
        if initial_boundary_conditions is None:             # If the initial-boundary conditions were not provided...

            # Set the initial-boundary conditions to be the stored value.
            initial_boundary_conditions = self.initial_boundary_conditions

        # Return the initial-boundary conditions.
        return initial_boundary_conditions


    # Implement a function to preprocess the number of samples.
    def preprocess_num_samples( self, num_samples = None, application = 'training' ):

        # Determine whether to use the stored total number of residual data.
        if num_samples is None:                             # If the number of samples that we want to draw was not specified...

            # Compute the total quantity of residual data from stored parameters.
            num_samples = self.data_percent2data_quantity( condition_type = 'residual', application = application )

        # Return the number of samples.
        return num_samples

    
    # Implement a function to preprocess the residual batch size.
    def preprocess_residual_batch_size( self, residual_batch_size = None, application = 'training' ):

        # Determine whether to use the stored residual batch size.
        if ( residual_batch_size is None ) and ( application.lower(  ) == 'training' ):                 # If the residual batch size was not provided...

            # Set the residual batch size to be the stored value.
            residual_batch_size = self.hyperparameters.residual_batch_size

        # Return the residual batch size.
        return residual_batch_size


    # Implement a function to preprocess the element volume percent.
    def preprocess_element_volume_percent( self, element_volume_percent = None ):

        # Determine whether to use the stored element volume percent.
        if element_volume_percent is None:              # If the element volume percent was not provided...

            # Set the element volume percent to be the stored value.
            element_volume_percent = self.hyperparameters.element_volume_percent

        # Return the element volume percent.
        return element_volume_percent


    # Implement a function to preprocess the element type.
    def preprocess_element_type( self, element_type = None ):

        # Determine whether to use the stored element type.
        if element_type is None:                        # If the element type was not provided...

            # Set the element type to be the stored value.
            element_type = self.hyperparameters.element_type

        # Return the element type.
        return element_type


    # Implement a function to preprocess the element computation option.
    def preprocess_element_computation_option( self, element_computation_option = None ):

        # Determine whether to use the stored element computation option.
        if element_computation_option is None:          # If no element computation option was provided...

            # Set the element computation option to be the stored value.
            element_computation_option = self.hyperparameters.element_computation_option

        # Return the element computation option.
        return element_computation_option


    # Implement a function to preprocess the integration order.
    def preprocess_integration_order( self, integration_order = None ):

        # Determine whether to use the stored integration order.
        if integration_order is None:                   # If the integration order was not provided...

            # Set the integration order to be the stored value.
            integration_order = self.hyperparameters.integration_order

        # Return the integration order.
        return integration_order


    # Implement a function to preprocess the number of plotting samples.
    def preprocess_num_plotting_samples( self, num_plotting_samples = None ):

        # Determine whether to use the stored number of plotting samples.
        if num_plotting_samples is None:                # If the number of plotting samples was not provided...

            # Set the number of plotting samples to be the stored value.
            num_plotting_samples = self.pinn_options.num_plotting_samples

        # Return the number of plotting samples.
        return num_plotting_samples

    
    # Implement a function to preprocess the number of hidden layers.
    def preprocess_num_hidden_layers( self, num_hidden_layers = None ):

        # Determine whether to use the stored number of hidden layers.
        if num_hidden_layers is None:               # If the number of hidden layers was not provided...

            # Set the number of hidden layers to be the stored value.
            num_hidden_layers = self.hyperparameters.num_hidden_layers

        # Return the number of hidden layers.
        return num_hidden_layers


    # Implement a function to preprocess the hidden layer widths.
    def preprocess_hidden_layer_widths( self, hidden_layer_widths ):

        # Determine whether to use the stored hidden layers widths.
        if hidden_layer_widths is None:             # If the hidden layer widths were not provided...

            # Set the hidden layer widths to be the stored value.
            hidden_layer_widths = self.hyperparameters.hidden_layer_widths

        # Return the hidden layer widths.
        return hidden_layer_widths

    
    # Implement a function to preprocess the network.
    def preprocess_network( self, network = None ):

        # Determine whether to use the stored network.
        if network is None:                     # If the network was not provided...

            # Set the network to be the stored value.
            network = self.network

        # Return the network.
        return network


    # Implement a function to preprocess the training flag.
    def preprocess_train_flag( self, train_flag = None ):

        # Determine whether to use the stored training flag.
        if train_flag is None:          # If the training flag was not provided...

            # Set the training flag to be the stored value.
            train_flag = self.pinn_options.train_flag

        # Return the train flag.
        return train_flag


    # Implement a function to preprocess the save directory.
    def preprocess_save_directory( self, save_directory = None ):

        # Determine whether to set the save directory to be the stored value.
        if save_directory is None:              # If no save directory was provided...

            # Set the save directory to be the stored value.
            save_directory = self.pinn_options.save_directory

        # Return the save directory.
        return save_directory


    # Implement a function to preprocess the load directory.
    def preprocess_load_directory( self, load_directory = None ):

        # Determine whether to set the load directory to be the stored value.
        if load_directory is None:              # If no load directory was provided...

            # Set the load directory to be the stored value.
            load_directory = self.pinn_options.load_directory

        # Return the load directory.
        return load_directory


    # Implement a function to preprocess the pde.
    def preprocess_pde( self, pde = None ):

        # Determine whether to use the stored pde.
        if pde is None:                 # If the pde was not provided...

            # Set the pde to be the stored value.
            pde = self.pde

        # Return the pde.
        return pde

    
    # Implement a function to preprocess flow functions.
    def preprocess_flow_functions( self, flow_functions = None ):

        # Determine whether to use the stored flow fields.
        if flow_functions is None:               # If the flow field was not provided...

            # Use the stored flows.
            flow_functions = self.flow_functions

        # Return the flow functions.
        return flow_functions


    # Implement a function to preprocess the plotting data.
    def preprocess_plotting_data( self, plotting_data = None ):

        # Determine whether to use the stored plotting data.
        if plotting_data is None:          # If the input data was not provided...

            # Use the stored plotting data.
            plotting_data = self.network.plotting_data

        # Return the plotting data.
        return plotting_data


    # Implement a function to preprocess the exploration volume percentage.
    def preprocess_exploration_volume_percentage( self, exploration_volume_percentage = None ):

        # Determine whether to use the stored exploration volume percentage.
        if exploration_volume_percentage is None:               # If the exploration volume percentage was not provided...

            # Set the exploration volume percentage to be the stored value.
            exploration_volume_percentage = self.pinn_options.exploration_volume_percentage

        # Return the exploration volume percentage.
        return exploration_volume_percentage


    # Implement a function to preprocess the unique volume percentage.
    def preprocess_unique_volume_percentage( self, unique_volume_percentage = None ):

        # Determine whether to use the stored unique volume percentage.
        if unique_volume_percentage is None:                    # If the unique volume percentage was not provided...

            # Set the unique volume percentage to be the stored value.
            unique_volume_percentage = self.pinn_options.unique_volume_percentage

        # Return the unique volume percentage.
        return unique_volume_percentage


    # Implement a function to preprocess the classification noise percentage.
    def preprocess_classification_noise_percentage( self, classification_noise_percentage = None ):

        # Determine whether to use the stored classification noise percentage.
        if classification_noise_percentage is None:             # If the classification noise percentage was not provided...

            # Set the classification noise percentage to be the stored value.
            classification_noise_percentage = self.pinn_options.classification_noise_percentage

        # Return the classification noise percentage.
        return classification_noise_percentage
        

    # Implement a function to preprocess the level set guess.
    def preprocess_level_set_guesses( self, level_set_guesses, num_guesses, domain, domain_subset_type = 'spatiotemporal' ):

        # Determine whether to use the default level set guesses.
        if level_set_guesses is None:               # If the level set guesses were not provided...

            # Generate level set guesses.
            level_set_guesses = domain.sample_domain( num_guesses, domain_subset_type )

        # Return the level set guesses.
        return level_set_guesses


    # Implement a function to preprocess the temporal step size.
    def preprocess_temporal_step_size( self, dt ):

        # Preprocess the temporal step size.
        if dt is None:                              # If the temporal step size was not provided...

            # Set the default temporal step size.
            dt = torch.tensor( 1e-3, dtype = torch.float32, device = self.device )

        # Return the temporal step size.
        return dt


    # Implement a function to preprocess the temporal integration span.
    def preprocess_temporal_integration_span( self, tspan ):

        # Preprocess the temporal integration span.
        if tspan is None:                           # If the temporal integration span was not provided...

            # Set the default temporal integration span.
            tspan = torch.tensor( [ 0, 1 ], dtype = torch.float32, device = self.device )

        # Return the temporal integration span.
        return tspan


    # Implement a function to preprocess plot times.
    def preprocess_plot_times( self, plot_times = None ):

        # Determine whether to set the default plot time.
        if plot_times is None:                                  # If no plot times were provided...

            # Set the plot times to be a zero tensor.
            plot_times = torch.tensor( 0, dtype = torch.float32, device = self.device )

        # Return the plot time.
        return plot_times


    # Implement a function to preprocess the number of timesteps.
    def preprocess_num_timesteps( self, num_timesteps = None ):

        # Determine whether to set the default number of timesteps.
        if num_timesteps is None:                           # If the number of timesteps was not provided...

            # Set the number of timesteps to be the default value.
            num_timesteps = self.num_timesteps

        # Return the number of timesteps.
        return num_timesteps


    # Implement a function to preprocess the level.
    def preprocess_level( self, level = None ):

        # Determine whether to use the default level.
        if level is None:           # If the level was not provided...

            # Set the level to be zero.
            level = torch.tensor( 0, dtype = torch.float32, device = self.device )

        # Return the level.
        return level
    

    # Implement a function to preprocess the number of level set guesses.
    def preprocess_num_guesses( self, num_guesses ):

        # Determine whether to use the default number of level set guesses.
        if num_guesses is None:                 # If the number of level set guesses was not provided.

            # Set the number of level set guesses to be the default value.
            num_guesses = torch.tensor( 1e2, dtype = torch.int64, device = self.device )

        # Return the number of guesses.
        return num_guesses


    # Implement a function to preprocess the newton tolerance.
    def preprocess_newton_tolerance( self, newton_tolerance = None ):

        # Determine whether to use the default network tolerance.
        if newton_tolerance is None:            # If the newton tolerance was not provided...

            # Set the newton tolerance to be the stored value.
            newton_tolerance = self.network.newton_tolerance

        # Return the newton tolerance.
        return newton_tolerance


    # Implement a function to preprocess the maximum number of newton iterations.
    def preprocess_newton_max_iterations( self, newton_max_iterations = None ):

        # Determine whether to use the default maximum number of newton iterations.
        if newton_max_iterations is None:           # If the maximum number of newton iterations were not provided...

            # Set the newton maximum iterations to be the stored value.
            newton_max_iterations = self.network.newton_max_iterations

        # Return the maximum number of newton iterations.
        return newton_max_iterations


    # Implement a function to preprocess the exploration radius.
    def preprocess_exploration_radius( self, exploration_radius = None, domain_subset_type = 'spatiotemporal' ):

        # Determine how to preprocess the exploration radius.
        if domain_subset_type.lower(  ) == 'spatial':               # If the domain subset type is spatial...

            # Preprocess the exploration radius.
            exploration_radius = self.network.preprocess_exploration_radius_spatial( exploration_radius )

        elif domain_subset_type.lower(  ) == 'spatiotemporal':      # If the domain subset type is spatiotemporal...

            # Preprocess the exploration radius.
            exploration_radius = self.network.preprocess_exploration_radius_spatiotemporal( exploration_radius )

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
            num_exploration_points = self.network.num_exploration_points

        # Return the number of exploration points.
        return num_exploration_points


    # Implement a function to preprocess the unique tolerance.
    def preprocess_unique_tolerance( self, unique_tolerance = None, domain_subset_type = 'spatiotemporal' ):

        # Determine how to preprocess the unique tolerance.
        if domain_subset_type.lower(  ) == 'spatiotemporal':                                                # If the domain subset type is spatiotemporal...

            # Preprocess the spatial unique tolerance.
            unique_tolerance = self.network.preprocess_unique_tolerance_spatial( unique_tolerance )

        elif domain_subset_type.lower(  ) == 'spatial':                                                     # If the domain subset type is spatial...

            # Preprocess the spatiotemporal unique tolerance.
            unique_tolerance = self.network.preprocess_unique_tolerance_spatiotemporal( unique_tolerance )

        else:                                                                                               # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid domain subset type: {domain_subset_type}' )

        # Return the unique tolerance.
        return unique_tolerance


    # Implement a function to preprocess the classification noise magnitude.
    def preprocess_classification_noise_magnitude( self, classification_noise_magnitude = None, domain_subset_type = 'spatiotemporal' ):

        # Determine how to preprocess the classification noise magnitude.
        if domain_subset_type.lower(  ) == 'spatial':                                                # If the domain subset type is spatial...

            # Preprocess the spatial classification noise magnitude.
            classification_noise_magnitude = self.network.preprocess_classification_noise_magnitude_spatial( classification_noise_magnitude )

        elif domain_subset_type.lower(  ) == 'spatiotemporal':                                                     # If the domain subset type is spatiotemporal...

            # Preprocess the spatiotemporal classification noise magnitude.
            classification_noise_magnitude = self.network.preprocess_classification_noise_magnitude_spatiotemporal( classification_noise_magnitude )

        else:                                                                                               # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid domain subset type: {domain_subset_type}' )

        # Return the classification noise magnitude.
        return classification_noise_magnitude


<<<<<<< HEAD
    # Implement a function to preprocess the number of noisy sample points per level set point.
    def preprocess_num_noisy_samples_per_level_set_point( self, num_noisy_samples_per_level_set_point = None ):

        # Determine whether to use the default number of noisy sample points per level set point.
        if num_noisy_samples_per_level_set_point is None:               # If the number of noisy samples per level set point was not provided...

            # Set the number of noisy sample points per level set point.
            num_noisy_samples_per_level_set_point = torch.tensor( 1, dtype = torch.float32, device = self.device )

        # Return the number of noisy sample points per level set point.
        return num_noisy_samples_per_level_set_point
    

    # Implement a function to preprocess the classification data.
    def preprocess_classification_data( self, classification_data, num_spatial_dimensions, num_timesteps, domain, plot_time, level, level_set_guesses, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, classification_noise_magnitude, num_noisy_samples_per_level_set_point, domain_subset_type ):
=======
    # Implement a function to preprocess the classification data.
    def preprocess_classification_data( self, classification_data, num_spatial_dimensions, num_timesteps, domain, plot_time, level, level_set_guesses, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, classification_noise_magnitude, domain_subset_type ):
>>>>>>> 55162c78e9fb0c13d60ea20df5463b1e4d4f30fb

        # Determine whether to generate the classification data at which to compute the classification loss.
        if classification_data is None:                 # If the classification data was not provided...

            # Determine whether to generate the classification data.
            if ( num_spatial_dimensions is not None ) and ( domain is not None ) and ( plot_time is not None ):

                # Generate the classification data.
<<<<<<< HEAD
                classification_data = self.network.generate_classification_data( num_spatial_dimensions, num_timesteps, domain, plot_time, level, level_set_guesses, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, classification_noise_magnitude, num_noisy_samples_per_level_set_point, domain_subset_type )
=======
                classification_data = self.network.generate_classification_data( num_spatial_dimensions, num_timesteps, domain, plot_time, level, level_set_guesses, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, classification_noise_magnitude, domain_subset_type )
>>>>>>> 55162c78e9fb0c13d60ea20df5463b1e4d4f30fb

            else:

                # Throw an error.
                raise ValueError( 'Either classification data or num_spatial_dimensions, domain, and plot_time must not be None.' )

        # Return the classification data.
        return classification_data


    #%% ------------------------------------------------------------ SETUP FUNCTIONS ------------------------------------------------------------

    # Implement a function to setup for generating initial-boundary condition data.
    def setup_initial_boundary_condition_data_generation( self, domain = None, initial_boundary_conditions = None ):

        # Preprocess the domain.
        domain = self.preprocess_domain( domain )

        # Preprocess the initial-boundary conditions.
        initial_boundary_conditions = self.preprocess_initial_boundary_conditions( initial_boundary_conditions )

        # Return the information required to generate initial-boundary condition data.
        return domain, initial_boundary_conditions


    # Implement a function to setup for generating residual data.
    def setup_residual_data_generation( self, num_samples = None, domain = None, application = 'training' ):

        # Preprocess the number of samples.
        num_samples = self.preprocess_num_samples( num_samples, application )

        # Preprocess the domain.
        domain = self.preprocess_domain( domain )

        # Define the residual data ID.
        id = torch.tensor( 1, dtype = torch.uint8, device = self.pinn_options.device )

        # Define the residual data name.
        name = 'residual'

        # Return the information necessary to generate residual data.
        return num_samples, domain, id, name


    # Implement a function to setup for the generation of variational data.
    def setup_variation_data_generation( self, domain = None ):

        # Preprocess the domain.
        domain = self.preprocess_domain( domain )

        # Define the variational data ID.
        id = torch.tensor( 1, dtype = torch.uint8, device = self.pinn_options.device )

        # Define the variational data name.
        name = 'variational'

        # Return the information necessary for generating variational data.
        return domain, id, name


    # Implement a function to setup for the generation of training / testing data.
    def setup_training_testing_data_generation( self, domain = None, initial_boundary_conditions = None, residual_batch_size = None, element_volume_percent = None, element_type = None, element_computation_option = None, integration_order = None, application = 'training' ):

        # Preprocess the domain.
        domain = self.preprocess_domain( domain )

        # Preprocess the initial-boundary conditions.
        initial_boundary_conditions = self.preprocess_initial_boundary_conditions( initial_boundary_conditions )

        # Preprocess the residual batch size.
        residual_batch_size = self.preprocess_residual_batch_size( residual_batch_size, application )

        # Preprocess the element volume percentage.
        element_volume_percent = self.preprocess_element_volume_percent( element_volume_percent )

        # Preprocess the element type.
        element_type = self.preprocess_element_type( element_type )

        # Preprocess the element computation options.
        element_computation_option = self.preprocess_element_computation_option( element_computation_option )

        # Preprocess the integration order.
        integration_order = self.preprocess_integration_order( integration_order )

        # Return the information necessary for generating training / testing data.
        return domain, initial_boundary_conditions, residual_batch_size, element_volume_percent, element_type, element_computation_option, integration_order


    # Implement a function to setup for the generation of plotting data.
    def setup_plotting_data_generation( self, num_plotting_samples = None, domain = None ):

        # Implement a function to preprocess the number of plotting samples.
        num_plotting_samples = self.preprocess_num_plotting_samples( num_plotting_samples )

        # Preprocess the domain.
        domain = self.preprocess_domain( domain )

        # Return the data necessary for setting up for the generation of plotting data.
        return num_plotting_samples, domain


    # Implement a function to setup for element scale computation.
    def setup_element_scale_computation( self, domain = None, element_volume_percent = None, element_type = None ):

        # Preprocess the domain.
        domain = self.preprocess_domain( domain )

        # Preprocess the element volume percentage.
        element_volume_percent = self.preprocess_element_volume_percent( element_volume_percent )

        # Preprocess the element type.
        element_type = self.preprocess_element_type( element_type )

        # Return the information necessary for element scale computation.
        return domain, element_volume_percent, element_type


    # Implement a function to setup for layer generation.
    def setup_layer_generation( self, num_inputs = None, num_outputs = None, num_hidden_layers = None, hidden_layer_widths = None ):

        # Preprocess the number of inputs.
        num_inputs = self.preprocess_num_inputs( num_inputs )

        # Preprocess the number of outputs.
        num_outputs = self.preprocess_num_outputs( num_outputs )

        # Preprocess the number of hidden layers.
        num_hidden_layers = self.preprocess_num_hidden_layers( num_hidden_layers )

        # Preprocess the hidden layer widths.
        hidden_layer_widths = self.preprocess_hidden_layer_widths( hidden_layer_widths )

        # Return the information necessary for layer generation.
        return num_inputs, num_outputs, num_hidden_layers, hidden_layer_widths


    # Implement a function to setup for training.
    def setup_training( self, network = None, train_flag = None ):

        # Preprocess the network.
        network = self.preprocess_network( network )

        # Preprocess the training flag.
        train_flag = self.preprocess_train_flag( train_flag )

        # Return the information necessary for training.
        return network, train_flag


    # Implement a function to setup for plotting training, testing, and plotting data.
    def setup_network_plotting( self, network = None, save_directory = None ):

        # Preprocess the network.
        network = self.preprocess_network( network )

        # Preprocess the save directory.
        save_directory = self.preprocess_save_directory( save_directory )

        # Return the information necessary to plot training, testing, and plotting data.
        return network, save_directory


    # Implement a function to setup for plotting training, testing, and plotting data.
    def setup_network_initial_final_plotting( self, domain = None, network = None, save_directory = None ):

        # Preprocess the domain.
        domain = self.preprocess_domain( domain )

        # Preprocess the network.
        network = self.preprocess_network( network )

        # Preprocess the save directory.
        save_directory = self.preprocess_save_directory( save_directory )

        # Return the information necessary to plot training, testing, and plotting data.
        return domain, network, save_directory


    # Implement a function to setup for pde plotting.
    def setup_pde_plotting( self, pde = None, save_directory = None ):

        # Preprocess the pde.
        pde = self.preprocess_pde( pde )

        # Preprocess the save directory.
        save_directory = self.preprocess_save_directory( save_directory )

        # Return the information required for plotting pde domains and initial-boundary conditions.
        return pde, save_directory


    # Implement a function to setup for flow field plotting.
    def setup_flow_field_plotting( self, plotting_data = None, flow_functions = None, save_directory = None ):

        # Preprocess the plotting data.
        plotting_data = self.preprocess_plotting_data( plotting_data )

        # Preprocess the flow functions.
        flow_functions = self.preprocess_flow_functions( flow_functions )

        # Preprocess the save directory.
        save_directory = self.preprocess_save_directory( save_directory )

        # Return the information required for flow field plotting.
        return plotting_data, flow_functions, save_directory


    # Implement a function to setup for roa boundary plotting.
    def setup_roa_boundary_plotting( self, plotting_data = None, domain = None, flow_functions = None, network = None, save_directory = None ):

        # Preprocess the plotting data.
        plotting_data = self.preprocess_plotting_data( plotting_data )

        # Preprocess the domain.
        domain = self.preprocess_domain( domain )

        # Preprocess the flow functions.
        flow_functions = self.preprocess_flow_functions( flow_functions )

        # Preprocess the network.
        network = self.preprocess_network( network )

        # Preprocess the save directory.
        save_directory = self.preprocess_save_directory( save_directory )

        # Return the information necessary for roa boundary plotting.
        return plotting_data, domain, flow_functions, network, save_directory


    # Implement a function to setup for classification loss computation.
<<<<<<< HEAD
    def setup_classification_loss( self, network = None, classification_data = None, level_set_guesses = None, classification_noise_magnitude = None, num_noisy_samples_per_level_set_point = None, unique_tolerance = None, num_exploration_points = None, exploration_radius = None, newton_max_iterations = None, newton_tolerance = None, num_guesses = None, level = None, plot_time = None, tspan = None, dt = None, num_spatial_dimensions = None, num_timesteps = None, domain = None, domain_subset_type = 'spatial' ):
=======
    def setup_classification_loss( self, network = None, classification_data = None, level_set_guesses = None, classification_noise_magnitude = None, unique_tolerance = None, num_exploration_points = None, exploration_radius = None, newton_max_iterations = None, newton_tolerance = None, num_guesses = None, level = None, plot_time = None, tspan = None, dt = None, num_spatial_dimensions = None, num_timesteps = None, domain = None, domain_subset_type = 'spatial' ):
>>>>>>> 55162c78e9fb0c13d60ea20df5463b1e4d4f30fb

        # Preprocess the integration time step size.
        dt = self.preprocess_temporal_step_size( dt )

        # Preprocess the integration temporal span.
        tspan = self.preprocess_temporal_integration_span( tspan )

        # Preprocess the plot time.
        plot_time = self.preprocess_plot_times( plot_time )

        # Preprocess the number of timesteps.
        num_timesteps = self.preprocess_num_timesteps( num_timesteps )

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

<<<<<<< HEAD
        # Preprocess the number of noisy samples per level set point.
        num_noisy_samples_per_level_set_point = self.preprocess_num_noisy_samples_per_level_set_point( num_noisy_samples_per_level_set_point )

=======
>>>>>>> 55162c78e9fb0c13d60ea20df5463b1e4d4f30fb
        # Preprocess the level set guesses.
        level_set_guesses = self.preprocess_level_set_guesses( level_set_guesses, num_guesses, domain, domain_subset_type )

        # Preprocess the classification data.
<<<<<<< HEAD
        classification_data = self.preprocess_classification_data( classification_data, num_spatial_dimensions, num_timesteps, domain, plot_time, level, level_set_guesses, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, classification_noise_magnitude, num_noisy_samples_per_level_set_point, domain_subset_type )
=======
        classification_data = self.preprocess_classification_data( classification_data, num_spatial_dimensions, num_timesteps, domain, plot_time, level, level_set_guesses, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, classification_noise_magnitude, domain_subset_type )
>>>>>>> 55162c78e9fb0c13d60ea20df5463b1e4d4f30fb

        # Preprocess the network.
        network = self.preprocess_network( network )

        # Return the classification loss parameters.
<<<<<<< HEAD
        return network, classification_data, level_set_guesses, classification_noise_magnitude, num_noisy_samples_per_level_set_point, unique_tolerance, num_exploration_points, exploration_radius, newton_max_iterations, newton_tolerance, num_guesses, level, plot_time, tspan, dt
=======
        return network, classification_data, level_set_guesses, classification_noise_magnitude, unique_tolerance, num_exploration_points, exploration_radius, newton_max_iterations, newton_tolerance, num_guesses, level, plot_time, tspan, dt
>>>>>>> 55162c78e9fb0c13d60ea20df5463b1e4d4f30fb


    #%% ------------------------------------------------------------ INITIAL-BOUNDARY CONDITION FUNCTIONS ------------------------------------------------------------

    # Implement a function to generate initial and boundary conditions from problem specifications.
    def create_initial_boundary_conditions( self, dimension_labels, ibc_types, ibc_dimensions, ibc_condition_functions, ibc_placements ):

        # Ensure that the initial-boundary condition problem specifications are compatible with one another.
        assert self.validate_ibc_compatibility( ibc_types, ibc_dimensions, ibc_condition_functions, ibc_placements )

        # Retrieve the number of initial-boundary conditions to generate.
        num_initial_boundary_conditions = torch.tensor( len( ibc_types ), dtype = torch.uint8, device = self.pinn_options.device )

        # Initialize a list to store the initial-boundary conditions.
        initial_boundary_conditions = [  ]

        # Create each of the initial-boundary condition objects.
        for k in range( num_initial_boundary_conditions ):                      # Iterate through each of the initial-boundary conditions...

            # Create the ID for this initial-boundary condition.
            id = torch.tensor( k + 1, dtype = torch.uint8, device = self.pinn_options.device )

            # Convert the label associated with this dimension to an initial-boundary condition general type.
            general_type = self.dimension_label2ibc_specific_type( dimension_labels[ ibc_dimensions[ k ].item(  ) ] )

            # Create the name for this initial-boundary condition.
            name = 'dim' + str( ibc_dimensions[ k ].item(  ) ) + ' ' + ibc_placements[ k ] + ' ' + ibc_types[ k ] + ' ' + general_type + ' condition'

            # Create this initial-boundary condition.
            initial_boundary_conditions.append( initial_boundary_condition_class( id, name, general_type, ibc_types[ k ], ibc_dimensions[ k ], ibc_condition_functions[ k ], ibc_placements[ k ], self.pinn_options.device ) )

        # Return the list of initial-boundary conditions.
        return initial_boundary_conditions

    
    # Implement a function to return the initial-boundary conditions objects that are initial conditions.
    def get_initial_conditions( self, initial_boundary_conditions ):

        # Collect the initial conditions from the initial-boundary conditions list.
        initial_conditions = [ initial_boundary_conditions[ k ] for k in range( len( initial_boundary_conditions ) ) if initial_boundary_conditions[ k ].general_type.lower(  ) == 'initial' ]

        # Return the initial conditions.
        return initial_conditions


    # Implement a function to return the initial-boundary conditions objects that are boundary conditions.
    def get_boundary_conditions( self, initial_boundary_conditions ):

        # Collect the boundary conditions from the initial-boundary conditions list.
        boundary_conditions = [ initial_boundary_conditions[ k ] for k in range( len( initial_boundary_conditions ) ) if initial_boundary_conditions[ k ].general_type.lower(  ) == 'boundary' ]

        # Return the initial conditions.
        return boundary_conditions


    # Implement a function to separate a list of initial-boundary conditions into its constituent initial and boundary conditions.
    def separate_initial_boundary_conditions( self, initial_boundary_conditions ):

        # Retrieve the initial conditions.
        initial_conditions = self.get_initial_conditions( initial_boundary_conditions )

        # Retrieve the boundary conditions.
        boundary_conditions = self.get_boundary_conditions( initial_boundary_conditions )

        # Return the initial and boundary conditions.
        return initial_conditions, boundary_conditions


    #%% ------------------------------------------------------------ TRAINING / TESTING / PLOTTING DATA FUNCTIONS ------------------------------------------------------------

    # Implement a function to generate initial-boundary condition training / testing data.
    def generate_initial_boundary_condition_data( self, num_samples = None, num_timesteps = None, domain = None, initial_boundary_conditions = None, application = 'training' ):

        # Determine whether to use the stored number of timesteps.
        if num_timesteps is None:               # If the number of timesteps was not provided...

            # Use the stored number of timesteps.
            num_timesteps = self.hyperparameters.num_timesteps

        # Setup for the generation of initial-boundary condition data.
        domain, initial_boundary_conditions = self.setup_initial_boundary_condition_data_generation( domain, initial_boundary_conditions )

        # Retrieve the number of initial-boundary conditions.
        num_initial_boundary_conditions = torch.tensor( len( initial_boundary_conditions ), dtype = torch.uint8, device = self.pinn_options.device )

        # Initialize an empty list to store the initial-boundary condition data sets.
        initial_boundary_condition_data = [  ]

        # Create the data associated with each initial-boundary condition.
        for k1 in range( num_initial_boundary_conditions ):                      # Iterate through each of the initial-boundary conditions...

            # Retrieve the number of condition functions.
            num_condition_functions = torch.tensor( len( initial_boundary_conditions[ k1 ].condition_functions ), dtype = torch.uint8, device = self.pinn_options.device )

            # Determine whether to use the stored total number of initial-boundary condition data.
            if num_samples is None:                             # If the number of samples that we want to draw was not specified...

                # Compute the total quantity of initial-boundary condition data from stored parameters.
                num_initial_boundary_condition_data_total = self.data_percent2data_quantity( condition_type = initial_boundary_conditions[ k1 ].general_type, application = application )

            else:                                               # Otherwise...

                # Use the provided value.
                num_initial_boundary_condition_data_total = num_samples

            # Compute the quantity of initial-boundary condition data per initial condition.
            num_data_per_initial_boundary_condition = ( num_initial_boundary_condition_data_total/num_initial_boundary_conditions ).to( torch.int32 )

            # Create the initial-boundary condition input data.
            input_data = domain.sample_domain_boundary( num_data_per_initial_boundary_condition, initial_boundary_conditions[ k1 ].dimension, placement_string = initial_boundary_conditions[ k1 ].placement, domain_type = 'spatiotemporal' )

            # Ensure that each network input is presented for the provided number of timesteps.
            input_data = torch.tile( torch.unsqueeze( input_data, dim = 2 ), [ 1, 1, num_timesteps ] )

            # Compute the output data.
            output_data = [ initial_boundary_conditions[ k1 ].condition_functions[ k2 ]( input_data ) for k2 in range( num_condition_functions ) ]

            # Compute the output derivative order associated with this condition type.
            output_derivative_order = self.condition_type2output_order( initial_boundary_conditions[ k1 ].specific_type )

            # Create this initial condition data set.
            initial_boundary_condition_data.append( ibc_data_class( initial_boundary_conditions[ k1 ].id, initial_boundary_conditions[ k1 ].name, initial_boundary_conditions[ k1 ].general_type, initial_boundary_conditions[ k1 ].dimension, domain.dimension_labels, input_data, output_data, output_derivative_order, device = self.pinn_options.device ) )

        # Return the initial-boundary condition data sets.
        return initial_boundary_condition_data


    # Implement a function to generate residual training / testing data.
    def generate_residual_data( self, num_samples = None, num_timesteps = None, domain = None, batch_size = None, application = 'training' ):

        # Determine whether to use the stored number of timesteps.
        if num_timesteps is None:                           # If the number of timesteps was not provided...

            # Use the stored number of timesteps.
            num_timesteps = self.hyperparameters.num_timesteps

        # Setup for the generation of residual data.
        num_samples, domain, id, name = self.setup_residual_data_generation( num_samples, domain, application )

        # Create the input data by taking a sample over the spatiotemporal domain.
        input_data = domain.sample_domain( num_samples, domain_type = 'spatiotemporal' )

        # Ensure that each network input is presented for the provided number of timesteps.
        input_data = torch.tile( torch.unsqueeze( input_data, dim = 2 ), [ 1, 1, num_timesteps ] )

        # Create the residual data.
        residual_data = residual_data_class( id, name, domain.dimension_labels, input_data, batch_size, device = self.pinn_options.device )

        # Return the residual data.
        return residual_data


    # Implement a function to generate the variational training / testing data.
    def generate_variational_data( self, domain = None, element_scale = None, integration_order = None, xs_element_centers = None, batch_size = None ):

        # Setup for the generation of variational data.
        domain, id, name = self.setup_variation_data_generation( domain )

        # Create the variational data.
        variational_data = variational_data_class( id, name, domain.dimension_labels, element_scale, integration_order, xs_element_centers, batch_size, device = self.pinn_options.device )

        # Return the variational data.
        return variational_data
        

    # Implement a function to generate the pinn training data.
    def generate_training_testing_data( self, num_samples = None, num_timesteps = None, domain = None, initial_boundary_conditions = None, residual_batch_size = None, element_volume_percent = None, element_type = None, element_computation_option = None, integration_order = None, application = 'training' ):

        # Determine whether to use the stored number of timesteps.
        if num_timesteps is None:               # If the number of timesteps was not provided...

            # Use the stored number of timesteps.
            num_timesteps = self.hyperparameters.num_timesteps

        # Setup for the generation of training / testing data.
        domain, initial_boundary_conditions, residual_batch_size, element_volume_percent, element_type, element_computation_option, integration_order = self.setup_training_testing_data_generation( domain, initial_boundary_conditions, residual_batch_size, element_volume_percent, element_type, element_computation_option, integration_order, application )

        # Retrieve the initial conditions from the initial-boundary conditions object.
        initial_conditions, boundary_conditions = self.separate_initial_boundary_conditions( initial_boundary_conditions )

        # Generate the initial condition data.
        initial_condition_data = self.generate_initial_boundary_condition_data( num_samples, num_timesteps, domain, initial_conditions, application )

        # Generate the boundary condition data.
        boundary_condition_data = self.generate_initial_boundary_condition_data( num_samples, num_timesteps, domain, boundary_conditions, application )

        # Generate the residual data.
        residual_data = self.generate_residual_data( num_samples, num_timesteps, domain, residual_batch_size, application )

        # Compute the element scale.
        element_scale = self.compute_element_scale( domain, element_volume_percent, element_type )

        # Compute the element centers from the residual data as specified by the element computation option.
        xs_element_centers = self.element_computation_option2xs_element_centers( element_computation_option, residual_data, application )

        # Generate the variational data.
        variational_data = self.generate_variational_data( domain, element_scale, integration_order, xs_element_centers, residual_batch_size )

        # Create the training / testing data object.
        training_testing_data = pinn_data_manager_class( initial_condition_data, boundary_condition_data, residual_data, variational_data, self.pinn_options.device )

        # Return the training / testing data object.
        return training_testing_data


    # Implement a function to generate the pinn plotting data.
    def generate_plotting_data( self, num_plotting_samples = None, num_timesteps = None, domain = None ):

        # Determine whether to use the stored number of timesteps.
        if num_timesteps is None:               # If the number of timesteps was not provided...

            # Set the number of timesteps to be the stored value.
            num_timesteps = self.hyperparameters.num_timesteps

        # Setup for the generation of plotting data.
        num_plotting_samples, domain = self.setup_plotting_data_generation( num_plotting_samples, domain )

        # Create a list of the grid vectors.
        grid_vectors = [ torch.linspace( domain.spatiotemporal_domain[ 0, k ], domain.spatiotemporal_domain[ 1, k ], num_plotting_samples, dtype = torch.float32, device = self.pinn_options.device ) for k in range( domain.num_spatiotemporal_dimensions ) ]

        # Create a grid from the grid vectors.
        grid_tensors = torch.meshgrid( grid_vectors, indexing = 'ij' )

        # Stack the grid tensors.
        grid = torch.cat( tuple( grid_tensors[ k ].unsqueeze( domain.num_spatiotemporal_dimensions ) for k in range( domain.num_spatiotemporal_dimensions ) ), dim = domain.num_spatiotemporal_dimensions )

        # Ensure that the plotting grid is repeated for each network timestep.
        grid = torch.tile( torch.unsqueeze( grid, dim = 4 ), dims = [ 1, 1, 1, 1, num_timesteps ] )

        # Return the grid.
        return grid


    #%% ------------------------------------------------------------ VALIDATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate the compatibility of the initial and boundary condition specifications.
    def validate_ibc_compatibility( self, ibc_types, ibc_dimensions, ibc_condition_functions, ibc_placements ):

        # Determine whether the ibc specs are compatible.
        valid_flag = ( len( ibc_types ) == ibc_dimensions.shape[ 0 ] ) and ( ibc_dimensions.shape[ 0 ] == len( ibc_condition_functions ) ) and ( ( len( ibc_condition_functions ) == len( ibc_placements ) ) )

        # Return the validity flag.
        return valid_flag


    #%% ------------------------------------------------------------ CONVERSION FUNCTIONS ------------------------------------------------------------

    # Implement a function to convert a dimension label to an initial-boundary condition general type.
    def dimension_label2ibc_specific_type( self, dimension_label ):

        # Determine the general initial-boundary condition type associated with this condition.
        if ( dimension_label.lower(  ) == 't' ) or ( dimension_label.lower(  ) == 'temporal' ):                           # If the dimension label is 'temporal'...
            
            # Set the general initial-boundary condition type to be 'initial.'
            general_type = 'initial'

        elif ( dimension_label.lower(  ) == 'x' ) or ( dimension_label.lower(  ) == 'spatial' ):                          # If the dimension label is 'spatial'...

            # Set the general initial-boundary condition type to be 'boundary.'
            general_type = 'boundary'

        else:                                                                                                               # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid dimension label: {dimension_label}' )

        # Return the initial-boundary condition type.
        return general_type


    # Implement a function to compute the amount of initial condition data.
    def data_percent2data_quantity( self, p = None, num_total = None, condition_type = 'initial', application = 'training' ):

        # Determine whether to use the stored data percent.
        if p is None:                       # If no data percent was provided...

            # Determine which stored value to use.
            if condition_type.lower(  ) == 'initial':                        # If we want to convert the data percentage to a data quantity for the initial condition data...

                # Use the stored percent initial condition data.
                p = self.hyperparameters.p_initial

            elif condition_type.lower(  ) == 'boundary':                        # If we want to convert the data percentage to a data quantity for the boundary condition data...

                # Use the stored percent boundary condition data.
                p = self.hyperparameters.p_boundary

            elif ( condition_type.lower(  ) == 'residual' ) or ( condition_type.lower(  ) == 'variational' ):                        # If we want to convert the data percentage to a data quantity for the residual data...

                # Use the stored percent residual data.
                p = self.hyperparameters.p_residual

            else:                                                           # Otherwise...

                # Throw an error.
                raise ValueError( 'Invalid condition type: {condition_type}' )

        # Determine whether to use the stored total training data quantity.
        if num_total is None:                       # If no total data quantity was provided...

            # Determine whether to use the stored training or testing data quantity.
            if application.lower(  ) == 'training':                             # If we want to reference the total quantity of training data...

                # Use the stored total training data quantity.
                num_total = self.hyperparameters.num_training_data

            elif application.lower(  ) == 'testing':                            # If we want to reference the total quantity of testing data...

                # Use the stored total testing data quantity.
                num_total = self.hyperparameters.num_testing_data

            else:                                                               # Otherwise...

                # Throw an error.
                raise ValueError( f'Invalid application type: {application}' )

        # Compute the quantity of data.
        num_data = ( p.to( torch.float32 )*num_total ).to( torch.int32 )

        # Return the quantity data.
        return num_data


    # Implement a function to determine the output derivative order associated with a specific condition type.
    def condition_type2output_order( self, condition_type ):

        # Determine how to define the output derivative order.
        if condition_type.lower(  ) == 'dirichlet':                     # If the initial condition type is 'dirichlet'...

            # Set the output derivative order to zero.
            output_derivative_order = torch.zeros( size = ( 1, 1 ), dtype = torch.uint8, device = self.pinn_options.device )

        elif condition_type.lower(  ) == 'neumann':                     # If the initial condition type is 'neumann'...

            # Set the output derivative order to one.
            output_derivative_order = torch.ones( size = ( 1, 1 ), dtype = torch.uint8, device = self.pinn_options.device )

        elif condition_type.lower(  ) == 'cauchy':                     # If the initial condition type is 'cauchy'...

            # Set the output derivative order to be zero, one.
            output_derivative_order = torch.tensor( [ 0, 1 ], dtype = torch.uint8, device = self.pinn_options.device )

        elif condition_type.lower(  ) == 'yuan-li':                     # IF the initial condition type is 'yuan-li'...

            # Set the output derivative order to be two.
            output_derivative_order = 2*torch.ones( ( 1, 1 ), dtype = torch.uint8, device = self.pinn_options.device )

        else:                                                                                   # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid specific condition type: {condition_type}' )

        # Return the output derivative order.
        return output_derivative_order


    # Implement a function to convert an element computation option to element center data.
    def element_computation_option2xs_element_centers( self, element_computation_option, residual_data, application = 'training' ):

        # Determine whether to specify element centers.
        if ( element_computation_option.lower(  ) in ( 'precompute', 'precomputed', 'pre-compute', 'pre-computed', 'before', 'before training', 'before_training', 'beforetraining' ) ) or ( application.lower(  ) == 'testing' ):              # If we want to compute the finite elements before training...

            # Set the element centers to be the residual data.
            xs_element_centers = residual_data.input_data

        elif element_computation_option.lower(  ) in ( 'dynamic', 'during', 'during training', 'during_training', 'duringtraining' ):                              # If we want to dynamically compute the finite elements during training...

            # Set the element centers to be None.
            xs_element_centers = None

        else:                                                       # Otherwise... ( i.e., the element computation type is not recognized... )

            # Throw an error.
            raise ValueError( f'Invalid element computation option: {element_computation_option}' )

        # Return the element centers.
        return xs_element_centers


    #%% ------------------------------------------------------------ UTILITY FUNCTIONS ------------------------------------------------------------

    # Implement a function to compute the scale of a finite element.
    def compute_element_scale( self, domain = None, element_volume_percent = None, element_type = None ):

        # Setup for element scale computation
        domain, element_volume_percent, element_type = self.setup_element_scale_computation( domain, element_volume_percent, element_type )

        # Compute the domain volume.
        domain_volume = domain.compute_volume(  )

        # Compute the element volume.
        element_volume = element_volume_percent*domain_volume

        # Determine how to compute the element scale.
        if element_type.lower(  ) == 'rectangular':                 # If the element type is rectangular...

            # Compute the domain ranges.
            domain_ranges = domain.compute_ranges(  )

            # Determine the critical dimension index ( i.e., the dimension with the largest range ).
            index = torch.argmax( domain_ranges )

            # Compute the domain dimension proportions.
            ps = domain_ranges/domain_ranges[ index ]

            # Compute the element size.
            element_size = torch.pow( element_volume/torch.prod( ps ), 1/domain.num_spatiotemporal_dimensions )

            # Compute the element scale.
            element_scale = torch.ones( domain.num_spatiotemporal_dimensions, dtype = torch.float32, device = self.pinn_options.device )

            # Set the scale of each element dimension.
            for k in range( domain.num_spatiotemporal_dimensions ):                     # Iterate through each of the spatiotemporal dimensions...

                # Set the scale of this element dimension.
                element_scale[ k ] = ps[ k ]*element_size

        # Return the element scale.
        return element_scale


    # Implement a function to compute the network layer architecture.
    def generate_layers( self, num_inputs, num_outputs, num_hidden_layers, hidden_layer_widths ):

        # Setup for layer generation.
        num_inputs, num_outputs, num_hidden_layers, hidden_layer_widths = self.setup_layer_generation( num_inputs, num_outputs, num_hidden_layers, hidden_layer_widths )

        # Create the network layers.
        layers = torch.tensor( [ num_inputs, *( num_hidden_layers*[ hidden_layer_widths ] ), num_outputs ], dtype = torch.int16, device = self.pinn_options.device )

        # Return the layers.
        return layers


    # Implement a function to convert a domain volume percentage to a radius.
    def domain_volume_percentage2radius( self, domain_volume_percentage, domain = None, domain_subset_type = 'spatiotemporal' ):

        # Preprocess the domain.
        domain = self.preprocess_domain( domain )

        # Determine which domain subset to use.
        if domain_subset_type.lower(  ) == 'spatiotemporal':                # If the domain subset type is spatiotemporal...

            # Set the domain subset to be the spatiotemporal domain.
            domain_subset = domain.spatiotemporal_domain

            # Retrieve the number of domain subset dimensions.
            num_dimensions = domain.num_spatiotemporal_dimensions

        elif domain_subset_type.lower(  ) == 'spatial':                     # If the domain subset type is spatial...

            # Set the domain subset to be the spatial domain.
            domain_subset = domain.spatial_domain

            # Retrieve the number of domain subset dimensions.
            num_dimensions = domain.num_spatial_dimensions

        elif domain_subset_type.lower(  ) == 'temporal':                    # If the domain subset type is temporal...

            # Set the domain subset to be the temporal domain.
            domain_subset = domain.temporal_domain

            # Retrieve the number of domain subset dimensions.
            num_dimensions = domain.num_temporal_dimensions

        else:                                                               # Otherwise... ( i.e., the domain subset type is not recognized... )

            # Throw an error.
            raise ValueError( f'Invalid domain subset type: {domain_subset_type}' )

        # Compute the domain volume.
        domain_volume = domain.compute_volume( domain_subset, domain.domain_type )

        # Compute the radius.
        radius = ( ( math.gamma( num_dimensions/2 + 1 )/( math.pi**( num_dimensions/2 ) ) )*domain_volume_percentage*domain_volume )**( 1/num_dimensions )

        # Return the radius.
        return radius


    # Implement a function to compute the exploration radius.
    def compute_exploration_radius( self, exploration_volume_percentage = None, domain = None, domain_subset_type = 'spatiotemporal' ):

        # Preprocess the exploration volume percentage.
        exploration_volume_percentage = self.preprocess_exploration_volume_percentage( exploration_volume_percentage )

        # Compute the exploration radius.
        exploration_radius = self.domain_volume_percentage2radius( exploration_volume_percentage, domain, domain_subset_type )

        # Return the exploration radius.
        return exploration_radius


    # Implement a function to compute the unique tolerance.
    def compute_unique_tolerance( self, unique_volume_percentage = None, domain = None, domain_subset_type = 'spatiotemporal' ):

        # Preprocess the unique volume percentage.
        unique_volume_percentage = self.preprocess_unique_volume_percentage( unique_volume_percentage )

        # Compute the unique tolerance.
        unique_tolerance = self.domain_volume_percentage2radius( unique_volume_percentage, domain, domain_subset_type )

        # Return the unique tolerance.
        return unique_tolerance


    # Implement a function to compute the classification noise magnitude.
    def compute_classification_noise_magnitude( self, classification_noise_percentage = None, domain = None, domain_subset_type = 'spatiotemporal' ):

        # Preprocess the classification noise percentage.
        classification_noise_percentage = self.preprocess_classification_noise_percentage( classification_noise_percentage )

        # Compute the classification noise magnitude.
        classification_noise_magnitude = self.domain_volume_percentage2radius( classification_noise_percentage, domain, domain_subset_type )

        # Return the classification noise magnitude.
        return classification_noise_magnitude


    #%% ------------------------------------------------------------ LOSS FUNCTIONS ------------------------------------------------------------

    # Implement a function to compute the classification loss.
<<<<<<< HEAD
    def compute_classification_loss( self, pde, network = None, classification_data = None, num_spatial_dimensions = None, num_timesteps = None, domain = None, plot_time = None, level = None, level_set_guesses = None, num_guesses = None, newton_tolerance = None, newton_max_iterations = None, exploration_radius = None, num_exploration_points = None, unique_tolerance = None, classification_noise_magnitude = None, num_noisy_samples_per_level_set_point = None, domain_subset_type = 'spatial', tspan = None, dt = None ):

        # Setup for the classification loss.
        network, classification_data, level_set_guesses, classification_noise_magnitude, num_noisy_samples_per_level_set_point, unique_tolerance, num_exploration_points, exploration_radius, newton_max_iterations, newton_tolerance, num_guesses, level, plot_time, tspan, dt = self.setup_classification_loss( network, classification_data, level_set_guesses, classification_noise_magnitude, num_noisy_samples_per_level_set_point, unique_tolerance, num_exploration_points, exploration_radius, newton_max_iterations, newton_tolerance, num_guesses, level, plot_time, tspan, dt, num_spatial_dimensions, num_timesteps, domain, domain_subset_type )

        # Compute the classification loss.
        classification_loss, num_classification_points = network.compute_classification_loss( pde, classification_data, num_spatial_dimensions, domain, plot_time, level, level_set_guesses, num_guesses, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, classification_noise_magnitude, num_noisy_samples_per_level_set_point, domain_subset_type, tspan, dt )

        # Return the classification loss.
        return classification_loss, num_classification_points
=======
    def compute_classification_loss( self, pde, network = None, classification_data = None, num_spatial_dimensions = None, num_timesteps = None, domain = None, plot_time = None, level = None, level_set_guesses = None, num_guesses = None, newton_tolerance = None, newton_max_iterations = None, exploration_radius = None, num_exploration_points = None, unique_tolerance = None, classification_noise_magnitude = None, domain_subset_type = 'spatial', tspan = None, dt = None ):

        # Setup for the classification loss.
        network, classification_data, level_set_guesses, classification_noise_magnitude, unique_tolerance, num_exploration_points, exploration_radius, newton_max_iterations, newton_tolerance, num_guesses, level, plot_time, tspan, dt = self.setup_classification_loss( network, classification_data, level_set_guesses, classification_noise_magnitude, unique_tolerance, num_exploration_points, exploration_radius, newton_max_iterations, newton_tolerance, num_guesses, level, plot_time, tspan, dt, num_spatial_dimensions, num_timesteps, domain, domain_subset_type )

        # Compute the classification loss.
        classification_loss = network.compute_classification_loss( pde, classification_data, num_spatial_dimensions, domain, plot_time, level, level_set_guesses, num_guesses, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, classification_noise_magnitude, domain_subset_type, tspan, dt )

        # Return the classification loss.
        return classification_loss
>>>>>>> 55162c78e9fb0c13d60ea20df5463b1e4d4f30fb


    #%% ------------------------------------------------------------ TRAINING FUNCTIONS ------------------------------------------------------------

    # Implement a function to train the stored neural network.
    def train( self, network = None, train_flag = None ):

        # Setup for training.
        network, train_flag = self.setup_training( network, train_flag )

        # Determine whether to train the network.
        if train_flag:                     # If we want to train the network...

            # Train the network.
            network.training_epochs, network.training_losses, network.testing_epochs, network.testing_losses = network.train( self.pde, network.training_data, network.testing_data, network.num_batches, network.num_epochs, network.derivative_required_for_residual, network.residual_code, network.epoch_print_frequency, network.verbose_flag )
<<<<<<< HEAD
=======
            # network.training_epochs, network.training_losses, network.testing_epochs, network.testing_losses, network.classification_loss = network.train( self.pde, network.training_data, network.testing_data, network.num_batches, network.num_epochs, network.derivative_required_for_residual, network.residual_code, network.epoch_print_frequency, network.verbose_flag )
            # network.training_epochs, network.training_losses, network.testing_epochs, network.testing_losses = network.train( self.pde, network.training_data, network.testing_data, network.num_batches, network.num_epochs, network.derivative_required_for_residual, network.residual_code, network.epoch_print_frequency, network.verbose_flag )
>>>>>>> 55162c78e9fb0c13d60ea20df5463b1e4d4f30fb

        # Return the network.
        return network


    #%% ------------------------------------------------------------ PRINTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to print a summary of the pinn information.
    def print( self, num_dashes = 20, decoration_flag = True ):

        # Print a header.
        self.printing_utilities.print_header( 'PINN SUMMARY', num_dashes, decoration_flag )

        # Print the pinn options, hyperparameters, and problem specifications.
        print( 'PINN Option / Hyperparameter / Problem Specification Information' )
        self.pinn_options.print( header_flag = False )
        self.hyperparameters.print( header_flag = False )
        self.problem_specifications.print( header_flag = False )
        print( '\n' )

        # Print domain information.
        print( 'Domain Information' )
        self.domain.print( header_flag = False )
        print( '\n' )

        # Print the flow functions.
        print( 'Flow Function Information' )
        print( f'Flow Functions: {self.flow_functions}' )
        print( '\n' )

        # Print the pde information.
        print( 'PDE Information' )
        self.pde.print( header_flag = False )
        print( '\n' )

        # Create the network object.
        print( 'Network Information' )
        self.network.print_summary( header_flag = False )

        # Print a footer.
        self.printing_utilities.print_footer( num_dashes, decoration_flag )


    #%% ------------------------------------------------------------ PLOTTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to plot the training data.
    def plot_training_data( self, network = None, projection_dimensions = None, projection_values = None, level = 0, fig = None, plot_type1 = 'all', plot_type2 = 'all', save_directory = None, as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Setup for plotting the training data.
        network, save_directory = self.setup_network_plotting( network, save_directory )

        # Plot the training data.
        figs, axes = network.plot_training_data( network.training_data, projection_dimensions, projection_values, level, fig, plot_type1, plot_type2, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figures and axes.
        return figs, axes


    # Implement a function to plot the testing data.
    def plot_testing_data( self, network = None, projection_dimensions = None, projection_values = None, level = 0, fig = None, plot_type1 = 'all', plot_type2 = 'all', save_directory = None, as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Setup for plotting the testing data.
        network, save_directory = self.setup_network_plotting( network, save_directory )

        # Plot the testing data.
        figs, axes = network.plot_testing_data( network.testing_data, projection_dimensions, projection_values, level, fig, plot_type1, plot_type2, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figures and axes.
        return figs, axes


    # Implement a function to plot the plotting data.
    def plot_plotting_data( self, network = None, projection_dimensions = None, projection_values = None, level = 0, dimension_labels = None, fig = None, save_directory = None, as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Setup for plotting the plotting data.
        network, save_directory = self.setup_network_plotting( network, save_directory )

        # Plot the plotting data.
        fig, ax = network.plot_plotting_data( network.plotting_data, projection_dimensions, projection_values, level, dimension_labels, fig, save_directory, as_surface, as_stream, as_contour, show_plot )
        
        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot the domain.
    def plot_domain( self, pde = None, projection_dimensions = None, projection_values = None, level = 0, fig = None, domain_type = 'spatiotemporal', save_directory = None, as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Setup for plotting the domain.
        pde, save_directory = self.setup_pde_plotting( pde, save_directory )

        # Plot the domain.
        figs, axes = pde.plot_domain( pde.domain, projection_dimensions, projection_values, level, fig, domain_type, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figures and axes.
        return figs, axes


    # Implement a function to plot the initial-boundary conditions.
    def plot_initial_boundary_condition( self, num_points_per_dimension, num_timesteps, pde = None, projection_dimensions = None, projection_values = None, level = 0, fig = None, save_directory = None, as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Preprocess the number of timesteps.
        num_timesteps = self.preprocess_num_timesteps( num_timesteps )

        # Setup for plotting the initial-boundary conditions.
        pde, save_directory = self.setup_pde_plotting( pde, save_directory )

        # Plot the initial boundary conditions.
        figs, axes = pde.plot_initial_boundary_conditions( num_points_per_dimension, num_timesteps, pde.domain, pde.initial_boundary_conditions, projection_dimensions, projection_values, level, fig, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figures and axes.
        return figs, axes


    # Implement a function to plot the network prediction.
    def plot_network_predictions( self, network_input_grid = None, network = None, projection_dimensions = None, projection_values = None, level = 0, dimension_labels = None, fig = None, save_directory = None, as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Setup for plotting the network predictions.
        network, save_directory = self.setup_network_plotting( network, save_directory )

        # Plot the plotting data.
        fig, ax = network.plot_network_predictions( network_input_grid, projection_dimensions, projection_values, level, dimension_labels, fig, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot the network prediction at a specific time.
    def plot_network_prediction_at_time( self, network_input_grid = None, network = None, plot_times = None, projection_dimensions = None, projection_values = None, level = 0, dimension_labels = None, fig = None, save_directory = None, as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Setup for plotting the network predictions at a specific time.
        network, save_directory = self.setup_network_plotting( network, save_directory )

        # Plot the network prediction at the specified time.
        fig, ax = network.plot_network_prediction_at_time( network_input_grid, plot_times, projection_dimensions, projection_values, level, dimension_labels, fig, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figure.
        return fig, ax


    # Implement a function to plot the network prediction at the initial time.
    def plot_network_initial_prediction( self, network_input_grid = None, domain = None, network = None, projection_dimensions = None, projection_values = None, level = 0, dimension_labels = None, fig = None, save_directory = None, as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Setup for plotting the network initial prediction.
        domain, network, save_directory = self.setup_network_initial_final_plotting( domain, network, save_directory )

        # Set the plot time to be the initial times.
        plot_times = domain.temporal_domain[ 0, : ]

        # Plot the network predictions as the specified time.
        fig, ax = self.plot_network_prediction_at_time( network_input_grid, network, plot_times, projection_dimensions, projection_values, level, dimension_labels, fig, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figure and axis objects.
        return fig, ax


    # Implement a function to plot the network prediction at the final time.
    def plot_network_final_prediction( self, network_input_grid = None, domain = None, network = None, projection_dimensions = None, projection_values = None, level = 0, dimension_labels = None, fig = None, save_directory = None, as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Setup for plotting the network final prediction.
        domain, network, save_directory = self.setup_network_initial_final_plotting( domain, network, save_directory )

        # Set the plot time to be the final times.
        plot_times = domain.temporal_domain[ 1, : ]

        # Plot the network predictions as the specified time.
        fig, ax = self.plot_network_prediction_at_time( network_input_grid, network, plot_times, projection_dimensions, projection_values, level, dimension_labels, fig, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figure and axis objects.
        return fig, ax


    # Implement a function to plot the network level set.
    def plot_network_level_set( self, domain = None, network = None, level = None, level_set_guess = None, newton_tolerance = None, newton_max_iterations = None, exploration_radius = None, num_exploration_points = None, unique_tolerance = None, projection_dimensions = None, projection_values = None, fig = None, dimension_labels = None, save_directory = r'.', as_surface = False, as_stream = False, as_contour = False, show_plot = False ):

        # Setup for plotting the network predictions.
        domain, network, save_directory = self.setup_network_initial_final_plotting( domain, network, save_directory )

        # Set the domain subset type.
        domain_subset_type = 'spatiotemporal'

        # Plot the plotting data.
        fig, ax = network.plot_network_level_set( domain.num_spatial_dimensions, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, domain_subset_type, projection_dimensions, projection_values, fig, dimension_labels, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot the network level set at a specific time.
    def plot_network_level_set_at_time( self, domain = None, network = None, num_timesteps = None, plot_times = None, level = None, level_set_guess = None, num_guesses = None, newton_tolerance = None, newton_max_iterations = None, exploration_radius = None, num_exploration_points = None, unique_tolerance = None, projection_dimensions = None, projection_values = None, fig = None, dimension_labels = None, save_directory = r'.', as_surface = False, as_stream = False, as_contour = False, show_plot = False ):

        # Preprocess the number of timesteps.
        if num_timesteps is None:           # If the number of timesteps was not provided...

            # Set the number of timesteps to be the stored value.
            num_timesteps = self.hyperparameters.num_timesteps

        # Setup for plotting the network predictions.
        domain, network, save_directory = self.setup_network_initial_final_plotting( domain, network, save_directory )

        # Set the domain subset type.
        domain_subset_type = 'spatial'

        # Implement a function to preprocess the level set guess.
        level_set_guess = self.preprocess_level_set_guesses( level_set_guess, num_guesses, domain, domain_subset_type )
        
        # Plot the network level set.
        fig, ax = network.plot_network_level_set_at_time( domain.num_spatial_dimensions, num_timesteps, domain, plot_times, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, domain_subset_type, projection_dimensions, projection_values, fig, dimension_labels, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot the network level set at the initial time.
    def plot_network_initial_level_set( self, domain = None, network = None, num_timesteps = None, level = None, level_set_guess = None, num_guesses = None, newton_tolerance = None, newton_max_iterations = None, exploration_radius = None, num_exploration_points = None, unique_tolerance = None, projection_dimensions = None, projection_values = None, fig = None, dimension_labels = None, save_directory = r'.', as_surface = False, as_stream = False, as_contour = False, show_plot = False ):

        # Preprocess the number of timesteps.
        if num_timesteps is None:           # If the number of timesteps was not provided...

            # Set the number of timesteps to be the stored value.
            num_timesteps = self.hyperparameters.num_timesteps

        # Setup for plotting the network initial prediction.
        domain, network, save_directory = self.setup_network_initial_final_plotting( domain, network, save_directory )

        # Set the plot time to be the initial times.
        plot_times = domain.temporal_domain[ 0, : ]

        # Plot the network predictions as the specified time.
        fig, ax = self.plot_network_level_set_at_time( domain, network, num_timesteps, plot_times, level, level_set_guess, num_guesses, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, projection_dimensions, projection_values, fig, dimension_labels, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figure and axis objects.
        return fig, ax


    # Implement a function to plot the network level set at the final time.
    def plot_network_final_level_set( self, domain = None, network = None, num_timesteps = None, level = None, level_set_guess = None, num_guesses = None, newton_tolerance = None, newton_max_iterations = None, exploration_radius = None, num_exploration_points = None, unique_tolerance = None, projection_dimensions = None, projection_values = None, fig = None, dimension_labels = None, save_directory = r'.', as_surface = False, as_stream = False, as_contour = False, show_plot = False ):

        # Preprocess the number of timesteps.
        if num_timesteps is None:           # If the number of timesteps was not provided...

            # Set the number of timesteps to be the stored value.
            num_timesteps = self.hyperparameters.num_timesteps

        # Setup for plotting the network final prediction.
        domain, network, save_directory = self.setup_network_initial_final_plotting( domain, network, save_directory )

        # Set the plot time to be the final times.
        plot_times = domain.temporal_domain[ 1, : ]

        # Plot the network predictions as the specified time.
        fig, ax = self.plot_network_level_set_at_time( domain, network, num_timesteps, plot_times, level, level_set_guess, num_guesses, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, projection_dimensions, projection_values, fig, dimension_labels, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figure and axis objects.
        return fig, ax


    # Implement a function to plot the training results.
    def plot_training_results( self, network = None, save_directory = None, show_plot = False ):

        # Setup for plotting the training results.
        network, save_directory = self.setup_network_plotting( network, save_directory )

        # Plot the plotting data.
        fig, ax = network.plot_training_results( network.training_losses, network.testing_epochs, network.testing_losses, save_directory, show_plot )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot the flow field.
    def plot_flow_field( self, plotting_data = None, flow_functions = None, projection_dimensions = None, projection_values = None, level = 0, fig = None, input_labels = None, title_string = 'Flow Field', save_directory = None, as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Setup for flow field plotting.
        plotting_data, flow_functions, save_directory = self.setup_flow_field_plotting( plotting_data, flow_functions, save_directory )

        # Plot the flow field.
        figs, axes = self.plotting_utilities.plot( plotting_data, flow_functions, projection_dimensions, projection_values, level, fig, input_labels, title_string, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figures and axes.
        return figs, axes


    # Implement a function to plot the ROA boundary prediction.
    def plot_roa_boundary( self, plotting_data = None, domain = None, network = None, flow_functions = None, projection_dimensions = None, projection_values = None, level = 0, fig = None, input_labels = None, title_string = 'ROA Boundary Prediction', save_directory = None, show_plot = False ):

        # Setup for roa boundary plotting.
        plotting_data, domain, flow_functions, network, save_directory = self.setup_roa_boundary_plotting( plotting_data, domain, flow_functions, network, save_directory )

        # Plot the flow field.
        fig, ax = self.plot_flow_field( plotting_data, flow_functions, projection_dimensions, projection_values, level, fig, input_labels, title_string, save_directory, as_surface = False, as_stream = True, as_contour = False, show_plot = show_plot )

        # Generate the final network prediction.
        fig, ax = self.plot_network_final_prediction( plotting_data, domain, network, None, None, level, None, fig[ 0 ], save_directory, as_surface = False, as_stream = False, as_contour = True, show_plot = show_plot )

        # Return the figure and axis.
        return fig, ax
        

<<<<<<< HEAD
    # Implement a function to plot the classification data.
    def plot_network_classifications( self, network = None, fig = None, dimension_labels = None, save_directory = None, show_plot = False ):

        # Setup for classification plotting.
        network, save_directory = self.setup_network_plotting( network, save_directory )

        # Plot the classification data.
        fig, ax = network.plot_classifications( network.classification_data, network.classification_data_forecast, network.actual_classifications, network.network_classifications, fig, dimension_labels, save_directory, as_surface = False, as_stream = False, as_contour = False, show_plot = show_plot )

        # Return the figure and axis.
        return fig, ax


=======
>>>>>>> 55162c78e9fb0c13d60ea20df5463b1e4d4f30fb
    #%% ------------------------------------------------------------ SAVE & LOAD FUNCTIONS ------------------------------------------------------------

    # Implement a function to save the pinn.
    def save( self, save_directory = None, file_name = r'pinn.pkl' ):

        # Preprocess the save directory.
        save_directory = self.preprocess_save_directory( save_directory )

        # Determine whether to save the pinn object.
        if self.pinn_options.save_flag:                                     # If we want to save the pinn object...

            # Save the pinn object.
            self.save_load_utilities.save( self, save_directory, file_name )


    # Implement a function to load pinn.
    def load( self, load_directory = None, file_name = r'pinn.pkl' ):

        # Preprocess the load directory.
        load_directory = self.preprocess_load_directory( load_directory )

        # Determine whether to load the pinn object.
        if self.pinn_options.load_flag:                                     # If we want to load the pinn object...

            # Load the pinn options.
            self = self.save_load_utilities.load( load_directory, file_name )

        # Return the pinn options.
        return self
<<<<<<< HEAD
=======

>>>>>>> 55162c78e9fb0c13d60ea20df5463b1e4d4f30fb
