####################################################################################### 
# THIS SOURCE CODE IS PROPERTY OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. 
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND 
# CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION 
# ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN 
# LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE,
# CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL 
# PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641. 
####################################################################################### 


#%% ------------------------------------------------------------ HYPERPARAMETERS CLASS ------------------------------------------------------------

# This file implements a class for storing and managing hyperparameters information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.


# Import custom libraries.
from save_load_utilities_class import save_load_utilities_class as save_load_utilities_class
from printing_utilities_class import printing_utilities_class as printing_utilities_class


#%% ------------------------------------------------------------ HYPERPARAMETERS CLASS ------------------------------------------------------------

# Implement the hyperparameters class.
class hyperparameters_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Define the class constructor.
    def __init__( self, neuron_parameters, synapse_parameters, num_timesteps, activation_function, num_hidden_layers, hidden_layer_widths, num_training_data, num_testing_data, p_initial, p_boundary, p_residual, num_epochs, residual_batch_size, learning_rate, integration_order, element_volume_percent, element_type, element_computation_option, c_IC, c_BC, c_residual, c_variational, c_monotonicity, save_path = None, load_path = None ):

        # Create an instance of the save-load utilities class.
        self.save_load_utilities = save_load_utilities_class(  )

        # Create an instance of the printing utilities class.
        self.printing_utilities = printing_utilities_class(  )

        # Store the neuron & synapse parameters.
        self.neuron_parameters = neuron_parameters
        self.synapse_parameters = synapse_parameters

        # Store the number of timesteps for which each input will be presented to the network.
        self.num_timesteps = num_timesteps

        # Store the network parameters.
        self.activation_function = activation_function
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_widths = hidden_layer_widths

        # Store the training data information.
        self.num_training_data = num_training_data
        self.num_testing_data = num_testing_data

        # Define the percent of training and testing data that should be sampled from the initial condition, the boundary condition, and the interior of the domain.
        self.p_initial = p_initial
        self.p_boundary = p_boundary
        self.p_residual = p_residual

        # Store the mini-batching and epoch information.
        self.num_epochs = num_epochs
        self.residual_batch_size = residual_batch_size

        # Store the optimizer parameters.
        self.learning_rate = learning_rate

        # Store the variational loss parameters.
        self.integration_order = integration_order
        self.element_volume_percent = element_volume_percent
        self.element_type = element_type
        self.element_computation_option = element_computation_option

        # Store the loss coefficients.
        self.c_IC = c_IC
        self.c_BC = c_BC
        self.c_residual = c_residual
        self.c_variational = c_variational
        self.c_monotonicity = c_monotonicity

        # Store the save and load paths.
        self.save_path = save_path
        self.load_path = load_path


    #%% ------------------------------------------------------------ PINN FUNCTIONS ------------------------------------------------------------

    # Implement a function to print the hyperparameters.
    def print( self, num_dashes = 20, decoration_flag = True ):

        # Print a header.
        self.printing_utilities.print_header( 'HYPERPARAMETERS SUMMARY', num_dashes, decoration_flag )

        # Print the network parameters.
        print( 'Network Parameters' )
        print( f'Activation Function: {self.activation_function}' )
        print( f'# of Hidden Layers: {self.num_hidden_layers}' )
        print( f'Hidden Layer Widths: {self.hidden_layer_widths}' )
        print( '\n' )

        # Print the training / testing data information.
        print( 'Training / Testing Data Information' )
        print( f'# of Training Data: {self.num_training_data}' )
        print( f'# of Testing Data: {self.num_testing_data}' )
        print( f'Percent Initial Data: {self.p_initial}' )
        print( f'Percent Boundary Data: {self.p_boundary}' )
        print( f'Percent Residual Data: {self.p_residual}' )
        print( '\n' )

        # Print the training information.
        print( 'Training Information' )
        print( f'# of Epochs: {self.num_epochs}' )
        print( f'Residual Batch Size: {self.residual_batch_size}' )
        print( f'Learning Rate: {self.learning_rate}' )
        print( '\n' )

        # Print the variational loss parameters.
        print( 'Variational Loss Information' )
        print( f'Integration Order: {self.integration_order}' )
        print( f'Element Volume Percentage: {self.element_volume_percent}' )
        print( f'Element Type: {self.element_type}' )
        print( f'Element Computation Option: {self.element_computation_option}' )
        print( '\n' )

        # Print the loss coefficients.
        print( 'Loss Coefficients' )
        print( f'Initial Condition Loss Coefficient: {self.c_IC}' )
        print( f'Boundary Condition Loss Coefficient: {self.c_BC}' )
        print( f'Residual Loss Coefficient: {self.c_residual}' )
        print( f'Variational Loss Coefficient: {self.c_variational}' )
        print( '\n' )

        # Print save and load information.
        print( 'Save / Load Information' )
        print( f'Save Path: {self.save_path}' )
        print( f'Load Path: {self.load_path}' )

        # Print a footer.
        self.printing_utilities.print_footer( num_dashes, decoration_flag )


    #%% ------------------------------------------------------------ SAVE & LOAD FUNCTIONS ------------------------------------------------------------

    # Implement a function to save the hyper-parameters.
    def save( self, save_path = None, file_name = r'hyperparameters.pkl' ):

        # Determine whether to use the stored save path.
        if save_path is None:                               # If the save path was not provided...

            # Use the stored save path.
            save_path = self.save_path

        # Save the hyper-parameters.
        self.save_load_utilities.save( self, save_path, file_name )


    # Implement a function to load hyper-parameters.
    def load( self, load_path = None, file_name = r'hyperparameters.pkl' ):

        # Determine whether to use the stored load path.
        if load_path is None:                               # If the load path was not provided...

            # Use the stored load path.
            load_path = self.load_path

        # Load the hyperparameters.
        self = self.save_load_utilities.load( load_path, file_name )

        # Return the hyperparameters.
        return self

