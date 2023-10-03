#%% ------------------------------------------------------------ PINN DATA MANAGER CLASS ------------------------------------------------------------

# This file implements a class for storing and managing pinn data manager information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch

# Import custom libraries.
from ibc_data_class import ibc_data_class as ibc_data_class
from residual_data_class import residual_data_class as residual_data_class
from variational_data_class import variational_data_class as variational_data_class
from finite_element_class import finite_element_class as finite_element_class
from printing_utilities_class import printing_utilities_class as printing_utilities_class


#%% ------------------------------------------------------------ PINN DATA MANAGER CLASS ------------------------------------------------------------

# Implement the pinn data manager class.
class pinn_data_manager_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self, initial_condition_data, boundary_condition_data, residual_data, variational_data, device = 'cpu' ):

        # Create an instance of the printing utilities class.
        self.printing_utilities = printing_utilities_class(  )

        # Store the device.
        self.device = device

        # Store the initial condition data.
        self.initial_condition_data = self.validate_initial_condition_data( initial_condition_data )
        self.num_initial_conditions = torch.tensor( len( self.initial_condition_data ), dtype = torch.uint8, device = self.device )

        # Store the boundary condition data.
        self.boundary_condition_data = self.validate_boundary_condition_data( boundary_condition_data )
        self.num_boundary_conditions = torch.tensor( len( self.boundary_condition_data ), dtype = torch.uint8, device = self.device )

        # Store the residual data.
        self.residual_data = self.validate_residual_data( residual_data )
        self.residual_batch_size = self.residual_data.batch_size

        # Store the variational data.
        self.variational_data = self.validate_variational_data( variational_data )
        self.variational_batch_size = self.residual_data.batch_size

        # Compute the amount of each type of data.
        self.num_initial_condition_points = torch.sum( torch.cat( tuple( self.initial_condition_data[ k ].num_data_points.unsqueeze( 0 ) for k in range( self.num_initial_conditions ) ) ), dtype = torch.int32 )
        self.num_boundary_condition_points = torch.sum( torch.cat( tuple( self.boundary_condition_data[ k ].num_data_points.unsqueeze( 0 ) for k in range( self.num_boundary_conditions ) ) ), dtype = torch.int32 )
        self.num_residual_points = self.residual_data.num_data_points

        # Compute the total amount of data.
        self.num_data_points = self.num_initial_condition_points + self.num_boundary_condition_points + self.num_residual_points

        # Compute the number of initial and boundary condition points per condition.
        self.num_points_per_initial_condition = self.initial_condition_data[ 0 ].num_data_points
        self.num_points_per_boundary_condition = self.boundary_condition_data[ 0 ].num_data_points

        # Compute the percentage of the total amount of data that each data set comprises.
        self.initial_data_percent = self.num_initial_condition_points/self.num_data_points
        self.boundary_data_percent = self.num_boundary_condition_points/self.num_data_points
        self.residual_data_percent = self.num_residual_points/self.num_data_points

        # Compute the total number of batches, the initial batch size, and the boundary batch size.
        self.num_batches, self.initial_condition_batch_size, self.boundary_condition_batch_size = self.initialize_batch_info( self.num_points_per_initial_condition, self.num_points_per_boundary_condition, self.num_residual_points, self.residual_batch_size )

        # Set the initial and boundary condition batch size.
        self.initial_condition_data = self.validate_initial_condition_data_batch_size( self.initial_condition_data, self.initial_condition_batch_size )
        self.boundary_condition_data = self.validate_boundary_condition_data_batch_size( self.boundary_condition_data, self.boundary_condition_batch_size )


    #%% ------------------------------------------------------------ PREPROCESS FUNCTIONS ------------------------------------------------------------

    # Implement a function to preprocess the variational data.
    def preprocess_variational_data( self, variational_data = None ):

        # Determine whether to use the stored variational data.
        if variational_data is None:                # If the variational data was not provided...

            # Set the variational data to be the stored value.
            variational_data = self.variational_data

        # Return the variational data.
        return variational_data


    # Implement a function to preprocess the batch number.
    def preprocess_batch_number( self, batch_number = None ):

        # Determine whether to use the default batch number.
        if batch_number is None:                # If the batch number was not provided...

            # Set the batch number to be zero.
            batch_number = torch.tensor( 0, dtype = torch.int64, device = self.device )

        # Return the batch number.
        return batch_number


    # Implement a function to preprocess the initial condition data.
    def preprocess_initial_condition_data( self, initial_condition_data = None ):

        # Determine whether to use the stored initial condition data.
        if initial_condition_data is None:                              # If initial condition data was not provided...

            # Use the stored initial condition data.
            initial_condition_data = self.initial_condition_data

        # Return the initial condition data.
        return initial_condition_data


    # Implement a function to preprocess the initial condition batch size.
    def preprocess_initial_condition_batch_size( self, initial_condition_batch_size = None ):

        # Determine whether to use the stored initial condition batch size.
        if initial_condition_batch_size is None:                        # If the initial condition batch size is None...

            # Use the stored initial condition batch size.
            initial_condition_batch_size = self.initial_condition_batch_size

        # Return the initial condition batch size.
        return initial_condition_batch_size


    # Implement a function to preprocess the boundary condition data.
    def preprocess_boundary_condition_data( self, boundary_condition_data = None ):

        # Determine whether to use the stored boundary condition data.
        if boundary_condition_data is None:                              # If boundary condition data was not provided...

            # Use the stored boundary condition data.
            boundary_condition_data = self.boundary_condition_data

        # Return the boundary condition data.
        return boundary_condition_data


    # Implement a function to preprocess the boundary condition batch size.
    def preprocess_boundary_condition_batch_size( self, boundary_condition_batch_size = None ):

        # Determine whether to use the stored boundary condition batch size.
        if boundary_condition_batch_size is None:                        # If the boundary condition batch size is None...

            # Use the stored boundary condition batch size.
            boundary_condition_batch_size = self.boundary_condition_batch_size

        # Return the boundary condition batch size.
        return boundary_condition_batch_size


    # Implement a function to preprocess the residual data.
    def preprocess_residual_data( self, residual_data = None ):

        # Determine whether to use the stored residual data.
        if residual_data is None:                              # If residual data was not provided...

            # Use the stored residual data.
            residual_data = self.residual_data

        # Return the residual data.
        return residual_data


    # Implement a function to preprocess the residual batch size.
    def preprocess_residual_batch_size( self, residual_batch_size = None ):

        # Determine whether to use the stored residual batch size.
        if residual_batch_size is None:                              # If residual batch size was not provided...

            # Use the stored residual batch size.
            residual_batch_size = self.residual_batch_size

        # Return the residual batch size.
        return residual_batch_size


    # Implement a function to preprocess the batch size.
    def preprocess_variational_batch_size( self, batch_size = None, variational_data = None ):

        # Setup the variational data.
        variational_data = self.preprocess_variational_data( variational_data )

        # Determine whether to use the stored batch size.
        if batch_size is None:                  # If the batch size was not provided...

            # Set the batch size to be the stored value.
            batch_size = variational_data.batch_size

        # Return the batch size.
        return batch_size


    # Implement a function to preprocess the number of points per initial condition.
    def preprocess_num_points_per_initial_condition( self, num_points_per_initial_condition = None ):

        # Determine whether to use the stored number of points per initial condition.
        if num_points_per_initial_condition is None:         # If the number of points per initial condition was not provided...

            # Set the number of points per initial condition to be the stored value.
            num_points_per_initial_condition = self.num_points_per_initial_condition

        # Return the number of points per initial condition.
        return num_points_per_initial_condition


    # Implement a function to preprocess the number of points per boundary condition.
    def preprocess_num_points_per_boundary_condition( self, num_points_per_boundary_condition = None ):

        # Determine whether to use the stored number of points per boundary condition.
        if num_points_per_boundary_condition is None:         # If the number of points per boundary condition was not provided...

            # Set the number of points per boundary condition to be the stored value.
            num_points_per_boundary_condition = self.num_points_per_boundary_condition

        # Return the number of points per boundary condition.
        return num_points_per_boundary_condition


    # Implement a function to preprocess the number of residual points.
    def preprocess_num_residual_points( self, num_residual_points = None ):

        # Determine whether to use the stored number of residual points.
        if num_residual_points is None:             # If the number of residual points was not provided...

            # Set the number of residual points to be the stored value.
            num_residual_points = self.num_residual_points

        # Return the number of residual points.
        return num_residual_points


    #%% ------------------------------------------------------------ SETUP FUNCTIONS ------------------------------------------------------------

    # Implement a function to setup finite elements.
    def setup_elements( self, variational_data = None, batch_option = 'keep', batch_number = None, batch_size = None ):

        # Setup the variational data.
        variational_data = self.preprocess_variational_data( variational_data )

        # Setup the batch number.
        batch_number = self.preprocess_batch_number( batch_number )

        # Setup the batch size.
        batch_size = self.preprocess_variational_batch_size( batch_size, variational_data )

        # Return the element setup information.
        return variational_data, batch_number, batch_size


    # Implement a function to setup for initial condition data and initial condition batch size validation.
    def setup_initial_condition_data_batch_size_validation( self, initial_condition_data = None, initial_condition_batch_size = None ):

        # Preprocess the initial condition data.
        initial_condition_data = self.preprocess_initial_condition_data( initial_condition_data )

        # Preprocess the initial condition batch size.
        initial_condition_batch_size = self.preprocess_initial_condition_batch_size( initial_condition_batch_size )

        # Return the initial condition data and the initial condition batch size.
        return initial_condition_data, initial_condition_batch_size


    # Implement a function to setup for boundary condition data and boundary condition batch size validation.
    def setup_boundary_condition_data_batch_size_validation( self, boundary_condition_data = None, boundary_condition_batch_size = None ):

        # Preprocess the boundary condition data.
        boundary_condition_data = self.preprocess_boundary_condition_data( boundary_condition_data )

        # Preprocess the boundary condition batch size.
        boundary_condition_batch_size = self.preprocess_boundary_condition_batch_size( boundary_condition_batch_size )

        # Return the boundary condition data and the boundary condition batch size.
        return boundary_condition_data, boundary_condition_batch_size


    #%% ------------------------------------------------------------ SET FUNCTIONS ------------------------------------------------------------

    # Implement a function to set initial condition data.
    def set_initial_condition_data( self, initial_condition_data, set_flag = True ):

        # Determine whether to set the initial condition data.
        if set_flag:                # If we want to set the initial condition data...

            # Set the initial condition data.
            self.initial_condition_data = initial_condition_data


    # Implement a function to set boundary condition data.
    def set_boundary_condition_data( self, boundary_condition_data, set_flag = True ):

        # Determine whether to set the boundary condition data.
        if set_flag:                # If we want to set the boundary condition data...

            # Set the boundary condition data.
            self.boundary_condition_data = boundary_condition_data


    # Implement a function to set residual data.
    def set_residual_data( self, residual_data, set_flag = True ):

        # Determine whether to set the residual data.
        if set_flag:                # If we want to set the residual data...

            # Set the residual data.
            self.residual_data = residual_data


    # Implement a function to set variational data.
    def set_variational_data( self, variational_data, set_flag = True ):

        # Determine whether to set the variational data.
        if set_flag:                # If we want to set the variational data...

            # Set the variational data.
            self.variational_data = variational_data


    # Implement a function to set the number of batches.
    def set_num_batches( self, num_batches, set_flag = True ):

        # Determine whether to set the batch data.
        if set_flag:            # If we want to set the number of batches...

            # Set the number of batches.
            self.num_batches = num_batches


    # Implement a function to set the initial condition batch size.
    def set_initial_condition_batch_size( self, initial_condition_batch_size, set_flag = True ):

        # Determine whether to set the initial condition batch size.
        if set_flag:            # If we want to set the initial condition batch size...

            # Set the initial condition batch size.
            self.initial_condition_batch_size = initial_condition_batch_size


    # Implement a function to set the boundary condition batch size.
    def set_boundary_condition_batch_size( self, boundary_condition_batch_size, set_flag = True ):

        # Determine whether to set the boundary condition batch size.
        if set_flag:            # If we want to set the boundary condition batch size...

            # Set the boundary condition batch size.
            self.boundary_condition_batch_size = boundary_condition_batch_size


    #%% ------------------------------------------------------------ IS VALID FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate initial-boundary condition data.
    def is_ibc_data_valid( self, ibc_data ):

        # Determine whether the given ibc data is valid.
        if isinstance( ibc_data, ibc_data_class ):                        # If the ibc data is itself a ibc data object...

            # Set the valid flag to true.
            valid_flag = True

        elif isinstance( ibc_data, list ):                                 # If the ibc data is a list...

            # Ensure that each of the list entries are ibc data objects.
            valid_flag = all( isinstance( ibc_data[ k ], ibc_data_class ) for k in range( len( ibc_data ) ) )

        else:                                                               # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate residual data.
    def is_residual_data_valid( self, residual_data ):

        # Determine whether the given residual data is valid.
        if isinstance( residual_data, residual_data_class ):                        # If the residual data is itself a residual data object...

            # Set the valid flag to true.
            valid_flag = True

        elif isinstance( residual_data, list ):                                 # If the residual data is a list...

            # Ensure that each of the list entries are residual data objects.
            valid_flag = all( isinstance( residual_data[ k ], residual_data_class ) for k in range( len( residual_data ) ) )

        else:                                                               # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate variational data.
    def is_variational_data_valid( self, variational_data ):

        # Determine whether the given variational data is valid.
        if isinstance( variational_data, variational_data_class ):                        # If the variational data is itself a residual data object...

            # Set the valid flag to true.
            valid_flag = True

        elif isinstance( variational_data, list ):                                 # If the variational data is a list...

            # Ensure that each of the list entries are variational data objects.
            valid_flag = all( isinstance( variational_data[ k ], variational_data_class ) for k in range( len( variational_data ) ) )

        else:                                                               # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    #%% ------------------------------------------------------------ VALIDATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to set the initial condition data.
    def validate_initial_condition_data( self, initial_condition_data, set_flag = False ):

        # Determine whether to set the initial condition data.
        if self.is_ibc_data_valid( initial_condition_data ):                   # If the initial condition data is valid...

            # Determine whether to embed the initial condition data in a list.
            if isinstance( initial_condition_data, ibc_data_class ):           # If the initial condition data is itself a pinn data object...

                # Embed the initial condition data in a list.
                initial_condition_data = [ initial_condition_data ]

        else:                                                                   # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid initial condition data: {initial_condition_data}' )

        # Set the initial condition data (as required).
        self.set_initial_condition_data( initial_condition_data, set_flag )

        # Return the initial condition data.
        return initial_condition_data


    # Implement a function to set the boundary condition data.
    def validate_boundary_condition_data( self, boundary_condition_data, set_flag = False ):

        # Determine whether to set the boundary condition data.
        if self.is_ibc_data_valid( boundary_condition_data ):                   # If the boundary condition data is valid...

            # Determine whether to embed the boundary condition data in a list.
            if isinstance( boundary_condition_data, ibc_data_class ):           # If the boundary condition data is itself a pinn data object...

                # Embed the boundary condition data in a list.
                boundary_condition_data = [ boundary_condition_data ]

        else:                                                                   # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid boundary condition data: {boundary_condition_data}' )
    
        # Set the boundary condition data (as required).
        self.set_boundary_condition_data( boundary_condition_data, set_flag )

        # Return the boundary condition data.
        return boundary_condition_data


    # Implement a function to set the residual data.
    def validate_residual_data( self, residual_data, set_flag = False ):

        # Determine whether to set the residual data.
        if not self.is_residual_data_valid( residual_data ):                   # If the residual data is not valid...

            # Throw an error.
            raise ValueError( f'Invalid residual data: {residual_data}' )
    
        # Set the residual data (as required).
        self.set_residual_data( residual_data, set_flag )

        # Return the residual data.
        return residual_data


    # Implement a function to set the variational data.
    def validate_variational_data( self, variational_data, set_flag = False ):

        # Determine whether to set the variational data.
        if not self.is_variational_data_valid( variational_data ):                   # If the variational data is not valid...

            # Throw an error.
            raise ValueError( f'Invalid variational data: {variational_data}' )
    
        # Set the variational data (as required).
        self.set_variational_data( variational_data, set_flag )

        # Return the variational data.
        return variational_data


    # Implement a function to set the initial condition batch size.
    def validate_initial_condition_data_batch_size( self, initial_condition_data = None, initial_condition_batch_size = None, set_flag = False ):

        # Setup for initial condition data and initial condition batch size validation.
        initial_condition_data, initial_condition_batch_size = self.setup_initial_condition_data_batch_size_validation( initial_condition_data, initial_condition_batch_size )

        # Determine the number of initial conditions.
        if isinstance( initial_condition_data, list ):

            # Retrieve the stored number of initial conditions.
            num_initial_conditions = torch.tensor( len( initial_condition_data ), dtype = torch.uint8, device = self.device )

        elif torch.is_tensor( initial_condition_data ) and ( initial_condition_data != 0 ):

            # Set the number of initial conditions to one.
            num_initial_conditions = torch.tensor( 1, dtype = torch.uint8, device = self.device )

        else:   

            # Throw an error.
            raise ValueError( f'Invalid initial condition data: {initial_condition_data}' )

        # Set the batch size for each initial condition.
        for k in range( num_initial_conditions ):                  # Iterate through each of the initial conditions...

            # Set the batch size of this initial condition.
            initial_condition_data[ k ].batch_size = initial_condition_data[ k ].set_batch_size( initial_condition_batch_size )

        # Set the initial condition data (as required).
        self.set_initial_condition_data( initial_condition_data, set_flag )

        # Return the stored initial condition data.
        return initial_condition_data


    # Implement a function to set the boundary condition batch size.
    def validate_boundary_condition_data_batch_size( self, boundary_condition_data = None, boundary_condition_batch_size = None, set_flag = False ):

        # Setup for boundary condition data and boundary condition batch size validation.
        boundary_condition_data, boundary_condition_batch_size = self.setup_boundary_condition_data_batch_size_validation( boundary_condition_data, boundary_condition_batch_size )

        # Determine the number of boundary conditions.
        if isinstance( boundary_condition_data, list ):

            # Retrieve the stored number of boundary conditions.
            num_boundary_conditions = torch.tensor( len( boundary_condition_data ), dtype = torch.uint8, device = self.device )

        elif torch.is_tensor( boundary_condition_data ) and ( boundary_condition_data != 0 ):

            # Set the number of boundary conditions to one.
            num_boundary_conditions = torch.tensor( 1, dtype = torch.uint8, device = self.device )

        else:   

            # Throw an error.
            raise ValueError( f'Invalid boundary condition data: {boundary_condition_data}' )

        # Set the batch size for each boundary condition.
        for k in range( num_boundary_conditions ):                  # Iterate through each of the boundary conditions...

            # Set the batch size of this boundary condition.
            boundary_condition_data[ k ].batch_size = boundary_condition_data[ k ].set_batch_size( boundary_condition_batch_size )

        # Set the boundary condition data (as required).
        self.set_boundary_condition_data( boundary_condition_data, set_flag )

        # Return the stored boundary condition data.
        return boundary_condition_data


    #%% ------------------------------------------------------------ SETUP FUNCTIONS ------------------------------------------------------------

    # Implement a function to setup for data shuffling.
    def setup_data_shuffling( self, initial_condition_data, boundary_condition_data, residual_data, variational_data ):

        # Preprocess the initial condition data.
        initial_condition_data = self.preprocess_initial_condition_data( initial_condition_data )

        # Preprocess the boundary condition data.
        boundary_condition_data = self.preprocess_boundary_condition_data( boundary_condition_data )

        # Preprocess the residual data.
        residual_data = self.preprocess_residual_data( residual_data )

        # Preprocess the variational data.
        variational_data = self.preprocess_variational_data( variational_data )

        # Return the data for shuffling.
        return initial_condition_data, boundary_condition_data, residual_data, variational_data


    # Implement a function to setup for the computation of initial condition batch data.
    def setup_initial_condition_batch_computation( self, initial_condition_data = None, batch_number = None, initial_condition_batch_size = None ):

        # Preprocess the initial condition data.
        initial_condition_data = self.preprocess_initial_condition_data( initial_condition_data )

        # Compute the number of initial conditions.
        num_initial_conditions = torch.tensor( len( initial_condition_data ), dtype = torch.uint8, device = self.device )

        # Preprocess the batch number.
        batch_number = self.preprocess_batch_number( batch_number )

        # Preprocess the initial condition batch size.
        initial_condition_batch_size = self.preprocess_initial_condition_batch_size( initial_condition_batch_size )

        # Return the data required for computing the initial condition data.
        return initial_condition_data, num_initial_conditions, batch_number, initial_condition_batch_size


    # Implement a function to setup for the computation of boundary condition batch data.
    def setup_boundary_condition_batch_computation( self, boundary_condition_data = None, batch_number = None, boundary_condition_batch_size = None ):

        # Preprocess the boundary condition data.
        boundary_condition_data = self.preprocess_boundary_condition_data( boundary_condition_data )

        # Compute the number of boundary conditions.
        num_boundary_conditions = torch.tensor( len( boundary_condition_data ), dtype = torch.uint8, device = self.device )

        # Preprocess the batch number.
        batch_number = self.preprocess_batch_number( batch_number )

        # Preprocess the boundary condition batch size.
        boundary_condition_batch_size = self.preprocess_boundary_condition_batch_size( boundary_condition_batch_size )

        # Return the data required for computing the boundary condition data.
        return boundary_condition_data, num_boundary_conditions, batch_number, boundary_condition_batch_size


    # Implement a function to setup for the computational of residual batch data.
    def setup_residual_batch_computation( self, residual_data = None, batch_number = None, residual_batch_size = None ):

        # Preprocess the residual data.
        residual_data = self.preprocess_residual_data( residual_data )

        # Preprocess the batch number.
        batch_number = self.preprocess_batch_number( batch_number )

        # Preprocess the residual batch size.
        residual_batch_size = self.preprocess_residual_batch_size( residual_batch_size )

        # Return the data required for computing a batch of residual data.
        return residual_data, batch_number, residual_batch_size


    # Implement a function to setup for the computation of variational batch data.
    def setup_variational_batch_computation( self, variational_data = None, batch_number = None, variational_batch_size = None ):

        # Preprocess the variational data.
        variational_data = self.preprocess_variational_data( variational_data )

        # Preprocess the batch number.
        batch_number = self.preprocess_batch_number( batch_number )

        # Preprocess the variational batch size.
        variational_batch_size = self.preprocess_variational_batch_size( variational_batch_size )

        # Return the data necessary for computing a batch of variational data.
        return variational_data, batch_number, variational_batch_size


    # Implement a function to setup for the computation of initial, boundary, residual, and variational data batches.
    def setup_all_batch_computations( self, initial_condition_data = None, boundary_condition_data = None, residual_data = None, variational_data = None, batch_number = None, initial_condition_batch_size = None, boundary_condition_batch_size = None, residual_batch_size = None, variational_batch_size = None ):
        
        # Preprocess the initial condition data.
        initial_condition_data = self.preprocess_initial_condition_data( initial_condition_data )

        # Preprocess the boundary condition data.
        boundary_condition_data = self.preprocess_boundary_condition_data( boundary_condition_data )

        # Preprocess the residual data.
        residual_data = self.preprocess_residual_data( residual_data )

        # Preprocess the variational data.
        variational_data = self.preprocess_variational_data( variational_data )

        # Preprocess the batch number.
        batch_number = self.preprocess_batch_number( batch_number )

        # Preprocess the initial condition batch size.
        initial_condition_batch_size = self.preprocess_initial_condition_batch_size( initial_condition_batch_size )

        # Preprocess the boundary condition batch size.
        boundary_condition_batch_size = self.preprocess_boundary_condition_batch_size( boundary_condition_batch_size )

        # Preprocess the residual batch size.
        residual_batch_size = self.preprocess_residual_batch_size( residual_batch_size )

        # Preprocess the variational batch size.
        variational_batch_size = self.preprocess_variational_batch_size( variational_batch_size )

        # Return the data required for all batch computations.
        return initial_condition_data, boundary_condition_data, residual_data, variational_data, batch_number, initial_condition_batch_size, boundary_condition_batch_size, residual_batch_size, variational_batch_size


    # Implement a function to setup for batch info initialization.
    def setup_batch_info_initialization( self, num_points_per_initial_condition = None, num_points_per_boundary_condition = None, num_residual_points = None, residual_batch_size = None ):

        # Preprocess the number of points per initial condition.
        num_points_per_initial_condition = self.preprocess_num_points_per_initial_condition( num_points_per_initial_condition )

        # Preprocess the number of points per boundary condition.
        num_points_per_boundary_condition = self.preprocess_num_points_per_boundary_condition( num_points_per_boundary_condition )

        # Preprocess the number of residual points.
        num_residual_points = self.preprocess_num_residual_points( num_residual_points )

        # Preprocess the residual batch size.
        residual_batch_size = self.preprocess_residual_batch_size( residual_batch_size )

        # Return the info necessary for batch info initialization.
        return num_points_per_initial_condition, num_points_per_boundary_condition, num_residual_points, residual_batch_size


    # Implement a function to setup for plotting.
    def setup_plotting( self, initial_condition_data = None, boundary_condition_data = None, residual_data = None ):

        # Preprocess the initial condition data.
        initial_condition_data = self.preprocess_initial_condition_data( initial_condition_data )

        # Preprocess the boundary condition data.
        boundary_condition_data = self.preprocess_boundary_condition_data( boundary_condition_data )

        # Preprocess the residual data.
        residual_data = self.preprocess_residual_data( residual_data )

        # Return the data required for plotting.
        return initial_condition_data, boundary_condition_data, residual_data


    #%% ------------------------------------------------------------ ELEMENT FUNCTIONS ------------------------------------------------------------

    # Implement a function to generate elements.
    def generate_elements( self, xs_element_centers, variational_data = None, replace_flag = False, batch_option = 'keep', batch_number = None, batch_size = None, set_flag = False ):

        # Setup the element information.
        variational_data, batch_number, batch_size = self.setup_elements( variational_data, batch_option, batch_number, batch_size )

        # Generate the elements.
        variational_data.finite_elements, variational_data.xs_integration_points_batch, variational_data.G_basis_values_batch, variational_data.W_integration_weights_batch, variational_data.sigma_jacobian_batch = variational_data.generate_elements( xs_element_centers, variational_data.finite_elements, replace_flag, batch_option, batch_number, batch_size )

        # Set the variational data (as required).
        self.set_variational_data( variational_data, set_flag )

        # Return the variational data.
        return variational_data


    # Implement a function to delete elements.
    def delete_elements( self, indexes, variational_data = None, batch_option = 'keep', batch_number = None, batch_size = None, set_flag = False ):

        # Setup the element information.
        variational_data, batch_number, batch_size = self.setup_elements( variational_data, batch_option, batch_number, batch_size )

        # Delete the elements.
        variational_data.finite_elements, variational_data.xs_integration_points_batch, variational_data.G_basis_values_batch, variational_data.W_integration_weights_batch, variational_data.sigma_jacobian_batch = variational_data.delete_elements( indexes, variational_data.finite_elements, batch_option, batch_number, batch_size )

        # Set the variational data (as required).
        self.set_variational_data( variational_data, set_flag )

        # Return the variational data.
        return variational_data


    #%% ------------------------------------------------------------ SHUFFLE FUNCTIONS ------------------------------------------------------------

    # Implement a function to shuffle ibc data.
    def shuffle_ibc_data( self, ibc_data, shuffle_indexes = None ):

        # Compute the number of initial conditions.
        num_conditions = torch.tensor( len( ibc_data ), dtype = torch.uint8, device = self.device )

        # Initialize a list to store the shuffle indexes.
        shuffle_indexes_list = [  ]

        # Shuffle the ibc data.
        for k in range( num_conditions ):                   # Iterate through each of the initial conditions...

            # Shuffle the data associated with this ibc.
            ibc_data[ k ].input_data, ibc_data[ k ].output_data, these_shuffle_indexes = ibc_data[ k ].shuffle_data( ibc_data[ k ].input_data, ibc_data[ k ].output_data, shuffle_indexes )

            # Append the shuffle indexes.
            shuffle_indexes_list.append( these_shuffle_indexes )

        # Return the ibc data.
        return ibc_data, shuffle_indexes_list


    # Implement a function to shuffle initial condition data.
    def shuffle_initial_data( self, initial_condition_data = None, shuffle_indexes = None, set_flag = False ):

        # Preprocess the initial condition data.
        initial_condition_data = self.preprocess_initial_condition_data( initial_condition_data )

        # Shuffle the initial condition data.
        initial_condition_data, shuffle_indexes = self.shuffle_ibc_data( initial_condition_data, shuffle_indexes )

        # Set the initial condition data (as required).
        self.set_initial_condition_data( initial_condition_data, set_flag )

        # Return the initial condition data.
        return initial_condition_data, shuffle_indexes


    # Implement a function to shuffle boundary condition data.
    def shuffle_boundary_data( self, boundary_condition_data = None, shuffle_indexes = None, set_flag = False ):

        # Preprocess the boundary condition data.
        boundary_condition_data = self.preprocess_boundary_condition_data( boundary_condition_data )

        # Shuffle the boundary condition data.
        boundary_condition_data, shuffle_indexes = self.shuffle_ibc_data( boundary_condition_data, shuffle_indexes )

        # Set the boundary condition data (as required).
        self.set_boundary_condition_data( boundary_condition_data, set_flag )

        # Return the boundary condition data.
        return boundary_condition_data, shuffle_indexes


    # Implement a function to shuffle the residual data.
    def shuffle_residual_data( self, residual_data = None, shuffle_indexes = None, set_flag = False ):

        # Preprocess the residual data.
        residual_data = self.preprocess_residual_data( residual_data )

        # Shuffle the residual data.
        residual_data.input_data, shuffle_indexes = residual_data.shuffle_data( residual_data.input_data, shuffle_indexes )

        # Set the residual data (as required).
        self.set_residual_data( residual_data, set_flag )

        # Return the residual data.
        return residual_data, shuffle_indexes


    # Implement a function to shuffle the variational data.
    def shuffle_variational_data( self, variational_data = None, shuffle_indexes = None, set_flag = False ):

        # Preprocess the variational data.
        variational_data = self.preprocess_variational_data( variational_data )

        # Shuffle the variational data.
        variational_data.finite_elements, shuffle_indexes = variational_data.shuffle_data( variational_data.finite_elements, shuffle_indexes )

        # Set the variational data (as required).
        self.set_variational_data( variational_data, set_flag )

        # Return the variational data.
        return variational_data, shuffle_indexes


    # Implement a function to shuffle all of the initial, boundary, and residual pinn data in the data manager.
    def shuffle_all_data( self, initial_condition_data = None, boundary_condition_data = None, residual_data = None, variational_data = None, shuffle_indexes = None, set_flag = False ):

        # Setup for data shuffling.
        initial_condition_data, boundary_condition_data, residual_data, variational_data = self.setup_data_shuffling( initial_condition_data, boundary_condition_data, residual_data, variational_data )

        # Shuffle the initial, boundary, residual, and variational data.
        initial_condition_data, _ = self.shuffle_initial_data( initial_condition_data, shuffle_indexes )
        boundary_condition_data, _ = self.shuffle_boundary_data( boundary_condition_data, shuffle_indexes )
        residual_data, shuffle_indexes = self.shuffle_residual_data( residual_data, shuffle_indexes )
        variational_data, _ = self.shuffle_variational_data( variational_data, shuffle_indexes )

        # Set the initial condition, boundary condition, residual, and variational data.
        self.set_initial_condition_data( initial_condition_data, set_flag )
        self.set_boundary_condition_data( boundary_condition_data, set_flag )
        self.set_residual_data( residual_data, set_flag )
        self.set_variational_data( variational_data, set_flag )

        # Return the initial, boundary, and residual data.
        return initial_condition_data, boundary_condition_data, residual_data, variational_data


    #%% ------------------------------------------------------------ BATCH FUNCTIONS ------------------------------------------------------------

    # Implement a function to retrieve a batch of the initial condition data.
    def compute_initial_batch_data( self, initial_condition_data = None, batch_number = None, initial_condition_batch_size = None, set_flag = False ):

        # Setup for computing the initial condition batch data.
        initial_condition_data, num_initial_conditions, batch_number, initial_condition_batch_size = self.setup_initial_condition_batch_computation( initial_condition_data, batch_number, initial_condition_batch_size )

        # Get the batch data from each initial condition.
        for k in range( num_initial_conditions ):              # Iterate through each of the new initial conditions...

            # Retrieve the batch data from this initial condition.
            initial_condition_data[ k ].input_data_batch, initial_condition_data[ k ].output_data_batch = initial_condition_data[ k ].compute_batch_data( initial_condition_data[ k ].input_data, initial_condition_data[ k ].output_data, batch_number, initial_condition_batch_size )

        # Set the initial condition data (as required).
        self.set_initial_condition_data( initial_condition_data, set_flag )

        # Return the initial batch data.
        return initial_condition_data


    # Implement a function to retrieve a batch of the boundary condition data.
    def compute_boundary_batch_data( self, boundary_condition_data = None, batch_number = None, boundary_condition_batch_size = None, set_flag = False ):

        # Setup for computing the boundary condition batch data.
        boundary_condition_data, num_boundary_conditions, batch_number, boundary_condition_batch_size = self.setup_boundary_condition_batch_computation( boundary_condition_data, batch_number, boundary_condition_batch_size )

        # Get the batch data from each boundary condition.
        for k in range( num_boundary_conditions ):              # Iterate through each of the new boundary conditions...

            # Retrieve the batch data from this boundary condition.
            boundary_condition_data[ k ].input_data_batch, boundary_condition_data[ k ].output_data_batch = boundary_condition_data[ k ].compute_batch_data( boundary_condition_data[ k ].input_data, boundary_condition_data[ k ].output_data, batch_number, boundary_condition_batch_size )

        # Set the boundary condition data (as required).
        self.set_boundary_condition_data( boundary_condition_data, set_flag )

        # Return the boundary batch data.
        return boundary_condition_data


    # Implement a function to retrieve a batch of the residual data.
    def compute_residual_batch_data( self, residual_data = None, batch_number = None, residual_batch_size = None, set_flag = False ):

        # Setup for the residual batch data computation.
        residual_data, batch_number, residual_batch_size = self.setup_residual_batch_computation( residual_data, batch_number, residual_batch_size )

        # Retrieve the residual batch data.
        residual_data.input_data_batch = residual_data.compute_batch_data( residual_data.input_data, batch_number, residual_batch_size )

        # Set the residual data (as required).
        self.set_residual_data( residual_data )

        # Return the residual data.
        return residual_data


    # Implement a function to retrieve a batch of the variational data.
    def compute_variational_batch_data( self, variational_data = None, batch_number = None, variational_batch_size = None, set_flag = False ):

        # Setup for computing the variational batch data.
        variational_data, batch_number, variational_batch_size = self.setup_variational_batch_computation( variational_data, batch_number, variational_batch_size )

        # Retrieve the variational batch data.
        variational_data.xs_integration_points_batch, variational_data.G_basis_values_batch, variational_data.W_integration_weights_batch, variational_data.sigma_jacobian_batch = variational_data.compute_batch_data( variational_data.finite_elements, batch_number, variational_batch_size )

        # Set the variational batch data (as required).
        self.set_variational_data( variational_data, set_flag )

        # Return the variational data.
        return variational_data


    # Implement a function to retrieve batches of all of the data.
    def compute_all_batch_data( self, initial_condition_data = None, boundary_condition_data = None, residual_data = None, variational_data = None, batch_number = None, initial_condition_batch_size = None, boundary_condition_batch_size = None, residual_batch_size = None, variational_batch_size = None, set_flag = False ):

        # Setup all of the data required to compute initial, boundary, residual, and variational data batches.
        initial_condition_data, boundary_condition_data, residual_data, variational_data, batch_number, initial_condition_batch_size, boundary_condition_batch_size, residual_batch_size, variational_batch_size = self.setup_all_batch_computations( initial_condition_data, boundary_condition_data, residual_data, variational_data, batch_number, initial_condition_batch_size, boundary_condition_batch_size, residual_batch_size, variational_batch_size )

        # Retrieve batches from the initial condition, boundary condition, residual, and variational data sets.
        initial_condition_data = self.compute_initial_batch_data( initial_condition_data, batch_number, initial_condition_batch_size )
        boundary_condition_data = self.compute_boundary_batch_data( boundary_condition_data, batch_number, boundary_condition_batch_size )
        residual_data = self.compute_residual_batch_data( residual_data, batch_number, residual_batch_size )
        variational_data = self.compute_variational_batch_data( variational_data, batch_number, variational_batch_size )

        # Set the initial condition, boundary condition, residual, and variational data (as required).
        self.set_initial_condition_data( initial_condition_data )
        self.set_boundary_condition_data( boundary_condition_data )
        self.set_residual_data( residual_data )
        self.set_variational_data( variational_data )

        # Return the bath data.
        return initial_condition_data, boundary_condition_data, residual_data, variational_data


    # Implement a function to compute the batch information.
    def initialize_batch_info( self, num_points_per_initial_condition = None, num_points_per_boundary_condition = None, num_residual_points = None, residual_batch_size = None, set_flag = False ):

        # Setup for batch info initialization.
        num_points_per_initial_condition, num_points_per_boundary_condition, num_residual_points, residual_batch_size = self.setup_batch_info_initialization( num_points_per_initial_condition, num_points_per_boundary_condition, num_residual_points, residual_batch_size )

        # Determine how to compute the number of batches.
        if residual_batch_size is None:                                                                             # If the residual batch size is None...

            # Set the number of batches to one.
            num_batches = torch.tensor( 1, dtype = torch.int32, device = self.device )

        elif torch.is_tensor( residual_batch_size ) and ( residual_batch_size != 0 ):                               # If the residual batch size is a non-empty tensor...

            # Compute the number of batches.
            num_batches = torch.round( num_residual_points/residual_batch_size ).to( torch.int32 )

        else:                                                                                                       # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid residual batch size: {residual_batch_size}' )

        # Compute the initial condition batch size.
        initial_condition_batch_size = ( num_points_per_initial_condition/num_batches ).to( torch.int32 )

        # Compute the boundary condition batch size.
        boundary_condition_batch_size = ( num_points_per_boundary_condition/num_batches ).to( torch.int32 )

        # Set the batch info (as required).
        self.set_num_batches( num_batches, set_flag )
        self.set_initial_condition_batch_size( initial_condition_batch_size, set_flag )
        self.set_boundary_condition_batch_size( boundary_condition_batch_size, set_flag )

        # Return the number of batches.
        return num_batches, initial_condition_batch_size, boundary_condition_batch_size


    #%% ------------------------------------------------------------ PRINTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to print print the pinn data stored in the pinn data manager.
    def print_summary( self, num_dashes = 20, decoration_flag = True ):

        # Print a header.
        self.printing_utilities.print_header( 'PINN DATA MANAGER SUMMARY', num_dashes, decoration_flag )

        # Print general information.
        print( 'General Information' )
        print( f'Device: {self.device}' )
        print( f'# of Inputs: {self.initial_condition_data[ 0 ].input_data.shape[ 1 ]}' )
        print( f'# of Outputs: {self.initial_condition_data[ 0 ].output_data.shape[ 1 ]}' )
        print( f'Total # of Data Points: {self.num_data_points}' )
        print( f'Total # of Batches: {self.num_batches}' )
        print( '\n' )

        # Print initial condition information.
        print( 'Initial Condition Information' )
        print( f'# of Initial Conditions: {self.num_initial_conditions}' )
        print( f'# of Initial Conditions Points Per Initial Condition: {self.num_points_per_initial_condition}' )
        print( f'Total # of Initial Condition Points: {self.num_initial_condition_points}' )
        print( f'Initial Condition Batch Size: {self.initial_condition_batch_size}' )
        print( f'Initial Condition Data: {self.initial_condition_data}' )
        print( '\n' )

        # Print boundary condition information.
        print( 'Boundary Condition Information' )
        print( f'# of Boundary Conditions: {self.num_boundary_conditions}' )
        print( f'# of Boundary Conditions Points Per Boundary Condition: {self.num_points_per_boundary_condition}' )
        print( f'Total # of Boundary Condition Points: {self.num_boundary_condition_points}' )
        print( f'Boundary Condition Batch Size: {self.boundary_condition_batch_size}' )
        print( f'Boundary Condition Data: {self.boundary_condition_data}' )
        print( '\n' )

        # Print residual information.
        print( 'Residual Information' )
        print( f'Total # of Residual Points: {self.num_residual_condition_points}' )
        print( f'Residual Batch Size: {self.residual_batch_size}' )
        print( f'Residual Data: {self.residual_condition_data}' )

        # Print a footer.
        self.printing_utilities.print_footer( num_dashes, decoration_flag )


    # Implement a function to print summaries of the constituent pinn data.
    def print_constituent_summaries( self, num_dashes = 20, decoration_flag = True ):

        # Print a header.
        self.printing_utilities.print_header( 'PINN DATA CONSTITUENT SUMMARIES', num_dashes, decoration_flag )

        # Print the initial condition data summary.
        self.initial_condition_data.print( decoration_flag )

        # Print the boundary condition data summary.
        self.boundary_condition_data.print( decoration_flag )

        # Print the residual data summary.
        self.residual_data.print( decoration_flag )

        # Print a footer.
        self.printing_utilities.print_footer( num_dashes, decoration_flag )


    # Implement a function to adaptively print pinn manager data.
    def print( self, header_flag = True, print_type = 'manager' ):

        # Determine how to print the pinn manager data.
        if print_type.lower(  ) in ( 'manager', 'summary' ):                   # If the print type is manager...

            # Print the manager data.
            self.print_summary( header_flag )

        elif print_type.lower(  ) in ( 'constituent', 'summaries' ):             # If the print type is constituent...

            # Print the constituent data summaries.
            self.print_constituent_summaries( header_flag )

        else:                                                   # Otherwise... ( i.e., the print type is not recognized... )

            # Throw an error.
            raise ValueError( f'Invalid print type: {print_type}' )


    #%% ------------------------------------------------------------ PLOTTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to plot all of the constituent pinn data.
    def plot( self, initial_condition_data = None, boundary_condition_data = None, residual_data = None, projection_dimensions = None, projection_values = None, level = 0, fig = None, plot_type1 = 'all', plot_type2 = 'all', save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Plot type 1 should be one of: 'initial', 'boundary', 'residual, or 'all'.
        # Plot type 2 should be one of: 'batch' or 'all'.

        # Setup for plotting.
        initial_condition_data, boundary_condition_data, residual_data = self.setup_plotting( initial_condition_data, boundary_condition_data, residual_data )

        # Compute the number of initial conditions.
        num_initial_conditions = torch.tensor( len( initial_condition_data ), dtype = torch.uint8, device = self.device )

        # Compute the number of boundary conditions.
        num_boundary_conditions = torch.tensor( len( boundary_condition_data ), dtype = torch.uint8, device = self.device )

        # Determine whether the first plotting type option is valid.
        if ( plot_type1.lower(  ) == 'initial' ) or ( plot_type1.lower(  ) == 'boundary' ) or ( plot_type1.lower(  ) == 'residual' ) or ( plot_type1.lower(  ) == 'all' ):

            # Initialize empty figure and axes lists.
            figs = [  ]
            axes = [  ]

            # Determine whether to plot the initial condition data.
            if ( plot_type1.lower(  ) == 'all' ) or ( plot_type1.lower(  ) == 'initial' ):                  # If we want to plot the initial condition data...

                # Plot the data associated with each initial condition.
                for k in range( num_initial_conditions ):                                              # Iterate through each of the initial conditions...

                    # Plot the initial condition data.
                    fig_initial, ax_initial = initial_condition_data[ k ].plot( None, None, initial_condition_data[ k ].dimension_labels, projection_dimensions, projection_values, level, fig, plot_type2, save_directory, as_surface, as_stream, as_contour, show_plot )

                    # Add the initial condition figure and axes to the lists.
                    figs.append( fig_initial )
                    axes.append( ax_initial )

            # Determine whether to plot the boundary condition data.
            if ( plot_type1.lower(  ) == 'all' ) or ( plot_type1.lower(  ) == 'boundary' ):                 # If we want to plot the boundary condition data...

                # Plot the data associated with each boundary condition.
                for k in range( num_boundary_conditions ):                                             # Iterate through each of the boundary conditions...

                    # Plot the boundary condition data.
                    fig_boundary, ax_boundary = boundary_condition_data[ k ].plot( None, None, boundary_condition_data[ k ].dimension_labels, projection_dimensions, projection_values, level, fig, plot_type2, save_directory, as_surface, as_stream, as_contour, show_plot )

                    # Add the boundary condition figure and axes to the lists.
                    figs.append( fig_boundary )
                    axes.append( ax_boundary )

            # Determine whether to plot the residual data.
            if ( plot_type1.lower(  ) == 'all' ) or ( plot_type1.lower(  ) == 'residual' ):                 # If we want to plot the residual data...

                # Plot the residual condition data.
                fig_residual, ax_residual = residual_data.plot( None, residual_data.dimension_labels, projection_dimensions, projection_values, level, None, fig, plot_type2, save_directory, as_surface, as_stream, as_contour, show_plot )

                # Add the residual condition figure and axes to the lists.
                figs.append( fig_residual )
                axes.append( ax_residual )

        else:                                                                                               # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid primary plotting type: {plot_type1}' )

        # Return the figure and axis objects.
        return figs, axes

