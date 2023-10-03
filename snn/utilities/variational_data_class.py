#%% ------------------------------------------------------------ VARIATIONAL DATA CLASS ------------------------------------------------------------

# This file implements a class for storing and managing variational data information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Import custom libraries.
from plotting_utilities_class import plotting_utilities_class as plotting_utilities_class
from pinn_data_class import pinn_data_class as pinn_data_class
from finite_element_class import finite_element_class as finite_element_class


#%% ------------------------------------------------------------ VARIATIONAL DATA CLASS ------------------------------------------------------------

# Implement the variational data class.
class variational_data_class( pinn_data_class ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self, id, name, dimension_labels, element_scale, integration_order, xs_element_centers = None, batch_size = None, device = 'cpu' ):

        # Construct the parent class.
        super(  ).__init__( id, name, dimension_labels, batch_size, device )

        # Create an instance of the finite element class.
        self.finite_elements = finite_element_class( element_scale, integration_order, element_type = 'rectangular', device = self.device )

        # Determine whether to create elements.
        if xs_element_centers is not None:                      # If element center locations were provided...

            # Create the desired elements.
            self.finite_elements.xs_integration_points, self.finite_elements.G_basis_values, self.finite_elements.W_integration_weights, self.finite_elements.sigma_jacobian, self.finite_elements.num_elements = self.finite_elements.append_elements( xs_element_centers )

        # Set the initial batch number to zero.
        batch_number = torch.tensor( 0, dtype = torch.int32, device = self.device )

        # Compute the batch data.
        self.xs_integration_points_batch, self.G_basis_values_batch, self.W_integration_weights_batch, self.sigma_jacobian_batch = self.compute_batch_data( self.finite_elements, batch_number, self.batch_size )


    #%% ------------------------------------------------------------ PRE-PROCESS FUNCTIONS ------------------------------------------------------------

    # Implement a function preprocess the finite elements.
    def preprocess_finite_elements( self, finite_elements = None ):

        # Determine whether to use the stored finite elements.
        if finite_elements is None:                 # If the finite elements were not provided...

            # Set the finite elements to be the stored values.
            finite_elements = self.finite_elements

        # Return the finite elements.
        return finite_elements


    # Implement a function to preprocess the batch number.
    def preprocess_batch_number( self, batch_number = None ):

        # Determine whether to use the default batch number.
        if batch_number is None:                # If the batch number was not provided...

            # Set the batch number to be zero.
            batch_number = torch.tensor( 0, dtype = torch.int64, device = self.device )

        # Return the batch number.
        return batch_number


    # Implement a function to preprocess the batch size.
    def preprocess_batch_size( self, batch_size = None ):

        # Determine whether to use the stored batch size.
        if batch_size is None:                  # If the batch size was not provided...

            # Set the batch size to be the stored value.
            batch_size = self.batch_size

        # Return the batch size.
        return batch_size


    #%% ------------------------------------------------------------ SETUP FUNCTIONS ------------------------------------------------------------

    # # Implement a function to setup the shuffle indexes.
    # def setup_shuffle_indexes( self, shuffle_indexes = None, finite_elements = None ):

    #     # Setup the finite elements.
    #     finite_elements = self.preprocess_finite_elements( finite_elements )

    #     # Determine whether to generate shuffle shuffle_indexes.
    #     if shuffle_indexes is None:                     # If shuffle shuffle_indexes where not provided...

    #         # Generate shuffled shuffle_indexes.
    #         shuffle_indexes = torch.randperm( finite_elements.num_elements, dtype = torch.int64, device = self.device )

    #     # Return the shuffle indexes.
    #     return shuffle_indexes


    # # Implement a function to setup the number of elements.
    # def setup_num_elements( self, num_elements = None, finite_elements = None ):

    #     # Setup the finite elements.
    #     finite_elements = self.preprocess_finite_elements( finite_elements )

    #     # Determine whether to use the stored number of elements.
    #     if num_elements is None:                                 # If the number of elements was not provided...

    #         # Use the stored number of elements.
    #         num_elements = finite_elements.num_elements

    #     # Return the number of elements.
    #     return num_elements


    # Implement a function to setup finite elements.
    def setup_elements( self, finite_elements = None, batch_number = None, batch_size = None ):

        # Setup the finite elements.
        finite_elements = self.preprocess_finite_elements( finite_elements )

        # Setup the batch number.
        batch_number = self.preprocess_batch_number( batch_number )

        # Setup the batch size.
        batch_size = self.preprocess_batch_size( batch_size )

        # Return the element setup information.
        return finite_elements, batch_number, batch_size


    #%% ------------------------------------------------------------ SETUP FUNCTIONS ------------------------------------------------------------

    # Implement a function to set the element batch data.
    def set_element_batch_data( self, xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch, set_flag = True ):

        # Determine whether to set the element batch data.
        if set_flag:                                # If we want to set the element batch data...

            # Set the element batch data.
            self.xs_integration_points_batch = xs_integration_points_batch
            self.G_basis_values_batch = G_basis_values_batch
            self.W_integration_weights_batch = W_integration_weights_batch
            self.sigma_jacobian_batch = sigma_jacobian_batch


    # Implement a function to set the finite elements.
    def set_finite_elements( self, finite_elements, set_flag = True ):

        # Determine whether to set the finite elements.
        if set_flag:                # If we want to set the finite elements...

            # Set the finite elements.
            self.finite_elements = finite_elements


    # Implement a function to set variational data.
    def set_variational_data( self, finite_elements, xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch, set_flag = True ):

        # Set the finite elements.
        self.set_finite_elements( finite_elements, set_flag )

        # Set the element batch data.
        self.set_element_batch_data( xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch, set_flag )


    #%% ------------------------------------------------------------ BATCH FUNCTIONS ------------------------------------------------------------

    # Implement a function to saturate the upper batch index.
    def saturate_upper_batch_index( self, upper_batch_index, finite_elements = None ):

        # Setup the finite elements.
        finite_elements = self.preprocess_finite_elements( finite_elements )

        # Saturate the upper batch index.
        upper_batch_index = finite_elements.saturate_upper_batch_index( upper_batch_index, finite_elements.num_elements )

        # Return the upper batch index.
        return upper_batch_index


    # Implement a function to stage a batch of the data.
    def compute_batch_data( self, finite_elements = None, batch_number = None, batch_size = None, set_flag = False ):

        # Setup the elements.
        finite_elements, batch_number, batch_size = self.setup_elements( finite_elements, batch_number, batch_size )

        # Compute the batch data associated with these finite elements.
        xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch = finite_elements.compute_batch_data( finite_elements.xs_integration_points, finite_elements.G_basis_values, finite_elements.W_integration_weights, finite_elements.sigma_jacobian, finite_elements.num_elements, batch_number, batch_size )

        # Set the element batch data (as required).
        self.set_element_batch_data( xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch, set_flag )

        # Return the input data batch.
        return xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch


    # Implement a function to retrieve the batch data in a way determined by the specified batch option.
    def get_batch_data( self, finite_elements = None, batch_number = None, batch_size = None, batch_option = 'keep' ):

        # Setup the elements.
        finite_elements, batch_number, batch_size = self.setup_elements( finite_elements, batch_number, batch_size )

        # Determine how to handle the elements batches.
        if batch_option.lower(  ) in ( 'leave', 'keep', 'ignore', 'current', 'existing' ):                 # If we want to keep the existing element batch data...

            xs_integration_points_batch = self.xs_integration_points_batch
            G_basis_values_batch = self.G_basis_values_batch
            W_integration_weights_batch = self.W_integration_weights_batch
            sigma_jacobian_batch = self.sigma_jacobian_batch

        elif batch_option.lower(  ) in ( 'remove', 'delete', 'none' ):                      # If we want to remove the existing element batch data without replacing it...

            xs_integration_points_batch = None
            G_basis_values_batch = None
            W_integration_weights_batch = None
            sigma_jacobian_batch = None

        elif batch_option.lower(  ) in ( 'replace', 'new', 'draw', 'compute' ):                # If we want to replace the existing element batch data by drawing a batch from the new element data...

            # Compute the element batch data associated with 
            xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch = self.compute_batch_data( finite_elements, batch_number, batch_size )

        else:                                                                       # Otherwise... ( i.e., the batch option is not recognized... )

            # Throw an error.
            raise ValueError( f'Invalid batch option: {batch_option}' )

        # Return the element batch data.
        return xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch


    #%% ------------------------------------------------------------ ELEMENT FUNCTIONS ------------------------------------------------------------

    # Implement a function to generate elements.
    def generate_elements_lazy( self, xs_element_centers, finite_elements = None, replace_flag = False, set_flag = False ):

        # Setup the finite elements.
        finite_elements = self.preprocess_finite_elements( finite_elements )

        # Generate the elements.
        finite_elements.xs_integration_points, finite_elements.G_basis_values, finite_elements.W_integration_weights, finite_elements.sigma_jacobian, finite_elements.num_elements = finite_elements.generate_elements( xs_element_centers, replace_flag )

        # Set the finite elements (as required).
        self.set_finite_elements( finite_elements, set_flag )

        # Return the finite elements.
        return finite_elements


    # Implement a function to generate elements.
    def generate_elements( self, xs_element_centers, finite_elements = None, replace_flag = False, batch_option = 'keep', batch_number = None, batch_size = None, set_flag = False ):

        # Setup the elements.
        finite_elements, batch_number, batch_size = self.setup_elements( finite_elements, batch_number, batch_size )

        # Generate the elements.
        finite_elements = self.generate_elements_lazy( xs_element_centers, finite_elements, replace_flag )

        # Get the element batch data according to the given batch option.
        xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch = self.get_batch_data( finite_elements, batch_number, batch_size, batch_option )

        # Set the variational data (as required).
        self.set_variational_data( finite_elements, xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch, set_flag )

        # Return the finite elements.
        return finite_elements, xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch


    # Implement a function to delete elements.
    def delete_elements_lazy( self, indexes, finite_elements = None, set_flag = False ):

        # Setup the finite elements.
        finite_elements = self.preprocess_finite_elements( finite_elements )

        # Delete the specified elements.
        finite_elements.xs_integration_points, finite_elements.G_basis_values, finite_elements.W_integration_weights, finite_elements.sigma_jacobian, finite_elements.num_elements = finite_elements.delete_elements( indexes )

        # Set the finite elements (as required).
        self.set_finite_elements( finite_elements, set_flag )

        # Return the finite elements.
        return finite_elements


    # Implement a function to delete elements.
    def delete_elements( self, indexes, finite_elements = None, batch_option = 'keep', batch_number = None, batch_size = None, set_flag = False ):

        # Setup the elements.
        finite_elements, batch_number, batch_size = self.setup_elements( finite_elements, batch_number, batch_size )

        # Delete the specified elements.
        finite_elements = self.delete_elements_lazy( indexes, finite_elements )

        # Get the element batch data according to the given batch option.
        xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch = self.get_batch_data( finite_elements, batch_number, batch_size, batch_option )

        # Set the variational data (as required).
        self.set_variational_data( finite_elements, xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch, set_flag )

        # Return the finite elements and associated batch data.
        return finite_elements, xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch


    #%% ------------------------------------------------------------ SHUFFLE FUNCTIONS ------------------------------------------------------------

    # Implement a function to shuffle the input data.
    def shuffle_data( self, finite_elements = None, shuffle_indexes = None, set_flag = False ):

        # Setup the finite elements.
        finite_elements = self.preprocess_finite_elements( finite_elements )

        # Shuffle the finite element data.
        finite_elements.xs_integration_points, finite_elements.G_basis_values, finite_elements.W_integration_weights, finite_elements.sigma_jacobian, shuffle_indexes = finite_elements.shuffle_data( finite_elements.xs_integration_points, finite_elements.G_basis_values, finite_elements.W_integration_weights, finite_elements.sigma_jacobian, shuffle_indexes )

        # Set the finite elements (as required).
        self.set_finite_elements( finite_elements, set_flag )

        # Return the shuffled data.
        return finite_elements, shuffle_indexes


    #%% ------------------------------------------------------------ PRINTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to print the variational data information.
    def print( self, num_dashes = 20, decoration_flag = True ):

        # Print a header.
        self.printing_utilities.print_header( 'VARIATIONAL DATA SUMMARY', num_dashes, decoration_flag )

        # Print the identification information.
        print( 'General Information' )
        print( f'Device: {self.device}' )
        print( f'ID: {self.id}' )
        print( f'Name: {self.name}' )
        print( f'Dimension Labels: {self.dimension_labels}' )
        print( f'Batch size: {self.batch_size}' )

        # Print the finite element information.
        self.finite_elements.print( header_flag = False )

        # Compute the batch data.
        print( 'Batch Information' )
        print( f'xs Integration Points Batch: {self.xs_integration_points_batch}' )
        print( f'G Basis Values Batch: {self.G_basis_values_batch}' )
        print( f'W Integration Weights Batch: {self.W_integration_weights_batch}' )
        print( f'sigma Jacobian Batch: {self.sigma_jacobian_batch}' )

        # Print a footer.
        self.printing_utilities.print_footer( num_dashes, decoration_flag )

