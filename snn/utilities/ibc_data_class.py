#%% ------------------------------------------------------------ IBC DATA CLASS ------------------------------------------------------------

# This file implements a class for storing and managing ibc data information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Import custom libraries.
from plotting_utilities_class import plotting_utilities_class as plotting_utilities_class
from pinn_data_class import pinn_data_class as pinn_data_class


#%% ------------------------------------------------------------ IBC DATA CLASS ------------------------------------------------------------

# Implement the ibc data class.
class ibc_data_class( pinn_data_class ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self, id, name, condition_type, dimension, dimension_labels, input_data, output_data, output_derivative_order, batch_size = None, device = 'cpu' ):

        # Construct the parent class.
        super(  ).__init__( id, name, dimension_labels, batch_size, device )

        # Set the initial-boundary condition type and application dimension.
        self.condition_type = self.validate_condition_type( condition_type )
        self.dimension = self.validate_dimension( dimension )

        # Set the input and output data.
        self.input_data = self.validate_input_data( input_data )
        self.output_data = self.validate_output_data( output_data )
        self.num_data_points = torch.tensor( self.input_data.shape[ 0 ], dtype = torch.int32, device = self.device )

        # Ensure that the input and output data are compatible.
        assert self.is_input_output_data_compatible( self.input_data, self.output_data )

        # Set the output derivative order and number of output sources.
        self.output_derivative_order = self.validate_output_derivative_order( output_derivative_order )
        self.num_output_sources = self.compute_num_output_sources( self.output_data )

        # Ensure that the output data and output types are compatible.
        assert self.is_output_data_derivative_order_compatible( self.output_data, self.output_derivative_order )

        # Stage a batch of initial-boundary data.
        self.input_data_batch, self.output_data_batch = self.compute_batch_data( self.input_data, self.output_data, batch_number = torch.tensor( 0, dtype = torch.uint8, device = self.device ), batch_size = self.batch_size )


    #%% ------------------------------------------------------------ PREPROCESS FUNCTIONS ------------------------------------------------------------

    # Implement a function to preprocess input data.
    def preprocess_input_data( self, input_data = None ):

        # Determine whether to use the store input data value.
        if input_data is None:                 # If the input data is None..

            # Use the stored input data.
            input_data = self.input_data

        # Return the input data.
        return input_data


    # Implement a function to preprocess output data.
    def preprocess_output_data( self, output_data = None ):

        # Determine whether to use the store output data value.
        if output_data is None:                 # If the output data is None..

            # Use the stored output data.
            output_data = self.output_data

        # Return the output data.
        return output_data


    # Implement a function to preprocess the number of data points.
    def preprocess_num_data_points( self, num_data_points = None ):

        # Determine whether to use the stored number of data points.
        if num_data_points is None:                 # If the number of data points was not provided...

            # Use the stored number of data points.
            num_data_points = self.num_data_points

        # Return the number of data points.
        return num_data_points


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


    # Implement a function to preprocess shuffle indexes.
    def preprocess_shuffle_indexes( self, shuffle_indexes = None, num_data_points = None ):

        # Preprocess the number of data points.
        num_data_points = self.preprocess_num_data_points( num_data_points )

        # Determine whether to generate shuffle shuffle_indexes.
        if shuffle_indexes is None:                     # If shuffle shuffle_indexes where not provided...

            # Generate shuffled shuffle_indexes.
            shuffle_indexes = torch.randperm( num_data_points, dtype = torch.int64, device = self.device )

        # Return the shuffle indexes.
        return shuffle_indexes


    # Implement a function to preprocess dimension labels.
    def preprocess_dimension_labels( self, dimension_labels = None ):

        # Determine whether to use the stored dimension labels.
        if dimension_labels is None:                # If the dimension labels were not provided...

            # Use the stored dimension labels.
            dimension_labels = self.dimension_labels

        # Return the dimension labels.
        return dimension_labels


    # Implement a function to preprocess plotting input and output data.
    def preprocess_input_output_data( self, input_data = None, output_data = None, plot_type = 'all' ):

        # Determine whether to use the stored input and output data.
        if ( input_data is None ) and ( output_data is None ):                                                                                  # If the input and output data was not provided...

            # Retrieve the plotting data based on the plot type (all vs batch).
            input_data, output_data = self.get_plotting_data( plot_type )

        elif ( ( input_data is not None ) and ( output_data is None ) ) or ( ( input_data is None ) and ( output_data is not None ) ):          # If either the input data or the output data was provided but no the other...

            # Throw an error.
            raise ValueError( 'Invalid input and output data.  The provided input and output data must either both be None or not None.' )

        # Return the input and output data.
        return input_data, output_data


    #%% ------------------------------------------------------------ IS VALID FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate the condition type.
    def is_condition_type_valid( self, condition_type ):

        # Determine whether the given condition type is valid.
        if isinstance( condition_type, str ) and ( ( condition_type.lower(  ) == 'initial' ) or ( condition_type.lower(  ) == 'boundary' ) or ( condition_type.lower(  ) == 'residual' ) or ( condition_type.lower(  ) == 'ic' ) or ( condition_type.lower(  ) == 'bc' ) or ( condition_type.lower(  ) == 'res' ) ):                        # If the condition type is valid...

            # Set the valid flag to true.
            valid_flag = True

        else:                                           # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the dimension.
    def is_dimension_valid( self, dimension ):

        # Determine whether the given dimension is valid.
        if torch.is_tensor( dimension ) and ( dimension.numel(  ) != 0 ) and ( dimension.dtype == torch.uint8 ) and ( dimension >= 0 ):                     # If the dimension is a tensor, non-empty, and contains a uint8 that is greater than or equal to zero...

            # Set the valid flag to true.
            valid_flag = True

        elif dimension is None:                                                                                                                             # If the dimension is None...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                                                                                                               # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the input data.
    def is_input_data_valid( self, input_data ):

        # Determine whether the input data is valid.
        if torch.is_tensor( input_data ) and ( input_data.numel(  ) != 0 ) and ( input_data.dim(  ) <= 3 ):                         # If the input data is a non-empty torch tensor...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                                                                                       # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the output data.
    def is_output_data_valid( self, output_data, condition_type = None ):

        # Determine whether to use the stored condition type.
        if condition_type is None:                                  # If a condition type was not provided...

            # Use the stored condition type.
            condition_type = self.condition_type

        # Determine whether the output data is valid.
        if isinstance( output_data, list ):                         # If the output data is a non-empty torch tensor...

            # Ensure that each of the list entries are non-empty tensors.
            valid_flag = all( ( torch.is_tensor( output_data[ k ] ) and ( output_data[ k ].numel(  ) != 0 ) ) for k in range( len( output_data ) ) )

        elif torch.is_tensor( output_data ) and ( output_data.numel(  ) != 0 ):                     # If the output data is itself a non-empty tensor...

            # Set the valid flag to true.
            valid_flag = True

        elif ( output_data is None ) and ( condition_type.lower(  ) == 'residual' ):                # If the output data is None and the condition type is residual...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                                                       # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the output derivative order.
    def is_output_derivative_order_valid( self, output_derivative_order, condition_type = None ):

        # Determine whether to use the stored condition type.
        if condition_type is None:                                  # If a condition type was not provided...

            # Use the stored condition type.
            condition_type = self.condition_type

        # Determine whether the output data is valid.
        if torch.is_tensor( output_derivative_order ) and ( output_derivative_order.numel(  ) != 0 ) and ( output_derivative_order.dim(  ) <= 2 ):                         # If the output data is a non-empty torch tensor...

            # Set the valid flag to true.
            valid_flag = True

        elif isinstance( output_derivative_order, list ) and ( not output_derivative_order ):                                                                               # If the output data is an empty list...

            # Set the valid flag to true.
            valid_flag = True

        elif ( output_derivative_order is None ) and ( condition_type.lower(  ) == 'residual' ):                # If the output data is None and the condition type is residual...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                                                       # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the compatibility of the input and output data.
    def is_input_output_data_compatible( self, input_data, output_data, condition_type = None ):

        # Determine whether to use the stored condition type.
        if condition_type is None:                                  # If a condition type was not provided...

            # Use the stored condition type.
            condition_type = self.condition_type

        # Determine whether the input data is valid.
        valid_flag = self.is_input_data_valid( input_data )

        # Determine whether to determine whether the output data is valid.
        if valid_flag:                                                         # If the prior validity checks passed...

            # Determine whether the output data is valid.            
            valid_flag &= self.is_output_data_valid( output_data )

        # Determine whether to determined whether the input and output data have the same number of data points.
        if valid_flag:                                                         # If the prior validity checks passed...

            # Determine whether to evaluate the output data as a list or as a tensor.
            if isinstance( output_data, list ):                                 # If the output data is a list...

                # Ensure that each of the list entries are valid.
                valid_flag = all( ( input_data.shape[ 0 ] == output_data[ k ].shape[ 0 ] ) for k in range( len( output_data ) ) )

            elif torch.is_tensor( output_data ):                                # If the output data is a tensor...

                # Determine whether the input and output data have the same number of data points.
                valid_flag &= input_data.shape[ 0 ] == output_data.shape[ 0 ]

            elif ( output_data is None ) and ( condition_type.lower(  ) == 'residual' ):                                           # If the output data is None...

                # Set the valid flag to true.
                valid_flag = True

            else:                                                               # Otherwise...

                # Throw an error.
                raise ValueError( f'Output data type not recognized.' )

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the compatibility of the output data and the output types.
    def is_output_data_derivative_order_compatible( self, output_data, output_derivative_order, condition_type = None ):

        # Determine whether to use the stored condition type.
        if condition_type is None:                                  # If a condition type was not provided...

            # Use the stored condition type.
            condition_type = self.condition_type

        # Determine whether the output data is valid.
        valid_flag = self.is_output_data_valid( output_data )

        # Determine whether to determine whether the output derivative order is valid.
        if valid_flag:                                                                          # If the prior validity checks passed...

            # Determine whether the output derivative order is valid.
            valid_flag &= self.is_output_derivative_order_valid( output_derivative_order )

        # Determine whether to determine whether the output data and output derivative order are compatible.
        if valid_flag:                                                                                                                  # If the prior validity checks passed...

            # Determine how to compute the number of output data sources from the output data.
            if torch.is_tensor( output_data ):                                                                                          # If the output data is a tensor...

                # Set the number of output data sources to be one.
                num_output_sources1 = torch.tensor( 1, dtype = torch.uint8, device = self.device )

            elif isinstance( output_data, list ):                                                                                       # If the output data is a list...

                # Set the number of output data sources to be the length of the output data list.
                num_output_sources1 = torch.tensor( len( output_data ), dtype = torch.uint8, device = self.device )

            elif output_data is None:                                                                                                   # If the output data is None...

                # Set the number of output data sources to be zero.
                num_output_sources1 = torch.tensor( 0, dtype = torch.uint8, device = self.device )

            else:                                                                                                                       # Otherwise...

                # Throw an error.
                raise ValueError( 'Invalid output data: {output_data}' )

            # Determine how to compute the number of output data sources from the output derivative order.
            if torch.is_tensor( output_derivative_order ):                                                                                          # If the output derivative order is a tensor...

                # Set the number of output data sources to be the number of elements in the tensor.
                num_output_sources2 = torch.tensor( output_derivative_order.numel(  ), dtype = torch.uint8, device = self.device )

            elif isinstance( output_derivative_order, list ):                                                                                       # If the output derivative order is a list...

                # Set the number of output data sources to be the length of the output derivative order list.
                num_output_sources2 = torch.tensor( len( output_derivative_order ), dtype = torch.uint8, device = self.device )

            elif output_derivative_order is None:                                                                                                   # If the output derivative order is None...

                # Set the number of output data sources to be zero.
                num_output_sources2 = torch.tensor( 0, dtype = torch.uint8, device = self.device )

            else:                                                                                                                                   # Otherwise...

                # Throw an error.
                raise ValueError( 'Invalid output derivative order: {output_derivative_order}' )

            # Determine whether the output data and output derivative order are compatible.
            if num_output_sources1 == num_output_sources2:                                      # If the number of output sources associated with the output data and output derivative order are compatible...

                # Set the valid flag to true.
                valid_flag = True

            else:                                                                               # Otherwise...

                # Set the valid flag to false.
                valid_flag = False

        # Return the validity flag.
        return valid_flag


    #%% ------------------------------------------------------------ GET FUNCTIONS ------------------------------------------------------------

    # Implement a function to get plotting data.
    def get_plotting_data( self, plot_type = 'all' ):

        # Retrieve the data to plot.
        if plot_type.lower(  ) == 'all':                             # If we want to plot all of the data in this data set...

            # Set the input and output plotting data.
            input_plotting_data = self.input_data
            output_plotting_data = self.output_data

        elif plot_type.lower(  ) == 'batch':                         # If we want to plot the batch data in this data set...

            # Set the input and output plotting data.
            input_plotting_data = self.input_data_batch
            output_plotting_data = self.output_data_batch

        else:                                                       # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid plotting type: {plot_type}' )

        # Return the plotting data.
        return input_plotting_data, output_plotting_data


    #%% ------------------------------------------------------------ SET FUNCTIONS ------------------------------------------------------------

    # Implement a function to set the condition type.
    def set_condition_type( self, condition_type, set_flag = True ):

        # Determine whether to set the condition type.
        if set_flag:                    # If we want to set the condition type...

            # Set the condition type.
            self.condition_type = condition_type


    # Implement a function to set the dimension.
    def set_dimension( self, dimension, set_flag = True ):

        # Determine whether to set the dimension.
        if set_flag:                    # If we want to set the dimension...

            # Set the dimension.
            self.dimension = dimension


    # Implement a function to set the input data.
    def set_input_data( self, input_data, set_flag = True ):

        # Determine whether to set the input data.
        if set_flag:                    # If we want to set the input data...

            # Set the input data.
            self.input_data = input_data


    # Implement a function to set the input data batch.
    def set_input_data_batch( self, input_data_batch, set_flag = True ):

        # Determine whether to set the input data batch.
        if set_flag:                    # If we want to set the input data batch...

            # Set the input data batch.
            self.input_data_batch = input_data_batch


    # Implement a function to set the output data.
    def set_output_data( self, output_data, set_flag = True ):

        # Determine whether to set the output data.
        if set_flag:                    # If we want to set the output data...

            # Set the output data.
            self.output_data = output_data


    # Implement a function to set the output data batch.
    def set_output_data_batch( self, output_data_batch, set_flag = True ):

        # Determine whether to set the output data batch.
        if set_flag:                    # If we want to set the output data batch...

            # Set the output data batch.
            self.output_data_batch = output_data_batch


    # Implement a function to set the output derivative order.
    def set_output_derivative_order( self, output_derivative_order, set_flag = True ):

        # Determine whether to set the output derivative order.
        if set_flag:                    # If we want to set the output derivative order...

            # Set the output derivative order.
            self.output_derivative_order = output_derivative_order


    # Implement a function to set the number of output sources.
    def set_num_output_sources( self, num_output_sources, set_flag = True ):

        # Determine whether to set the number of output sources.
        if set_flag:                    # If we want to set the number of output sources.

            # Set the number of output sources.
            self.num_output_sources = num_output_sources


    #%% ------------------------------------------------------------ VALIDATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate the condition type.
    def validate_condition_type( self, condition_type, set_flag = False ):

        # Determine how to process the condition type.
        if not self.is_condition_type_valid( condition_type ):                          # If the condition type is not valid...

            # Throw an error.
            raise ValueError( f'Invalid condition type: {condition_type}' )

        # Set the condition type (as required).
        self.set_condition_type( condition_type, set_flag )

        # Return the condition type.
        return condition_type


    # Implement a function to validate the dimension.
    def validate_dimension( self, dimension, set_flag = False ):

        # Determine how to process the dimension.
        if not self.is_dimension_valid( dimension ):                          # If the dimension is not valid...

            # Throw an error.
            raise ValueError( f'Invalid dimension: {dimension}' )

        # Set the dimension (as required).
        self.set_dimension( dimension, set_flag )

        # Return the dimension.
        return dimension


    # Implement a function to validate the input data.
    def validate_input_data( self, input_data, set_flag = False ):

        # Determine how to process the input data.
        if self.is_input_data_valid( input_data ):                          # If the input data is valid...

            # Set the input data.
            input_data = self.augment_input_output_data_tensor( input_data )

        else:                                                   # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid input data: {input_data}' )

        # Set the input data (as required).
        self.set_input_data( input_data, set_flag )

        # Return the input data.
        return input_data


    # Implement a function to validate the output data.
    def validate_output_data( self, output_data, set_flag = False ):

        # Determine how to process the output data.
        if self.is_output_data_valid( output_data ):                            # If the output data is valid...

            # Ensure that the output data is embedded in a list if it is not already.
            if torch.is_tensor( output_data ):                                  # If the output data is a tensor...

                # Embed the output data tensor in a list.
                output_data = [ output_data ]

            # Set the output data.
            output_data = self.augment_input_output_data( output_data )

        else:                                                                   # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid output data: {output_data}' )

        # Set the output data (as required).
        self.set_output_data( output_data, set_flag )

        # Return the output data.
        return output_data


    # Implement a function to validate the output derivative order.
    def validate_output_derivative_order( self, output_derivative_order, set_flag = False ):

        # Determine how to process the output derivative order.
        if not self.is_output_derivative_order_valid( output_derivative_order ):                          # If the output derivative order is not valid...

            # Throw an error.
            raise ValueError( f'Invalid condition order: {output_derivative_order}' )

        # Set the output derivative order (as required).
        self.set_output_derivative_order( output_derivative_order, set_flag )

        # Return the output derivative order.
        return output_derivative_order


    #%% ------------------------------------------------------------ SETUP FUNCTIONS ------------------------------------------------------------

    # Implement a function to setup for data shuffling.
    def setup_data_shuffling( self, input_data = None, output_data = None, shuffle_indexes = None ):

        # Preprocess the input data.
        input_data = self.preprocess_input_data( input_data )

        # Compute the number of data points associated with the input data.
        num_data_points = torch.tensor( input_data.shape[ 0 ], dtype = torch.int32, device = self.device )

        # Preprocess the output data.
        output_data = self.preprocess_output_data( output_data )

        # Compute the number of output sources.
        num_output_sources = self.compute_num_output_sources( output_data )

        # Preprocess the shuffle indexes.
        shuffle_indexes = self.preprocess_shuffle_indexes( shuffle_indexes, num_data_points )

        # Return the shuffling data.
        return input_data, output_data, num_output_sources, shuffle_indexes


    # Implement a function to setup for batch computing.
    def setup_batch_computing( self, input_data = None, output_data = None, batch_number = None, batch_size = None ):

        # Preprocess the input data.
        input_data = self.preprocess_input_data( input_data )

        # Compute the number of data points associated with the input data.
        num_data_points = torch.tensor( input_data.shape[ 0 ], dtype = torch.int32, device = self.device )

        # Preprocess the output data.
        output_data = self.preprocess_output_data( output_data )

        # Compute the number of output sources.
        num_output_sources = self.compute_num_output_sources( output_data )

        # Preprocess the batch number.
        batch_number = self.preprocess_batch_number( batch_number )

        # Preprocess the batch size.
        batch_size = self.preprocess_batch_size( batch_size )

        # Return the batch computation data.
        return input_data, num_data_points, output_data, num_output_sources, batch_number, batch_size


    #%% ------------------------------------------------------------ COMPUTE FUNCTIONS ------------------------------------------------------------

    # Implement a function to set the number of output sources.
    def compute_num_output_sources( self, output_data = None, set_flag = False ):

        # Preprocess the output data.
        output_data = self.preprocess_output_data( output_data )

        # Determine how to compute the number of output sources.
        if torch.is_tensor( output_data ):                              # If the output data is a tensor...

            # Set the number of output sources to one.
            num_output_sources = torch.tensor( 1, dtype = torch.uint8, device = self.device )

        elif isinstance( output_data, list ):                           # If the output data is a list...

            # Set the number of output sources to be the length of the list.
            num_output_sources = torch.tensor( len( output_data ), dtype = torch.uint8, device = self.device )

        elif output_data is None:                                       # If the output data is None...

            # Set the number of output sources to be zero.
            num_output_sources = torch.tensor( 0, dtype = torch.uint8, device = self.device )

        else:                                                           # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid output data: {output_data}' )

        # Set the number of output sources (as required).
        self.set_num_output_sources( num_output_sources, set_flag )

        # Return the number of output sources.
        return num_output_sources


    #%% ------------------------------------------------------------ BASIC FUNCTIONS ------------------------------------------------------------

    # Implement a function to ensure that a tensor is two dimensional.
    def augment_input_output_data_tensor( self, data ):

        # Determine whether to unsqueeze the first dimension.
        if data.dim(  ) == 0:                                     # If the data is zero dimensional...

            # Unsqueeze the first dimension.
            data.unsqueeze_( 1 )

        # Determine whether to unsqueeze the second dimension.
        if data.dim(  ) == 1:                                     # If the data is one dimensional...
            
            # Unsqueeze the first dimension.
            data.unsqueeze_( 2 )

        # Return the data.
        return data


    # Implement a function to ensure that a tensor or list of tensors is two dimensional.
    def augment_input_output_data( self, data ):

        # Determine whether to treat the data as a list or as a tensor.
        if isinstance( data, list ):                                # If the data is a list...

            # Retrieve the number of list entires.
            num_entries = len( data )

            # Augment each of the list entries.
            for k in range( num_entries ):                          # Iterate through each of the list entries...

                # Augment this list entry.
                data[ k ] = self.augment_input_output_data_tensor( data[ k ] )

        elif torch.is_tensor( data ):                               # If the data is a tensor...

            # Augment this tensor.
            data = self.augment_input_output_data_tensor( data )

        elif data is None:                                          # If the data is None...

            # Do nothing.
            pass

        else:                                                       # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid data: {data}' )

        # Return the augmented data.
        return data


    #%% ------------------------------------------------------------ SHUFFLE FUNCTIONS ------------------------------------------------------------

    # Implement a function to shuffle the input and output data.
    def shuffle_data( self, input_data = None, output_data = None, shuffle_indexes = None, set_flag = False ):

        # Setup for data shuffling.
        input_data, output_data, num_output_sources, shuffle_indexes = self.setup_data_shuffling( input_data, output_data, shuffle_indexes )

        # Shuffle the input data.
        input_data = input_data[ shuffle_indexes, ... ]

        # Determine whether there is output data to shuffle.
        if output_data:                        # If the output data list is not empty...

            # Shuffle the output data.
            output_data = [ output_data[ k ][ shuffle_indexes, ... ] for k in range( num_output_sources ) ]

        # Set the input and output data (as required).
        self.set_input_data( input_data, set_flag )
        self.set_output_data( output_data, set_flag )

        # Return the shuffled data.
        return input_data, output_data, shuffle_indexes


    #%% ------------------------------------------------------------ PLOTTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to generate a figure title.
    def generate_figure_title( self, num_inputs, num_outputs, source_index ):

        # Determine how to set the title string based on the number of inputs.
        if ( num_inputs == 1 ):                                                                                     # If the number of inputs is one...

            # Determine how to set the title string based on the number of outputs.
            if ( num_outputs == 0 ):                                                                                # If the number of outputs is zero...

                # Set the title string.
                title_string = f'{self.name}: Input Data'

            elif ( num_outputs == 1 ) or ( num_outputs == 2 ) or ( num_outputs == 3 ):                              # If the number of outputs is one, two, or three...

                # Set the title string.
                title_string = f'{self.name} (Output Source {source_index + 1}): Output Data vs Input Data'

            else:                                                                                                   # Otherwise...

                # Set the title string.
                title_string = ''

        elif ( num_inputs == 2 ):                                                                                   # If the number of inputs is two...

            # Determine how to set the title string based on the number of outputs.
            if ( num_outputs == 0 ):                                                                                # If the number of outputs is zero...

                # Set the title string.
                title_string = f'{self.name}: Input Data'

            elif ( num_outputs == 1 ) or ( num_outputs == 2 ) or ( num_outputs == 3 ):                              # If the number of outputs is one, two, or three...

                # Set the title string.
                title_string = f'{self.name} (Output Source {source_index + 1}): Output Data vs Input Data'

            else:                                                                                                   # Otherwise...

                # Set the title string.
                title_string = ''

        elif ( num_inputs == 3 ):                                                                                   # If the number of inputs is three...

            # Determine how to set the title string based on the number of outputs.
            if ( num_outputs == 0 ):                                                                                # If the number of outputs is zero...

                # Set the title string.
                title_string = f'{self.name}: Input Data'

            elif ( num_outputs == 1 ):                                                                              # If the number of outputs is one...

                # Set the title string.
                title_string = f'{self.name} (Output Source {source_index + 1}): Output Data vs Input Data'

            elif ( num_outputs == 2 ) or ( num_outputs == 3 ):                                                      # If the number of outputs is two or three...

                # Set the title string.
                title_string = f'{self.name} ( Output Source {source_index + 1} ),'

            else:                                                                                                   # Otherwise...

                # Set the title string.
                title_string = ''
        
        else:                                                                                                       # Otherwise...
            
            # Set the title string.
            title_string = ''
        
        # Return the title string.
        return title_string


    #%% ------------------------------------------------------------ BATCH FUNCTIONS ------------------------------------------------------------

    # Implement a function to saturate the upper batch index.
    def saturate_upper_batch_index( self, upper_batch_index, num_data_points = None ):

        # Preprocess the number of data points.
        num_data_points = self.preprocess_num_data_points( num_data_points )

        # Ensure that the upper batch index is valid.
        if upper_batch_index > num_data_points:                                           # If the upper batch index is greater than the quantity of training data...

            # Set the upper batch index to be the total quantity of training data.
            upper_batch_index = num_data_points

        # Return the upper batch index.
        return upper_batch_index


    # Implement a function to stage a batch of the data.
    def compute_batch_data( self, input_data = None, output_data = None, batch_number = None, batch_size = None, set_flag = False ):

        # Setup for batch computation.
        input_data, num_data_points, output_data, num_output_sources, batch_number, batch_size = self.setup_batch_computing( input_data, output_data, batch_number, batch_size )

        # Determine how to stage the batch.
        if batch_size is not None:                          # If the batch size is not None...

            # Compute the batch indexes.
            lower_batch_index = batch_number*batch_size
            upper_batch_index = ( batch_number + 1 )*batch_size

            # Saturate the upper batch index.
            upper_batch_index = self.saturate_upper_batch_index( upper_batch_index, num_data_points )

            # Retrieve a batch of the input data.
            input_data_batch = input_data[ lower_batch_index:upper_batch_index, ... ]

            # Determine whether to retrieve a batch of the output data.
            if isinstance( output_data, list ) and output_data:                           # If the output data is a non-empty list...

                # Retrieve a batch of the output data from the list.
                output_data_batch = [ output_data[ k ][ lower_batch_index:upper_batch_index, ... ] for k in range( num_output_sources ) ]

            elif torch.is_tensor( output_data ) and ( output_data.numel(  ) != 0 ):       # If the output data is a non-empty tensor...

                # Retrieve a batch of the output data from the tensor.
                output_data_batch = output_data[ lower_batch_index:upper_batch_index, ... ]

            elif output_data is None:                                                          # If the output data is None...

                # Set the output data batch to be None.
                output_data_batch = None

            else:                                       # Otherwise...

                # Throw an error.
                raise ValueError( f'Invalid output data: {self.output_data}' )

        else:                                                                                       # Otherwise...

            # Stage all of the data.
            input_data_batch = input_data
            output_data_batch = output_data

        # Set the input and output data batch (as required).
        self.set_input_data( input_data_batch, set_flag )
        self.set_output_data( output_data_batch, set_flag )

        # Return the input and output data batches.
        return input_data_batch, output_data_batch


    #%% ------------------------------------------------------------ PRINTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to print the residual data information.
    def print( self, num_dashes = 20, decoration_flag = True ):

        # Print a header.
        self.printing_utilities.print_header( f'{self.condition_type.upper(  )} CONDITION DATA SUMMARY', num_dashes, decoration_flag )

        # Print the identification information.
        print( 'Basic Information' )
        print( f'Device: {self.device}' )
        print( f'ID: {self.id}' )
        print( f'Name: {self.name}' )
        print( f'Dimension Labels: {self.dimension_labels}' )
        print( f'Batch size: {self.batch_size}' )
        print( '\n' )

        # Print the general information.
        print( f'General Information' )
        print( f'Condition Application Dimension: {self.dimension}' )
        print( f'Total # Data Points: {self.num_data_points}' )
        print( f'Batch # Data Points: {self.input_data_batch.shape[ 0 ]}' )
        print( '\n' )

        # Print the input information.
        print( 'Input Information' )
        print( f'# of Input Dimensions: {self.input_data.shape[ 1 ]} ' )
        print( f'Input Dimension Labels: {self.dimension_labels}' )
        print( f'Input Data: {self.input_data}' )
        print( f'Input Data Batch: {self.input_data_batch}' )
        print( '\n' )

        # Print the output information.
        print( 'Output Information' )
        print( f'# of Output Dimensions: {self.output_data.shape[ 1 ]} ' )
        print( f'Output Data: {self.output_data}' )
        print( f'Output Data Batch: {self.output_data_batch}' )
        print( f'Output Derivative Order: {self.output_derivative_order}' )

        # Print a footer.
        self.printing_utilities.print_footer( num_dashes, decoration_flag )


    #%% ------------------------------------------------------------ PLOTTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to plot the input-output data.
    def plot( self, input_data = None, output_data = None, dimension_labels = None, projection_dimensions = None, projection_values = None, level = 0, fig = None, plot_type = 'all', save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Preprocess the input and output data.
        input_data, output_data = self.preprocess_input_output_data( input_data, output_data, plot_type )

        # Preprocess the dimension labels.
        dimension_labels = self.preprocess_dimension_labels( dimension_labels )

        # Compute the number of output sources.
        num_output_sources = self.compute_num_output_sources( output_data )

        # Compute the input axis labels from the dimension labels.
        input_labels = self.plotting_utilities.dimension_labels2axis_labels( dimension_labels )

        # Create empty lists to store the output figures and axes.
        figs = [  ]
        axes = [  ]

        # Plot the input and output data.
        for k in range( num_output_sources ):                      # Iterate through each of the output sources...

            # Generate the title string.
            title_string = self.generate_figure_title( input_data.shape[ -1 ], output_data[ k ].shape[ -1 ], k )

            # Plot the input and output data associated with this output data source.
            fig, ax = self.plotting_utilities.plot( input_data, output_data[ k ], projection_dimensions, projection_values, level, fig, input_labels, title_string, save_directory, as_surface, as_stream, as_contour, show_plot )

            # Append the current figure and axes to the figure and axes list.
            figs.append( fig )
            axes.append( ax )

        # Return the figure and axis.
        return figs, axes

