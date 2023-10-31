####################################################################################### 
# THIS SOURCE CODE IS PROPERTY OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. 
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND 
# CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION 
# ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN 
# LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE,
# CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL 
# PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641. 
####################################################################################### 


#%% ------------------------------------------------------------ RESIDUAL DATA CLASS ------------------------------------------------------------

# This file implements a class for storing and managing residual data information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Import custom libraries.
from plotting_utilities_class import plotting_utilities_class as plotting_utilities_class
from pinn_data_class import pinn_data_class as pinn_data_class


#%% ------------------------------------------------------------ RESIDUAL DATA CLASS ------------------------------------------------------------

# Implement the residual data class.
class residual_data_class( pinn_data_class ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self, id, name, dimension_labels, input_data, batch_size = None, device = 'cpu' ):

        # Construct the parent class.
        super(  ).__init__( id, name, dimension_labels, batch_size, device )

        # Set the pinn input data.
        self.input_data = self.validate_input_data( input_data )
        self.num_data_points = torch.tensor( self.input_data.shape[ 0 ], dtype = torch.int32, device = self.device )

        # Stage a batch of initial-boundary data.
        self.input_data_batch = self.compute_batch_data( self.input_data, batch_number = torch.tensor( 0, dtype = torch.uint8, device = self.device ), batch_size = self.batch_size )


    #%% ------------------------------------------------------------ PREPROCESS FUNCTIONS ------------------------------------------------------------

    # Implement a function to preprocess the input data.
    def preprocess_input_data( self, input_data = None ):

        # Determine whether to use the stored input data.
        if input_data is None:                              # If the input data was not provided...

            # Set the input data to be the stored value.
            input_data = self.input_data

        # Return the input data.
        return input_data


    # Implement a function to preprocess the number of data points.
    def preprocess_num_data_points( self, num_data_points = None ):

        # Determine whether to use the stored number of data points.
        if num_data_points is None:                                 # If the number of data points was not provided...

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


    # Implement a function to preprocess the shuffle indexes.
    def preprocess_shuffle_indexes( self, shuffle_indexes = None, num_data_points = None ):

        # Preprocess the number of data points.
        num_data_points = self.preprocess_num_data_points( num_data_points )

        # Determine whether to generate shuffle shuffle_indexes.
        if shuffle_indexes is None:                     # If shuffle shuffle_indexes where not provided...

            # Generate shuffled shuffle_indexes.
            shuffle_indexes = torch.randperm( num_data_points, dtype = torch.int64, device = self.device )

        # Return the shuffle indexes.
        return shuffle_indexes


    # Implement a function to preprocess the plotting data.
    def preprocess_plotting_data( self, plotting_data = None, plot_type = 'all' ):

        # Determine whether to use the stored input and output data.
        if ( plotting_data is None ):              # If the plotting data was not provided...

            # Retrieve the plotting data based on the plot type (all vs batch).
            plotting_data = self.get_plotting_data( plot_type )

        # Return the plotting data.
        return plotting_data


    # Implement a function to preprocess dimension labels.
    def preprocess_dimension_labels( self, dimension_labels = None ):

        # Determine whether to use the stored dimension labels.
        if dimension_labels is None:                # If the dimension labels were not provided...

            # Set the dimension labels to be the stored value.
            dimension_labels = self.dimension_labels

        # Return the dimension labels.
        return dimension_labels


    # Implement a function to preprocess the name.
    def preprocess_name( self, name = None ):

        # Determine whether to use the stored dimension labels.
        if name is None:                        # If the name was not provided...

            # Set the name to be the stored value.
            name = self.name

        # Return the name.
        return name


    #%% ------------------------------------------------------------ SETUP FUNCTIONS ------------------------------------------------------------

    # Implement a function to setup for plotting.
    def setup_plotting( self, plotting_data = None, plot_type = 'all', dimension_labels = None, name = None ):

        # Implement a function to preprocess the input data.
        plotting_data = self.preprocess_plotting_data( plotting_data, plot_type )

        # Implement a function to preprocess the dimension labels.
        dimension_labels = self.preprocess_dimension_labels( dimension_labels )
        
        # Implement a function to preprocess the name.
        name = self.preprocess_name( name )

        # Return the information required for plotting.
        return plotting_data, dimension_labels, name


    #%% ------------------------------------------------------------ IS VALID FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate the input data.
    def is_input_data_valid( self, input_data ):

        # Determine whether the input data is valid.
        if torch.is_tensor( input_data ) and ( input_data.numel(  ) != 0 ) and ( input_data.dim(  ) <= 2 ):                         # If the input data is a non-empty torch tensor...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                                                                                       # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    #%% ------------------------------------------------------------ GET FUNCTIONS ------------------------------------------------------------

    # Implement a function to get plotting data.
    def get_plotting_data( self, plot_type = 'all' ):

        # Retrieve the data to plot.
        if plot_type.lower(  ) == 'all':                             # If we want to plot all of the data in this data set...

            # Set the input and output plotting data.
            input_plotting_data = self.input_data

        elif plot_type.lower(  ) == 'batch':                         # If we want to plot the batch data in this data set...

            # Set the input and output plotting data.
            input_plotting_data = self.input_data_batch

        else:                                                       # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid plotting type: {plot_type}' )

        # Return the plotting data.
        return input_plotting_data


    #%% ------------------------------------------------------------ SET FUNCTIONS ------------------------------------------------------------

    # Implement a function to set the input data.
    def set_input_data( self, input_data, set_flag = True ):

        # Determine whether to set the input data.
        if set_flag:                # If we want to set the input data...

            # Set the input data.
            self.input_data = input_data


    # Implement a function to set the input data batch.
    def set_input_data_batch( self, input_data_batch, set_flag = True ):

        # Determine whether to set the input data batch.
        if set_flag:            # If we want to set the input data batch...

            # Set the input data batch.
            self.input_data_batch = input_data_batch


    #%% ------------------------------------------------------------ VALIDATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to set the input data.
    def validate_input_data( self, input_data, set_flag = False ):

        # Determine whether to set the input data.
        if self.is_input_data_valid( input_data ):                          # If the input data is valid...

            # Set the input data.
            input_data = self.augment_input_data_tensor( input_data )

        else:                                                   # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid input data: {input_data}' )

        # Set the input data (as required).
        self.set_input_data( input_data, set_flag)

        # Return the stored input data.
        return input_data


    #%% ------------------------------------------------------------ SETUP FUNCTIONS ------------------------------------------------------------

    # Implement a function to setup the batch data.
    def setup_batch_data( self, input_data = None, batch_number = None, batch_size = None ):

        # Preprocess the input data.
        input_data = self.preprocess_input_data( input_data )

        # Set the number of data points to be that associated with the given input data.
        num_data_points = input_data.shape[ 0 ]

        # Preprocess the batch number.
        batch_number = self.preprocess_batch_number( batch_number )

        # Preprocess the batch size.
        batch_size = self.preprocess_batch_size( batch_size )

        # Return the batch data.
        return input_data, num_data_points, batch_number, batch_size


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
    def compute_batch_data( self, input_data = None, batch_number = None, batch_size = None, set_flag = False ):

        # Setup the batch data.
        input_data, num_data_points, batch_number, batch_size = self.setup_batch_data( input_data, batch_number, batch_size )

        # Determine how to stage the batch.
        if batch_size is not None:                          # If the batch size is not None...

            # Compute the batch indexes.
            lower_batch_index = ( batch_number*batch_size ).to( batch_size.dtype )
            upper_batch_index = ( ( batch_number + 1 )*batch_size ).to( batch_size.dtype )

            # Saturate the upper batch index.
            upper_batch_index = self.saturate_upper_batch_index( upper_batch_index, num_data_points )

            # Retrieve a batch of the input data.
            input_data_batch = input_data[ lower_batch_index:upper_batch_index, ... ]

        else:                                                                                       # Otherwise...

            # Stage all of the data.
            input_data_batch = input_data

        # Set the input data batch (as required).
        self.set_input_data_batch( input_data_batch, set_flag )

        # Return the input data batch.
        return input_data_batch


    #%% ------------------------------------------------------------ BASIC FUNCTIONS ------------------------------------------------------------

    # Implement a function to ensure that a tensor is two dimensional.
    def augment_input_data_tensor( self, data ):

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


    #%% ------------------------------------------------------------ SHUFFLE FUNCTIONS ------------------------------------------------------------

    # Implement a function to shuffle the input data.
    def shuffle_data( self, input_data = None, shuffle_indexes = None, set_flag = False ):

        # Preprocess the input data.
        input_data = self.preprocess_input_data( input_data )

        # Set the number of data points to be that associated with the given input data.
        num_data_points = input_data.shape[ 0 ]

        # Preprocess the shuffle indexes.
        shuffle_indexes = self.preprocess_shuffle_indexes( shuffle_indexes, num_data_points )

        # Shuffle the input data.
        input_data = input_data[ shuffle_indexes, ... ]

        # Set the input data (as required).
        self.set_input_data( input_data, set_flag )

        # Return the shuffled data.
        return input_data, shuffle_indexes
        

    #%% ------------------------------------------------------------ PRINTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to print the residual data information.
    def print( self, num_dashes = 20, decoration_flag = True ):

        # Print a header.
        self.printing_utilities.print_header( 'RESIDUAL DATA SUMMARY', num_dashes, decoration_flag )

        # Print the identification information.
        print( 'General Information' )
        print( f'Device: {self.device}' )
        print( f'ID: {self.id}' )
        print( f'Name: {self.name}' )
        print( f'Dimension Labels: {self.dimension_labels}' )
        print( f'Batch size: {self.batch_size}' )
        print( '\n' )

        # Print the residual data.
        print( 'Data Information' )
        print( f'# Data Points: {self.num_data_points}' )
        print( f'Input Data: {self.input_data}' )
        print( f'Input Data Batch: {self.input_data_batch}' )

        # Print a footer.
        self.printing_utilities.print_footer( num_dashes, decoration_flag )


    #%% ------------------------------------------------------------ PLOTTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to plot the input data.
    def plot( self, input_data = None, dimension_labels = None, projection_dimensions = None, projection_values = None, level = 0, name = None, fig = None, plot_type = 'all', save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Setup for plotting.
        input_data, dimension_labels, name = self.setup_plotting( input_data, plot_type, dimension_labels, name )

        # Compute the input axis labels from the dimension labels.
        input_labels = self.plotting_utilities.dimension_labels2axis_labels( dimension_labels )

        # Compute the title string.
        title_string = f'{name}: Input Data'

        # Plot the input and output data associated with this output data source.
        fig, ax = self.plotting_utilities.plot( input_data, [  ], projection_dimensions, projection_values, level, fig, input_labels, title_string, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figure and axis.
        return fig, ax
