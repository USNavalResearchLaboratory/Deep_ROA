####################################################################################### 
# THIS SOURCE CODE IS PROPERTY OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. 
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND 
# CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION 
# ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN 
# LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE,
# CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL 
# PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641. 
####################################################################################### 


#%% ------------------------------------------------------------ PLOTTING UTILITIES CLASS ------------------------------------------------------------

# This file implements a class for storing and managing plotting utilities information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch
import matplotlib.pyplot as plt
import warnings

# Import custom libraries.
from tensor_utilities_class import tensor_utilities_class as tensor_utilities_class
from function_utilities_class import function_utilities_class as function_utilities_class


#%% ------------------------------------------------------------ SAVE-LOAD UTILITIES CLASS ------------------------------------------------------------

# Implement the plotting utilities class.
class plotting_utilities_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self ):

        # Create an instance of the tensor utilities class.
        self.tensor_utilities = tensor_utilities_class(  )

        # Create an instance of the function utilities class.
        self.function_utilities = function_utilities_class(  )


    #%% ------------------------------------------------------------ VALIDATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate plotting data.
    def validate_data( self, data, recourse = 'ignore' ):

        # Determine whether the data is valid.
        if torch.is_tensor( data ) and ( data.numel(  ) != 0 ):                 # If the data is valid...

            # Set the valid flag to true.
            valid_flag = True

        elif ( recourse.lower(  ) == 'ignore' ):                                # If the recourse option is set to 'ignore'...

            # Set the valid flag to false.
            valid_flag = False

        elif ( recourse.lower(  ) == 'warning' ):                               # If the recourse option is set to 'warning'...

            # Set the valid flag to false.
            valid_flag = False

            # Throw an warning.
            warnings.warn( f'Invalid plotting data detected: {data}' )

        elif ( recourse.lower(  ) == 'error' ):                                 # If the recourse option is set to 'error'...

            # Throw an error.
            raise ValueError( f'Invalid plotting data detected: {data}' )

        else:                                                                   # Otherwise... (i.e., invalid data and recourse option not recognized...)

            # Throw an error.
            raise ValueError( f'Invalid recourse option: {recourse}' )

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the compatibility of input and output data.
    def validate_data_compatibility( self, input_data, output_data ):

        # Determine whether the input and output data is gridded.
        input_gridded_flag = self.tensor_utilities.is_grid_flat( input_data )
        output_gridded_flag = self.tensor_utilities.is_grid_flat( output_data )

        # Determine whether the input and output data are compatible.
        if ( input_gridded_flag and output_gridded_flag ) or ( ~input_gridded_flag and ~output_gridded_flag ):                  # If the input and output data are either both gridded or both nongridded...

            # Set the input and output data compatibility flag to true.
            compatibility_flag = True

        else:                                                                                                                   # Otherwise... (i.e., the input and output data are not compatible...)

            # Set the input and output data compatibility flag to false.
            compatibility_flag = False

        # Return the compatibility flag.
        return compatibility_flag


    #%% ------------------------------------------------------------ CONVERSION FUNCTIONS ------------------------------------------------------------

    # Implement a function to retrieve the number of temporal and spatial dimensions.
    def dimension_labels2num_spatiotemporal_dimensions( self, dimension_labels ):

        # Compute the number of temporal dimensions.
        num_temporal_dimensions = sum( [ label.lower(  ) == 't' for label in dimension_labels ] )

        # Compute the number of spatial dimensions.
        num_spatial_dimensions = sum( [ label.lower(  ) == 'x' for label in dimension_labels ] )

        # Compute the number of spatiotemporal dimensions.
        num_spatiotemporal_dimensions = num_temporal_dimensions + num_spatial_dimensions

        # Return the number of spatiotemporal dimensions.
        return num_temporal_dimensions, num_spatial_dimensions, num_spatiotemporal_dimensions


    # Implement a function to convert dimension labels to axis labels.
    def dimension_labels2axis_labels( self, dimension_labels ):

        # Retrieve the number of spatiotemporal dimensions.
        _, _, num_spatiotemporal_dimensions = self.dimension_labels2num_spatiotemporal_dimensions( dimension_labels )

        # Initialize a list of axis labels.
        axis_labels = [  ]

        # Initialize two counter variables.
        t_counter = 0
        x_counter = 0

        # Create the axis labels.
        for k in range( num_spatiotemporal_dimensions ):                # Iterate through each of the spatiotemporal dimensions...

            # Retrieve the label associated with this dimension

            # Determine how to create this axis label.
            if dimension_labels[ k ].lower(  ) == 't':                  # If the dimension label is a 't'...

                # Advance the t counter.
                t_counter += 1

                # Set the variable number equal to the t counter.
                var_num = t_counter

            elif dimension_labels[ k ].lower(  ) == 'x':                # If the dimension label is a 'x'...

                # Advance the x counter.
                x_counter += 1

                # Set the variable number equal to the x counter.
                var_num = x_counter

            else:                                                       # Otherwise...

                # Throw an error.
                raise ValueError( f'Invalid dimension label: {dimension_labels[ k ]}' )

            # Append this axis label to the list of axis labels.
            axis_labels.append( f'Input Data: {dimension_labels[ k ]}{var_num}' )

        # Return the axis labels.
        return axis_labels


    #%% ------------------------------------------------------------ PRE-PROCESSING FUNCTIONS ------------------------------------------------------------

    # Implement a function to pre-process data.
    def preprocess_data( self, data, flatten_flag = True ):

        # Ensure that the data is valid before pre-processing.
        assert self.validate_data( data )

        # Determine whether it is necessary to flatten the data grid.
        if flatten_flag and ~self.tensor_utilities.is_grid_flat( data ):                    # If the data is gridded and we want to flatten the data...

            # Flatten the data grid.
            data = self.tensor_utilities.flatten_grid( data )

        # Return the data.
        return data


    # Implement a function to pre-process the input and output data.
    def preprocess_input_output_data( self, input_data, output_data, flatten_flag = True ):

        # Ensure that both the input and output data are valid before pre-preprocessing.
        assert self.validate_data( input_data )
        assert self.validate_data( output_data )

        # Ensure that the input and output data are compatible.
        assert self.validate_data_compatibility( input_data, output_data )

        # Pre-process the input and output data.
        input_data = self.preprocess_data( input_data, flatten_flag )
        output_data = self.preprocess_data( output_data, flatten_flag )

        # Return the input and output data.
        return input_data, output_data

    
    # Implement a function to process tensor for plotting.
    def plot_process( self, data ):

        # Return the tensor data processed for plotting.
        return data.detach(  ).cpu(  ).numpy(  )


    # Implement a function to compute the number of subplot rows and columns.
    def get_subplot_rc_nums( self, num_plots, more_rows_flag = True ):

        # Compute the square root of the number of plots.
        square_root = torch.sqrt( num_plots )

        # Determine how to compute the number of rows and columns.
        if ( int( square_root ) == square_root ):           # If creating a square plot is possible...

            # Compute the number of rows and columns.
            num_rows = int( square_root )
            num_cols = int( square_root )

        else:

            # Compute the number of columns to use.
            num_cols = int( torch.floor( torch.log2( num_plots ) ) )

            # Compute the number of rows to use.
            num_rows = int( torch.ceil( num_plots/num_cols ) )

            # Determine whether we need to swap the rows and columns.
            if ( more_rows_flag & ( num_rows < num_cols ) ):

                # Swap the rows and columns.
                num_rows, num_cols = num_cols, num_rows

        # Return the number of rows and number of columns.
        return num_rows, num_cols


    #%% ------------------------------------------------------------ STANDARD PLOTTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to plot 1in 0out data.
    def plot_1in_0out_data( self, input_data, fig = None, input_labels = [ 'in1' ], title_string = '1i0o Data Plot', save_directory = r'.', show_plot = False ):

        # Ensure that the input data is valid.
        assert self.validate_data( input_data )

        # Determine whether to set the input labels to a default value.
        if input_labels is None:                    # If the input labels variable is None...

            # Set the input labels to be a default value.
            input_labels = [ 'in1' ]

        # Determine whether to create a new figure.
        if fig is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input data.
            fig = plt.figure(  ); plt.xlabel( input_labels[ 0 ] ), plt.ylabel( f'Dummy Axis [-]' ), plt.title( title_string )

        else:                               # Otherwise... ( i.e., a figure was provided... )

            # Set the given figure to be active.
            plt.figure( fig.number )

        # Retrieve the current axes.
        ax = plt.gca(  )

        # Plot the input data.
        plt.plot( self.plot_process( input_data ), self.plot_process( torch.zeros_like( input_data ) ) )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Determine whether to show the plot.
        if show_plot:                      # If we want to show the figure...

            # Show the figure.
            plt.show( block = False )

        # Return the figure and axis.
        return fig, ax
        

    # Implement a function to plot 1in 1out data.
    def plot_1in_1out_data( self, input_data, output_data, fig = None, input_labels = [ 'in1' ], title_string = '1i1o Data Plot', save_directory = r'.', show_plot = False, line_style = '-' ):

        # Ensure that the input and output data is valid.
        assert self.validate_data( input_data )
        assert self.validate_data( output_data )

        # Determine whether to set the input labels to a default value.
        if input_labels is None:                    # If the input labels variable is None...

            # Set the input labels to be a default value.
            input_labels = [ 'in1' ]

        # Determine whether to create a new figure.
        if fig is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input and output data.
            fig = plt.figure(  ); plt.xlabel( input_labels[ 0 ] ), plt.ylabel( f'Output Data [-]' ), plt.title( title_string )

        else:                               # Otherwise... ( i.e., a figure was provided... )

            # Set the given figure to be active.
            plt.figure( fig.number )

        # Retrieve the current axes.
        ax = plt.gca(  )

        # Plot the input and output data.
        plt.plot( self.plot_process( input_data ), self.plot_process( output_data ), line_style )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Determine whether to show the plot.
        if show_plot:                      # If we want to show the figure...

            # Show the figure.
            plt.show( block = False )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot 1in 2out data.
    def plot_1in_2out_data( self, input_data, output_data, fig = None, input_labels = [ 'in1' ], title_string = '1i2o Data Plot', save_directory = r'.', show_plot = False ):

        # Ensure that the input and output data is valid.
        assert self.validate_data( input_data )
        assert self.validate_data( output_data )

        # Determine whether to set the input labels to a default value.
        if input_labels is None:                    # If the input labels variable is None...

            # Set the input labels to be a default value.
            input_labels = [ 'in1' ]

        # Determine whether to create a new figure.
        if fig is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input and output data.
            fig = plt.figure(  ); plt.xlabel( input_labels[ 0 ] + r' | Output Data: Dim1 [-]' ), plt.ylabel( r'Output Data: Dim2 [-]' ), plt.title( title_string )

        else:                               # Otherwise... ( i.e., a figure was provided... )

            # Set the given figure to be active.
            plt.figure( fig.number )

        # Retrieve the current axes.
        ax = plt.gca(  )

        # Create a quiver plot of the input and output data.
        plt.quiver( self.plot_process( input_data ), self.plot_process( torch.zeros_like( input_data ) ), self.plot_process( output_data[ :, 0 ] ), self.plot_process( output_data[ :, 1 ] ) )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Determine whether to show the plot.
        if show_plot:                      # If we want to show the figure...

            # Show the figure.
            plt.show( block = False )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot 1in 3out data.
    def plot_1in_3out_data( self, input_data, output_data, fig = None, input_labels = [ 'in1' ], title_string = '1i3o Data Plot', save_directory = r'.', show_plot = False ):

        # Ensure that the input and output data is valid.
        assert self.validate_data( input_data )
        assert self.validate_data( output_data )

        # Determine whether to set the input labels to a default value.
        if input_labels is None:                    # If the input labels variable is None...

            # Set the input labels to be a default value.
            input_labels = [ 'in1' ]

        # Determine whether to create a new figure.
        if fig is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input and output data.
            fig, ax = plt.subplots( subplot_kw = { 'projection': '3d' } ); ax.set_xlabel( input_labels[ 0 ] + r' | Output Data: Dim1 [-]' ), ax.set_ylabel( r'Output Data: Dim2 [-]' ), ax.set_zlabel( r'Output Data: Dim3 [-]' ), ax.set_title( title_string )

        else:                               # Otherwise... ( i.e., a figure was provided... )

            # Set the given figure to be active.
            plt.figure( fig.number )

            # Retrieve the current axes.
            ax = plt.gca(  )

        # Create a quiver plot of the input and output data.
        ax.quiver( self.plot_process( input_data ), self.plot_process( torch.zeros_like( input_data ) ), self.plot_process( torch.zeros_like( input_data ) ), self.plot_process( output_data[ :, 0 ] ), self.plot_process( output_data[ :, 1 ] ), self.plot_process( output_data[ :, 2 ] ) )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Determine whether to show the plot.
        if show_plot:                      # If we want to show the figure...

            # Show the figure.
            plt.show( block = False )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot 2in 0out data.
    def plot_2in_0out_data( self, input_data, fig = None, input_labels = [ 'in1', 'in2' ], title_string = '2i0o Data Plot', save_directory = r'.', show_plot = False ):

        # Determine whether to set the input labels to a default value.
        if input_labels is None:                    # If the input labels variable is None...

            # Set the input labels to be a default value.
            input_labels = [ 'in1', 'in2' ]

        # Preprocess the input data.
        input_data = self.preprocess_data( input_data )

        # Determine whether to create a new figure.
        if fig is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input data.
            fig = plt.figure(  ); plt.xlabel( input_labels[ 0 ] ), plt.ylabel( input_labels[ 1 ] ), plt.title( title_string )

        else:                               # Otherwise... ( i.e., a figure was provided... )

            # Set the given figure to be active.
            plt.figure( fig.number )

        # Retrieve the current axes.
        ax = plt.gca(  )

        # Plot the input and output data.
        plt.plot( self.plot_process( input_data[ :, 0 ] ), self.plot_process( input_data[ :, 1 ] ) )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Determine whether to show the plot.
        if show_plot:                      # If we want to show the figure...

            # Show the figure.
            plt.show( block = False )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot 2in 1out data.
    def plot_2in_1out_data( self, input_data, output_data, level = 0, fig = None, input_labels = [ 'in1', 'in2' ], title_string = '2i1o Data Plot', save_directory = r'.', as_surface = True, as_contour = True, show_plot = False ):

        # Determine whether to set the input labels to a default value.
        if input_labels is None:                    # If the input labels variable is None...

            # Set the input labels to be a default value.
            input_labels = [ 'in1', 'in2' ]

        # Determine whether to create a new figure.
        if fig is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input and output data.
            fig, ax = plt.subplots( subplot_kw = { 'projection': '3d' } ); ax.set_xlabel( input_labels[ 0 ] ), ax.set_ylabel( input_labels[ 1 ] ), ax.set_zlabel( 'Output Data [-]' ), ax.set_title( title_string )

        else:                               # Otherwise... ( i.e., A figure was provided... )
          
            # Determine whether the provided fig is embedded in a list.
            if isinstance( fig, list ):                 # If the figure is embedded in a list...

                # Unembed the figure from the list.
                fig = fig[ 0 ]

            # Set the given figure to be active.
            plt.figure( fig.number )

            # Retrieve the current axes.
            ax = plt.gca(  )

        # Determine how to plot the 2in 1out data.
        if as_surface:                      # If we want to plot the data as a surface...

            # Preprocess the input and output data.
            input_data, output_data = self.preprocess_input_output_data( input_data, output_data, flatten_flag = False )

            # Plot the data as a surface.
            ax.plot_surface( self.plot_process( input_data[ ..., 0 ] ), self.plot_process( input_data[ ..., 1 ] ), self.plot_process( output_data[ ..., 0 ] ) )

            # Determine whether to plot the contour.
            if as_contour:                      # If we want to plot a contour...

                # fig, ax = plt.subplots( subplot_kw = { 'projection': '3d' } )

                # Plot the given level set.
                ax.contour( self.plot_process( input_data[ ..., 0 ] ), self.plot_process( input_data[ ..., 1 ] ), self.plot_process( output_data[ ..., 0 ] ), levels = [ self.plot_process( level ) ], colors = 'red', linewidths = 2.0 )

        elif as_contour:                    # If we want to plot a contour...

            # Preprocess the input and output data.
            input_data, output_data = self.preprocess_input_output_data( input_data, output_data, flatten_flag = False )

            # Plot the given level set.
            # ax.contour( self.plot_process( input_data[ ..., 0 ] ), self.plot_process( input_data[ ..., 1 ] ), self.plot_process( output_data[ ..., 0 ] ), levels = [ self.plot_process( level ) ], colors = 'red', linewidths = 2.0 )
            ax.contour( self.plot_process( input_data[ ..., 0 ] ), self.plot_process( input_data[ ..., 1 ] ), self.plot_process( output_data[ ..., 0 ] ), levels = [ self.plot_process( level ) ], linewidths = 2.0 )

        else:

            # Preprocess the input and output data.
            input_data, output_data = self.preprocess_input_output_data( input_data, output_data, flatten_flag = True )

            # Plot the input and output data.
            ax.scatter3D( self.plot_process( input_data[ :, 0 ] ), self.plot_process( input_data[ :, 1 ] ), self.plot_process( output_data ) )
            # ax.scatter3D( self.plot_process( input_data[ ..., 0 ] ), self.plot_process( input_data[ ..., 1 ] ), self.plot_process( output_data[ ..., 0 ] ) )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Determine whether to show the plot.
        if show_plot:                      # If we want to show the figure...

            # Show the figure.
            plt.show( block = False )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot 2in 2out data.
    def plot_2in_2out_data( self, input_data, output_data, fig = None, input_labels = [ 'in1', 'in2' ], title_string = '2i2o Data Plot', save_directory = r'.', as_stream = True, show_plot = False ):

        # Determine whether to set the input labels to a default value.
        if input_labels is None:                    # If the input labels variable is None...

            # Set the input labels to be a default value.
            input_labels = [ 'in1', 'in2' ]

        # Determine whether to create a new figure.
        if fig is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input and output data.
            fig = plt.figure(  ); plt.xlabel( input_labels[ 0 ] + r' | Output Data: Dim1 [-]' ), plt.ylabel( input_labels[ 1 ] + r' | Output Data: Dim2 [-]' ), plt.title( title_string )

        else:                               # Otherwise... ( i.e., A figure was provided... )
          
            # Set the given figure to be active.
            plt.figure( fig.number )

        # Retrieve the current axes.
        ax = plt.gca(  )

        # Determine how to plot the 2in2out data.
        if as_stream:                               # If we want to create a stream plot...

            # Preprocess the input and output data.
            input_data, output_data = self.preprocess_input_output_data( input_data, output_data, flatten_flag = False )

            # Create a stream plot of the input and output data.
            # ax.streamplot( self.plot_process( input_data[ ..., 0 ].T ), self.plot_process( input_data[ ..., 1 ].T ), self.plot_process( output_data[ ..., 0 ].T ), self.plot_process( output_data[ ..., 1 ].T ) )
            ax.streamplot( self.plot_process( input_data[ ..., 0 ].T ), self.plot_process( input_data[ ..., 1 ].T ), self.plot_process( output_data[ ..., 0 ].T ), self.plot_process( output_data[ ..., 1 ].T ), density = 1.0, linewidth = 1.0, arrowsize = 0.75 )

        else:                                       # Otherwise... ( i.e., if we want to create a quiver plot... )

            # Preprocess the input and output data.
            input_data, output_data = self.preprocess_input_output_data( input_data, output_data, flatten_flag = True )

            # Create a quiver plot of the input and output data.
            plt.quiver( self.plot_process( input_data[ :, 0 ] ), self.plot_process( input_data[ :, 1 ] ), self.plot_process( output_data[ :, 0 ] ), self.plot_process( output_data[ :, 1 ] ) )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Determine whether to show the plot.
        if show_plot:                      # If we want to show the figure...

            # Show the figure.
            plt.show( block = False )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot 2in 3out data.
    def plot_2in_3out_data( self, input_data, output_data, fig = None, input_labels = [ 'in1', 'in2' ], title_string = '2i3o Data Plot', save_directory = r'.', show_plot = False ):

        # Determine whether to set the input labels to a default value.
        if input_labels is None:                    # If the input labels variable is None...

            # Set the input labels to be a default value.
            input_labels = [ 'in1', 'in2' ]

        # Preprocess the input and output data.
        input_data, output_data = self.preprocess_input_output_data( input_data, output_data )

        # Determine whether to create a new figure.
        if fig is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input and output data.
            fig, ax = plt.subplots( subplot_kw = { 'projection': '3d' } ); ax.set_xlabel( input_labels[ 0 ] + r' | Output Data: Dim1 [-]' ), ax.set_ylabel( input_labels[ 1 ] + r' | Output Data: Dim2 [-]' ), ax.set_zlabel( 'Output Data: Dim3 [-]' ), ax.set_title( title_string )

        else:                               # Otherwise... ( If a figure was provided... )

            # Set the given figure to be active.
            plt.figure( fig.number )

            # Retrieve the current axes.
            ax = plt.gca(  )

        # Create a quiver plot of the input and output data.
        ax.quiver( self.plot_process( input_data[ :, 0 ] ), self.plot_process( input_data[ :, 1 ] ), self.plot_process( input_data[ :, 2 ] ), self.plot_process( output_data[ :, 0 ] ), self.plot_process( output_data[ :, 1 ] ), self.plot_process( output_data[ :, 2 ] ) )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Determine whether to show the plot.
        if show_plot:                      # If we want to show the figure...

            # Show the figure.
            plt.show( block = False )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot 3in 1out data.
    def plot_3in_0out_data( self, input_data, fig = None, input_labels = [ 'in1', 'in2', 'in3' ], title_string = '3i0o Data Plot', save_directory = r'.', show_plot = False ):

        # Determine whether to set the input labels to a default value.
        if input_labels is None:                    # If the input labels variable is None...

            # Set the input labels to be a default value.
            input_labels = [ 'in1', 'in2', 'in3' ]

        # Preprocess the input data.
        input_data = self.preprocess_data( input_data )

        # Determine whether to create a new figure.
        if fig is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input and output data.
            fig, ax = plt.subplots( subplot_kw = { 'projection': '3d' } ); ax.set_xlabel( input_labels[ 0 ] ), ax.set_ylabel( input_labels[ 1 ] ), ax.set_zlabel( input_labels[ 2 ] ), ax.set_title( title_string )

        else:                               # Otherwise... ( If a figure was provided... )

            # Set the given figure to be active.
            plt.figure( fig.number )

            # Retrieve the current axes.
            ax = plt.gca(  )

        # Plot the input and output data.
        ax.scatter3D( self.plot_process( input_data[ :, 0 ] ), self.plot_process( input_data[ :, 1 ] ), self.plot_process( input_data[ :, 2 ] ) )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Determine whether to show the plot.
        if show_plot:                      # If we want to show the figure...

            # Show the figure.
            plt.show( block = False )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot 3in 1out data.
    def plot_3in_1out_data( self, input_data, output_data, fig = None, input_labels = [ 'in1', 'in2', 'in3' ], title_string = '3i1o Data Plot', save_directory = r'.', show_plot = False ):

        # Determine whether to set the input labels to a default value.
        if input_labels is None:                    # If the input labels variable is None...

            # Set the input labels to be a default value.
            input_labels = [ 'in1', 'in2', 'in3' ]

        # Preprocess the input and output data.
        input_data, output_data = self.preprocess_input_output_data( input_data, output_data )

        # Determine whether to create a new figure.
        if fig is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input and output data.
            fig, ax = plt.subplots( subplot_kw = { 'projection': '3d' } ); ax.set_xlabel( input_labels[ 0 ] ), ax.set_ylabel( input_labels[ 1 ] ), ax.set_zlabel( input_labels[ 2 ] ), ax.set_title( title_string )

        else:                               # Otherwise... ( If a figure was provided... )

            # Set the given figure to be active.
            plt.figure( fig.number )

            # Retrieve the current axes.
            ax = plt.gca(  )

        # Plot the input and output data.
        ax.scatter3D( self.plot_process( input_data[ :, 0 ] ), self.plot_process( input_data[ :, 1 ] ), self.plot_process( input_data[ :, 2 ] ), c = self.plot_process( output_data ) )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Determine whether to show the plot.
        if show_plot:                      # If we want to show the figure...

            # Show the figure.
            plt.show( block = False )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot 3in 2out data.
    def plot_3in_2out_data( self, input_data, output_data, figs = None, input_labels = [ 'in1', 'in2', 'in3' ], title_string = '3i2o Data Plot', save_directory = r'.', show_plot = False ):

        # Determine whether to set the input labels to a default value.
        if input_labels is None:                    # If the input labels variable is None...

            # Set the input labels to be a default value.
            input_labels = [ 'in1', 'in2', 'in3' ]

        # Preprocess the input and output data.
        input_data, output_data = self.preprocess_input_output_data( input_data, output_data )

        # Determine whether to create a new figure.
        if figs is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input data and the first dimension of the output data.
            fig1, ax1 = plt.subplots( subplot_kw = { 'projection': '3d' } ); ax1.set_xlabel( input_labels[ 0 ] ), ax1.set_ylabel( input_labels[ 1 ] ), ax1.set_zlabel( input_labels[ 2 ] ), ax1.set_title( title_string + ' ( Output Dim0 )' )

        else:                               # Otherwise... ( i.e., if a figure was provided... )

            # Retrieve the first provided figure.
            fig1 = figs[ 0 ]

            # Make the first figure active.
            plt.figure( fig1.number )

            # Retrieve the associated axis.
            ax1 = plt.gca(  )

        # Plot the input data and the first dimension of the output data.
        ax1.scatter3D( self.plot_process( input_data[ :, 0 ] ), self.plot_process( input_data[ :, 1 ] ), self.plot_process( input_data[ :, 2 ] ), c = self.plot_process( output_data[ :, 0 ] ) )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Determine whether to create a new figure.
        if figs is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input data and the first dimension of the output data.
            fig2, ax2 = plt.subplots( subplot_kw = { 'projection': '3d' } ); ax2.set_xlabel( input_labels[ 0 ] ), ax2.set_ylabel( input_labels[ 1 ] ), ax2.set_zlabel( input_labels[ 2 ] ), ax2.set_title( title_string + ' ( Output Dim1 )' )

        else:                               # Otherwise... ( i.e., if a figure was provided... )

            # Retrieve the second provided figure.
            fig2 = figs[ 1 ]

            # Make the second figure active.
            plt.figure( fig2.number )

            # Retrieve the associated axis.
            ax2 = plt.gca(  )

        # Plot the input data and the first dimension of the output data.
        ax2.scatter3D( self.plot_process( input_data[ :, 0 ] ), self.plot_process( input_data[ :, 1 ] ), self.plot_process( input_data[ :, 2 ] ), c = self.plot_process( output_data[ :, 1 ] ) )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Store the figures and axes into lists.
        fig = [ fig1, fig2 ]
        ax = [ ax1, ax2 ]

        # Determine whether to show the plot.
        if show_plot:                      # If we want to show the figure...

            # Show the figure.
            plt.show( block = False )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot 3in 3out data.
    def plot_3in_3out_data( self, input_data, output_data, figs = None, input_labels = [ 'in1', 'in2', 'in3' ], title_string = '3i3o Data Plot', save_directory = r'.', show_plot = False ):

        # Determine whether to set the input labels to a default value.
        if input_labels is None:                    # If the input labels variable is None...

            # Set the input labels to be a default value.
            input_labels = [ 'in1', 'in2', 'in3' ]

        # Preprocess the input and output data.
        input_data, output_data = self.preprocess_input_output_data( input_data, output_data )

        # Determine whether to create a new figure.
        if figs is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input data and the first dimension of the output data.
            fig1, ax1 = plt.subplots( subplot_kw = { 'projection': '3d' } ); ax1.set_xlabel( input_labels[ 0 ] ), ax1.set_ylabel( input_labels[ 1 ] ), ax1.set_zlabel( input_labels[ 2 ] ), ax1.set_title( title_string + ' ( Output Dim0 )' )

        else:                               # Otherwise... ( i.e., if a figure was provided... )

            # Retrieve the first provided figure.
            fig1 = figs[ 0 ]

            # Make the first figure active.
            plt.figure( fig1.number )

            # Retrieve the associated axis.
            ax1 = plt.gca(  )

        # Plot the input data and the first dimension of the output data.
        ax1.scatter3D( self.plot_process( input_data[ :, 0 ] ), self.plot_process( input_data[ :, 1 ] ), self.plot_process( input_data[ :, 2 ] ), c = self.plot_process( output_data[ :, 0 ] ) )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Determine whether to create a new figure.
        if figs is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input data and the first dimension of the output data.
            fig2, ax2 = plt.subplots( subplot_kw = { 'projection': '3d' } ); ax2.set_xlabel( input_labels[ 0 ] ), ax2.set_ylabel( input_labels[ 1 ] ), ax2.set_zlabel( input_labels[ 2 ] ), ax2.set_title( title_string + ' ( Output Dim1 )' )

        else:                               # Otherwise... ( i.e., if a figure was provided... )

            # Retrieve the second provided figure.
            fig2 = figs[ 1 ]

            # Make the second figure active.
            plt.figure( fig2.number )

            # Retrieve the associated axis.
            ax2 = plt.gca(  )

        # Plot the input data and the first dimension of the output data.
        ax2.scatter3D( self.plot_process( input_data[ :, 0 ] ), self.plot_process( input_data[ :, 1 ] ), self.plot_process( input_data[ :, 2 ] ), c = self.plot_process( output_data[ :, 1 ] ) )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Determine whether to create a new figure.
        if figs is None:                     # If no figure was provided...

            # Create a figure to store a plot of the input data and the first dimension of the output data.
            fig3, ax3 = plt.subplots( subplot_kw = { 'projection': '3d' } ); ax3.set_xlabel( input_labels[ 0 ] ), ax3.set_ylabel( input_labels[ 1 ] ), ax3.set_zlabel( input_labels[ 2 ] ), ax3.set_title( title_string + ' ( Output Dim2 )' )

        else:                               # Otherwise... ( i.e., if a figure was provided... )

            # Retrieve the third provided figure.
            fig3 = figs[ 2 ]

            # Make the third figure active.
            plt.figure( fig3.number )

            # Retrieve the associated axis.
            ax3 = plt.gca(  )

        # Plot the input data and the first dimension of the output data.
        ax3.scatter3D( self.plot_process( input_data[ :, 0 ] ), self.plot_process( input_data[ :, 1 ] ), self.plot_process( input_data[ :, 2 ] ), c = self.plot_process( output_data[ :, 2 ] ) )

        # Save the figure.
        plt.savefig( save_directory + '/' + f'Figure_{plt.gcf(  ).number}.png' )

        # Store the figures and axes into lists.
        fig = [ fig1, fig2, fig3 ]
        ax = [ ax1, ax2, ax3 ]

        # Determine whether to show the plot.
        if show_plot:                      # If we want to show the figure...

            # Show the figure.
            plt.show( block = False )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot the input-output data.
    def plot_standard_data( self, input_data, output_data, level = 0, fig = None, input_labels = None, title_string = 'Standard Data Plot', save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False, D1_style = '-' ):

        # Retrieve the number of input and output dimensions.
        num_input_dimensions = self.tensor_utilities.get_number_of_dimensions( input_data )
        num_output_dimensions = self.tensor_utilities.get_number_of_dimensions( output_data )

        # Determine whether to embed the number of output dimensions in a list.
        if not ( isinstance( num_output_dimensions, list ) or isinstance( num_output_dimensions, tuple ) ):                # If the number of output dimensions is not a list or tuple...

            # Embed the number of output dimensions in a list.
            num_output_dimensions = [ num_output_dimensions ]

        # Retrieve the number of output data sources.
        num_output_sources = self.tensor_utilities.get_number_of_sources( output_data, input_data.device )

        # Determine whether to set the input labels to a default value.
        if input_labels is None:                    # If the input labels variable is None...

            # Set the input labels to be a default value.
            input_labels = [ f'in{k}' for k in range( num_input_dimensions ) ]

        # Determine which plot to create based on the number of input dimensions.
        if ( num_input_dimensions == 1 ):                               # If the input is a scalar...

            # Determine whether to set the number of output dimensions to zero.
            if not num_output_dimensions:                                       # If the number of output dimensions is an empty list...

                # Create a plot for the single input no output data.
                figs, axes = self.plot_1in_0out_data( input_data, fig, input_labels, title_string, save_directory, show_plot )

            else:

                figs = [  ]
                axes = [  ]

            # Create a plot for each of the output data sources.
            for k in range( num_output_sources ):                       # Iterate through each of the output sources...

                # Determine whether to set the number of output dimensions to zero.
                if num_output_dimensions[ k ] == 1:                     # If this output source is a scalar... 

                    # Create a plot for single input single output data.
                    fig, ax = self.plot_1in_1out_data( input_data, output_data, fig, input_labels, title_string, save_directory, show_plot, D1_style )

                elif num_output_dimensions[ k ] == 2:                   # If this output source is a vector of dimension 2...

                    # Create a plot for single input two output data.
                    fig, ax = self.plot_1in_2out_data( input_data, output_data, fig, input_labels, title_string, save_directory, show_plot )

                elif num_output_dimensions[ k ] == 3:                   # If this output source is a vector of dimension 3...

                    # Create a plot for single input three output data.
                    fig, ax = self.plot_1in_3out_data( input_data, output_data, fig, input_labels, title_string, save_directory, show_plot )

                else:                                                   # Otherwise... (If the number of output dimensions is not one for which we have an established plotting protocol...)

                    # Throw a warning.
                    warnings.warn( f'No plotting protocol for pinn data sets with {num_input_dimensions} inputs and {num_output_dimensions[ k ]} outputs has been established.  Setting figure and axis objects to None.' )

                    # Set the figure and axis objects to None.
                    fig = None
                    ax = None

                # Add this figure and axis to the figures and axes lists.
                figs.append( fig )
                axes.append( ax )

        elif ( num_input_dimensions == 2 ):                               # If the input is a 2D vector...

            # Determine whether to set the number of output dimensions to zero.
            if not num_output_dimensions:                                       # If the number of output dimensions is an empty list...

                # Create a plot for the two input no output data.
                figs, axes = self.plot_2in_0out_data( input_data, fig, input_labels, title_string, save_directory, show_plot )

            else:

                figs = [  ]
                axes = [  ]

            # Create a plot for each of the output data sources.
            for k in range( num_output_sources ):                       # Iterate through each of the output sources...

                # Determine which plot to create based on the number of output dimension.                
                if num_output_dimensions[ k ] == 1:                     # If this output source is a scalar... 

                    # Create a plot for two input single output data.
                    fig, ax = self.plot_2in_1out_data( input_data, output_data, level, fig, input_labels, title_string, save_directory, as_surface, as_contour, show_plot )

                elif num_output_dimensions[ k ] == 2:                   # If this output source is a vector of dimension 2...

                    # Create a plot for two input two output data.
                    fig, ax = self.plot_2in_2out_data( input_data, output_data, fig, input_labels, title_string, save_directory, as_stream, show_plot )

                elif num_output_dimensions[ k ] == 3:                   # If this output source is a vector of dimension 3...

                    # Create a plot for two input three output data.
                    fig, ax = self.plot_2in_3out_data( input_data, output_data, fig, input_labels, title_string, save_directory, show_plot )
                
                else:                                                   # Otherwise... (If the number of output dimensions is not one for which we have an established plotting protocol...)

                    # Throw a warning.
                    warnings.warn( f'No plotting protocol for pinn data sets with {num_input_dimensions} inputs and {num_output_dimensions[ k ]} outputs has been established.  Setting figure and axis objects to None.' )

                    # Set the figure and axis objects to None.
                    fig = None
                    ax = None

                # Add this figure and axis to the figures and axes lists.
                figs.append( fig )
                axes.append( ax )

        elif ( num_input_dimensions == 3 ):                             # If the input is a 3D vector...

            # Determine whether to set the number of output dimensions to zero.
            if not num_output_dimensions:                                       # If the number of output dimensions is an empty list...

                # Create a plot for the three input no output data.
                figs, axes = self.plot_3in_0out_data( input_data, fig, input_labels, title_string, save_directory, show_plot )

            else:

                figs = [  ]
                axes = [  ]

            # Create a plot for each of the output data sources.
            for k in range( num_output_sources ):                       # Iterate through each of the output sources...

                # Determine which plot to create based on the number of output dimension.
                if num_output_dimensions[ k ] == 1:                     # If this output source is a scalar... 

                    # Create a plot for three input single output data.
                    fig, ax = self.plot_3in_1out_data( input_data, output_data, fig, input_labels, title_string, save_directory, show_plot )

                elif num_output_dimensions[ k ] == 2:                   # If this output source is a vector of dimension 2...

                    # Create a plot for three input two output data.
                    fig, ax = self.plot_3in_2out_data( input_data, output_data, fig, input_labels, title_string, save_directory, show_plot )

                elif num_output_dimensions[ k ] == 3:                   # If this output source is a vector of dimension 3...

                    # Create a plot for the three input three output data.
                    fig, ax = self.plot_3in_3out_data( input_data, output_data, fig, input_labels, title_string, save_directory, show_plot )

                else:                                                   # Otherwise... (If the number of output dimensions is not one for which we have an established plotting protocol...)

                    # Throw a warning.
                    warnings.warn( f'No plotting protocol for pinn data sets with {num_input_dimensions} inputs and {num_output_dimensions[ k ]} outputs has been established.  Setting figure and axis objects to None.' )

                    # Set the figure and axis objects to None.
                    fig = None
                    ax = None

                # Add this figure and axis to the figures and axes lists.
                figs.append( fig )
                axes.append( ax )

        else:                                                                                   # Otherwise...

            # Throw a warning.
            warnings.warn( f'No plotting protocol for pinn data sets with {num_input_dimensions} inputs has been established.  Setting figure and axis objects to None.' )

            # Set the figure and axis to be none.
            figs = None
            axes = None

        # Return the figures and axes.
        return figs, axes


    # Implement a function to plot projected data.
    def plot_projected_data( self, input_data, output_data, projection_dimensions, projection_values, level = 0, fig = None, input_labels = None, title_string = 'Projected Data Plot', save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Project the input and output data.
        projected_input_data, projected_output_data = self.tensor_utilities.project_input_output_data( input_data, output_data, projection_dimensions, projection_values )

        # Determine whether it is necessary to modify the input labels.
        if input_labels and isinstance( input_labels, list ):               # If the input labels is a non-empty...

            # Retrieve the number of projected dimensions.
            num_projected_dimensions = len( projection_dimensions )

            # Remove each of the projected dimensions.
            for k in range( num_projected_dimensions ):                     # Iterate through each of the projected dimensions...

                # Remove the projected dimension from this list.
                del input_labels[ k ]

        # Plot the now projected data.
        figs, axes = self.plot_standard_data( projected_input_data, projected_output_data, level, fig, input_labels, title_string, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figures and axes.
        return figs, axes


    # Implement a function to plot general data (automatically determining whether to project the data).
    def plot_data( self, input_data, output_data, projection_dimensions = None, projection_values = None, level = 0, fig = None, input_labels = None, title_string = 'Data Plot', save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False, D1_style = '-' ):

        # Determine whether to project the data or to plot it in the standard way.
        if ( projection_dimensions is not None ) and ( projection_values is not None ):         # If we want to project the data...

            # Plot the projected data.
            figs, axes = self.plot_projected_data( input_data, output_data, projection_dimensions, projection_values, level, fig, input_labels, title_string, save_directory, as_surface = as_surface, as_stream = as_stream, as_contour = as_contour, show_plot = show_plot )

        elif ( projection_dimensions is None ) and ( projection_values is None ):               # If we do not want to project the data...

            # Plot the data in the standard way.
            figs, axes = self.plot_standard_data( input_data, output_data, level, fig, input_labels, title_string, save_directory, as_surface = as_surface, as_stream = as_stream, as_contour = as_contour, show_plot = show_plot, D1_style = D1_style )

        else:                                                                                   # Otherwise... ( i.e., it is not clear whether to project the data... )

            # Throw an error.
            raise ValueError( 'Cannot determine whether to project or not project the data.' )

        # Return the figure and axes.
        return figs, axes


    # Implement a function to plot a function given some input data (automatically determining whether to project the data).
    def plot_function( self, input_data, function, projection_dimensions = None, projection_values = None, level = 0, fig = None, input_labels = None, title_string = 'Function Plot', save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Generate the output data associated with the given function and its input data.
        output_data = self.function_utilities.evaluate_functions( input_data, function, as_tensor = True )

        # Plot the input and output data.
        figs, axes = self.plot_data( input_data, output_data, projection_dimensions, projection_values, level, fig, input_labels, title_string, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figures and axes.
        return figs, axes


    # Implement a function to plot a input and output data or input data and an output function (the correct plotting procedure is automatically detected).
    def plot( self, input_data, output, projection_dimensions = None, projection_values = None, level = 0, fig = None, input_labels = None, title_string = 'Input-Output Plot', save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False, D1_style = '-' ):

        # Determine whether the output is a data or a function to evaluate.
        if callable( output ) or ( isinstance( output, list ) and output and all( callable( output[ k ] ) for k in range( len( output ) ) ) ):                      # If the output is a function...

            # Plot the function output given the specified inputs.
            figs, axes = self.plot_function( input_data, output, projection_dimensions, projection_values, level, fig, input_labels, title_string, save_directory, as_surface, as_stream, as_contour, show_plot )

        else:                                       # If the output is data...

            # Plot the input and output data.
            figs, axes = self.plot_data( input_data, output, projection_dimensions, projection_values, level, fig, input_labels, title_string, save_directory, as_surface, as_stream, as_contour, show_plot, D1_style )

        # Return the figures and axes.
        return figs, axes


    #%% ------------------------------------------------------------ FINITE ELEMENT PLOTTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to plot the template element points.ax
    def plot_template_integration_points( self, xis, show_plot_flag = False ):

        # Determine how to plot the element points.
        if xis.shape[ -1 ] == 1:                        # If the number of spatiotemporal variables is one...

            # Create a figure for the element points.
            fig = plt.figure(  ); plt.xlabel( 'x' ); plt.ylabel( 'y' ); plt.title( 'Element Points' )
            plt.scatter( self.plot_process( torch.range( xis.shape[ 0 ], dtype = torch.int64, device = xis.device ) ), self.plot_process( xis ) )

            # Retrieve the current axes.
            ax = fig.gca(  )

        elif xis.shape[ -1 ] == 2:                      # If the number of spatiotemporal variables is two...

            # Create a figure for the element points.
            fig = plt.figure(  ); plt.xlabel( 'x' ); plt.ylabel( 'y' ); plt.title( 'Element Points' )
            plt.scatter( self.plot_process( xis[ :, 0 ] ), self.plot_process( xis[ :, 1 ] ) )

            # Retrieve the current axes.
            ax = fig.gca(  )

        elif xis.shape[ -1 ] == 3:                      # If the number of spatiotemporal variables is three...

            # Create a figure for the element points.
            fig = plt.figure(  ); ax = plt.axes( projection = '3d' ); ax.set_xlabel( 'x1' ); ax.set_ylabel( 'x2' ); ax.set_zlabel( 'x3' ); plt.title( 'Element Points' )
            ax.scatter( self.plot_process( xis[ :, 0 ] ), self.plot_process( xis[ :, 1 ] ), self.plot_process( xis[ :, 2 ] ) )

        else:                                                               # Otherwise...

            # State that we can't display plots for problems of this dimension.
            print( 'Need to add code to plot element points of greater than three dimensions.' )

        # Determine whether to show the plots.
        if show_plot_flag:                                                  # If we want to show the plots...

            # Show the plot.
            plt.show(  )

        # Return the figure and axis.
        return fig, ax


    # Implement a function to plot the basis function evaluation.
    def plot_basis_functions( self, xis, G, show_plot_flag = False ):     

        # Retrieve the basis function values associated with a single collocation point.
        g = torch.reshape( G[ 0, :, : ], ( G.shape[ 1 ], G.shape[ 2 ] ) )

        # Compute the number of basis functions
        num_basis_functions = g.shape[ 0 ]

        # Compute the number of subplot rows and columns.
        num_rows, num_cols = self.get_subplot_rc_nums( num_basis_functions )

        # Determine how to plot the basis functions.
        if xis.shape[ -1 ] == 3:                   # If this is a 3D problem...

            # Create subplots.
            fig, axes_temp = plt.subplots( nrows = num_rows, ncols = num_cols, subplot_kw = { 'projection': '3d' } )

            # Ensure that the axes are at least a single list.
            if not isinstance( axes_temp, list ):                    # If the axes object is not a list...

                # Embed the axes object into a list.
                axes_temp = [ axes_temp ]

            # Ensure that the axes are double lists.
            if not isinstance( axes_temp[ 0 ], list ):                  # If the axes object is only a single list...

                # Embed the axes object in a list.
                axes = [ [ axes_temp[k] ] for k in range( len( axes_temp ) ) ]

            else:

                # Store the temporary axes object permanently.
                axes = axes_temp

            # Initialize a counter variable.
            k3 = torch.tensor( 0, dtype = torch.int64, device = xis.device )

            # Plot the basis functions.
            for k1 in range( num_rows ):              # Iterate through the subplot rows...
                for k2 in range( num_cols ):          # Iterate through the subplot columns...

                    # Determine whether to plot this basis function.
                    if k3 < num_basis_functions:                   # If this index corresponds to a valid basis function...

                        # Plot the basis function.
                        axes[ k1 ][ k2 ].scatter3D( self.xis[ :, 0 ], self.xis[ :, 1 ], self.xis[ :, 2 ], c = g[ k3, : ], s = 20 )

                        # Format this subplot.
                        axes[ k1 ][ k2 ].set_title( 'Basis Function ' + str( k3 + 1 ) ); axes[ k1 ][ k2 ].set_xlabel( 'x' ); axes[ k1 ][ k2 ].set_ylabel( 'y' ); axes[ k1 ][ k2 ].set_zlabel( 't' )

                    # Advance the counter.
                    k3 += 1

        # Determine whether to show the plot.
        if show_plot_flag:                   # If we want to show the plot...
            
            # Show the plot.
            plt.show(  ) 

        # Return the figure and axes.
        return fig, axes

