####################################################################################### 
# THIS SOURCE CODE IS PROPERTY OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. 
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND 
# CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION 
# ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN 
# LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE,
# CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL 
# PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641. 
####################################################################################### 


#%% ------------------------------------------------------------ FUNCTION UTILITIES CLASS ------------------------------------------------------------

# This file implements a class for storing and managing function utilities information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch
import warnings

# Import custom libraries.
from tensor_utilities_class import tensor_utilities_class as tensor_utilities_class


#%% ------------------------------------------------------------ FUNCTION UTILITIES CLASS ------------------------------------------------------------

# Implement the function utilities class.
class function_utilities_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self ):

        # Create an instance of the tensor utilities class.
        self.tensor_utilities = tensor_utilities_class(  )


    #%% ------------------------------------------------------------ EVALUATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to evaluate a function over a flattened grid.
    def evaluate_function_over_flattened_grid( self, input_data, function ):

        # Generate the output data.
        output_data = function( input_data )

        # Return the output data.
        return output_data


    # Implement a function to evaluate a function over an expanded grid.
    def evaluate_function_over_expanded_grid( self, input_data, function ):

        # Flatten the input data.
        input_data_flattened = self.tensor_utilities.flatten_grid( input_data )

        # Evaluate the function over the flattened grid.
        output_data_flattened = self.evaluate_function_over_flattened_grid( input_data_flattened, function )

        # Expand the output data.
        output_data = self.tensor_utilities.expand_grid( output_data_flattened, input_data[ ..., 0 ].shape )

        # Return the output data.
        return output_data


    # Implement a function to evaluate a function over some input tensor (automatically detecting whether the input is a flattened or expanded grid).
    def evaluate_function( self, input_data, function ):

        # Determine how to evaluate the given function.
        if self.tensor_utilities.is_grid_flat( input_data ):                                         # If the input grid is flat...

            # Evaluate the function over the flattened grid.
            output_data = self.evaluate_function_over_flattened_grid( input_data, function )

        else:                                                                                       # Otherwise... ( i.e., the input grid is not flat... )

            # Evaluate the function over the expanded grid.
            output_data = self.evaluate_function_over_expanded_grid( input_data, function )

        # Return the output data.
        return output_data


    # Implement a function to evaluate a list of functions over some input tensor.
    def evaluate_functions( self, input_data, functions, as_tensor = True ):

        # Determine how to evaluate the functions.
        if callable( functions ):                   # If the functions argument is callable...

            # Evaluate the function.
            output_data = self.evaluate_function( input_data, functions )

        else:                                       # Otherwise... ( i.e., the functions argument is a list... )

            # Evaluate each of the functions.
            output_data = [ self.evaluate_function( input_data, functions[ k ] ) for k in range( len( functions ) ) ]

        # Determine whether to concatenate the output data.
        if as_tensor and isinstance( output_data, list ):                                   # If we want to concatenate the output data...

            # Concatenate the output data.
            output_data = torch.cat( tuple( output_data ), dim = -1 )

        # Return the output data.
        return output_data
