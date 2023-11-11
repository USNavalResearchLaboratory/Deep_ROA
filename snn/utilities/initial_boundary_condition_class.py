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
#%% ------------------------------------------------------------ INITIAL BOUNDARY CONDITION CLASS ------------------------------------------------------------

# This file implements a class for storing and managing initial and boundary conditions.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch

# Import custom libraries.
from plotting_utilities_class import plotting_utilities_class as plotting_utilities_class
from printing_utilities_class import printing_utilities_class as printing_utilities_class


#%% ------------------------------------------------------------ INITIAL BOUNDARY CONDITION CLASS ------------------------------------------------------------

# Implement the initial boundary condition class.
class initial_boundary_condition_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self, id, name, general_type, specific_type, dimension, condition_functions, placement, device = 'cpu' ):

        # Create an instance of the plotting utilities class.
        self.plotting_utilities = plotting_utilities_class(  )

        # Create an instance of the printing utilities class.
        self.printing_utilities = printing_utilities_class(  )

        # Store the device.
        self.device = device

        # Store the initial boundary condition information.
        self.id = self.validate_id( id )
        self.name = self.validate_name( name )
        self.general_type = self.validate_general_type( general_type )
        self.specific_type = self.validate_specific_type( specific_type )

        # Store the initial boundary condition application dimension.
        self.dimension = self.validate_dimension( dimension )

        # Store the initial boundary condition function.
        self.condition_functions = self.validate_condition_function( condition_functions )
        self.num_condition_functions = self.compute_number_of_condition_functions( self.condition_functions )

        # Store the initial boundary condition placement.
        self.placement = self.validate_placement( placement )


    #%% ------------------------------------------------------------ PREPROCESS FUNCTIONS ------------------------------------------------------------

    # Implement a function to preprocess the condition functions.
    def preprocess_condition_functions( self, condition_functions = None ):

        # Determine whether to use the stored condition functions.
        if condition_functions is None:                 # If the condition functions was not provided...

            # Set the condition functions to be the stored value.
            condition_functions = self.condition_functions

        # Return the condition functions.
        return condition_functions


    #%% ------------------------------------------------------------ IS VALID FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate the initial-boundary condition ID.
    def is_id_valid( self, id ):

        # Determine whether the given ID is valid.
        if torch.is_tensor( id ) and ( ( id.dtype == torch.uint8 ) or ( id.dtype == torch.int8 ) or ( id.dtype == torch.int16 ) or ( id.dtype == torch.int32 ) or ( id.dtype == torch.int64 ) ) and ( id > 0 ):                  # If the id is valid...

            # Set the valid flag to true.
            b_valid = True

        else:                                                                               # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    # Implement a function to validate the initial-boundary condition name.
    def is_name_valid( self, name ):

        # Determine whether the given name is valid.
        if isinstance( name, str ):                     # If the name is valid...

            # Set the valid flag to true.
            b_valid = True

        else:                                           # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    # Implement a function to validate the general initial-boundary condition type.
    def is_general_type_valid( self, general_type ):

        # Determine whether the given general type is valid.
        if isinstance( general_type, str ) and ( ( general_type.lower(  ) == 'initial' ) or ( general_type.lower(  ) == 'boundary' ) or ( general_type.lower(  ) == 'ic' ) or ( general_type.lower(  ) == 'bc' ) ):

            # Set the valid flag to true.
            b_valid = True

        else:                                           # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    # Implement a function to validate the specific initial-boundary condition type.
    def is_specific_type_valid( self, specific_type ):

        # Determine whether the given general type is valid.
        if isinstance( specific_type, str ) and ( ( specific_type.lower(  ) == 'dirichlet' ) or ( specific_type.lower(  ) == 'neumann' ) or ( specific_type.lower(  ) == 'cauchy' ) or ( specific_type.lower(  ) == 'yuan-li' ) ):                      # If specific type is recognized...

            # Set the valid flag to true.
            b_valid = True

        else:                                                                                                                                                                                               # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    # Implement a function to validate the initial-boundary condition dimension.
    def is_dimension_valid( self, dimension ):

        # Determine whether the given dimension is valid.
        if torch.is_tensor( dimension ) and ( ( dimension.dtype == torch.uint8 ) or ( dimension.dtype == torch.int8 ) or ( dimension.dtype == torch.int16 ) or ( dimension.dtype == torch.int32 ) or ( dimension.dtype == torch.int64 ) ) and ( dimension >= 0 ):                  # If the dimension is valid...

            # Set the valid flag to true.
            b_valid = True

        else:                                                                                                                                                                                                                                                                       # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    # Implement a function to validate the initial-boundary condition function.
    def is_condition_function_valid( self, condition_functions ):

        # Determine whether the condition function is valid.
        if callable( condition_functions ):                                                                     # If the condition functions variable is itself a callable function...
            
            # Set the valid flag to true.
            b_valid = True

        elif isinstance( condition_functions, list ) and len( condition_functions ) > 0:                        # If the condition functions variable is a list...

            # Retrieve the number of condition functions.
            num_condition_functions = len( condition_functions )

            # Initialize a loop variable.
            # k = 0
            k = torch.tensor( 0, dtype = torch.int64, device = self.device )

            # Set the valid flag to true.
            b_valid = True

            # Determine whether all of the condition functions are in fact functions.
            while b_valid and ( k < num_condition_functions ):                                                  # If all of the condition function entries have so far been functions and we have not yet checked all of the condition function entries...

                # Check whether this condition function entry is in fact a function.
                b_valid &= callable( condition_functions[ k ] )

                # Advance the loop variable.
                k += 1

        else:                                                                                                   # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    # Implement a function to validate the initial-boundary condition placement.
    def is_placement_valid( self, placement ):

        # Determine whether the given placement is valid.
        if isinstance( placement, str ) and ( ( placement.lower(  ) == 'left' ) or ( placement.lower(  ) == 'right' ) or ( placement.lower(  ) == 'lower' ) or ( placement.lower(  ) == 'upper' ) or ( placement.lower(  ) == 'bottom' ) or ( placement.lower(  ) == 'top' ) ):

            # Set the valid flag to true.
            b_valid = True

        else:                                           # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    #%% ------------------------------------------------------------ SET FUNCTIONS ------------------------------------------------------------

    # Implement a function to set ids.
    def set_id( self, id, set_flag = True ):

        # Determine whether to set the id.
        if set_flag:                # If we want to set the id...

            # Set the id.
            self.id = id


    # Implement a function to set names.
    def set_name( self, name, set_flag = True ):

        # Determine whether to set the name.
        if set_flag:                # If we want to set the name...

            # Set the name.
            self.name = name


    # Implement a function to set general types.
    def set_general_type( self, general_type, set_flag = True ):

        # Determine whether to set the general type.
        if set_flag:                # If we want to set the general type...

            # Set the general type.
            self.general_type = general_type


    # Implement a function to set specific types.
    def set_specific_type( self, specific_type, set_flag = True ):

        # Determine whether to set the specific_type.
        if set_flag:                # If we want to set the specific type...

            # Set the specific type.
            self.specific_type = specific_type


    # Implement a function to set dimensions.
    def set_dimension( self, dimension, set_flag = True ):

        # Determine whether to set the dimension.
        if set_flag:                # If we want to set the dimension...

            # Set the dimension.
            self.dimension = dimension


    # Implement a function to set condition functions.
    def set_condition_functions( self, condition_functions, set_flag = True ):

        # Determine whether to set the condition_functions.
        if set_flag:                # If we want to set the condition_functions...

            # Set the condition_functions.
            self.condition_functions = condition_functions


    # Implement a function to set placements.
    def set_placement( self, placement, set_flag = True ):

        # Determine whether to set the placement.
        if set_flag:                # If we want to set the placement...

            # Set the placement.
            self.placement = placement


    # Implement a function to set the number of condition functions.
    def set_num_condition_functions( self, num_condition_functions, set_flag = True ):

        # Determine whether to set the number of condition functions.
        if set_flag:                                # If we want to set the number of condition functions...

            # Set the number of condition functions.
            self.num_condition_functions = num_condition_functions
        

    #%% ------------------------------------------------------------ VALIDATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate the initial-boundary condition ID.
    def validate_id( self, id, set_flag = False ):

        # Determine whether to validate the initial-boundary condition ID.
        if not self.is_id_valid( id ):                          # If the ID is not valid...

            # Throw an error.
            raise ValueError( f'Invalid id: {id}' )

        # Set the id (as required).
        self.set_id( id, set_flag )

        # Return the ID.
        return id


    # Implement a function to validate the initial-boundary condition name.
    def validate_name( self, name, set_flag = False ):

        # Determine whether to set the initial-boundary condition name.
        if not self.is_name_valid( name ):                          # If the name is not valid...

            # Throw an error.
            raise ValueError( f'Invalid name: {name}' )

        # Set the name (as required).
        self.set_name( name, set_flag )

        # Return the name.
        return name


    # Implement a function to validate the initial-boundary condition general type.
    def validate_general_type( self, general_type, set_flag = False ):

        # Determine whether to set the initial-boundary condition general type.
        if not self.is_general_type_valid( general_type ):                          # If the general type is not valid...

            # Throw an error.
            raise ValueError( f'Invalid general type: {general_type}' )

        # Set the general type (as required).
        self.set_general_type( general_type, set_flag )

        # Return the general type.
        return general_type


    # Implement a function to validate the initial-boundary condition specific type.
    def validate_specific_type( self, specific_type, set_flag = False ):

        # Determine whether to set the initial-boundary condition specific type.
        if not self.is_specific_type_valid( specific_type ):                          # If the specific type is not valid...

            # Throw an error.
            raise ValueError( f'Invalid specific type: {specific_type}' )

        # Set the specific type (as required).
        self.set_specific_type( specific_type, set_flag )

        # Return the specific type.
        return specific_type


    # Implement a function to validate the initial-boundary condition dimension.
    def validate_dimension( self, dimension, set_flag = False ):

        # Determine whether to set the initial-boundary condition dimension.
        if not self.is_dimension_valid( dimension ):                          # If the dimension is not valid...

            # Throw an error.
            raise ValueError( f'Invalid dimension: {dimension}' )

        # Set the dimension.
        self.set_dimension( dimension, set_flag )

        # Return the dimension.
        return dimension


    # Implement a function to set the initial-boundary condition function.
    def validate_condition_function( self, condition_functions, set_flag = False ):

        # Determine whether to set the initial-boundary condition function.
        if self.is_condition_function_valid( condition_functions ):                             # If the condition function is valid...

            # Determine whether to embed the condition functions into a list.
            if callable( condition_functions ):                                                 # If the condition functions variable is itself a callable function...

                # Embed the condition functions into a list.
                condition_functions = [ condition_functions ]

        else:                                                   # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid condition function: {condition_functions}' )

        # Set the condition functions (as required).
        self.set_condition_functions( condition_functions, set_flag )

        # Return the condition function.
        return condition_functions


    # Implement a function to set the initial-boundary condition placement.
    def validate_placement( self, placement, set_flag = False ):

        # Determine whether to set the initial-boundary condition placement.
        if not self.is_placement_valid( placement ):                          # If the placement is not valid...

            # Throw an error.
            raise ValueError( f'Invalid placement: {placement}' )

        # Set the placement (as required).
        self.set_placement( placement, set_flag )

        # Return the placement.
        return placement


    #%% ------------------------------------------------------------ COMPUTE FUNCTIONS ------------------------------------------------------------

    # Implement a function to compute the number of condition functions.
    def compute_number_of_condition_functions( self, condition_functions = None, set_flag = False ):

        # Preprocess the condition functions.
        condition_functions = self.preprocess_condition_functions( condition_functions )

        # Determine how many condition functions there are.
        if callable( condition_functions ):                                                                         # If the conditions function object is itself callable...

            # Set the number of condition functions to one.
            num_condition_functions = torch.tensor( 1, dtype = torch.uint8, device = self.device )

        elif isinstance( condition_functions, list ) or isinstance( condition_functions, tuple ):                   # If the condition function object is a list or tuple...

            # Set the number of condition functions to be the length of the list or tuple.
            num_condition_functions = len( condition_functions )

        else:                                                                                                       # Otherwise... (i.e., the condition function object type is not recognized...)

            # Throw an error.
            raise ValueError( f'Invalid condition functions: {condition_functions}' )

        # Set the number of condition functions (as required).
        self.set_num_condition_functions( num_condition_functions, set_flag )

        # Return the number of condition functions.
        return num_condition_functions


    #%% ------------------------------------------------------------ PRINTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to print initial boundary condition information.
    def print( self, num_dashes = 20, decoration_flag = True ):

        # Print a header.
        self.printing_utilities.print_header( 'INITIAL-BOUNDARY CONDITION SUMMARY', num_dashes, decoration_flag )

        # Print the domain information
        print( 'General Information' )
        print( f'Device: {self.device}' )
        print( f'ID: {self.id}' )
        print( f'Name: {self.name}' )
        print( f'General Type: {self.general_type}' )
        print( f'Specific Type: {self.specific_type}' )
        print( f'Dimension: {self.dimension}' )
        print( f'Condition Functions: {self.condition_functions}' )
        print( f'# of Condition Functions: {self.num_condition_functions}' )
        print( f'Placement: {self.placement}' )

        # Print a footer.
        self.printing_utilities.print_footer( num_dashes, decoration_flag )


    #%% ------------------------------------------------------------ PLOTTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to plot the initial-boundary condition over some provided points.
    def plot( self, input_data, condition_functions = None, projection_dimensions = None, projection_values = None, level = 0, fig = None, input_labels = None, title_string = 'Initial-Boundary Condition Plot', save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Preprocess the condition functions.
        condition_functions = self.preprocess_condition_functions( condition_functions )

        # Compute the number of condition functions.
        num_condition_functions = self.compute_number_of_condition_functions( condition_functions )

        # Determine whether to create a list of None values for figures.
        if fig is None:                 # If no figure was provided...

            # Create a list of Nones.
            fig = [ None ]*num_condition_functions

        # Initialize a variable to store the figures and axes.
        figs = [  ]
        axes = [  ]

        # Plot each of the initial-boundary condition functions.
        for k in range( num_condition_functions ):                     # Iterate through each condition function...

            # Modify the title string.
            title_string_modified = title_string + f' (Cond. Number {k})'

            # Plot the initial-boundary condition function over the given input data.
            fig, ax = self.plotting_utilities.plot( input_data, condition_functions[ k ], projection_dimensions, projection_values, level, fig[ k ], input_labels, title_string_modified, save_directory, as_surface, as_stream, as_contour, show_plot )

            # Append these figures and axes to the list.
            figs.append( fig )
            axes.append( ax )

        # Return the figures and axes.
        return figs, axes

