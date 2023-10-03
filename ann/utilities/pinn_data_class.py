####################################################################################### 
# THIS SOURCE CODE IS PROPERTY OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. 
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND 
# CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION 
# ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN 
# LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE,
# CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL 
# PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641. 
####################################################################################### 


#%% ------------------------------------------------------------ PINN DATA CLASS ------------------------------------------------------------

# This file implements a class for storing and managing pinn data information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Import custom libraries.
from plotting_utilities_class import plotting_utilities_class as plotting_utilities_class
from printing_utilities_class import printing_utilities_class as printing_utilities_class


#%% ------------------------------------------------------------ PINN DATA CLASS ------------------------------------------------------------

# Implement the pinn data class.
class pinn_data_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self, id, name, dimension_labels, batch_size = None, device = 'cpu' ):

        # Create an instance of the plotting utilities class.
        self.plotting_utilities = plotting_utilities_class(  )

        # Create an instance of the printing utilities class.
        self.printing_utilities = printing_utilities_class(  )

        # Store the device.
        self.device = device

        # Set the pinn data information.
        self.id = self.validate_id( id )
        self.name = self.validate_name( name )
        self.dimension_labels = self.validate_dimension_labels( dimension_labels )

        # Set the batch size.
        self.batch_size = self.validate_batch_size( batch_size )


    #%% ------------------------------------------------------------ IS VALID FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate the id.
    def is_id_valid( self, id ):

        # Determine whether the given ID is valid.
        if torch.is_tensor( id ) and ( ( id.dtype == torch.uint8 ) or ( id.dtype == torch.int8 ) or ( id.dtype == torch.int16 ) or ( id.dtype == torch.int32 ) or ( id.dtype == torch.int64 ) ) and ( id > 0 ):                  # If the id is valid...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                                               # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the name.
    def is_name_valid( self, name ):

        # Determine whether the given name is valid.
        if isinstance( name, str ):                     # If the name is valid...

            # Set the valid flag to true.
            valid_flag = True

        else:                                           # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the dimension labels
    def is_dimension_labels_valid( self, dimension_labels ):

        # Determine whether the given dimension labels are valid.
        if isinstance( dimension_labels, list ):                        # If the dimension labels are a list...

            # Retrieve the number of dimension labels.
            num_dims = len( dimension_labels )

            # Initialize the loop counter variable.
            k = torch.tensor( 0, dtype = torch.int64, device = self.device )

            # Initialize the valid flag to true.
            valid_flag = True

            # Determine whether the dimension labels are valid.
            while valid_flag and ( k < num_dims ):                      # While all of the dimension labels that have been checked are valid and we have not yet checked all of the dimension labels...

                # Check whether this dimension label is valid.
                valid_flag &= ( dimension_labels[ k ].lower(  ) == 't' ) or ( dimension_labels[ k ].lower(  ) == 'x' )

                # Advance the loop counter.
                k += 1

        else:                                                           # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to validate the batch size.
    def is_batch_size_valid( self, batch_size ):

        # Determine whether the given batch size is valid.
        if torch.is_tensor( batch_size ) and ( batch_size.numel(  ) != 0 ):                 # If the batch size is a non-empty tensor...

            # Set the valid flag to true.
            valid_flag = True

        elif batch_size is None:                                                            # If the batch size is None...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                                               # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the validity flag.
        return valid_flag


    #%% ------------------------------------------------------------ SET FUNCTIONS ------------------------------------------------------------

    # Implement a function to set the id.
    def set_id( self, id, set_flag = True ):

        # Determine whether to set the id.
        if set_flag:                # If we want to set the id...

            # Set the id.
            self.id = id

    
    # Implement a function to set the name.
    def set_name( self, name, set_flag = True ):

        # Determine whether to set the name.
        if set_flag:                # If we want to set the name...

            # Set the name.
            self.name = name 

    
    # Implement a function to set the dimension labels.
    def set_dimension_labels( self, dimension_labels, set_flag = True ):

        # Determine whether to set the dimension labels.
        if set_flag:                # If we want to set the dimension labels...

            # Set the dimension labels.
            self.dimension_labels = dimension_labels


    # Implement a function to set the batch size.
    def set_batch_size( self, batch_size, set_flag = True ):

        # Determine whether to set the batch size.
        if set_flag:            # If we want to set the batch size...

            # Set the batch size.
            self.batch_size = batch_size


    #%% ------------------------------------------------------------ VALIDATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to set the id.
    def validate_id( self, id, set_flag = False ):

        # Determine whether the id is valid.
        if not self.is_id_valid( id ):                          # If the id is not valid...

            # Throw an error.
            raise ValueError( f'Invalid id: {id}' )

        # Set the id (as required).
        self.set_id( id, set_flag )

        # Return the id.
        return id


    # Implement a function to set the name.
    def validate_name( self, name, set_flag = False ):

        # Determine whether the name is valid.
        if not self.is_name_valid( name ):                          # If the name is not valid...

            # Throw an error.
            raise ValueError( f'Invalid name: {name}' )

        # Set the name (as required).
        self.set_name( name, set_flag )

        # Return the name.
        return name


    # Implement a function to set the dimension labels.
    def validate_dimension_labels( self, dimension_labels, set_flag = False ):

        # Determine whether the dimension labels are valid.
        if not self.is_dimension_labels_valid( dimension_labels ):                      # If the dimension labels are not valid...

            # Throw an error.
            raise ValueError( f'Invalid dimension labels: {dimension_labels}' )

        # Set the dimensional labels (as required).
        self.set_dimension_labels( dimension_labels, set_flag )

        # Return the dimension labels.
        return dimension_labels


    # Implement a function to set the batch size.
    def validate_batch_size( self, batch_size, set_flag = False ):

        # Determine whether the batch size is valid.
        if not self.is_batch_size_valid( batch_size ):                          # If the batch size is not valid...

            # Throw an error.
            raise ValueError( f'Invalid batch size: {batch_size}' )

        # Set the batch size.
        self.set_batch_size( batch_size, set_flag )

        # Return the batch size.
        return batch_size

    
    #%% ------------------------------------------------------------ PRINTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to print a summary of the pinn data.
    def print( self, num_dashes = 20, decoration_flag = True ):

        # Print a header.
        self.printing_utilities.print_header( f'PINN DATA {self.id} SUMMARY: {self.name}', num_dashes, decoration_flag )

        # Print the general information.
        print( f'General Information' )
        print( f'Device: {self.device}' )
        print( f'Input Dimension Labels: {self.dimension_labels}' )
        print( f'Batch Size: {self.batch_size}' )

        # Print a footer.
        self.printing_utilities.print_footer( num_dashes, decoration_flag )

