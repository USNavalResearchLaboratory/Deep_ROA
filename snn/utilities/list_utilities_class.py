#%% ------------------------------------------------------------ LIST UTILITIES CLASS ------------------------------------------------------------

# This file implements a class for storing and managing list utilities information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch
import warnings

# Import custom libraries.


#%% ------------------------------------------------------------ LIST UTILITIES CLASS ------------------------------------------------------------

# Implement the list utilities class.
class list_utilities_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self ):

        # Pass through the constructor.
        pass


    #%% ------------------------------------------------------------ BASIC FUNCTIONS ------------------------------------------------------------

    # Implement a function to insert a list of values at a list of locations.
    def insert_list_at_list( self, start_list, index_list, element_list, adapt_indexes_flag = True ):

        # Copy the starting list.
        new_list = start_list[ : ]

        # Copy the index list.
        index_list = index_list[ : ]

        # Retrieve the number of elements to insert.
        num_insertions = len( index_list )

        # Insert each element one at a time.
        for k1 in range( num_insertions ):                # Iterate through each of the insertion elements...

            # Insert this element.
            new_list.insert( index_list[ k1 ], element_list[ k1 ] )

            # Determine whether to advance the indexes.
            if adapt_indexes_flag:                   # If the user has specified that we should advance the indexes...

                # Advance the other insert locations (if necessary)
                for k2 in range( k1 + 1, num_insertions ):        # Iterate through each of the remaining insertion points...

                    # Determine whether to advance the insert locations.
                    if ( index_list[ k2 ] > index_list[ k1 ] ):           # If this index is after the current index...

                        # Advance the current index.
                        index_list[ k2 ] += 1

        # Return the new list.
        return new_list