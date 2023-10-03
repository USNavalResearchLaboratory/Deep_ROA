#%% ------------------------------------------------------------ PRINTING UTILITIES CLASS ------------------------------------------------------------

# This file implements a class for storing and managing printing utilities information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.

# Import custom libraries.


#%% ------------------------------------------------------------ SAVE-LOAD UTILITIES CLASS ------------------------------------------------------------

# Implement the printing utilities class.
class printing_utilities_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self ):

        # Pass through the constructor.
        pass

    #%% ------------------------------------------------------------ BASIC FUNCTIONS ------------------------------------------------------------

    # Implement a function to print a horizontal line.
    def print_horizontal_line( self, num_dashes = 20 ):

        # Print a horizontal line.
        print( '-'*num_dashes )
        print( '\n' )


    # Implement a function to print headers.
    def print_header( self, header_title, num_dashes = 20, header_flag = True ):

        # Determine whether to print a header.
        if header_flag:                     # If we want to print a header...

            # Determine whether to print a header.
            self.print_horizontal_line( num_dashes )
            print( header_title )
            self.print_horizontal_line( num_dashes )

        else:                               # Otherwise...

            # Print a new line.
            print( '\n' )


    # Implement a function to print footers.
    def print_footer( self, num_dashes = 20, footer_flag = True ):

        # Determine whether to print a footer.
        if footer_flag:                             # If we want to print a footer...

            # Print a horizontal line.
            self.print_horizontal_line( num_dashes )

        else:                                       # Otherwise...

            # Print a new line.
            print( '\n' )
