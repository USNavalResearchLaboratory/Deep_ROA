####################################################################################### 
# THIS SOURCE CODE IS PROPERTY OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. 
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND 
# CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION 
# ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN 
# LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE,
# CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL 
# PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641. 
####################################################################################### 


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
