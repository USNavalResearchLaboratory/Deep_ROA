#%% ------------------------------------------------------------ SAVE-LOAD UTILITIES CLASS ------------------------------------------------------------

# This file implements a class for storing and managing save-load utilities information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import dill as pickle

# Import custom libraries.


#%% ------------------------------------------------------------ SAVE-LOAD UTILITIES CLASS ------------------------------------------------------------

# Implement the save-load utilities class.
class save_load_utilities_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self, save_path = None, load_path = None ):

        # Store the save and load paths.
        self.save_path = save_path
        self.load_path = load_path


    #%% ------------------------------------------------------------ SAVE & LOAD FUNCTIONS ------------------------------------------------------------

    # Implement a function to save an object.
    def save( self, data, save_path = None, file_name = r'temp.pkl' ):

        # Determine whether to use a predefined save path.
        if save_path is None:                                           # If no save path was provided...

            # Determine which predefined save path to use.
            if self.save_path is not None:                              # If there is a stored save path...

                # Set the save path to be the stored save path.
                save_path = self.save_path

            else:                                                       # Otherwise...

                # Set the save path to be the current folder.
                save_path = r'.'

            # Use the stored save path.
            save_path = self.save_path

        # Create the  file path.
        file_path = save_path + '/' + file_name

        # Open ( or create ) a temporary file to store the object.
        with open( file_path, 'wb' ) as file:               # With the object file open...

            # Save the object.
            pickle.dump( data, file )


    # Implement a function to load an object.
    def load( self, load_path = None, file_name = r'temp.pkl' ):

        # Determine whether to use a predefined load path.
        if load_path is None:                                           # If no load path was provided...

            # Determine which predefined load path to use.
            if self.load_path is not None:                              # If there is a stored load path...

                # Use the stored load path.
                load_path = self.load_path

            else:                                                       # Otherwise...

                # Use the current folder.
                load_path = r'.'

        # Create the  file path.
        file_path = load_path + '/' + file_name

        # Open the object file.
        with open( file_path, 'rb' ) as file:                   # With the object file open...

            # Load the object.
            data = pickle.load( file )

        # Return the object.
        return data