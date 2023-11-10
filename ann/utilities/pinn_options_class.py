####################################################################################### 
# THIS SOURCE CODE IS PROPERTY OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. 
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND 
# CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION 
# ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN 
# LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE,
# CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL 
# PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641. 
####################################################################################### 


#%% ------------------------------------------------------------ PINN OPTIONS CLASS ------------------------------------------------------------

# This file implements a class for storing and managing pinn options information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.


# Import custom libraries.
from save_load_utilities_class import save_load_utilities_class as save_load_utilities_class
from printing_utilities_class import printing_utilities_class as printing_utilities_class


#%% ------------------------------------------------------------ PINN OPTIONS CLASS ------------------------------------------------------------

# Implement the pinn options class.
class pinn_options_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self, save_path, save_frequency, save_flag, load_path, load_flag, train_flag, batch_print_frequency, epoch_print_frequency, print_flag, num_plotting_samples, newton_tolerance, newton_max_iterations, exploration_volume_percentage, num_exploration_points, unique_volume_percentage, classification_noise_percentage, num_noisy_samples_per_level_set_point, classification_dt, classification_tfinal, plot_flag, device, verbose_flag ):

        # Create an instance of the save-load utilities class.
        self.save_load_utilities = save_load_utilities_class(  )

        # Create an instance of the printing utilities class.
        self.printing_utilities = printing_utilities_class(  )

        # Store the saving information.
        self.save_path = save_path
        self.save_frequency = save_frequency
        self.save_flag = save_flag

        # Store the loading information.
        self.load_path = load_path
        self.load_flag = load_flag

        # Store the training information.
        self.train_flag = train_flag

        # Store the printing information.
        self.batch_print_frequency = batch_print_frequency
        self.epoch_print_frequency = epoch_print_frequency
        self.print_flag = print_flag

        # Store the plotting information.
        self.num_plotting_samples = num_plotting_samples
        self.plot_flag = plot_flag

        # Store the newton parameters.
        self.newton_tolerance = newton_tolerance
        self.newton_max_iterations = newton_max_iterations
        
        # Store the exploration parameters.
        self.exploration_volume_percentage = exploration_volume_percentage
        self.num_exploration_points = num_exploration_points
        self.unique_volume_percentage = unique_volume_percentage

        # Store the classification parameters.
        self.num_noisy_samples_per_level_set_point = num_noisy_samples_per_level_set_point
        self.classification_noise_percentage = classification_noise_percentage
        self.classification_dt = classification_dt
        self.classification_tfinal = classification_tfinal

        # Store the computational device.
        self.device = device

        # Store the verbosity flag.
        self.verbose_flag = verbose_flag


    #%% ------------------------------------------------------------ PINN FUNCTIONS ------------------------------------------------------------

    # Implement a function to print the pinn options.
    def print( self, num_dashes = 20, decoration_flag = True ):

        # Print a header.
        self.printing_utilities.print_header( 'PINN OPTIONS SUMMARY', num_dashes, decoration_flag )

        # Print general information.
        print( 'General Information' )
        print( f'Device: {self.device}' )
        print( '\n' )

        # Print the saving information.
        print( 'Saving Information' )
        print( f'Save Path: {self.save_path}' )
        print( f'Save Frequency: {self.save_frequency}' )
        print( f'Save Flag: {self.save_flag}' )
        print( '\n' )

        # Print the loading information.
        print( 'Loading Information' )
        print( f'Load Path: {self.load_path}' )
        print( f'Load Flag: {self.load_flag}' )
        print( '\n' )

        # Print the training information.
        print( 'Training Information' )
        print( f'Training Flag: {self.train_flag}' )
        print( '\n' )

        # Print the printing information.
        print( 'Printing Information' )
        print( f'Batch Print Frequency: {self.batch_print_frequency}' )
        print( f'Epoch Print Frequency: {self.epoch_print_frequency}' )
        print( f'Print Flag: {self.print_flag}' )
        print( f'Verbosity Flag: {self.verbose_flag}' )
        print( '\n' )

        # Store the plotting information.
        print( 'Plotting Information' )
        print( f'# of Plotting Samples: {self.num_plotting_samples}' )
        print( f'Plot Flag: {self.plot_flag}' )

        # Print a footer.
        self.printing_utilities.print_footer( num_dashes, decoration_flag )



    #%% ------------------------------------------------------------ SAVE & LOAD FUNCTIONS ------------------------------------------------------------

    # Implement a function to save the pinn options.
    def save( self, save_path = None, file_name = r'pinn_options.pkl' ):

        # Determine whether to use the stored save path.
        if save_path is None:                               # If the save path was not provided...

            # Use the stored save path.
            save_path = self.save_path

        # Save the pinn options.
        self.save_load_utilities.save( self, save_path, file_name )


    # Implement a function to load pinn options.
    def load( self, load_path = None, file_name = r'pinn_options.pkl' ):

        # Determine whether to use the stored load path.
        if load_path is None:                               # If the load path was not provided...

            # Use the stored load path.
            load_path = self.load_path

        # Load the pinn options.
        self = self.save_load_utilities.load( load_path, file_name )

        # Return the pinn options.
        return self

