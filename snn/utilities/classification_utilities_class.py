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
#%% ------------------------------------------------------------ CLASSIFICATION UTILITIES CLASS ------------------------------------------------------------

# This file implements a class for storing and managing classification utilities information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch
import matplotlib.pyplot as plt

# Import custom libraries.
from tensor_utilities_class import tensor_utilities_class as tensor_utilities_class
from plotting_utilities_class import plotting_utilities_class as plotting_utilities_class


#%% ------------------------------------------------------------ CLASSIFICATION UTILITIES CLASS ------------------------------------------------------------

# Implement the classification utilities class.
class classification_utilities_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self ):

        # Create an instance of the tensor utilities class.
        self.tensor_utilities = tensor_utilities_class(  )

        # Create an instance of the plotting utilities class.
        self.plotting_utilities = plotting_utilities_class(  )


    #%% ------------------------------------------------------------ VALIDATION FUNCTIONS ------------------------------------------------------------




    #%% ------------------------------------------------------------ CLASSIFICATION POINTS FUNCTIONS ------------------------------------------------------------

    # Implement a function to generate classification points randomly.
    def generate_classification_points( self, domain, num_samples ):

        # Sample the classification points from the domain.
        classification_points = domain.sample_domain( num_samples )

        # Return the classification points.
        return classification_points


    # Implement a function to retrieve ROA boundary points from provided network input and output data.
    def generate_roa_boundary_points_data( self, network_input_data, network_output_data, level = None, device = 'cpu' ):

        # Generate a dummy figure.
        

        # Retrieve the contour set.
        contour_set = plt.contour( self.plotting_utilities.plot_process( network_input_data[ ..., 0 ] ), self.plotting_utilities.plot_process( network_input_data[ ..., 1 ] ), self.plotting_utilities.plot_process( network_output_data[ ..., 0 ] ), levels = [ self.plotting_utilities.plot_process( level ) ], colors = 'red', linewidths = 2.0 )

        # Retrieve the contour data.
        contour_data = torch.tensor( contour_set._get_allsegs_and_allkinds(  )[ 0 ][ 0 ][ 0 ], dtype = torch.float32, device = device )

        # Return the contour data.
        return contour_data


    # Implement a function to retrieve ROA boundary points from a provided network and its input data.
    def generate_roa_boundary_points_network( self, network_input_data, network, level = None, device = 'cpu' ):

        # Compute the network output.
        network_output_data = network.forward( network_input_data )

        # Generate the contour data.
        contour_data = self.generate_roa_boundary_points_data( network_input_data, network_output_data, level = level, device = device )

        # Return the contour data.
        return contour_data


    # Implement a function to retrieve ROA boundary points given user specified network information.
    def generate_roa_boundary_points( self, network_input_data, network_info, level = None, device = 'cpu' ):

        # Determine whether the network info is a network object.
        if type( network_info ).__name__ == 'neural_network_class':                # If the provided network info is a neural network...

            # Compute the contour data using the provided network.
            contour_data = self.generate_roa_boundary_points_network( network_input_data, network_info, level, device )

        elif torch.istensor( network_info ):                                # If the provided network info is a tensor...

            # Compute the contour data using the provided network output data.
            contour_data = self.generate_roa_boundary_points_data( network_input_data, network_info, level, device )

        else:                                                               # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid Network Info: {network_info}' )

        # Return the contour data.
        return contour_data


    # Implement a function to generate classification points along the ROA boundary.


    # Implement a function to generate classification points using the user specified algorithm.


    #%% ------------------------------------------------------------ NETWORK CLASSIFICATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to determine the network's classification of a point by referencing provided network input and output data.


    # Implement a function to determine the network's classification of a point by referencing a provided network and its input data.


    # Implement a function to determine the network's classification of a point given user specified network information.


    #%% ------------------------------------------------------------ 'TRUE' CLASSIFICATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to determine the true classification of a point by referencing a provided analytical or numerical solution.



    # Implement a function to determine an approximately 'true' classification of a point by determining whether its network classification changes over a finite time horizon.



    # Implement a function to determine the true classification of a point by checking a user provided energy function.

