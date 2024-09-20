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
# FINITE ELEMENT CLASS

# This script implements the finite element class which includes properties and methods related to constructing and integrating finite elements.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import custom libraries.
from tensor_utilities_class import tensor_utilities_class as tensor_utilities_class
from plotting_utilities_class import plotting_utilities_class as plotting_utilities_class
from printing_utilities_class import printing_utilities_class as printing_utilities_class


#%% ------------------------------------------------------------ FINITE ELEMENT CLASS ------------------------------------------------------------

# Implement the finite element class.
class finite_element_class:
    
    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the finite element class constructor.
    def __init__( self, element_scale, integration_order, element_type = 'rectangular', device = 'cpu' ):

        # Note that the residual matrix needs to be constructed outside of this class because it relies on the current network state.

        # Create an instance of the tensor utilities class.
        self.tensor_utilities = tensor_utilities_class(  )

        # Create an instance of the plotting utilities class.
        self.plotting_utilities = plotting_utilities_class(  )

        # Create an instance of the printing utilities class.
        self.printing_utilities = printing_utilities_class(  )

        # Store the device.
        self.device = device

        # Store the element scale.
        self.element_scale = self.validate_element_scale( element_scale )

        # Store the Gauss-Legendre integration order.
        self.integration_order = self.validate_integration_order( integration_order )

        # Store the element scale.
        self.element_type = self.validate_element_type( element_type )

        # Store the number of dimensions.
        self.num_dimensions = torch.tensor( self.element_scale.numel(  ), dtype = torch.uint8, device = self.device )

        # Compute the number of points per element.
        self.num_points_per_element = self.compute_num_points_per_element( self.integration_order, self.num_dimensions )

        # Compute the number of elements.
        self.num_elements = torch.tensor( 0, dtype = torch.int64, device = self.device )

        # Set the number of basis functions.
        self.num_basis_functions = self.compute_num_basis_functions( self.integration_order, self.num_dimensions )

        # Get the template element points, basis functions, and Gauss-Legendre weights.
        self.xis_1D_template_integration_points = self.compute_1D_template_integration_points( self.integration_order )                     # 1D Gauss-Legendre Element Points Template.
        self.gs_template_basis_functions = self.compute_template_basis_functions( self.integration_order )                                  # 1D Gauss-Legendre Basis Functions Template.
        self.ws_template_weights = self.compute_template_weights( self.integration_order )                                                  # 1D Gauss-Legendre Weights Template.

        # Construct the integration points for a single template element centered at the origin.
        self.xis_template_integration_points = self.compute_template_integration_points( self.xis_1D_template_integration_points, self.num_dimensions, self.integration_order )                                    # ND Gauss-Legendre Element Points.

        # Set the integration points to be None.
        self.xs_integration_points = None

        # Set the basis evaluation matrix to be None.
        self.G_basis_values = None

        # Set the integration weight matrix to None.
        self.W_integration_weights = None

        # Set the element scaling jacobian matrix to be None.
        self.sigma_jacobian = None


    #%% ------------------------------------------------------------ IS VALID FUNCTIONS ------------------------------------------------------------

    # Implement a function to check the validity of the gauss-legendre order.
    def is_integration_order_valid( self, integration_order ):

        # Determine whether the gauss-legendre order is valid.
        if ( integration_order == 1 ) or ( integration_order == 2 ):                  # If the gauss-legendre order is one or two...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                                               # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the validation flag.
        return valid_flag


    # Implement a function to check the validity of the number of elements.
    def is_num_elements_valid( self, num_elements ):

        # Determine whether the number of elements is valid.
        if ( num_elements > 0 ):                    # If the number of elements is greater than zero...

            # Set the valid flag to true.
            valid_flag = True

        else:                                       # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to check the validity of the number of basis functions.
    def is_num_basis_functions_valid( self, num_basis_functions, num_points_per_element = None ):

        # Determine whether to use the stored number of points per element.
        if num_points_per_element is None:              # If the number of points per element was not provided...

            # Set the number of points per element to be the stored number of points per element.
            num_points_per_element = self.num_points_per_element
        
        # Determine whether the number of basis functions is valid.
        if ( num_basis_functions >= 1 ) and ( num_basis_functions <= self.num_points_per_element ):                 # If the number of desired basis functions is greater than or equal to one and less than or equal to the number of points per element...

            # Set the valid flag to be true.
            valid_flag = True

        else:                                                                                                       # Otherwise...

            # Set the valid flag to be false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to check the validity of the element scale.
    def is_element_scale_valid( self, element_scale ):
        
        # Determine whether the element scale is valid.
        if ( element_scale.numel(  ) > 0 ):                     # If the number of element scale values is equal to the number of spatiotemporal dimensions or one...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                                                                                           # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to check the validity of element centers.
    def is_element_centers_valid( self, xs_element_centers, num_dimensions = None ):

        # Determine whether to use the stored number of dimensions.
        if num_dimensions is None:              # If the number of dimensions is None...

            # Set the number of dimensions to be the stored value.
            num_dimensions = self.num_dimensions

        # Determine whether the element centers are valid.
        if xs_element_centers.shape[ -1 ] == num_dimensions:            # If the number of element dimensions matches the number of dimensions...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                           # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    # Implement a function to check the validity of the element type.
    def is_element_type_valid( self, element_type ):

        # Determine whether the element type is valid.
        if element_type.lower(  ) in ( 'rectangular', 'rectangle', 'cartesian' ):             # If the element type is 'rectangular'...

            # Set the valid flag to true.
            valid_flag = True

        else:                                                   # Otherwise...

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag


    #%% ------------------------------------------------------------ PRE-PROCESS FUNCTIONS ------------------------------------------------------------

    # Implement a function to pre-process the element scale.
    def preprocess_element_scale( self, element_scale = None ):

        # Determine whether to use the stored element scale.
        if element_scale is None:                       # If the element scale was not provided...

            # Use the default element scale.
            element_scale = self.element_scale

        # Return the element scale.
        return element_scale


    # Implement a function to pre-process the integration order.
    def preprocess_integration_order( self, integration_order = None ):

        # Determine whether to use the stored gauss-legendre integration order.
        if integration_order is None:                # If no gauss-legendre integration order was provided...

            # Use the stored value gauss-legendre integration order.
            integration_order = self.integration_order

        # Return the integration order.
        return integration_order


    # Implement a function to pre-process the number of dimensions.
    def preprocess_num_dimensions( self, num_dimensions = None ):

        # Determine whether to use the stored number of spatiotemporal variables.
        if num_dimensions is None:        # If no number of spatiotemporal variables was provided...

            # Set the number of spatiotemporal variables to be the stored value.
            num_dimensions = self.num_dimensions

        # Return the number of dimensions.
        return num_dimensions


    # Implement a function to pre-process the number of elements.
    def preprocess_num_elements( self, num_elements ):

        # Determine whether to use the stored number of elements.
        if num_elements is None:        # If the number of elements was not provided...

            # Set the number of elements to be the stored value.
            num_elements = self.num_elements

        # Return the number of elements.
        return num_elements


    # Implement a function to pre-process the number of basis functions.
    def preprocess_num_basis_functions( self, num_basis_functions, integration_order, num_dimensions, set_flag ):

        # Determine whether to use the stored number of basis functions.
        if num_basis_functions is None:                # If no number of basis functions was provided...

            # Determine whether to use the number of basis functions or to generate new ones.
            if self.num_basis_functions is None:        # If there are no stored number of basis functions...

                # Compute the number of basis functions.
                num_basis_functions = self.compute_num_basis_functions( integration_order, num_dimensions, set_flag )

            else:                                   # Otherwise...

                # Set the number of basis functions to be the existing values.
                num_basis_functions = self.num_basis_functions

        # Return the number of basis functions.
        return num_basis_functions


    # Implement a function to pre-process the number of points per element.
    def preprocess_num_points_per_element( self, num_points_per_element, integration_order, num_dimensions, set_flag = False ):

        # Determine whether to use the stored number of points per element.
        if num_points_per_element is None:                # If no number of points per element was provided...

            # Determine whether to use the stored gauss-legendre template points or to generate new ones.
            if self.num_points_per_element is None:        # If there are no stored gauss-legendre template points...

                # Compute the number of points per element.
                num_points_per_element = self.compute_num_points_per_element( integration_order, num_dimensions, set_flag )

            else:                                   # Otherwise...

                # Set the number of points per element to be the existing values.
                num_points_per_element = self.num_points_per_element

        # Return the number of points per element.
        return num_points_per_element


    # Implement a function to pre-process the 1D template integration points.
    def preprocess_1D_template_integration_points( self, xis_1D_template_integration_points, integration_order, set_flag = False ):

        # Determine whether to use the existing gauss-legendre template points.
        if xis_1D_template_integration_points is None:                 # If the gauss-legendre template points were not provided...

            # Determine whether to use the stored gauss-legendre template points or to generate new ones.
            if self.xis_1D_template_integration_points is None:        # If there are no stored gauss-legendre template points...

                # Compute the gauss-legendre template points.
                xis_1D_template_integration_points = self.compute_1D_template_integration_points( integration_order, set_flag ) 

            else:                                                       # Otherwise...

                # Set the gauss-legendre template points to be the existing values.
                xis_1D_template_integration_points = self.xis_1D_template_integration_points

        # Return the 1D template integration points.
        return xis_1D_template_integration_points


    # Implement a function to pre-process the template integration points.
    def preprocess_template_integration_points( self, xis_template_integration_points, xis_1D_template_integration_points, num_dimensions, integration_order, set_flag = False ):

        # Determine whether to use the stored template element points.
        if xis_template_integration_points is None:                 # If no template element points were provided...

            # Determine whether to use the stored template element points.
            if self.xis_template_integration_points is None:        # If there are no stored template integration points...

                # Compute the template integration points.
                xis_template_integration_points = self.compute_template_integration_points( xis_1D_template_integration_points, num_dimensions, integration_order, set_flag )

            else:                                                   # Otherwise... ( i.e., if there are stored template integration points... )    

                # Set the template integration points to be the existing values.
                xis_template_integration_points = self.xis_template_integration_points

        # Return the template integration points.
        return xis_template_integration_points


    # Implement a function to pre-process the template basis functions.
    def preprocess_template_basis_functions( self, gs_template_basis_functions, integration_order, set_flag = False ):

        # Determine whether to use the stored template basis functions.
        if gs_template_basis_functions is None:                 # If the template basis functions were not provided...

            # Determine whether to use the stored template basis functions.
            if self.gs_template_basis_functions is None:        # If there are no stored template basis functions...

                # Compute the template basis functions.
                gs_template_basis_functions = self.compute_template_basis_functions( integration_order, set_flag )

            else:                                               # Otherwise... ( i.e., if there are stored template basis functions... )    

                # Use the stored template basis functions.
                gs_template_basis_functions = self.gs_template_basis_functions

        # Return the template basis functions.
        return gs_template_basis_functions


    # Implement a function to pre-process the template weights.
    def preprocess_template_weights( self, ws_template_weights, integration_order, set_flag = False ):

        # Determine how to process the template integration weights.
        if ws_template_weights is None:             # If the template weights was not provided...

            # Determine whether to set the template integration weights to be the stored value.
            if self.ws_template_weights is None:                    # If there are no stored template weights...

                # Determine whether to compute the template integration weights.
                ws_template_weights = self.compute_template_weights( integration_order, set_flag )

            else:                                                   # Otherwise... ( i.e., there are stored template weights... )

                # Set the template integration weights to be the stored value.
                ws_template_weights = self.ws_template_weights

        # Return the template integration weights.
        return ws_template_weights


    # Implement a function to pre-process the integration points.
    def preprocess_integration_points( self, xs_integration_points ):

        # Determine whether to use the stored integration points.
        if xs_integration_points is None:                       # If the integration points were not provided...

            # Set the integration points to be the stored value.
            xs_integration_points = self.xs_integration_points

        # Return the integration points.
        return xs_integration_points


    # Implement a function to pre-process the basis values.
    def preprocess_basis_values( self, G_basis_values ):

        # Determine whether to use the stored basis values.
        if G_basis_values is None:                              # If the basis values were not provided...

            # Set the basis values to be the stored value.
            G_basis_values = self.G_basis_values

        # Return the basis values.
        return G_basis_values


    # Implement a function to pre-process the integration weights.
    def preprocess_integration_weights( self, W_integration_weights ):

        # Determine whether to use the stored integration weights.
        if W_integration_weights is None:                       # If the integration weights were not provided...

            # Set the integration weights to be the stored value.
            W_integration_weights = self.W_integration_weights

        # Return the integration weights.
        return W_integration_weights


    # Implement a function to pre-process the jacobian.
    def preprocess_jacobian( self, sigma_jacobian ):

        # Determine whether to use the stored jacobian.
        if sigma_jacobian is None:                              # If the jacobian is None:

            # Set the jacobian to be the stored value.
            sigma_jacobian = self.sigma_jacobian

        # Return the jacobian.
        return sigma_jacobian
        

    # Implement a function to pre-process the shuffle indexes.
    def preprocess_shuffle_indexes( self, shuffle_indexes = None ):

        # Determine whether to generate shuffle shuffle_indexes.
        if shuffle_indexes is None:                     # If shuffle shuffle_indexes where not provided...

            # Generate shuffled shuffle_indexes.
            shuffle_indexes = torch.randperm( self.num_elements, dtype = torch.int64, device = self.device )

        # Return the shuffle indexes.
        return shuffle_indexes

    
    # Implement a function to preprocess the batch number.
    def preprocess_batch_number( self, batch_number = None ):

        # Determine whether to use the default batch number.
        if batch_number is None:                # If the batch number was not provided...

            # Set the batch number to be zero.
            batch_number = torch.tensor( 0, dtype = torch.int64, device = self.device )

        # Return the batch number.
        return batch_number


    # Implement a function to preprocess the batch size.
    def preprocess_batch_size( self, batch_size = None ):

        # Determine whether to use the default batch size.
        if batch_size is None:              # If the batch size was not provided...

            # Set the batch size.
            batch_size = self.num_elements

        # Return the batch size.
        return batch_size


    # Implement a function to preprocess the element data.
    def preprocess_element_data( self, xs_integration_points = None, G_basis_values = None, W_integration_weights = None, sigma_jacobian = None, num_elements = None ):

        # Preprocess the integration points.
        xs_integration_points = self.preprocess_integration_points( xs_integration_points )

        # Preprocess the basis values.
        G_basis_values = self.preprocess_basis_values( G_basis_values )

        # Preprocess the integration weights.
        W_integration_weights = self.preprocess_integration_weights( W_integration_weights )

        # Preprocess the jacobian.
        sigma_jacobian = self.preprocess_jacobian( sigma_jacobian )

        # Preprocess the number of elements.
        num_elements = self.preprocess_num_elements( num_elements )

        # Return the element data.
        return xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements


    #%% ------------------------------------------------------------ SETUP FUNCTIONS ------------------------------------------------------------

    # Implement a function to setup the number of basis functions.
    def setup_num_basis_functions( self, integration_order = None, num_dimensions = None ):

        # Pre-process the integration order.
        integration_order = self.preprocess_integration_order( integration_order )

        # Pre-process the number of dimensions.
        num_dimensions = self.preprocess_num_dimensions( num_dimensions )

        # Return the integration order and number of dimensions.
        return integration_order, num_dimensions


    # Implement a function to setup the number of points per element.
    def setup_num_points_per_element( self, integration_order = None, num_dimensions = None ):

        # Pre-process the integration order.
        integration_order = self.preprocess_integration_order( integration_order )

        # Pre-process the number of dimensions.
        num_dimensions = self.preprocess_num_dimensions( num_dimensions )

        # Return the integration order and the number of dimensions.
        return integration_order, num_dimensions


    # Implement a function to setup the template integration points.
    def setup_template_integration_points( self, xis_1D_template_integration_points = None, integration_order = None, num_dimensions = None, set_flag = False ):
        
        # Pre-process the integration order.
        integration_order = self.preprocess_integration_order( integration_order )

        # Pre-process the number of dimensions.
        num_dimensions = self.preprocess_num_dimensions( num_dimensions )

        # Pre-process the xis 1D template integration points.
        xis_1D_template_integration_points = self.preprocess_1D_template_integration_points( xis_1D_template_integration_points, integration_order, set_flag )

        # Return the integration order, number of dimensions, and 1D template integration points.
        return integration_order, num_dimensions, xis_1D_template_integration_points


    # Implement a function to setup the integration points.
    def setup_integration_points( self, xis_template_integration_points = None, xis_1D_template_integration_points = None, num_points_per_element = None, integration_order = None, num_dimensions = None, element_scale = None, set_flag = False ):

        # Pre-process the element scale.
        element_scale = self.preprocess_element_scale( element_scale )

        # Pre-process the number of dimensions.
        num_dimensions = self.preprocess_num_dimensions( num_dimensions )

        # Pre-process integration order.
        integration_order = self.preprocess_integration_order( integration_order )

        # Pre-process the number of points per element.
        num_points_per_element = self.preprocess_num_points_per_element( num_points_per_element, integration_order, num_dimensions, set_flag )
           
        # Pre-process the 1D template integration points.
        xis_1D_template_integration_points = self.preprocess_1D_template_integration_points( xis_1D_template_integration_points, integration_order, set_flag )

        # Pre-process the template integration points.
        xis_template_integration_points = self.preprocess_template_integration_points( xis_template_integration_points, xis_1D_template_integration_points, num_dimensions, integration_order, set_flag )

        # Return the template integration points, 1D template integration points, number of points per element, integration order, number of dimensions, and element scale.
        return element_scale, num_dimensions, integration_order, num_points_per_element, xis_1D_template_integration_points, xis_template_integration_points


    # Implement a function to setup the basis functions.
    def setup_basis_values( self, gs_template_basis_functions = None, xis_template_integration_points = None, xis_1D_template_integration_points = None, num_basis_functions = None, num_points_per_element = None, integration_order = None, num_dimensions = None, num_elements = None, set_flag = False ):

        # Pre-process the number of elements.
        num_elements = self.preprocess_num_elements( num_elements )

        # Pre-process the number of dimensions.
        num_dimensions = self.preprocess_num_dimensions( num_dimensions )

        # Pre-process the integration order.
        integration_order = self.preprocess_integration_order( integration_order )

        # Pre-process the number fo points per element.
        num_points_per_element = self.preprocess_num_points_per_element( num_points_per_element, integration_order, num_dimensions, set_flag )

        # Pre-process the number of basis functions.
        num_basis_functions = self.preprocess_num_basis_functions( num_basis_functions, integration_order, num_dimensions, set_flag )

        # Pre-process the 1D template integration points.
        xis_1D_template_integration_points = self.preprocess_1D_template_integration_points( xis_1D_template_integration_points, integration_order, set_flag )

        # Pre-process the template integration points.
        xis_template_integration_points = self.preprocess_template_integration_points( xis_template_integration_points, xis_1D_template_integration_points, num_dimensions, integration_order, set_flag )

        # Pre-process the template basis functions.
        gs_template_basis_functions = self.preprocess_template_basis_functions( gs_template_basis_functions, integration_order, set_flag )

        # Return the basis function information.
        return num_elements, num_dimensions, integration_order, num_points_per_element, num_basis_functions, xis_1D_template_integration_points, xis_template_integration_points, gs_template_basis_functions


    # Implement a function to setup the integration weights.
    def setup_integration_weights( self, ws_template_weights = None, num_basis_functions = None, integration_order = None, num_dimensions = None, num_elements = None, set_flag = False ):

        # Pre-process the number of elements.
        num_elements = self.preprocess_num_elements( num_elements )

        # Pre-process the number of dimensions.
        num_dimensions = self.preprocess_num_dimensions( num_dimensions )

        # Pre-process the integration order.
        integration_order = self.preprocess_integration_order( integration_order )

        # Pre-process the number of basis functions.
        num_basis_functions = self.preprocess_num_basis_functions( num_basis_functions, integration_order, num_dimensions, set_flag )

        # Pre-process the template integration weights.
        ws_template_weights = self.preprocess_template_weights( ws_template_weights, integration_order, set_flag )

        # Return the integration weight information.
        return num_elements, num_dimensions, integration_order, num_basis_functions, ws_template_weights


    # Implement a function to setup the jacobian.
    def setup_jacobian( self, element_scale = None, num_basis_functions = None, integration_order = None, num_dimensions = None, num_elements = None, set_flag = False ):

        # Pre-process the number of elements.
        num_elements = self.preprocess_num_elements( num_elements )

        # Pre-process the number of dimensions.
        num_dimensions = self.preprocess_num_dimensions( num_dimensions )

        # Pre-process the integration order.
        integration_order = self.preprocess_integration_order( integration_order )

        # Pre-process the number of basis functions.
        num_basis_functions = self.preprocess_num_basis_functions( num_basis_functions, integration_order, num_dimensions, set_flag )

        # Pre-process the element scale.
        element_scale = self.preprocess_element_scale( element_scale )

        # Return the jacobian information.
        return num_elements, num_dimensions, integration_order, num_basis_functions, element_scale


    # Implement a function to setup for data shuffling.
    def setup_shuffle_data( self, xs_integration_points = None, G_basis_values = None, W_integration_weights = None, sigma_jacobian = None, shuffle_indexes = None ):
        
        # Preprocess the integration points.
        xs_integration_points = self.preprocess_integration_points( xs_integration_points )

        # Preprocess the basis values.
        G_basis_values = self.preprocess_basis_values( G_basis_values )

        # Preprocess the integration weights.
        W_integration_weights = self.preprocess_integration_weights( W_integration_weights )

        # Preprocess the jacobian.
        sigma_jacobian = self.preprocess_jacobian( sigma_jacobian )

        # Setup the shuffle indexes.
        shuffle_indexes = self.preprocess_shuffle_indexes( shuffle_indexes )

        # Return the data required to setup for data shuffling.
        return xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, shuffle_indexes


    # Implement a function to setup for batch computation.
    def setup_batch_computation( self, xs_integration_points = None, G_basis_values = None, W_integration_weights = None, sigma_jacobian = None, num_elements = None, batch_number = None, batch_size = None ):

        # Preprocess the element data.
        xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements = self.preprocess_element_data( xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements )

        # Preprocess the batch number.
        batch_number = self.preprocess_batch_number( batch_number )

        # Preprocess the batch size.
        batch_size = self.preprocess_batch_size( batch_size )
    
        # Return the batch computation information.
        return xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements, batch_number, batch_size


    # Implement a function to setup for plotting basis function values.
    def setup_basis_function_plotting( self, xis_template_integration_points = None, G_basis_values = None ):

        # Preprocess the template integration points.
        xis_template_integration_points = self.preprocess_template_integration_points( xis_template_integration_points )

        # Preprocess the basis values.
        G_basis_values = self.preprocess_basis_values( G_basis_values )

        # Return the information necessary for plotting the basis values.
        return xis_template_integration_points, G_basis_values


    #%% ------------------------------------------------------------ SET FUNCTIONS ------------------------------------------------------------

    # Implement a function to set the element type.
    def set_element_type( self, element_type, set_flag = True ):

        # Determine whether to set the element type.
        if set_flag:                # If we want to set the element type...

            # Set the element type.
            self.element_type = element_type


    # Implement a function to set the integration order.
    def set_integration_order( self, integration_order, set_flag = True ):

        # Determine whether to set the integration order.
        if set_flag:                # If we want to set the integration order...

            # Set the integration order.
            self.integration_order = integration_order


    # Implement a function to set the number of basis functions.
    def set_num_basis_functions( self, num_basis_functions, set_flag = True ):

        # Determine whether to set the number of basis functions.
        if set_flag:                # If we want to set the number of basis functions...

            # Set the number of basis functions.
            self.num_basis_functions = num_basis_functions


    # Implement a function to set the number of points per elements.
    def set_num_points_per_element( self, num_points_per_element, set_flag = True ):

        # Determine whether to set the number of points per element.
        if set_flag:                # If we want to set the number of points per element...

            # Set the number of points per element.
            self.num_points_per_element = num_points_per_element


    # Implement a function to set the number of elements.
    def set_num_elements( self, num_elements, set_flag = True ):

        # Determine whether to set the number of elements.
        if set_flag:                # If we want to set the number of elements...

            # Set the number of elements.
            self.num_elements = num_elements


    # Implement a function to set the element scale.
    def set_element_scale( self, element_scale, set_flag = True ):

        # Determine whether to set the element scale.
        if set_flag:                # If we want to set the number of elements...

            # Set the element scale.
            self.element_scale = element_scale


    # Implement a function to set the 1D template integration points.
    def set_1D_template_integration_points( self, xis_1D_template_integration_points, set_flag = True ):

        # Determine whether to set the 1D template integration points.
        if set_flag:                # If we want to set the 1D template integration points...

            # Set the 1D template integration points.
            self.xis_1D_template_integration_points = xis_1D_template_integration_points


    # Implement a function to set the template integration points.
    def set_template_integration_points( self, xis_template_integration_points, set_flag = True ):

        # Determine whether to set the template integration points.
        if set_flag:                # If we want to set the template integration points...

            # Set the template integration points.
            self.xis_template_integration_points = xis_template_integration_points


    # Implement a function to set the integration points.
    def set_integration_points( self, xs_integration_points, set_flag = True ):

        # Determine whether to set the integration points.
        if set_flag:                # If we want to set the integration points...

            # Set the integration points.
            self.xs_integration_points = xs_integration_points


    # Implement a function to set the template weights.
    def set_template_weights( self, ws_template_weights, set_flag = True ):

        # Determine whether to set the template gauss-legendre weights.
        if set_flag:                # If we want to set the template gauss-legendre weights...

            # Set the gauss-legendre weights.
            self.ws_template_weights = ws_template_weights


    # Implement a function to set the template basis functions.
    def set_template_basis_functions( self, gs_template_basis_functions, set_flag = True ):

        # Determine whether to set the template gauss-legendre bases.
        if set_flag:                # If we want to set the template gauss-legendre bases...

            # Set the gauss-legendre template bases.
            self.gs_template_basis_functions = gs_template_basis_functions

    
    # Implement a function to set the basis values.
    def set_basis_values( self, G_basis_values, set_flag = True ):

        # Determine whether to set the basis function values.
        if set_flag:                # If we want to set the basis function values...

            # Set the basis function values.
            self.G_basis_values = G_basis_values


    # Implement a function to set the integration weights.
    def set_integration_weights( self, W_integration_weights, set_flag = True ):

        # Determine whether to set the integration weight values.
        if set_flag:                # If we want to set the integration weight values...

            # Set the integration weight values.
            self.W_integration_weights = W_integration_weights


    # Implement a function to set the jacobian.
    def set_jacobian( self, sigma_jacobian, set_flag = True ):

        # Determine whether to set the jacobian element scaling values.
        if set_flag:                # If we want to set the jacobian element scaling values...

            # Set the jacobian element scaling.
            self.sigma_jacobian = sigma_jacobian


    # Implement a function to set the element data.
    def set_element_data( self, xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements = None, set_flag = True ):

        # Determine whether to set the element data.
        if set_flag:                # If we want to set the element data...

            # Set the integration points.
            self.xs_integration_points = xs_integration_points
            
            # Set the basis values.
            self.G_basis_values = G_basis_values

            # Set the integration weights.
            self.W_integration_weights = W_integration_weights
            
            # Set the element scaling jacobian.
            self.sigma_jacobian = sigma_jacobian

            # Determine whether to set the number of elements.
            if num_elements is not None:                        # If the number of elements was provided...

                # Set the number of elements.
                self.num_elements = num_elements


    #%% ------------------------------------------------------------ VALIDATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate the element type.
    def validate_element_type( self, element_type, set_flag = False ):

        # Determine whether to set the element type.
        if not self.is_element_type_valid( element_type ):                              # If the element type is not valid...

            # Throw an error.
            raise ValueError( f'Invalid element type: {element_type}' )

        # Set the element type (as required).
        self.set_element_type( element_type, set_flag )

        # Return the element type.
        return element_type


    # Implement a function to validate the integration order.
    def validate_integration_order( self, integration_order, set_flag = False ):

        # Determine whether to set the gauss-legendre integration order.
        if not self.is_integration_order_valid( integration_order ):                  # If the gauss-legendre integration order is not valid...

            # Throw an error.
            raise ValueError( f'Invalid Gauss-Legendre integration order: {integration_order}' )

        # Determine whether to set the integration order (as required).
        self.set_integration_order( integration_order, set_flag )

        # Return the gauss-legendre integration order.
        return integration_order


    # Implement a function to validate the number of elements.
    def validate_num_elements( self, num_elements, set_flag = False ):

        # Determine whether to set the number of elements.
        if not self.is_num_elements_valid( num_elements ):                      # If the number of elements is not valid...

            # Throw an error.
            raise ValueError( f'Invalid number of elements: {num_elements}' )

        # Set the number of elements.
        self.set_num_elements( num_elements, set_flag )

        # Return the number of elements.
        return num_elements


    # Implement a function to validate the element scale.
    def validate_element_scale( self, element_scale, set_flag = False ):

        # Determine whether to set the element scale.
        if not self.is_element_scale_valid( element_scale ):                    # If the element scale is not valid...

            # Throw an error.
            raise ValueError( f'Invalid element scale: {element_scale}' )

        # Set the element scale.
        self.set_element_scale( element_scale, set_flag )

        # Return the element scale.
        return element_scale


    #%% ------------------------------------------------------------ COMPUTE FUNCTIONS ------------------------------------------------------------

    # Implement a function to set the number of basis functions.
    def compute_num_basis_functions( self, integration_order = None, num_dimensions = None, set_flag = False ):

        # Setup the number of basis functions.
        integration_order, num_basis_functions = self.setup_num_basis_functions( integration_order, num_dimensions )

        # Compute the number of basis functions.
        num_basis_functions = ( integration_order + 1 )**num_dimensions

        # Set the number of basis functions (as required).
        self.set_num_basis_functions( num_basis_functions, set_flag )

        # Return the number of basis functions.
        return num_basis_functions


    # Implement a function to set the number of points per element.
    def compute_num_points_per_element( self, integration_order = None, num_dimensions = None, set_flag = False ):

        # Setup the number of points per element.
        integration_order, num_dimensions = self.setup_num_points_per_element( integration_order, num_dimensions )

        # Compute the number of points per element.
        num_points_per_element = ( integration_order + 1 )**num_dimensions

        # Set the number of points per element (as required).
        self.set_num_points_per_element( num_points_per_element, set_flag )

        # Return the number of points per element.
        return num_points_per_element
        

    # Implement a function to compute the template Gauss-Legendre points.
    def compute_1D_template_integration_points( self, integration_order = None, set_flag = False ):

        # Note that these template Gauss-Legendre points are the template points that will be used for Gauss-Legendre integration over the element.
        # The integration points for Gauss-Legendre integration are chosen by taking the roots of the associated Gauss-Legendre polynomial
        # The 2nd degree Gauss-Legendre polynomial is used for integration of order one, the 3rd degree Gauss-Legendre polynomial is used for integration of order two, etc.
        # More integration orders could be implemented by using the roots of higher degree Gauss-Legendre polynomials.

        # Pre-process the integration order.
        integration_order = self.preprocess_integration_order( integration_order )

        # Determine how to compute the template Gauss-Legendre points.
        if integration_order == 1:                                  # If the Gauss-Legendre integration order is one...

            # Set the template Gauss-Legendre points.
            xis_1D_template_integration_points = ( 1/torch.sqrt( torch.tensor( 3, dtype = torch.float32, device = self.device ) ) )*torch.tensor( [ -1, 1 ], dtype = torch.float32, device = self.device )

        elif integration_order == 2:                                # If the Gauss-Legendre integration order is two...

            # Set the template Gauss-Legendre points.
            xis_1D_template_integration_points = torch.sqrt(  torch.tensor( 3/5, dtype = torch.float32, device = self.device ) )*torch.tensor( [ -1, 0, 1 ], dtype = torch.float32, device = self.device )

        else:                                                       # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid Gauss-Legendre integration order: {integration_order}.  Only integration orders of 1 and 2 are recognized.' )

        # Set the 1D template integration points (as required).
        self.set_1D_template_integration_points( xis_1D_template_integration_points, set_flag )

        # Return the template Gauss-Legendre points.
        return xis_1D_template_integration_points


    # Implement a function to compute the template Gauss-Legendre weights.
    def compute_template_weights( self, integration_order = None, set_flag = False ):

        # Similar to the Gauss-Legendre integration points, the Gauss-Legendre weights are chosen in a specific way provided by the associated GL theorem.
        # An algorithm for computing weights for higher order integration could be implemented.
        # Weights are often simply tabulated, so such a table is likely readily available.

        # Pre-process the integration order.
        integration_order = self.preprocess_integration_order( integration_order )

        # Determine how to compute the template Gauss-Legendre weights.
        if integration_order == 1:                                  # If the Gauss-Legendre integration order is one...           

            # Set the template Gauss-Legendre weights.
            ws_template_weights = torch.ones( 2, dtype = torch.float32, device = self.device )

        elif integration_order == 2:                                # If the Gauss-Legendre integration order is two...

            # Set the template Gauss-Legendre weights.
            ws_template_weights = ( 1/9 )*torch.tensor( [ 5, 8, 5 ], dtype = torch.float32, device = self.device )

        else:

            # Throw an error.
            raise ValueError( 'Gauss-Legendre integration order not recognized.  Only integration orders of 1 and 2 are recognized.' )

        # Set the template weights.
        self.set_template_weights( ws_template_weights, set_flag )

        # Return the template Gauss-Legendre weights.
        return ws_template_weights


    # Implement a function to compute the template Gauss-Legendre bases.
    def compute_template_basis_functions( self, integration_order = None, set_flag = False ):

        # As with the integration points and weights, the Gauss-Legendre basis functions are provided by the associated GL theorem.
        # These polynomials are frequently tabulated, so higher order methods could be implemented simply by referring to such a table.

        # Pre-process the integration order.
        integration_order = self.preprocess_integration_order( integration_order )

        # Determine how to compute the template Gauss-Legendre bases.
        if integration_order == 1:                                  # If the Gauss-Legendre integration order is one...           

            # Define the template basis functions.
            gs_gl_template1 = lambda x: 0.5*( 1 - x )
            gs_gl_template2 = lambda x: 0.5*( 1 + x )
            gs_template_basis_functions = [ gs_gl_template1, gs_gl_template2 ]

        elif integration_order == 2:                                # If the Gauss-Legendre integration order is two...

            # Define the template basis functions.
            gs_gl_template1 = lambda x: ( x/2 )*( x + 1 )
            gs_gl_template2 = lambda x: -( x**2 ) + 1
            gs_gl_template3 = lambda x: ( x/2 )*( x - 1 )
            gs_template_basis_functions = [ gs_gl_template1, gs_gl_template2, gs_gl_template3 ]

        else:

            # Throw an error.
            raise ValueError( 'Gauss-Legendre integration order not recognized.  Only integration orders of 1 and 2 are recognized.' )

        # Set the template basis functions.
        self.set_template_basis_functions( gs_template_basis_functions, set_flag )

        # Return the template Gauss-Legendre bases.
        return gs_template_basis_functions


    # Implement a function to compute the integration points of a single template element located at the origin.  
    def compute_template_integration_points( self, xis_1D_template_integration_points = None, num_dimensions = None, integration_order = None, set_flag = False ):

        # Note that the Gauss-Legendre template points are only a collection of scalar integration points.
        # In one dimension, these template points are sufficient to define the element points.
        # In higher dimensions, the template points most be combined to form points of the appropriate dimension.

        # Setup the template integration points.
        integration_order, num_dimensions, xis_1D_template_integration_points = self.setup_template_integration_points( xis_1D_template_integration_points, integration_order, num_dimensions, set_flag )

        # Define the local element domain.
        local_domain = [ xis_1D_template_integration_points ]*num_dimensions

        # Create a list of the element template points.
        # xis_template_integration_points = torch.meshgrid( *local_domain )
        xis_template_integration_points = torch.meshgrid( *local_domain, indexing = 'ij' )

        # Convert the element template points list to an element template points tensor.
        xis_template_integration_points = self.tensor_utilities.grid_list2grid_tensor( xis_template_integration_points )

        # Flatten the element template points.
        xis_template_integration_points = self.tensor_utilities.flatten_grid( xis_template_integration_points )
        # xis_template_integration_points = self.tensor_utilities.flatten_grid( torch.unsqueeze( xis_template_integration_points, dim = xis_template_integration_points.dim(  ) ) )
        # xis_template_integration_points = self.tensor_utilities.expand_grid( xis_template_integration_points,  )

        # Set the template integration points (as required).
        self.set_template_integration_points( xis_template_integration_points, set_flag )

        # Return the element template points.
        return xis_template_integration_points


    # Implement a function to generate all of the element integration points.
    def compute_integration_points( self, xs_element_centers, element_scale = None, num_dimensions = None, integration_order = None, num_points_per_element = None, xis_1D_template_integration_points = None, xis_template_integration_points = None, set_flag = False ):

        # Ensure that the element centers are valid.
        assert self.is_element_centers_valid( xs_element_centers )

        # Setup the integration points.
        element_scale, num_dimensions, integration_order, num_points_per_element, xis_1D_template_integration_points, xis_template_integration_points = self.setup_integration_points( xis_template_integration_points, xis_1D_template_integration_points, num_points_per_element, integration_order, num_dimensions, element_scale, set_flag )

        # Compute the number of elements.
        num_elements = torch.tensor( xs_element_centers.shape[ 0 ], dtype = torch.int64, device = self.device )

        # Create an tensor to store the integration points.
        xs_integration_points = torch.zeros( ( num_elements, num_points_per_element, num_dimensions ), dtype = torch.float32, device = self.device )

        # Compute the transformation matrix associated with each collocation point.
        for k in range( num_elements ):                    # Iterate through each collocation point...

            # Compute the translation matrix matrix.
            T = torch.eye( num_dimensions + 1, dtype = torch.float32, device = self.device )
            T[ :-1, -1 ] = xs_element_centers[ k, : ]

            # Compute the scaling matrix.
            S = torch.diag( torch.hstack( ( element_scale, torch.ones( 1, dtype = element_scale.dtype, device = self.device ) ) ) )

            # Compute the transformation matrix associated with this collocation point.
            H = torch.matmul( T, S )

            # Create the element points associated with this transformation matrix.
            xs_e = torch.matmul( H, torch.vstack( ( xis_template_integration_points.T, torch.ones( ( 1, num_points_per_element ), dtype = torch.float32, device = self.device ) ) ) )[ :-1, : ]

            # Store the integration points.
            xs_integration_points[ k, :, : ] = torch.reshape( xs_e.T, ( 1, xs_e.shape[ 1 ], xs_e.shape[ 0 ] ) )

        # Set the integration points (as required).
        self.set_integration_points( xs_integration_points, set_flag )

        # Return the integration points.
        return xs_integration_points


    # Implement a function to evaluate the basis functions at each local element point.
    def compute_basis_values( self, num_elements = None, num_dimensions = None, integration_order = None, num_points_per_element = None, num_basis_functions = None, xis_1D_template_integration_points = None, xis_template_integration_points = None, gs_template_basis_functions = None, set_flag = False ):
    
        # Setup basis value information.
        num_elements, num_dimensions, integration_order, num_points_per_element, num_basis_functions, xis_1D_template_integration_points, xis_template_integration_points, gs_template_basis_functions = self.setup_basis_values( gs_template_basis_functions, xis_template_integration_points, xis_1D_template_integration_points, num_basis_functions, num_points_per_element, integration_order, num_dimensions, num_elements, set_flag )

        # Create a template list of the indexes to use when combining the basis functions.
        index_template = torch.tensor( range( integration_order + 1 ), dtype = torch.uint8, device = self.device )

        # Create a list of the indexes to use when combining the basis functions.
        # indexes = torch.meshgrid( *[ index_template for _ in range( num_dimensions ) ] )
        indexes = torch.meshgrid( *[ index_template for _ in range( num_dimensions ) ], indexing = 'ij' )

        # Convert the element template points list to an element template points tensor.
        indexes = self.tensor_utilities.grid_list2grid_tensor( indexes )

        # Flatten the element template points.
        indexes = self.tensor_utilities.flatten_grid( indexes )

        # Create lists of each combination of basis functions.
        g_list = [ [ gs_template_basis_functions[ indexes[ k1, k2 ] ] for k2 in range( num_dimensions ) ] for k1 in range( num_points_per_element ) ]
        # g_list = [ [ gs_template_basis_functions[ int( indexes[ k1 ][ k2 ] ) ] for k2 in range( num_dimensions ) ] for k1 in range( num_points_per_element ) ]

        # Create an tensor to store the basis functions evaluated over the template element.
        G_basis_values = torch.zeros( ( 1, num_basis_functions, num_points_per_element ), dtype = torch.float32, device = self.device )

        # Compute the value of each basis function at each point in template element.
        for k1 in range( num_basis_functions ):                                # Iterate through each basis function...
            for k2 in range( num_points_per_element ):                     # Iterate through each template element point...

                # Initialize the basis function value associated with this basis function and template element point.
                g = torch.tensor( 1, dtype = torch.float32, device = self.device )

                # Compute the basis function value associated with this basis function evaluated at this template point.
                for k3 in range( num_dimensions ):                         # Iterate through each problem dimension...

                    # Compute the basis function value of this basis function at this template element point by taking the products.
                    g *= g_list[ k1 ][ k3 ]( xis_template_integration_points[ k2, k3 ] )

                # Store the basis function value associated with this basis function evaluated at this template point.
                G_basis_values[ 0, k1, k2 ] = g

        # Repeat the basis function values for each element (since the basis functions have the same values for each element).
        G_basis_values = torch.repeat_interleave( G_basis_values, num_elements, axis = 0 )

        # Set the basis values (as required).
        self.set_basis_values( G_basis_values, set_flag )

        # Return the basis values.
        return G_basis_values


    # Implement a function to compute the Gauss-Legendre weights.
    def compute_integration_weights( self, num_elements = None, num_dimensions = None, integration_order = None, num_basis_functions = None, ws_template_weights = None, set_flag = False ):

        # Setup the integration weight information.
        num_elements, num_dimensions, integration_order, num_basis_functions, ws_template_weights = self.setup_integration_weights( ws_template_weights, num_basis_functions, integration_order, num_dimensions, num_elements, set_flag )

        # Create a grid of the template integration weights.
        ws_gl = torch.meshgrid( *[ ws_template_weights for _ in range( num_dimensions ) ], indexing = 'ij' )

        # Convert the template integration weights grid list to a grid tensor.
        ws_gl = self.tensor_utilities.grid_list2grid_tensor( ws_gl )

        # Compute the composite template integration weights for a single element (constant for all elements).
        ws_gl = torch.prod( self.tensor_utilities.flatten_grid( ws_gl ), axis = 1 )

        # Compute the integration weight matrix.
        W_integration_weights = torch.repeat_interleave( torch.repeat_interleave( torch.reshape( ws_gl, ( 1, 1, ws_gl.numel(  ) ) ), num_elements, axis = 0 ), num_basis_functions.to( torch.int32 ), axis = 1 )

        # Set the integration weights.
        self.set_integration_weights( W_integration_weights, set_flag )

        # Return the integration weight matrix.
        return W_integration_weights


    # Implement a function to compute the element scaling Jacobian.
    def compute_jacobian( self, num_elements = None, num_dimensions = None, integration_order = None, num_basis_functions = None, element_scale = None, set_flag = False ):

        # Setup the jacobian information.
        num_elements, num_dimensions, integration_order, num_basis_functions, element_scale = self.setup_jacobian( element_scale, num_basis_functions, integration_order, num_dimensions, num_elements, set_flag )

        # Repeat the element scales for each basis function (n_c x n_b x n_e).
        element_scales_mat = torch.repeat_interleave( torch.repeat_interleave( torch.reshape( element_scale, ( 1, 1, num_dimensions ) ), num_elements, axis = 0 ), num_basis_functions.to( torch.int32 ), axis = 1 )

        # Compute the jacobian scaling (n_c x n_b).
        sigma_jacobian = ( 1/2 )*torch.sum( element_scales_mat, axis = -1 )

        # Set the jacobian (as required).
        self.set_jacobian( sigma_jacobian, set_flag )

        # Return the jacobian element scaling.
        return sigma_jacobian


    #%% ------------------------------------------------------------ APPEND FUNCTIONS ------------------------------------------------------------

    # Implement a function to append integration points.
    def append_integration_points( self, xs_integration_points, set_flag = False ):

        # Determine whether to append the integration points.
        if self.xs_integration_points is not None:              # If the integration points is not None...
        
            # Concatenate the integration points.
            xs_integration_points = torch.cat( ( self.xs_integration_points, xs_integration_points ), dim = 0 )

        # Set the integration points (as required).
        self.set_integration_points( xs_integration_points, set_flag )

        # Return the appended integration points.
        return xs_integration_points


    # Implement a function to append the basis values.
    def append_basis_values( self, G_basis_values, set_flag = False ):

        # Determine whether to append the basis values.
        if self.G_basis_values is not None:                     # If the basis values is not None...

            # Concatenate the basis values.
            G_basis_values = torch.cat( ( self.G_basis_values, G_basis_values ), dim = 0 )

        # Set the basis values (as required).
        self.set_basis_values( G_basis_values, set_flag )

        # Return the appended basis values.
        return G_basis_values


    # Implement a function to append the integration weights.
    def append_integration_weights( self, W_integration_weights, set_flag = False ):

        # Determine whether to append the integration weights.
        if self.W_integration_weights is not None:              # If the integration weights is not None...

            # Concatenate the integration weights.
            W_integration_weights = torch.cat( ( self.W_integration_weights, W_integration_weights ), dim = 0 )
        
        # Set the integration weights (as required).
        self.set_integration_weights( W_integration_weights, set_flag )

        # Return the appended integration weights.
        return W_integration_weights


    # Implement a function to append the jacobian.
    def append_jacobian( self, sigma_jacobian, set_flag = False ):

        # Determine whether to append the element scaling jacobian matrix.
        if self.sigma_jacobian is not None:                     # If the element scaling jacobian is not None...

            # Concatenate the element scaling jacobian matrix.
            sigma_jacobian = torch.cat( ( self.sigma_jacobian, sigma_jacobian ), dim = 0 )

        # Set the jacobian (as required).
        self.set_jacobian( sigma_jacobian, set_flag )

        # Return the appended jacobian.
        return sigma_jacobian


    # Implement a function to append the number of elements.
    def append_num_elements( self, num_elements, set_flag = False ):

        # Increase the number of elements.
        num_elements = self.num_elements + num_elements
        
        # Determine whether to set the number of elements.
        self.set_num_elements( num_elements, set_flag )

        # Return the number of elements.
        return num_elements


    # Implement a function to append element data.
    def append_element_data( self, xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements, set_flag = False ):

        # Append the integration points.
        self.append_integration_points( xs_integration_points, set_flag )

        # Append the basis values.
        self.append_basis_values( G_basis_values, set_flag )

        # Append the integration weights.
        self.append_integration_weights( W_integration_weights, set_flag )

        # Append the jacobian.
        self.append_jacobian( sigma_jacobian, set_flag )

        # Append the number of elements.
        self.append_num_elements( num_elements, set_flag )

        # Return the appended element data.
        return xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements


    #%% ------------------------------------------------------------ ELEMENT FUNCTIONS ------------------------------------------------------------

    # Implement a function to generate new element data.
    def generate_element_data( self, xs_element_centers ):

        # Construct the integration points for every element in the space.
        xs_integration_points = self.compute_integration_points( xs_element_centers[ ..., -1 ] )                             # ND Gauss-Legendre Integration Points.

        # Ensure that the integration points are repeated for each network timestep.
        xs_integration_points = torch.tile( torch.unsqueeze( xs_integration_points, dim = 3 ), [ 1, 1, 1, xs_element_centers.shape[ -1 ] ] )

        # Define the number of elements.
        num_elements = torch.tensor( xs_element_centers.shape[ 0 ], dtype = torch.int64, device = self.device )

        # Construct the basis evaluation matrix.
        G_basis_values = self.compute_basis_values( num_elements )                                        # Basis Functions Evaluated at the Integration Points.

        # Construct the integration weight matrix.
        W_integration_weights = self.compute_integration_weights( num_elements )

        # Construct the element scaling jacobian matrix.
        sigma_jacobian = self.compute_jacobian( num_elements )

        # Return the integration points, basis evaluation matrix, integration weight matrix, and element scaling jacobian matrix.
        return xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements


    # Implement a function to replace existing elements.
    def replace_elements( self, xs_element_centers, set_flag = False ):

        # Compute the element data.
        xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements = self.generate_element_data( xs_element_centers )

        # Set the element data (as required).
        self.set_element_data( xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements, set_flag )

        # Return the element data.
        return xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements


    # Implement a function to append to existing element data.
    def append_elements( self, xs_element_centers, set_flag = False ):

        # Compute the element data.
        xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements = self.generate_element_data( xs_element_centers )

        # Append the element data.
        xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements = self.append_element_data( xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements )

        # Set the element data (as required).
        self.set_element_data( xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements, set_flag )

        # Return the element data.
        return xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements


    # Implement a function to generate elements.
    def generate_elements( self, xs_element_centers, replace_flag = False, set_flag = False ):

        # Determine how to generate the elements.
        if replace_flag:                    # If we want to replace the existing elements...

            # Replace the existing elements.
            xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements = self.replace_elements( xs_element_centers, set_flag )

        else:                               # If we want to append to the existing elements...

            # Append to the existing elements.
            xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements = self.append_elements( xs_element_centers, set_flag )

        # Return the element data.
        return xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements


    # Implement a function to delete elements.
    def delete_elements( self, indexes = None, set_flag = False ):

        # Determine which indexes to delete.
        if indexes is None:                     # If the indexes were not provided...

            # Delete all of the stored element points.
            xs_integration_points = None

            # Delete all of the stored basis values.
            G_basis_values = None

            # Delete all of the stored integration weights.
            W_integration_weights = None

            # Delete all of the stored jacobian entries.
            sigma_jacobian = None

            # Set the number of elements to zero.
            num_elements = torch.tensor( 0, dtype = torch.int64, device = self.device )

        else:                                   # Otherwise... ( i.e., if indexes were provided... )

            # Delete the specified integration points.
            xs_integration_points = self.tensor_utilities.delete_entries( self.xs_integration_points, indexes, dim = 0 )

            # Delete the specified basis values.
            G_basis_values = self.tensor_utilities.delete_entries( self.G_basis_values, indexes, dim = 0 )

            # Delete the specified integration weights.
            W_integration_weights = self.tensor_utilities.delete_entries( self.W_integration_weights, indexes, dim = 0 )

            # Delete the specified jacobian entries.
            sigma_jacobian = self.tensor_utilities.delete_entries( self.sigma_jacobian, indexes, dim = 0 )

            # Compute the number of elements.
            num_elements = self.num_elements - indexes.numel(  )

        # Set the element data (as required).
        self.set_element_data( xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements, set_flag )

        # Return the element data.
        return xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements


    #%% ------------------------------------------------------------ SHUFFLE FUNCTIONS ------------------------------------------------------------

    # Implement a function to shuffle the input data.
    def shuffle_data( self, xs_integration_points = None, G_basis_values = None, W_integration_weights = None, sigma_jacobian = None, shuffle_indexes = None, set_flag = False ):

        # Setup for shuffling the element information.
        xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, shuffle_indexes = self.setup_shuffle_data( xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, shuffle_indexes )

        # Determine whether to shuffle the integration points.
        if xs_integration_points is not None:               # If the integration points is not None...

            # Shuffle the integration points.
            xs_integration_points = xs_integration_points[ shuffle_indexes, ... ]

        # Determine whether to shuffle the basis values.
        if G_basis_values is not None:                      # If the basis values is not None...

            # Shuffle the basis values.
            G_basis_values = G_basis_values[ shuffle_indexes, ... ]

        # Determine whether to shuffle the integration weights.
        if W_integration_weights is not None:               # If the integration weights is not None...

            # Shuffle the integration weights.
            W_integration_weights = W_integration_weights[ shuffle_indexes, ... ]
        
        # Determine whether to shuffle the jacobian.
        if sigma_jacobian is not None:                      # If the jacobian is not None...

            # Shuffle the jacobian.
            sigma_jacobian = sigma_jacobian[ shuffle_indexes, ... ]

        # Set the element information (as required).
        self.set_element_data( xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, None, set_flag )

        # Return the shuffled data.
        return xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, shuffle_indexes


    #%% ------------------------------------------------------------ BATCH FUNCTIONS ------------------------------------------------------------

    # Implement a function to saturate the upper batch index.
    def saturate_upper_batch_index( self, upper_batch_index, num_elements = None ):

        # Preprocess the number of elements.
        num_elements = self.preprocess_num_elements( num_elements )

        # Ensure that the upper batch index is valid.
        if upper_batch_index > num_elements:                                           # If the upper batch index is greater than the number of elements...

            # Set the upper batch index to be the number of points.
            upper_batch_index = num_elements

        # Return the upper batch index.
        return upper_batch_index


    # Implement a function to stage a batch of the data.
    def compute_batch_data( self, xs_integration_points = None, G_basis_values = None, W_integration_weights = None, sigma_jacobian = None, num_elements = None, batch_number = None, batch_size = None ):

        # Setup for batch computation.
        xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements, batch_number, batch_size = self.setup_batch_computation( xs_integration_points, G_basis_values, W_integration_weights, sigma_jacobian, num_elements, batch_number, batch_size )

        # Determine whether there are elements to batch.
        if num_elements > 0:                                    # If there are any elements...

            # Determine how to stage the batch.
            if batch_size is not None:                          # If the batch size was provided...

                # Compute the batch indexes.
                lower_batch_index = batch_number*batch_size
                upper_batch_index = ( batch_number + 1 )*batch_size

                # Saturate the upper batch index.
                upper_batch_index = self.saturate_upper_batch_index( upper_batch_index, num_elements )

                # Retrieve a batch of the finite element data.
                xs_integration_points_batch = xs_integration_points[ lower_batch_index:upper_batch_index, ... ]
                G_basis_values_batch = G_basis_values[ lower_batch_index:upper_batch_index, ... ]
                W_integration_weights_batch = W_integration_weights[ lower_batch_index:upper_batch_index, ... ]
                sigma_jacobian_batch = sigma_jacobian[ lower_batch_index:upper_batch_index, ... ]

            else:                                                                                       # Otherwise... ( i.e., the batch size was not provided... )

                # Stage all of the data.
                xs_integration_points_batch = xs_integration_points
                G_basis_values_batch = G_basis_values
                W_integration_weights_batch = W_integration_weights
                sigma_jacobian_batch = sigma_jacobian

        else:                                                                                           # Otherwise... ( i.e., there are no elements... )

            # Set the element batch information to None.
            xs_integration_points_batch = None
            G_basis_values_batch = None
            W_integration_weights_batch = None
            sigma_jacobian_batch = None

        # Return the input data batch.
        return xs_integration_points_batch, G_basis_values_batch, W_integration_weights_batch, sigma_jacobian_batch


    #%% ------------------------------------------------------------ PRINTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to print element summary information.
    def print( self, num_dashes = 20, decoration_flag = True ):

        # Print a header.
        self.printing_utilities.print_header( 'FINITE ELEMENT SUMMARY', num_dashes, decoration_flag )

        # Print the basic element information.
        print( 'Basic Information' )
        print( f'# of Elements: {self.num_elements}' )
        print( f'# of Dimensions: {self.num_dimensions}' )
        print( f'Element Type: {self.element_type}' )
        print( f'Element Scale: {self.element_scale}' )
        print( f'Integration Order: {self.integration_order}' )
        print( f'# of Points per Element: {self.num_points_per_element}' )
        print( f'# of Basis Functions: {self.num_basis_functions}' )
        print( '\n' )

        # Print the integration point information.
        print( 'Integration Point Information' )
        print( f'xis 1D Template Integration Points: {self.xis_1D_template_integration_points}' )
        print( f'xis Template Integration Points: {self.xis_template_integration_points}' )
        print( f'xs Integration Points: {self.xs_integration_points}' )
        print( '\n' )
        
        # Print the basis function information.
        print( 'Basis Function Information' )
        print( f'gs Template Basis Functions: {self.gs_template_basis_functions}' )
        print( f'G Basis Values: {self.G_basis_values}' )
        print( '\n' )

        # Print the integration weight information.
        print( 'Integration Weight Information' )
        print( f'ws Template Weights: {self.ws_template_weights}' )
        print( f'W Integration Weights: {self.W_integration_weights}' )
        print( '\n' )

        # Print the jacobian information.
        print( 'Jacobian Information' )
        print( f'sigma Jacobian: {self.sigma_jacobian}' )

        # Print a footer.
        self.printing_utilities.print_footer( num_dashes, decoration_flag )


    #%% ------------------------------------------------------------ PLOTTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to plot the template element points.ax
    def plot_template_integration_points( self, xis_template_integration_points = None, show_plot_flag = False ):

        # Preprocess the template integration points.
        xis_template_integration_points = self.preprocess_template_integration_points( xis_template_integration_points )

        # Plot the element points.
        fig, ax = self.plotting_utilities.plot_template_integration_points( xis_template_integration_points, show_plot_flag )

        # Return the figure and axes.
        return fig, ax


    # Implement a function to plot the basis function evaluation.
    def plot_basis_functions( self, xis_template_integration_points = None, G_basis_values = None, show_plot_flag = False ):     

        # Setup for plotting the basis functions.
        xis_template_integration_points, G_basis_values = self.setup_basis_function_plotting( xis_template_integration_points, G_basis_values )

        # Plot the basis functions.
        fig, axes = self.plotting_utilities.plot_basis_functions( xis_template_integration_points, G_basis_values, show_plot_flag )

        # Return the figure and axes.
        return fig, axes

