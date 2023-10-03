#%% ------------------------------------------------------------ DOMAIN CLASS ------------------------------------------------------------

# This file implements a class for storing and managing domain information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch

# Import custom libraries.
from save_load_utilities_class import save_load_utilities_class as save_load_utilities_class
from tensor_utilities_class import tensor_utilities_class as tensor_utilities_class
from plotting_utilities_class import plotting_utilities_class as plotting_utilities_class
from printing_utilities_class import printing_utilities_class as printing_utilities_class


#%% ------------------------------------------------------------ DOMAIN CLASS ------------------------------------------------------------

# Implement the domain class.
class domain_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self, temporal_domain, spatial_domain, domain_type, device = 'cpu' ):

        # Create an instance of the plotting utilities class.
        self.tensor_utilities = tensor_utilities_class(  )

        # Create an instance of the plotting utilities class.
        self.plotting_utilities = plotting_utilities_class(  )

        # Create an instance of the printing utilities class.
        self.printing_utilities = printing_utilities_class(  )

        # Create an instance of the save-load utilities class.
        self.save_load_utilities = save_load_utilities_class(  )

        # Store the device.
        self.device = device

        # Store the domain type.
        self.domain_type = domain_type

        # Store the spatiotemporal domains.
        self.temporal_domain = self.validate_temporal_domain( temporal_domain )
        self.spatial_domain = self.validate_spatial_domain( spatial_domain )
        self.spatiotemporal_domain = torch.cat( ( self.temporal_domain, self.spatial_domain ), dim = 1 )

        # Store the number of spatiotemporal dimensions.
        self.num_temporal_dimensions = temporal_domain.shape[ 1 ]
        self.num_spatial_dimensions = spatial_domain.shape[ 1 ]
        self.num_spatiotemporal_dimensions = self.num_temporal_dimensions + self.num_spatial_dimensions

        # Store the dimension labels.
        self.dimension_labels = self.generate_dimension_labels(  )


    #%% ------------------------------------------------------------ PREPROCESS FUNCTIONS ------------------------------------------------------------

    # Implement a function to preprocess a temporal domain.
    def preprocess_temporal_domain( self, temporal_domain = None ):

        # Determine whether to use the stored temporal domain.
        if temporal_domain is None:                 # If the temporal domain was not provided....

            # Set the temporal domain to be the stored value.
            temporal_domain = self.temporal_domain

        # Return the temporal domain.
        return temporal_domain


    # Implement a function to preprocess a spatial domain.
    def preprocess_spatial_domain( self, spatial_domain = None ):

        # Determine whether to use the stored spatial domain.
        if spatial_domain is None:                 # If the spatial domain was not provided....

            # Set the spatial domain to be the stored value.
            spatial_domain = self.spatial_domain

        # Return the spatial domain.
        return spatial_domain


    # Implement a function to preprocess a spatiotemporal domain.
    def preprocess_spatiotemporal_domain( self, spatiotemporal_domain = None ):

        # Determine whether to use the stored spatiotemporal domain.
        if spatiotemporal_domain is None:                 # If the spatiotemporal domain was not provided....

            # Set the spatiotemporal domain to be the stored value.
            spatiotemporal_domain = self.spatiotemporal_domain

        # Return the spatiotemporal domain.
        return spatiotemporal_domain


    # Implement a function to preprocess the domain type.
    def preprocess_domain_type( self, domain_type = None ):

        # Determine whether to use the stored domain type.
        if domain_type is None:                     # If the domain type was not provided...

            # Use the stored domain type.
            domain_type = self.domain_type

        # Return the domain type.
        return domain_type


    # Implement a function to preprocess the number of temporal dimensions.
    def preprocess_num_temporal_dimensions( self, num_temporal_dimensions = None ):

        # Determine whether to use the stored number of temporal dimensions.
        if num_temporal_dimensions is None:                                             # If the number of temporal dimension was not provided...

            # Set the number of temporal dimensions to be the stored value.
            num_temporal_dimensions = self.num_temporal_dimensions

        # Return the number of temporal dimensions.
        return num_temporal_dimensions


    # Implement a function to preprocess the number of spatial dimensions.
    def preprocess_num_spatial_dimensions( self, num_spatial_dimensions = None ):

        # Determine whether to use the stored number of spatial dimensions.
        if num_spatial_dimensions is None:                                             # If the number of spatial dimension was not provided...

            # Set the number of spatial dimensions to be the stored value.
            num_spatial_dimensions = self.num_spatial_dimensions

        # Return the number of spatial dimensions.
        return num_spatial_dimensions


    # Implement a function to preprocess the dimension labels.
    def preprocess_dimension_labels( self, dimension_labels = None ):

        # Determine whether to use the stored dimension labels.
        if dimension_labels is None:                # If the dimension labels were not provided...

            # Use the stored dimension labels.
            dimension_labels = self.dimension_labels

        # Return the dimension labels.
        return dimension_labels


    #%% ------------------------------------------------------------ IS VALID FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate a temporal domain.
    def is_temporal_domain_valid( self, temporal_domain ):

        # Determine whether the temporal domain is valid.
        if torch.is_tensor( temporal_domain ) and ( temporal_domain.numel(  ) != 0 ) and ( ( temporal_domain.dim(  ) == 1 ) or ( temporal_domain.dim(  ) == 2 ) ):                     # If the temporal domain is valid...

            # Set the valid flag to true.
            b_valid = True

        else:                                                                                                                                                                           # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    # Implement a function to validate a spatial domain.
    def is_spatial_domain_valid( self, spatial_domain ):

        # Determine whether the spatial domain is valid.
        if torch.is_tensor( spatial_domain ) and ( spatial_domain.numel(  ) != 0 ) and ( ( spatial_domain.dim(  ) == 1 ) or ( spatial_domain.dim(  ) == 2 ) ):                     # If the spatial domain is valid...

            # Set the valid flag to true.
            b_valid = True

        else:                                                                                                                                                                       # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    #%% ------------------------------------------------------------ GET FUNCTIONS ------------------------------------------------------------

    # Implement a function to get the specified domain.
    def get_domain( self, domain_type = 'spatiotemporal' ):

        # Determine which domain to retrieve.
        if domain_type.lower(  ) == 'temporal':                         # If we want to retrieve the temporal domain...

            # Retrieve the temporal domain.
            domain = self.temporal_domain

        elif domain_type.lower(  ) == 'spatial':                        # If we want to retrieve the spatial domain...    

            # Retrieve the spatial domain.
            domain = self.spatial_domain

        elif domain_type.lower(  ) == 'spatiotemporal':                 # If we want to retrieve the spatiotemporal domain...

            # Retrieve the spatiotemporal domain.
            domain = self.spatiotemporal_domain

        else:                                                           # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid domain type: {domain_type}' )

        # Return the domain.
        return domain


    # Implement a function to generate a grid from the domains.
    def get_domain_grid( self, domain_type = 'spatiotemporal' ):

        # Retrieve the specified domain.
        domain = self.get_domain( domain_type )

        # Generate the domain grid.
        domain_grid = self.tensor_utilities.generate_grid( domain, 2*torch.ones( domain.shape[ -1 ], dtype = torch.int16, device = self.device ) )

        # Return the domain grid.
        return domain_grid


    #%% ------------------------------------------------------------ SET FUNCTIONS ------------------------------------------------------------

    # Implement a function to set the temporal domain.
    def set_temporal_domain( self, temporal_domain, set_flag = True ):

        # Determine whether to set the temporal domain.
        if set_flag:            # If we want to set the temporal domain....

            # Set the temporal domain.
            self.temporal_domain = temporal_domain


    # Implement a function to set the spatial domain.
    def set_spatial_domain( self, spatial_domain, set_flag = True ):

        # Determine whether to set the spatial domain.
        if set_flag:            # If we want to set the spatial domain....

            # Set the spatial domain.
            self.spatial_domain = spatial_domain


    # Implement a function to set the spatiotemporal domain.
    def set_spatiotemporal_domain( self, spatiotemporal_domain, set_flag = True ):

        # Determine whether to set the spatiotemporal domain.
        if set_flag:            # If we want to set the spatiotemporal domain....

            # Set the spatiotemporal domain.
            self.spatiotemporal_domain = spatiotemporal_domain


    #%% ------------------------------------------------------------ VALIDATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate the temporal domain.
    def validate_temporal_domain( self, temporal_domain ):

        # Determine whether to set the temporal domain.
        if self.is_temporal_domain_valid( temporal_domain ):                    # If the temporal domain is valid...

            # Determine whether it is necessary to unsqueeze the second dimension.
            if temporal_domain.dim(  ) == 1:                        # If the temporal domain is one dimensional...

                # Unsqueeze the second dimension.
                temporal_domain.unsqueeze_( 1 )

        else:                                                                   # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid temporal domain: {temporal_domain}' )

        # Set the temporal domain (as required).
        self.set_temporal_domain( temporal_domain )

        # Return the temporal domain.
        return temporal_domain


    # Implement a function to validate the spatial domain.
    def validate_spatial_domain( self, spatial_domain ):

        # Determine whether to set the spatial domain.
        if self.is_spatial_domain_valid( spatial_domain ):                      # If the spatial domain is valid...

            # Determine whether it is necessary to unsqueeze the second dimension.
            if spatial_domain.dim(  ) == 1:                                     # If the spatial domain is one dimensional...

                # Unsqueeze the second dimension.
                spatial_domain.unsqueeze_( 1 )

        else:                                                                   # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid spatial domain: {spatial_domain}' )

        # Set the spatial domain (as required).
        self.set_spatial_domain( spatial_domain )

        # Return the spatial domain.
        return spatial_domain


    #%% ------------------------------------------------------------ COMPUTE FUNCTIONS ------------------------------------------------------------

    # Implement a function to compute the ranges of the domain.
    def compute_ranges( self, spatiotemporal_domain = None ):

        # Preprocess the spatiotemporal domain.
        spatiotemporal_domain = self.preprocess_spatiotemporal_domain( spatiotemporal_domain )

        # Compute the domain ranges.
        ranges = spatiotemporal_domain[ 1, : ] - spatiotemporal_domain[ 0, : ]

        # Return the domain ranges.
        return ranges
        

    # Implement a function to compute the volume of the domain.
    def compute_volume( self, spatiotemporal_domain = None, domain_type = None ):

        # Preprocess the spatiotemporal domain.
        spatiotemporal_domain = self.preprocess_spatiotemporal_domain( spatiotemporal_domain )

        # Preprocess the domain type.
        domain_type = self.preprocess_domain_type( domain_type )

        # Determine how to compute the volume of the domain.
        if domain_type.lower(  ) == 'cartesian':                # If the domain type is cartesian...

            # Compute the volume.
            volume = torch.prod( spatiotemporal_domain[ 1, : ] - spatiotemporal_domain[ 0, : ] )

        else:                                                   # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid domain type: {domain_type}' )

        # Return the volume.
        return volume


    #%% ------------------------------------------------------------ UTILITY FUNCTIONS ------------------------------------------------------------

    # Implement a function to label the dimension types.
    def generate_dimension_labels( self, num_temporal_dimensions = None, num_spatial_dimensions = None ):

        # Preprocess the number of temporal dimensions.
        num_temporal_dimensions = self.preprocess_num_temporal_dimensions( num_temporal_dimensions )

        # Preprocess the number of spatial dimensions.
        num_spatial_dimensions = self.preprocess_num_spatial_dimensions( num_spatial_dimensions )

        # Generate the dimension labels.
        dimension_labels = [ 't' ]*num_temporal_dimensions + [ 'x' ]*num_spatial_dimensions

        # Return the dimension labels.
        return dimension_labels


    # Implement a function to sample data from the domain.
    def sample_domain( self, num_samples, domain_type = 'spatiotemporal' ):

        # Retrieve the sample domain.
        sample_domain = self.get_domain( domain_type )

        # Generate a sample from the specified domain.
        data_sample = ( sample_domain[ 1, : ] - sample_domain[ 0, : ] )*torch.rand( size = ( num_samples, sample_domain.shape[ 1 ] ), dtype = torch.float32, device = self.device ) + sample_domain[ 0, : ]

        # Return the sample from the specified domain.
        return data_sample


    # Implement a function to sample a specific boundary of the domain.
    def sample_domain_boundary( self, num_samples, dimension, placement_string = 'lower', domain_type = 'spatiotemporal' ):

        # Sample the complete domain.
        data_sample = self.sample_domain( num_samples, domain_type = domain_type )

        # Determine the first index we should use to retrieve the value we want to substitute into the data sample.
        if ( ( placement_string.lower(  ) == 'left' ) or ( placement_string.lower(  ) == 'lower' ) or ( placement_string.lower(  ) == 'bottom' ) ):              # If we want to sample from the lower boundary...

            # Set the placement index to zero.
            placement_index = torch.tensor( 0, dtype = torch.uint8, device = self.device )

        elif ( ( placement_string.lower(  ) == 'right' ) or ( placement_string.lower(  ) == 'upper' ) or ( placement_string.lower(  ) == 'top' ) ):              # If we want to sample from the upper boundary...

            # Set the placement index to one.
            placement_index = torch.tensor( 1, dtype = torch.uint8, device = self.device )

        else:                                                                                                                               # Otherwise...

            # Throw an error.
            raise ValueError( f'Invalid placement: {placement_string}' )

        # Determine which value to substitute into the sample based on the specified domain of interest.
        if domain_type.lower(  ) == 'temporal':                         # If we are interested in sampling from the temporal domain...

            # Set the value of interest to be the specified boundary of the given dimension of the temporal domain.
            boundary_value = self.temporal_domain[ placement_index, dimension ]

        elif domain_type.lower(  ) == 'spatial':                        # If we are interested in sampling from the spatial domain...

            # Set the value of interest to be the specified boundary of the given dimension of the spatial domain.
            boundary_value = self.spatial_domain[ placement_index, dimension ]

        elif domain_type.lower(  ) == 'spatiotemporal':                 # If we are interested in sampling from the spatiotemporal domain...

            # Set the value of interest to be the specified boundary of the given dimension of the spatial domain.
            boundary_value = self.spatiotemporal_domain[ placement_index.item(  ), dimension.item(  ) ]

        else:                                                           # Otherwise...

            # Throw an error.
            raise ValueError( f'Boundary value: {boundary_value}' )

        # Substitute the boundary value into the specified dimension of the data sample.
        data_sample[ :, dimension.item(  ) ] = boundary_value*torch.ones( num_samples, dtype = torch.float32, device = self.device )

        # Return the data sample.
        return data_sample


    # Implement a function to determine whether a point is contained in a domain.
    def contains_scalar( self, point, domain_type = 'spatiotemporal' ):

        # Retrieve the domain associated with this domain type.
        domain = self.get_domain( domain_type )

        # Compute the number of dimensions associated with this domain.
        num_dimensions = domain.shape[ 1 ]

        # Squeeze the input point.
        point = point.squeeze(  )

        # Set the contained flag to true.
        contained_flag = True

        # Initialize the loop counter variable to zero.
        k = torch.tensor( 0, dtype = torch.int64, device = self.device )

        # Determine whether the given point is contained in the given domain.
        while contained_flag and ( k < num_dimensions ):                    # While the point appears to be contained and we have not yet checked every dimension of the domain...

            # Determine whether the kth dimension of the point is contained in the kth dimension of the domain.
            if ( point[ k ] < domain[ 0, k ] ) or ( point[ k ] > domain[ 1, k ] ):

                # Set the contained flag to false.
                contained_flag = False

            # Advance the loop counter.
            k += 1

        # Return the contained flag.
        return contained_flag


    # Implement a function to determine whether a set of points is contained in a domain.
    def contains( self, points, domain_type = 'spatiotemporal' ):

        # Compute the number of points.
        num_points = points.shape[ 0 ]

        # Preallocate a variable to store the contained flags.
        contained_flags = torch.empty( num_points, dtype = torch.bool, device = self.device )

        # Determine whether each point is contained in the domain.
        for k in range( num_points ):           # Iterate through each of the points...

            # Determine whether this point is contained in the domain.
            contained_flags[ k ] = self.contains_scalar( points[ k, : ], domain_type )

        # Return the contained flag.
        return contained_flags


    #%% ------------------------------------------------------------ SAVE & LOAD FUNCTIONS ------------------------------------------------------------

    # Implement a function to save a domain object.
    def save( self, save_path = None, file_name = r'domain.pkl' ):

        # Save the domain object.
        self.save_load_utilities.save( self, save_path, file_name )


    # Implement a function to load a domain object.
    def load( self, load_path = None, file_name = r'domain.pkl' ):

        # Load the domain object.
        self = self.save_load_utilities.load( load_path, file_name )

        # Return the domain object.
        return self


    #%% ------------------------------------------------------------ PRINTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to print a summary of the domain information.
    def print( self, num_dashes = 20, decoration_flag = True ):

        # Print a header.
        self.printing_utilities.print_header( 'DOMAIN SUMMARY', num_dashes, decoration_flag )

        # Print out the domain info.
        print( 'Temporal Information' )
        print( f'# of Temporal Dimensions: {self.num_temporal_dimensions}' )
        print( f'Temporal Domain: {self.temporal_domain}' )
        print( '\n' )

        print( 'Spatial Information' )
        print( f'# of Spatial Dimensions: {self.num_spatial_dimensions}' )
        print( f'Spatial Domain: {self.spatial_domain}' )
        print( '\n' )

        print( 'Spatiotemporal Information' )
        print( f'Total # of Spatiotemporal Dimensions: {self.num_spatiotemporal_dimensions}' )
        print( f'Spatiotemporal Domain: {self.spatiotemporal_domain}' )

        # Print a footer.
        self.printing_utilities.print_footer( num_dashes, decoration_flag )


    #%% ------------------------------------------------------------ PLOTTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to plot the domain.
    def plot( self, projection_dimensions = None, projection_values = None, level = 0, dimension_labels = None, fig = None, domain_type = 'spatiotemporal', save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Preprocess the dimension labels.
        dimension_labels = self.preprocess_dimension_labels( dimension_labels )

        # Retrieve the domain grid.
        domain_grid = self.get_domain_grid( domain_type )

        # Set the plot title.
        title_string = f'Domain: {domain_type}'

        # Generate the input labels.
        input_labels = self.plotting_utilities.dimension_labels2axis_labels( dimension_labels )

        # Plot the desired domain.
        figs, axes = self.plotting_utilities.plot( domain_grid, [  ], projection_dimensions, projection_values, level, fig, input_labels, title_string, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figures and axes.
        return figs, axes

