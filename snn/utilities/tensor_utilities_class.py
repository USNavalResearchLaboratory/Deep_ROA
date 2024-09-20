####################################################################################### 
# THIS SOURCE CODE IS PROPERTY OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. 
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND 
# CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION 
# ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN 
# LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE,
# CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL 
# PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641. 
####################################################################################### 


#%% ------------------------------------------------------------ TENSOR UTILITIES CLASS ------------------------------------------------------------

# This file implements a class for storing and managing tensor utilities information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch
import warnings
import math


#%% ------------------------------------------------------------ TENSOR UTILITIES CLASS ------------------------------------------------------------

# Implement the tensor utilities class.
class tensor_utilities_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self ):

        # Pass through the constructor.
        pass


    #%% ------------------------------------------------------------ BASIC FUNCTIONS ------------------------------------------------------------

    # Implement a function to convert an absolute tensor index to a dimensional tensor index.
    def absolute_index2dimensional_index( self, absolute_index, dimensions ):

        # Given an index absolute_index corresponding to a values index in a flattened tensor, this function returns the corresponding index m that would represent this same value if the tensor were reshaped to have dimensions dimensions.

        # Retrieve the dimension of this problem.
        num_dimensions = dimensions.numel(  )

        # Initialize a list of dimensional indexes.
        dimensional_indexes = torch.empty( num_dimensions, dtype = torch.int64, device = absolute_index.device )

        # Compute the index associated with each dimension.
        for k in range( num_dimensions - 1 ):                               # Iterate through each dimension (less one)...

            # Compute the number of elements before this dimension.
            num_elements = torch.prod( dimensions[ 0:-( k + 1 ) ] )

            # Compute the index associated with this dimension.
            i = absolute_index // num_elements

            # Compute the new desired index.
            absolute_index %= num_elements

            # Append the current index to the list of dimensional indexes.
            dimensional_indexes[ k ] = i

        # Append the final index to the list of dimensional indexes.
        dimensional_indexes[ -1 ] = absolute_index

        # Reverse the list of dimensional indexes.
        dimensional_indexes = torch.flip( dimensional_indexes, 0 )

        # Return the list of dimensional indexes.
        return dimensional_indexes


    # Implement a function to convert a row / column based index into an absolute matrix dimension for 2D tensors.
    def dimensional_index2absolute_index( self, dimensional_indexes, dimensions ):

        # Given a list of indexes corresponding to a specific value in an tensor and the shape of this tensor dimensions, this function returns the index that would be associated with this same value were the tensor to be flattened.

        # Retrieve the dimension of this problem.
        num_dimensions = dimensions.numel(  )

        # Add a place holder one to the dimensions.
        dimensions = torch.cat( ( torch.ones( 1, dtype = torch.int64, device = dimensional_indexes.device ), dimensional_indexes ), dim = 0 )

        # Initialize the absolute index to zero.
        absolute_index = torch.tensor( 0, dtype = torch.int64, device = dimensional_indexes.device )

        # Compute the absolute index one term at a time.
        for k in range( num_dimensions ):                    # Iterate through each dimension...

            # Add the absolute index term contributed by this dimension to the total.
            absolute_index += dimensional_indexes[ k ]*torch.prod( dimensions[ 0:( k + 1 ) ] )

        # Return the absolute index.
        return absolute_index


    # Implement a function to remove the specified indexes of a tensor.
    def delete_entries( data, indexes, dim = None ):

        # Determine how to create the mask tensor.
        if dim is None:             # If the dimension is None...

            # Create a mask tensor of true values.
            mask = torch.ones( data.shape, dtype = torch.bool )
        
        else:                       # Otherwise...

            # Create a mask tensor of true values.
            mask = torch.ones( data.shape[ dim ], dtype = torch.bool )

        # Set the specified indexes of the mask to be false.
        mask[ indexes ] = False

        # Delete the desired entries from the data.
        data = data[ mask ]

        # Return the data with the desired entries removed.
        return data


    # Implement a function to shuffle tensors.
    def shuffle_data( self, data, dim = 0 ):

        # Determine whether to imbed the data in a list.
        if not ( isinstance( data, list ) or isinstance( data, tuple ) ):           # If the provided data is a list or tuple...

            # Embed the data in a list.
            data = [ data ]

            # Set the embedded flag to true.
            expose_flag = True

        else:                                                                       # Otherwise...

            # Set the embedded flag to false.
            expose_flag = False

        # Generate shuffled indexes.
        indexes = torch.randperm( data[ 0 ].shape[ dim ], dtype = torch.int64, device = data.device )

        # Shuffle the data.
        data = [ data[ k ][ indexes, ... ] for k in range( len( data ) ) ]

        # Determine whether to expose the data.
        if expose_flag:               # If we want to expose the data (e.g., because )...

            # Expose the data.
            data = data[ 0 ]

        # Return the shuffled data.
        return data


    #%% ------------------------------------------------------------ VALIDATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate the compatibility of domain and number of points tensors.
    def validate_domain_num_points_compatibility( self, domain, num_points ):

        # Determine whether the domain and number of points objects is valid.
        if ( torch.is_tensor( domain ) and domain.shape[ -1 ] != 0 ) and ( torch.is_tensor( num_points ) and num_points.numel(  ) != 0 ):           # If the domain and number of points are non-empty tensors...

            # Determine whether the number of domain elements and the number of number of points are equal.
            valid_flag = domain.shape[ -1 ] == num_points.numel(  )

        else:                                                                                                                                       # Otherwise... ( i.e., the domain and number of points are not compatible... )

            # Set the valid flag to false.
            valid_flag = False

        # Return the valid flag.
        return valid_flag

    
    # Implement a function to check that given values and subgrid indexes are compatible.
    def validate_values_subgrid_indexes_compatibility( self, values, subgrid_indexes ):

        # Determine whether there are the same number of values and subgrid indexes.
        valid_flag = values.numel(  ) == subgrid_indexes.numel(  )

        # Return the validity flag.
        return valid_flag

    # Implement a function to check that a given nontemporal grid and subgrid indexes are compatible.
    def validate_nontemporal_grid_subgrid_indexes_compatibility( self, grid, subgrid_indexes ):

        # Determine whether all of the subgrid indexes are in bounds.
        if all( ( subgrid_indexes < grid.shape[ -1 ] ) ) and all( ( subgrid_indexes >= 0 ) ):                       # If all of the subgrid indexes are in bounds...

            # Set the validation flag to true.
            valid_flag = True

        else:                                                                                                       # Otherwise... ( i.e., not all of the subgrids are in bounds... )

            # Set the validation flag to false.
            valid_flag = False

        # Return the validation flag.
        return valid_flag


    # Implement a function to check that a given temporal grid and subgrid indexes are compatible.
    def validate_temporal_grid_subgrid_indexes_compatibility( self, grid, subgrid_indexes ):

        # Determine whether all of the subgrid indexes are in bounds.
        if all( ( subgrid_indexes < grid.shape[ -2 ] ) ) and all( ( subgrid_indexes >= 0 ) ):                       # If all of the subgrid indexes are in bounds...

            # Set the validation flag to true.
            valid_flag = True

        else:                                                                                                       # Otherwise... ( i.e., not all of the subgrids are in bounds... )

            # Set the validation flag to false.
            valid_flag = False

        # Return the validation flag.
        return valid_flag


    # Implement a function to check that a given grid and subgrid indexes are compatible.
    def validate_grid_subgrid_indexes_compatibility( self, grid, subgrid_indexes, temporal_grid_flag = None ):

        # Determine whether it is necessary to determine whether the grid is temporal.
        if temporal_grid_flag is None:              # If the temporal grid flag was not provided...

            # Detetermine whether the grid is temporal.
            temporal_grid_flag = self.is_grid_temporal( grid, grid_expanded_flat_type = None )

        # Determine how to validate the given grid and subgrid indexes.
        if temporal_grid_flag:                          # If the grid is temporal...

            # Validate the temporal grid and subgrid index compatibility.
            valid_flag = self.validate_temporal_grid_subgrid_indexes_compatibility( grid, subgrid_indexes )

        else:                                           # Otherwise...

            # Validate the nontemporal grid and subgrid index compatibility.
            valid_flag = self.validate_nontemporal_grid_subgrid_indexes_compatibility( grid, subgrid_indexes )

        # Return the validation flag.
        return valid_flag


    #%% ------------------------------------------------------------ PROJECTION FUNCTIONS ------------------------------------------------------------

    # Implement a function to retrieve the slice indexes associated with a grid tensor, desired target dimensions, and the associated desired values.
    def get_slice_indexes( self, grid, target_dims, target_values, recourse = 'ignore' ):

        # Preallocate a tensor to store the slice indexes.
        slice_indexes = torch.empty_like( target_dims, dtype = torch.uint8, device = grid.device )

        # Compute each of the slice indexes.
        for k in range( target_dims.numel(  ) ):                        # Iterate through each of the target dimensions (note: there is one slice index per target dimension)...

            # Determine the slice index.
            min_value = torch.min( torch.abs( grid[ ..., k ] - target_values[ k ] ), dim = k )[ 0 ][ (0,)*( grid.dim() - 2 ) ]
            slice_indexes[ k ] = torch.argmin( torch.abs( grid[ ..., k ] - target_values[ k ] ), dim = k )[ (0,)*( grid.dim() - 2 ) ]

            # Determine how to proceed given the minimum value.
            if ( min_value != 0 ):                  # If the minium value is zero...

                # Determine how to respond to a non-exact match.
                if ( recourse.lower(  ) == 'ignore' ):                      # If the non-zero recourse is set to ignore...

                    # Do nothing.
                    pass

                elif ( recourse.lower(  ) == 'warning' ):                   # If the non-zero recourse is set to warning...

                    # Throw a warning.
                    warnings.warn( f'An exact match for the {k}th target value {target_values[ k ]} associated with dimension {target_dims[ k ]} can not be found in grid {grid}.  Using nearest value instead.' )

                elif ( recourse.lower(  ) == 'error' ):                     # If the non-zero recourse is set to error...

                    # Throw an error.
                    raise ValueError( f'An exact match for the {k}th target value {target_values[ k ]} associated with dimension {target_dims[ k ]} can not be found in grid {grid}.' )

                else:                                                       # If the non-zero recourse is set invalid...

                    # Throw an error.
                    raise ValueError( f'Invalid recourse: {recourse}' )

        # Return the slice indexes.
        return slice_indexes


    # Implement a function to project arbitrary grid tensors.
    def project_tensor( self, grid, projection_dimensions, projection_values ):

        # grid = A torch tensor grid, such as those produced by ndgrid.
        # projection_dimensions = A torch tensor vector that stores the dimensions on which we would like to project the data.
        # projection_values = A torch tensor vector of the same shape as projection_dimensions that specifies the values at which to project onto each of the dimensions in projection_dimensions.

        # Ensure that there are no duplicate projection dimensions.
        assert torch.sort( projection_dimensions )[ 0 ] == torch.unique( projection_dimensions, sorted == True )

        # Ensure that the projection dimensions and projection values are compatible.
        assert projection_dimensions.numel(  ) == projection_values.numel(  )

        # Compute the slice indexes associated with the target values in the given dimensions.
        projection_slices = self.get_slice_indexes( grid, projection_dimensions, projection_values )

        # Initialize the projected grid tensor by cloning the original grid tensor.
        projected_grid = torch.clone( grid )

        # Perform each of the projections.
        for k in range( projection_dimensions.numel(  ) ):                      # Iterate through each of the projections...

            # Project the grid tensor onto this dimension.
            projected_grid = torch.unsqueeze( torch.select( projected_grid, projection_dimensions[ k ], projection_slices[ k ] ), projection_dimensions[ k ] )

            # Remove the projected subgrid.
            projected_grid = torch.cat( ( projected_grid[ ..., :projection_dimensions[ k ].item(  ) ], projected_grid[ ..., projection_dimensions[ k ].item(  ) + 1: ] ), dim = -1 )

        # Squeeze the projected grid.
        projected_grid.squeeze_(  )

        # Return the projected grid tensor.
        return projected_grid


    # Implement a function to project input and output grids.
    def project_input_output_data( self, input_data, output_data, projection_dimensions, projection_values ):

        # Retrieve the number of input and output dimensions.
        num_input_dimensions = self.get_number_of_dimensions( input_data )

        # Define the number of projection input dimensions.
        num_input_dimensions_projected = num_input_dimensions - projection_dimensions.numel(  )

        # Construct the grid from the input and output data.
        grid = torch.cat( ( input_data, output_data ), dim = -1 )

        # Project the data grid.
        projected_grid = self.project_tensor( grid, projection_dimensions, projection_values )

        # Retrieve the input and output grids.
        projected_input_data = projected_grid[ ..., :num_input_dimensions_projected ]
        projected_output_data = projected_grid[ ..., num_input_dimensions_projected: ]

        # Return the projected input and output data.
        return projected_input_data, projected_output_data


    #%% ------------------------------------------------------------ NEWTON'S METHOD FUNCTIONS ------------------------------------------------------------

    # Implement a function to perform newton's method.
    def newtons_method_scalar( self, f, x, tol, max_iterations ):

        # Determine whether the input needs its requires grad flag updated.
        if not x.requires_grad:                 # If the input does not require grad...

            # Ensure that the input point has its require derivative flag set to true.
            x.requires_grad = True

        # Compute the starting function value.
        y = f( x )

        # Initialize a counter variable.
        k = torch.tensor( 0, dtype = torch.int64, device = x.device )

        # Perform newton's method to estimate the root.
        while ( k < max_iterations ) and ( torch.linalg.norm( y, ord = 2 ) > tol ):             # While the maximum number of iterations have not yet been performed and the root has not yet been estimated with acceptable accuracy...

            # Compute the derivative of the output with respect to the input.
            jacobian = self.compute_data_jacobians( x, y )

            # Compute the new root estimate.
            x -= torch.transpose( torch.linalg.pinv( jacobian[ 0, :, : ] )*y, 0, 1 )

            # Compute the new associated function value.
            y = f( x )

            # Advance the counter variable.
            k += 1

        # Return the root.
        return x


    # Implement a function to perform newton's method.
    def newtons_method( self, f, xs, tol, max_iterations ):

        # Determine whether the input needs its requires grad flag updated.
        if not xs.requires_grad:                 # If the input does not require grad...

            # Ensure that the input point has its require derivative flag set to true.
            xs.requires_grad = True

        # Retrieve the number of starting points.
        num_points = xs.shape[ 0 ]

        # Create a tensor to store the indexes associated with the input points.
        indexes = torch.arange( num_points, device = xs.device )

        # Compute the starting function value.
        ys = f( xs )

        # Preallocate tensors to store the complete roots and their values.
        xs_complete = torch.zeros_like( xs )
        ys_complete = torch.zeros_like( ys )

        # Compute the initial convergence flags.
        convergence_flags = torch.linalg.norm( ys, ord = 2, dim = 1 ) < tol

        # Store any converged roots and their associated values.
        convergence_indexes = indexes[ convergence_flags ]
        xs_complete[ convergence_flags ] = xs[ convergence_flags ]
        ys_complete[ convergence_flags ] = ys[ convergence_flags ]

        # Remove the converged roots and their associated values.
        indexes = indexes[ ~convergence_flags ]
        xs = xs[ ~convergence_flags ]

        # Determine whether to re-evaluate our function.
        if ( xs.numel(  ) != 0 ):               # If the number of input values is non-zero...

            # Evaluate the function.
            ys = f( xs )

        else:                                                           # Otherwise... (If the number of input values is zero...)

            # Set the function evaluation to be empty.
            ys = ys_complete[ False, 0, : ]

        # Initialize a counter variable.
        k = torch.tensor( 0, dtype = torch.int64, device = xs.device )

        # Perform newton's method to estimate the root.
        while ( k < max_iterations ) and ( xs.numel(  ) != 0 ):             # While the maximum number of iterations have not yet been performed and the root has not yet been estimated with acceptable accuracy...

            # Compute the derivative of the output with respect to the input.
            jacobians = self.compute_data_jacobians( xs, ys )

            # Compute the new root estimate.
            xs -= torch.linalg.lstsq( jacobians, ys ).solution

            # Compute the new associated function value.
            ys = f( xs )

            # Compute the initial convergence flags.
            convergence_flags = torch.linalg.norm( ys, ord = 2, dim = 1 ) < tol

            # Store any converged roots and their associated values.
            convergence_indexes = indexes[ convergence_flags ]
            xs_complete[ convergence_indexes ] = xs[ convergence_flags ]
            ys_complete[ convergence_indexes ] = ys[ convergence_flags ]

            # Remove the converged roots and their associated values.
            indexes = indexes[ ~convergence_flags ]
            xs = xs[ ~convergence_flags ]

            # Determine whether to re-evaluate our function.
            if ( xs.numel(  ) != 0 ):               # If the number of input values is non-zero...

                # Evaluate the function.
                ys = f( xs )

            else:                                                           # Otherwise... (If the number of input values is zero...)

                # Set the function evaluation to be empty.
                ys = ys_complete[ False, 0, : ]

            # Advance the counter variable.
            k += 1

        # Add any remaining incomplete points and roots to the set of complete ones.
        xs_complete[ indexes ] = xs
        ys_complete[ indexes ] = ys

        # Determine the convergence status of each of the points.
        convergence_flags = torch.linalg.norm( ys_complete, ord = 2, dim = 1 ) < tol

        # Return the roots.
        return xs_complete, ys_complete, convergence_flags



    #%% ------------------------------------------------------------ GRID FUNCTIONS ------------------------------------------------------------

    # Implement a function to convert a grid list to a grid tensor.
    def grid_list2grid_tensor( self, grid_list ):

        # This function converts a grid list to a grid tensor.  A grid list is a list of meshgrid tensor objects.  A grid tensor is an tensor of meshgrid objects concatenated along the dimension one higher than the dimension of the grids.

        # Retrieve the number of grids.
        num_grids = len( grid_list )

        # Retrieve the grid dimensions.
        grid_dims = grid_list[ 0 ].shape

        # Preallocate an tensor to store the grid values.
        grid_tensor = torch.zeros( ( *grid_dims, num_grids ), dtype = grid_list[ 0 ].dtype, device = grid_list[ 0 ].device )

        # Store each grid in the grid list along the last dimension in the grid tensor.
        for k in range( num_grids ):                        # Iterate through each of the grids...

            # Store this grid in the grid tensor.
            grid_tensor[ ..., k ] = grid_list[ k ]

        # Return the grid tensor.
        return grid_tensor


    # Implement a function to convert a grid tensor to a grid list.
    def grid_tensor2grid_list( self, grid_tensor ):

        # This function converts a grid tensor to a grid list.  A grid tensor is an tensor of meshgrid objects concatenated along the dimension one higher than the dimension of the grids.  A grid list is a list of meshgrid tensor objects.

        # Store each grid in the grid tensor as a separate entry in the grid list.  Note that each grid is stored along the last dimension of the grid tensor.
        return [ grid_tensor[ ..., k ] for k in range( grid_tensor.shape[ -1 ] ) ]


    # Implement a function to convert the grid vectors to a grid.
    def grid_vectors2grid( self, grid_vectors ):

        # Retrieve the number of grid vectors.
        num_grid_vectors = len( grid_vectors )

        # Convert the grid vectors into a tensor grid.
        grid_tensors = torch.meshgrid( grid_vectors, indexing = 'ij' )

        # Convert the grid tensors into a grid.
        grid = torch.cat( tuple( grid_tensors[ k ].unsqueeze( num_grid_vectors ) for k in range( num_grid_vectors ) ), dim = num_grid_vectors )

        # Return the grid.
        return grid


    # Implement a function to convert the grid to grid vectors.
    def grid2grid_vectors( self, grid ):

        # Retrieve the number of subgrids.
        num_subgrids = grid.shape[ -1 ]

        # Initialize the grid vectors list.
        grid_vectors = [  ]

        for k in range( num_subgrids ):                 # Iterate through each of the subgrids...

            # Retrieve the grid vectors.
            grid_vectors.append( grid[ ..., k ].unique(  ) )

        # Return the grid vectors.
        return grid_vectors


    # Implement a function to generate grid vectors.
    def generate_grid_vectors( self, domain, num_points ):

        # Ensure that the domain and number of points tensors are compatible.
        assert self.validate_domain_num_points_compatibility( domain, num_points )

        # Generate each grid vector.
        grid_vectors = [ torch.linspace( domain[ 0, k ], domain[ 1, k ], num_points[ k ], dtype = torch.float32, device = domain.device ) for k in range( domain.shape[ -1 ] ) ]

        # Return the grid vectors.
        return grid_vectors


    # Implement a function to generate a grid.
    def generate_grid( self, domain, num_points ):

        # Generate each grid vector.
        grid_vectors = self.generate_grid_vectors( domain, num_points )

        # Convert the grid vectors into a grid.
        grid = self.grid_vectors2grid( grid_vectors )

        # Return the grid.
        return grid


    # Implement a function to flatten a non-temporal grid.
    def flatten_nontemporal_grid( self, expanded_nontemporal_grid ):

        # Preallocate a tensor to store the network input tensor.
        flattened_nontemporal_grid = torch.empty( size = ( expanded_nontemporal_grid[ ..., 0 ].numel(  ), expanded_nontemporal_grid.shape[ -1 ] ), dtype = expanded_nontemporal_grid.dtype, device = expanded_nontemporal_grid.device )

        # Store the grid data in the input data tensor.
        for k in range( expanded_nontemporal_grid.shape[ -1 ] ):                   # Iterate through each of the final grid dimension elements...

            # Store the input data associated with this input.
            flattened_nontemporal_grid[ :, k ] = expanded_nontemporal_grid[ ..., k ].ravel(  )

        # Return the input data.
        return flattened_nontemporal_grid


    # Implement a function to flatten a temporal grid.
    def flatten_temporal_grid( self, expanded_temporal_grid ):

        # Preallocate a tensor to store the network input tensor.
        flattened_temporal_grid = torch.empty( size = ( expanded_temporal_grid[ ..., 0, -1 ].numel(  ), expanded_temporal_grid.shape[ -2 ], expanded_temporal_grid.shape[ -1 ] ), dtype = expanded_temporal_grid.dtype, device = expanded_temporal_grid.device )

        # Store the grid data in the input data tensor.
        for k2 in range( expanded_temporal_grid.shape[ -1 ] ):                          # Iterate through each of the timesteps...
            for k1 in range( expanded_temporal_grid.shape[ -2 ] ):                      # Iterate through each of the spatial dimensions...

                # Store the input data associated with this input.
                flattened_temporal_grid[ :, k1, k2 ] = expanded_temporal_grid[ ..., k1, k2 ].ravel(  )

        # Return the input data.
        return flattened_temporal_grid
    

    # Implement a function to convert a grid tensor to a network input tensor.
    def flatten_grid( self, expanded_grid ):

        # Determine how to flatten the grid.
        if self.is_grid_temporal( expanded_grid, grid_expanded_flat_type = self.get_grid_expanded_flat_type( expanded_grid ) ):

            # Flatten the temporal grid.
            flattened_grid = self.flatten_temporal_grid( expanded_grid )

        else:                                           # Otherwise...

            # Flatten the nontemporal grid.
            flattened_grid = self.flatten_nontemporal_grid( expanded_grid )

        # Return the flattened grid.
        return flattened_grid


    # Implement a function to expand a nontemporal grid.
    def expand_nontemporal_grid( self, flattened_grid, subgrid_dims ):

        # Preallocate a tensor to store the output grid.
        expanded_grid = torch.empty( size = tuple( subgrid_dims ) + ( flattened_grid.shape[ 1 ], ), dtype = torch.float32, device = flattened_grid.device )

        # Store the output data in the grid.
        for k in range( flattened_grid.shape[ 1 ] ):                # Iterate through each of the network output channels...

            # Store the output data associated with this channel in the grid.
            expanded_grid[ ..., k ] = flattened_grid.reshape( subgrid_dims )

        # Return the output grid.
        return expanded_grid


    # Implement a function to expand a temporal grid.
    def expand_temporal_grid( self, flattened_grid, subgrid_dims ):

        # Preallocate a tensor to store the output grid.
        expanded_grid = torch.empty( size = tuple( subgrid_dims ) + ( flattened_grid.shape[ 1 ], flattened_grid.shape[ 2 ] ), dtype = torch.float32, device = flattened_grid.device )

        # Store the output data in the grid.
        for k2 in range( flattened_grid.shape[ 2 ] ):               # Iterate through each of the network timesteps...
            for k1 in range( flattened_grid.shape[ 1 ] ):           # Iterate through each of the network output channels...

                # Store the output data associated with this channel in the grid.
                expanded_grid[ ..., k1, k2 ] = flattened_grid[ :, k1, k2 ].reshape( subgrid_dims )

        # Return the output grid.
        return expanded_grid


    # Implement a function to convert a network output tensor to a grid tensor.
    def expand_grid( self, flattened_grid, subgrid_dims ):

        # Determine how to exapdn the grid.
        if self.is_grid_temporal( flattened_grid, grid_expanded_flat_type = self.get_grid_expanded_flat_type( flattened_grid ) ):           # If the grid is temporal...

            # Compute the expanded grid.
            expanded_grid = self.expand_temporal_grid( flattened_grid, subgrid_dims )

        else:                                                                                                                               # Otherwise...

            # Compute the expanded grid.
            expanded_grid = self.expand_nontemporal_grid( flattened_grid, subgrid_dims )

        # Return the output grid.
        return expanded_grid


    # Implement a function to ensure that a flattened grid is a tensor.
    def ensure_flattened_grid_is_tensor( self, flattened_grid ):

        # Ensure that the flattened grid input and flattened grid output are tensors.
        if isinstance( flattened_grid, list ) or isinstance( flattened_grid, tuple ):           # If the flattened grid is a list or tuple...

            # Retrieve the number of dimensions.
            num_dimensions = len( flattened_grid )

            # Retrieve the number of grid points.
            num_grid_points = flattened_grid[ 0 ].shape[ 0 ]

            # Initialize an empty tensor to store the flattened grid tensor.
            flattened_grid_tensor = torch.empty( ( num_grid_points, num_dimensions ), dtype = torch.float32, device = flattened_grid[ 0 ].device )

            # Store the flatten grid list entries into the flattened grid tensor.
            for k in range( num_dimensions ):               # Iterate through each of the dimensions...

                # Store the flattened grid list entries associated with this dimension.
                flattened_grid_tensor[ :, k ] = flattened_grid[ k ]

        elif torch.is_tensor( flattened_grid ):                                                       # If the flattened grid is a tensor...

            # Set the flattened grid tensor to be the provided flattened grid.
            flattened_grid_tensor = flattened_grid
        
        else:                                                                                               # Otherwise... ( i.e., the flattened grid type is not recognized... )

            # Throw an error.
            raise ValueError( f'Invalid flattened grid: {flattened_grid}' )

        # Return the flattened grid tensor.
        return flattened_grid_tensor
    

    #%% ------------------------------------------------------------ DERIVATIVE FUNCTIONS ------------------------------------------------------------

    # Implement a function to compute the derivatives of a set of input and output points from a scalar function.
    def compute_data_derivatives( self, inputs, outputs, output_factor = None ):

        # Determine whether to use the default output factor.
        if output_factor is None:           # If the output factor was not provided...

            # Set the output factor to be the default value.
            output_factor = torch.ones_like( outputs )

        # Compute the derivatives of numerator tensor with respect to the denominator tensor.
        outputs.backward( gradient = output_factor, create_graph = False, inputs = inputs )

        # Clone the derivatives so that the derivative calculation can be zeroed out.
        derivatives = torch.clone( inputs.grad )

        # Zero out the derivative.
        inputs.grad = torch.zeros_like( inputs.grad )

        # Return the derivative.
        return derivatives


     # Implement a function to compute the derivatives of a scalar function at a set of input points.
    def compute_function_derivatives( self, function, function_inputs ):

        # Compute the numerator tensors.
        function_outputs = function( function_inputs )

        # Compute the derivatives of the function output tensors with respect to the function input tensors.
        derivatives = self.compute_data_derivatives( function_inputs, function_outputs )

        # Return the derivatives.
        return derivatives


    # Implement a function to compute the gradient of a set of flat input and output data points from a function with multiple inputs.
    def compute_flat_data_gradients( self, flat_grid_input, flat_grid_output ):

        # Ensure that the flattened grid input and flattened grid output are tensors.
        flat_grid_input = self.ensure_flattened_grid_is_tensor( flat_grid_input )
        flat_grid_output = self.ensure_flattened_grid_is_tensor( flat_grid_output )

        # Compute the number of grid points and input dimensions.
        num_grid_points = flat_grid_input.shape[ 0 ]
        num_inputs = flat_grid_input.shape[ -1 ]

        # Preallocate a tensor to stored the gradients.
        gradients = torch.empty( ( num_grid_points, num_inputs ), dtype = torch.float32, device = flat_grid_input.device )

        # Compute the derivatives.
        for k in range( num_inputs ):              # Iterate through each of the inputs...

                # Compute the gradients of this output with respect to this input.
                gradients[ :, k ] = self.compute_data_derivatives( flat_grid_input[ :, k ], flat_grid_output )

        # Return the gradients.
        return gradients


    # Implement a function to compute the gradient of a grid of data points ( flattened or otherwise ).
    def compute_data_gradients( self, grid_input, grid_output ):

        # Determine whether the grid input needs to be flattened.
        if not self.is_grid_flat( grid_input ):            # If the input grid is expanded...

            # Flattened the expanded input grid.
            grid_input = self.flatten_grid( grid_input )

        # Determine whether the grid output needs to be flattened.
        if not self.is_grid_flat( grid_output ):            # If the output grid is expanded...

            # Flattened the expanded output grid.
            grid_input = self.flatten_grid( grid_input )

        # Compute the gradients associated with the flattened input and output grids.
        gradients = self.compute_flat_data_gradients( grid_input, grid_output )

        # Return the gradients.
        return gradients
    

    # Implement a function to compute the gradient of a function with multiple inputs at a set of input points ( flattened or otherwise ).
    def compute_function_gradients( self, function, function_inputs ):

        # Compute the output data associated with the given function and inputs.
        function_outputs = function( function_inputs )

        # Compute the gradients of input and output data.
        gradients = self.compute_data_gradients( function_inputs, function_outputs )

        # Return the gradients.
        return gradients


    # Implement a function to compute the jacobian of a set of flat input and output data points from a function with multiple inputs and multiple outputs.
    def compute_flat_data_jacobians( self, flat_grid_input, flat_grid_output ):

        # Ensure that the flattened grid input and flattened grid output are tensors.
        flat_grid_input = self.ensure_flattened_grid_is_tensor( flat_grid_input )
        flat_grid_output = self.ensure_flattened_grid_is_tensor( flat_grid_output )

        # Compute the number of inputs and outputs.
        num_inputs = flat_grid_input.shape[ -1 ]
        num_outputs = flat_grid_output.shape[ -1 ]

        # Compute the number of flattened grid points.
        num_grid_points = flat_grid_input.shape[ 0 ]

        # Preallocate a tensor to stored the jacobians.
        jacobians = torch.empty( ( num_grid_points, num_outputs, num_inputs ), dtype = torch.float32, device = flat_grid_input.device )

        # Compute the jacobians.
        for k in range( num_outputs ):                 # Iterate through each of the outputs...

                # Reset the output factor.
                output_factor = torch.zeros_like( flat_grid_output )

                # Set the corresponding column of the output factor matrix to ones.
                output_factor[ :, k ] = torch.ones( num_grid_points, dtype = output_factor.dtype, device = output_factor.device )

                # Compute the derivative of this output with respect to this input.
                jacobians[ :, k, : ] = self.compute_data_derivatives( flat_grid_input, flat_grid_output, output_factor )

        # Return the jacobians.
        return jacobians


    # Implement a function to compute the jacobian of a grid of data points ( flattened or otherwise ).
    def compute_data_jacobians( self, grid_input, grid_output ):

        # Determine whether the grid input needs to be flattened.
        if not self.is_grid_flat( grid_input ):            # If the input grid is expanded...

            # Flattened the expanded input grid.
            grid_input = self.flatten_grid( grid_input )

        # Determine whether the grid output needs to be flattened.
        if not self.is_grid_flat( grid_output ):            # If the output grid is expanded...

            # Flattened the expanded output grid.
            grid_output = self.flatten_grid( grid_output )

        # Compute the jacobians associated with the flattened input and output grids.
        jacobians = self.compute_flat_data_jacobians( grid_input, grid_output )

        # Return the jacobians.
        return jacobians


    # Implement a function to compute the jacobian of a function with multiple inputs and outputs at a set of input points ( flattened or otherwise ).
    def compute_function_jacobians( self, function, function_inputs ):

        # Compute the output data associated with the given function and inputs.
        function_outputs = function( function_inputs )

        # Compute the jacobians of input and output data.
        jacobians = self.compute_data_jacobians( function_inputs, function_outputs )

        # Return the jacobians.
        return jacobians


    #%% ------------------------------------------------------------ INTEGRATION FUNCTIONS ------------------------------------------------------------

    # Implement a function to perform a single step of fourth order Runge-Kutta integration.
    def RK4_step( self, f, t, x, dt ):

        # Compute half the step size.
        dt_half = dt/2

        # Compute the middle time.
        t_mid = t + dt_half

        # Compute the final time.
        t_final = t + dt

        # Compute the RK4 intermediate derivative estimates.
        k1 = f( t, x )
        k2 = f( t_mid, x + dt_half*k1 )
        k3 = f( t_mid, x + dt_half*k2 )
        k4 = f( t_final, x + dt*k3 )

        # Compute the RK4 derivative estimate.
        dxdt = ( 1/6 )*( k1 + 2*k2 + 2*k3 + k4 )

        # Compute the change in the state.
        dx = dt*dxdt

        # Compute the next state by applying the RK4 derivative estimate.
        x += dx

        # Return the state estimate and state derivative estimate after a single RK4 step.
        return x, dx


    # Implement a function to perform fourth order Runge-Kutta integration.
    def RK4_integration( self, f, tspan, x0, dt ):

        # Initialize the integration time.
        t = torch.unsqueeze( torch.repeat_interleave( torch.unsqueeze( tspan[ 0 ], dim = 0 ), x0.shape[ 0 ], dim = 0 ), dim = 1 )

        # Initialize the ODE state.
        x = x0.detach(  ).clone(  )

        # Integrate the ODE.
        while all( t[ :, 0 ] < tspan[ -1 ] ):                          # While we have not yet integrated through the specified times...

            # Perform a single RK4 step.
            x, dx = self.RK4_step( f, t, x, dt )

            # Advance the integration time.
            t += dt

        # Return the final state.
        return x, dx
        

    #%% ------------------------------------------------------------ SUBSPACE FUNCTIONS ------------------------------------------------------------

    # Implement a function to compute the intersection of two subspaces.
    def compute_subspace_intersection( self, U, V ):

        # Compute the number of dimensions of each subspace.
        num_dimensions = U.shape[ 0 ]

        # Compute the number of basis vectors in each of the original subspaces.
        num_U_vectors = U.shape[ -1 ]
        num_V_vectors = V.shape[ -1 ]

        # Compute the number of basis vectors associated with the intersection of these two subspaces.
        num_W_vectors = num_U_vectors + num_V_vectors - num_dimensions

        # Determine how to compute the subspace intersection.
        if num_W_vectors < 0:                                               # If the number of subspace intersection basis vectors is less than zero...

            # Set the subspace intersection to be empty.
            W = torch.tensor( [  ], dtype = torch.float32, device = U.device )

        elif num_W_vectors == 0:                                            # If the number of subspace basis vectors is zero...

            # Define the system matrix.
            A = torch.cat( ( U, -V ), dim = 1 )

            # Determine whether the system matrix is invertible.
            if torch.linalg.matrix_rank( A ) == A.shape[ 0 ]:               # If the system matrix is invertible...

                # Define the right-hand side matrix.
                b = torch.zeros( ( num_dimensions, 1 ), dtype = torch.float32, device = U.device )

                # Solve for the intersection coefficients.
                x = torch.linalg.solve( A, b )

                # Solve for the intersection point.
                W = U*x[ :num_U_vectors ]

            else:                                                           # Otherwise... ( i.e., the system matrix is singular... )

                # Set the intersection subspace to be empty.
                W = torch.tensor( [  ], dtype = torch.float32, device = U.device )

        elif num_W_vectors >= 1:                                            # If the number of subspace basis vectors is one...

            # Define the system matrix.
            A = torch.cat( ( U, -V ), dim = 1 )
            A = A[ :, :-num_W_vectors ]

            # Determine whether the system matrix is invertible.
            if torch.linalg.matrix_rank( A ) == A.shape[ 0 ]:               # If the system matrix is invertible...

                # Define the right-hand side matrix.
                B = -A[ :, -num_W_vectors: ]

                # Solve for the intersection coefficients.
                X = torch.linalg.solve( A, B )

                # Augment the intersection coefficients.
                X = torch.cat( ( X, torch.eye( num_W_vectors ) ), dim = 0 )

                # Compute the intersection subspace basis.
                W = U*X

            else:                                                           # Otherwise... ( i.e., the system matrix is singular... )

                # Set the intersection subspace to be empty.
                W = torch.tensor( [  ], dtype = torch.float32, device = U.device )

        # Return the intersection subspace.
        return W


    # Implement a function to compute the intersection of multiple subspaces.
    def compute_subspace_intersections( self, subspaces ):

        # Compute the number of subspaces.
        num_subspaces = subspaces.shape[ 0 ]

        # Set the subspace intersection to be the first subspace.
        W = subspaces[ 0 ]

        # Initialize a counter variable.
        k = torch.tensor( 0, dtype = torch.int64, device = subspaces.device )

        # Compute the intersection of all of the subspaces.
        while ( k < ( num_subspaces - 1 ) ) and ( W.numel != 0 ):                   # While we have not yet iterated through each of the subspaces and the subspace intersection is not yet empty...

            # Compute the intersection of all of the previous subspaces and the next subspace.
            W = self.compute_subspace_intersection( W, subspaces[ k + 1 ] )

            # Advance the loop variable.
            k += 1

        # Return the subspace intersection.
        return W


    # Implement a function to compute the orthogonal subspace associated with a single gradient.
    def gradient2orthogonal_subspace( self, gradient ):

        # Compute the number of dimensions of the gradient.
        num_dimensions = gradient.numel(  )

        # Initialize the orthogonal subspace.
        orthogonal_subspace = torch.cat( ( torch.eye( num_dimensions - 1, dtype = torch.float32, device = gradient.device ), torch.zeros( ( 1, num_dimensions - 1 ), dtype = torch.float32, device = gradient.device ) ), dim = 0 )

        # Fill in the final row of the orthogonal subspace.
        for k in range( num_dimensions - 1 ):                   # Iterate through each column of the orthogonal subspace...

            # Fill in this column of the last row of the orthogonal subspace.
            orthogonal_subspace[ -1, k ] = - gradient[ k ]/gradient[ -1 ]

        # Normalize the columns of the orthogonal subspace.
        orthogonal_subspace = orthogonal_subspace/torch.linalg.norm( orthogonal_subspace, ord = 2, dim = 0 )

        # Return the orthogonal subspace.
        return orthogonal_subspace


    # Implement a function to compute the orthogonal subspaces associated with a tensor of gradients (each row is a different gradient, each column is a different dimension) (this convention is chosen to match torch's preferred orientation).
    def gradients2orthogonal_subspaces( self, gradients ):

        # Compute the number of dimensions.
        num_dimensions = gradients.shape[ -1 ]

        # Compute the number of gradients.
        num_gradients = gradients.shape[ 0 ]

        # Create the orthogonal subspace template.
        orthogonal_subspace_template = torch.cat( ( torch.eye( num_dimensions - 1, dtype = torch.float32, device = gradients.device ), torch.zeros( ( 1, num_dimensions - 1 ), dtype = torch.float32, device = gradients.device ) ), dim = 0 )

        # Initialize the orthogonal subspace.
        orthogonal_subspace = torch.repeat_interleave( torch.unsqueeze( orthogonal_subspace_template, dim = 0 ), num_gradients, 0 )

        # Compute the orthogonal subspace values.
        orthogonal_subspace[ :, -1, : ] = gradients[ :, :-1 ]/torch.unsqueeze( gradients[ :, -1 ], dim = -1 )

        # Normalize the columns of the orthogonal subspace.
        orthogonal_subspace = orthogonal_subspace/torch.repeat_interleave( torch.linalg.norm( orthogonal_subspace, ord = 2, dim = 1, keepdim = True ), num_dimensions, dim = 1 )

        # Return the orthogonal subspace.
        return orthogonal_subspace


    # Implement a function to compute the orthogonal subspace associated with a jacobian.
    def jacobian2orthogonal_subspace( self, jacobian ):

        # Compute the orthogonal subspaces associated with each of the gradients contained in the jacobian.
        orthogonal_subspaces = self.gradients2orthogonal_subspaces( jacobian )

        # Compute the intersection of the orthogonal subspaces.
        orthogonal_subspaces_intersection = self.compute_subspace_intersections( orthogonal_subspaces )
        
        # Return the orthogonal subspaces intersection.
        return orthogonal_subspaces_intersection


    # Implement a function to compute the orthogonal subspaces associated with a tensor of jacobians.
    def jacobians2orthogonal_subspaces( self, jacobians ):

        # Compute the number of dimensions.
        num_dimensions = jacobians.shape[ 1 ]

        # Compute the number of jacobians.
        num_jacobians = jacobians.shape[ 0 ]

        # Initialize a tensor to store the orthogonal subspaces.
        orthogonal_subspaces = torch.empty( ( num_jacobians, num_dimensions, num_dimensions - 1 ), dtype = torch.float32, device = jacobians.device )

        # Compute the orthogonal subspaces associated with each jacobian.
        for k in range( num_jacobians ):                # Iterate through each of the jacobians...

            # Compute the orthogonal subspace associated with this jacobian.
            orthogonal_subspaces[ k, :, : ] = torch.unsqueeze( self.jacobian2orthogonal_subspace( jacobians[ k, :, : ] ), dim = 0 )

        # Return the orthogonal subspaces.
        return orthogonal_subspaces


    # Implement a function to convert spherical coordinates to cartesian coordinates.
    def spherical_coordinates2cartesian_coordinates( self, spherical_coordinates ):

        # Retrieve the number of points.
        num_points = spherical_coordinates.shape[ 0 ]

        # Initialize the cartesian coordinates to be the radius.
        cartesian_coordinates = torch.unsqueeze( spherical_coordinates[ :, 0 ], dim = 1 )*torch.ones_like( spherical_coordinates )

        # Multiply by the cosine of the angular dimensions.
        cartesian_coordinates *= torch.cat( ( torch.cos( spherical_coordinates[ :, 1: ] ), torch.ones( ( num_points, 1 ), dtype = torch.float32, device = spherical_coordinates.device ) ), dim = 1 )
        
        # Multiply by the cumulative product of the sine of the angular dimensions.
        cartesian_coordinates *= torch.cat( ( torch.ones( ( num_points, 1 ), dtype = torch.float32, device = spherical_coordinates.device ), torch.cumprod( torch.sin( spherical_coordinates[ :, 1: ] ), dim = 1 ) ), dim = 1 )

        # Return the cartesian coordinates.
        return cartesian_coordinates


    # Implement a function to generate a random point in spherical coordinates.
    def generate_spherical_sample_point( self, radius, num_dimensions ):

        # Preallocate a tensor to store the spherical coordinates of a random sample point.
        spherical_coordinate = torch.empty( num_dimensions, dtype = torch.float32, device = radius.device )

        # Initialize the first component of the coordinate to be the given radius.
        spherical_coordinate[ 0 ] = radius*torch.rand( 1, dtype = torch.float32, device = radius.device )

        # Compute the spherical coordinate components associated with each angular coordinate.
        for k in range( num_dimensions - 1 ):                   # Iterate through each angular coordinate...

            spherical_coordinate[ k + 1 ] = math.pi*torch.rand( 1, dtype = torch.float32, device = radius.device )

        # Determine whether there is a final angular coordinate to double.
        if num_dimensions > 1:              # If there is more than one dimension...

            # Double the final angular coordinate.
            spherical_coordinate[ -1 ] *= 2

        elif num_dimensions == 1:           # If there is only one dimension...

            # Randomly generate whether to switch the sign associated with this point.
            sign_flag = torch.randint( 2, ( 1, ), device = radius.device )

            # Determine whether to switch the sign on this point.
            if sign_flag:               # If we want to switch signs on this point...

                # Switch the sign associated with this point.
                spherical_coordinate[ 0 ] *= -1

        # Return the spherical coordinate.
        return spherical_coordinate


    # Implement a function to generate a given number of random points in spherical coordinates.
    def generate_spherical_sample_points( self, radius, num_dimensions, num_points, match_sign = False ):

        # Initialize the spherical coordinates.
        spherical_coordinates = math.pi*torch.ones( ( num_points, num_dimensions ), dtype = torch.float32, device = radius.device )

        # Set the first dimension coordinates to be the radius.
        spherical_coordinates[ :, 0 ] = radius*torch.ones( num_points, dtype = torch.float32, device = radius.device )

        # Determine whether there is a final angular coordinate to double.
        if num_dimensions > 1:              # If there is more than one dimension...

            # Double the final angular coordinate.
            spherical_coordinates[ :, -1 ] *= 2

        elif num_dimensions == 1:           # If there is only one dimension...

            # Randomly generate whether to switch the sign associated with this point.
            signs = torch.randint( 2, ( num_points, ), device = radius.device )

            # Adjust the sign on the first dimension as necessary.
            spherical_coordinates[ :, 0 ] *= ( -1 )**signs

        # Randomly generate magnitudes in this spherical space.
        # magnitudes = torch.rand( ( num_points, num_dimensions ), dtype = torch.float32, device = radius.device )
        magnitudes = torch.normal( torch.zeros( num_points, num_dimensions ), ( 1/3 )*torch.ones( num_points, num_dimensions ) ).to( radius.device )

        # Adjust the magnitudes of the spherical coordinates.
        spherical_coordinates *= magnitudes

        # Determine whether to match the sign of the spherical coordinates with the sign of the radius.
        if match_sign:                  # If we want to match the sign of the spherical coordinates with the sign of the radius.

            # Ensure the the sign of the spherical coordinates matches that of the radius.
            spherical_coordinates = torch.sign( radius )*torch.abs( spherical_coordinates )

        # Return the spherical coordinates.
        return spherical_coordinates


    # Implement a function to convert a cartesian coordinate to a subspace coordinate.
    def cartesian_coordinate2subspace_coordinate( self, subspace, cartesian_coordinate ):

        # Determine whether to unsqueeze the column dimension of the cartesian coordinates.
        if cartesian_coordinate.dim(  ) == 1:           # If the cartesian coordinate is only one dimensional...

            # Unsqueeze the column dimension.
            cartesian_coordinate = torch.unsqueeze( cartesian_coordinate, dim = 1 )

        # Compute the subspace coordinate.
        subspace_coordinate = subspace*cartesian_coordinate

        # Remove the column dimension.
        subspace_coordinate = torch.squeeze( subspace_coordinate )

        # Return the subspace coordinate.
        return subspace_coordinate


    # Implement a function to convert cartesian coordinates to subspace coordinates
    def cartesian_coordinates2subspace_coordinates( self, subspace, cartesian_coordinates ):

        # Compute the subspace coordinates.
        # subspace_coordinates = subspace*torch.transpose( cartesian_coordinates, 0, 1 )
        subspace_coordinates = torch.bmm( subspace, torch.repeat_interleave( torch.unsqueeze( torch.transpose( cartesian_coordinates, 0, 1 ), dim = 0 ), subspace.shape[ 0 ], dim = 0 ) )

        # Return the subspace coordinates.
        return subspace_coordinates


    # Implement a function to generate a sample from a subspace.
    def generate_subspace_sample_points( self, subspace, radius, num_points, match_sign = False ):

        # Retrieve the dimension of the subspace.
        num_dimensions = subspace.shape[ -1 ]

        # Generate the desired number of spherical sample points.
        spherical_coordinates = self.generate_spherical_sample_points( radius, num_dimensions, num_points, match_sign )

        # Convert the spherical coordinates to cartesian coordinates.
        cartesian_coordinates = self.spherical_coordinates2cartesian_coordinates( spherical_coordinates )

        # Convert the cartesian coordinates to subspace coordinates.
        subspace_coordinates = self.cartesian_coordinates2subspace_coordinates( subspace, cartesian_coordinates )

        # Return the subspace coordinates.
        return subspace_coordinates


    #%% ------------------------------------------------------------ LEVEL SET FUNCTIONS ------------------------------------------------------------

    # Implement a function to compute a set of nearby level set points given an existing level set point.
    def generate_nearby_level_set_points( self, function, level_set_seed, radius, num_points ):

        # Compute the jacobian associated with this level set seed.
        jacobian = self.compute_function_jacobians( function, level_set_seed )

        # Compute the level set subspace. THIS ASSUMES THAT THE FUNCTION ONLY HAS A SINGLE SCALAR OUTPUT SUCH AS IN THE CASE OF OUR ROA STABILITY ESTIMATE.
        # level_set_subspace = self.jacobian2orthogonal_subspace( jacobian )
        level_set_subspace = self.gradients2orthogonal_subspaces( jacobian[ :, 0, : ] )

        # Generate the nearby level set points in the subspace.
        level_set_subspace_points = self.generate_subspace_sample_points( level_set_subspace, radius, num_points ) 

        # Compute the nearby level set points.
        level_set_points = torch.unsqueeze( level_set_seed, dim = -1 ) + level_set_subspace_points

        # Restructure the nearby level set points.
        level_set_points = torch.squeeze( torch.vstack( torch.tensor_split( level_set_points, level_set_points.shape[ -1 ], dim = -1 ) ), dim = -1 )

        # print( '-------------------------------------------------------------------------------------' )
        # print( f'jacobian = {jacobian/torch.linalg.norm( jacobian, ord = 2, dim = 0, keepdim = True )}' )
        # print( f'level_set_subspace = {level_set_subspace/torch.linalg.norm( level_set_subspace, ord = 2, dim = 0, keepdim = True )}' )
        # print( f'level_set_subspace_points = {level_set_subspace_points/torch.linalg.norm( level_set_subspace_points, ord = 2, dim = 1, keepdim = True )}' )
        # print( f'level_set_seed = {level_set_seed}' )
        # print( f'level_set_points = {level_set_points}' )
        # print( '-------------------------------------------------------------------------------------' )

        # Return the nearby level set points.
        return level_set_points


    # Implement a function to estimate a complete level set given a grid.
    def generate_level_set( self, level_function, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance ):

        # Compute the adjusted level function.
        adjusted_level_function = lambda x: level_function( x ) - level

        # Retrieve the number of input dimensions.
        num_input_dimensions = level_set_guess.shape[ -1 ]

        # Compute the level set seed.
        level_set_seed, _, convergence_flag = self.newtons_method( adjusted_level_function, level_set_guess, newton_tolerance, newton_max_iterations )

        # Retrieve the converged level set seeds.
        level_set_seed = level_set_seed[ convergence_flag, : ]

        # Initialize a tensor to store the unexplored and explored level set points.
        unexplored_level_set_points = level_set_seed
        explored_level_set_points = torch.rand( ( 1, num_input_dimensions ), dtype = torch.float32, device = level_set_seed.device )

        # Initialize a counter to count the number of explorations.
        count = torch.tensor( 0, dtype = torch.int64, device = level_set_seed.device )

        # Generate as many unique level set points as possible.
        while unexplored_level_set_points.numel(  ) != 0:                   # While there are still unexplored level set points...

            # Generate nearby level set point estimates.
            level_set_points = self.generate_nearby_level_set_points( adjusted_level_function, unexplored_level_set_points, exploration_radius, num_exploration_points )

            # Pull the nearby level set point estimates back to the level set via Newton's method.
            level_set_points, _, convergence_flags = self.newtons_method( adjusted_level_function, level_set_points, newton_tolerance, newton_max_iterations )

            # Add these new nearby level set points to the tensor of new unexplored level set points.
            new_unexplored_level_set_points = level_set_points[ convergence_flags, : ]

            # Add this unexplored level set point to the tensor of explored level set points.
            explored_level_set_points = torch.cat( ( explored_level_set_points, unexplored_level_set_points ), dim = 0 )

            # Determine which new unexplored level set points have actually not already been explored.
            explored_flags = self.are_points_in_adaptive_batch( new_unexplored_level_set_points, explored_level_set_points, unique_tolerance )

            # Retrieve the new unexplored level set points that have actually not already been explored.
            unexplored_level_set_points = new_unexplored_level_set_points[ ~explored_flags, : ]

            # Advance the counter.
            count += 1

        # Remove the dummy new explored level set point.
        explored_level_set_points = explored_level_set_points[ 1:, : ]

        # Return the explored level set points.
        return explored_level_set_points

    # # Implement a function to generate points nearby a level set.
    # def generate_noisy_level_set( self, level_function, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, noise_magnitude, num_noisy_samples_per_level_set_point, spatial_domain ):

    #     # Compute the level set points.
    #     level_set_points = self.generate_level_set( level_function, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance )

    #     # Determine whether there is a level set to process.
    #     if level_set_points.numel(  ) != 0:                 # If the level set points exist...

    #         # Retrieve the number of level set points.
    #         num_level_set_points = level_set_points.shape[ 0 ]

    #         # Retrieve the number of dimensions.
    #         num_dims = level_set_points.shape[ 1 ]

    #         # Compute the level function jacobian at the level set points.
    #         level_set_jacobians = self.compute_function_jacobians( level_function, level_set_points )
    #         level_set_jacobian_subspaces = torch.transpose( level_set_jacobians, 1, 2 )

    #         # Generate sample points in the jacobian subspaces.
    #         jacobian_subspace_points = self.generate_subspace_sample_points( level_set_jacobian_subspaces, noise_magnitude, num_level_set_points )
    #         # jacobian_subspace_points = self.generate_subspace_sample_points( level_set_jacobian_subspaces, noise_magnitude, 1 )

    #         # Take the diagonal of the jacobian subspace points.
    #         jacobian_subspace_points = torch.transpose( torch.diagonal( jacobian_subspace_points, dim1 = 0, dim2 = 2 ), 0, 1 )

    #         # Compute the noisy level set points.
    #         # level_set_points_noisy = torch.unsqueeze( level_set_points, dim = -1 ) + jacobian_subspace_points
    #         level_set_points_noisy = level_set_points + jacobian_subspace_points

    #         # # Restructure the noisy level set points.
    #         # level_set_points_noisy = torch.squeeze( torch.vstack( torch.tensor_split( level_set_points_noisy, level_set_points_noisy.shape[ -1 ], dim = -1 ) ), dim = -1 )

    #     else:                                               # Otherwise...

    #         # Set the noisy level set points to be empty.
    #         level_set_points_noisy = level_set_points

    #     # Return the noisy level set points.
    #     return level_set_points_noisy



    # Implement a function to generate points nearby a level set.
    def generate_noisy_level_set( self, level_function, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance, noise_magnitude, num_noisy_samples_per_level_set_point, spatial_domain ):

        # Compute the level set points.
        level_set_points = self.generate_level_set( level_function, level, level_set_guess, newton_tolerance, newton_max_iterations, exploration_radius, num_exploration_points, unique_tolerance )

        # Determine whether there is a level set to process.
        if level_set_points.numel(  ) != 0:                 # If the level set points exist...

            # Retrieve the number of level set points.
            num_level_set_points = level_set_points.shape[ 0 ]

            # Retrieve the number of dimensions.
            num_dims = level_set_points.shape[ 1 ]

            # Compute the level function jacobian at the level set points.
            level_set_jacobians = self.compute_function_jacobians( level_function, level_set_points )                                                       # "Direction" of steepest ascent.
            level_set_jacobians_normalized = level_set_jacobians/torch.linalg.vector_norm( level_set_jacobians, ord = 2, dim = 2, keepdim = True )          # Normalized "direction" of steepest ascent.
            level_set_jacobian_subspaces = torch.transpose( level_set_jacobians_normalized, 1, 2 )                                                          # Post-processing so that the normalized "direction" of steepest ascent looks how the generate_subspace_sample_points() wants.

            # Ensure that the jacobian subspaces have the appropriate shape for the coming computations.
            level_set_jacobian_subspaces_augmented = torch.repeat_interleave( level_set_jacobian_subspaces, 2, dim = 2 )

            # Ensure that the spatial domain has an appropriate shape for the coming computations.
            spatial_domain_augmented = torch.repeat_interleave( torch.unsqueeze( torch.transpose( spatial_domain, dim0 = 0, dim1 = 1 ), dim = 0 ), num_level_set_points, dim = 0 )

            # Ensure that the level set point have the appropriate shape for the coming computations.
            level_set_points_augmented = torch.repeat_interleave( torch.unsqueeze( level_set_points, dim = 2 ), 2, dim = 2)
            
            # Compute the augmented distribution widths.
            distribution_widths_augmented = ( spatial_domain_augmented - level_set_points_augmented )/level_set_jacobian_subspaces_augmented

            # Preallocate a tensor to store the distribution widths.
            distribution_widths = torch.empty( ( num_level_set_points, 2 ), dtype = level_set_points.dtype, device = level_set_points.device )

            # Compute the distribution widths.
            for k in range( num_level_set_points ):             # Iterate through each of the level set points...

                # Retrieve the positive distribution widths associated with this point.
                distribution_widths[ k, 0 ] = -torch.min( torch.abs( distribution_widths_augmented[ k, ... ][ distribution_widths_augmented[ k, ... ] < 0 ] ) )
                distribution_widths[ k, 1 ] = torch.min( torch.abs( distribution_widths_augmented[ k, ... ][ distribution_widths_augmented[ k, ... ] > 0 ] ) )

            # Preallocate a tensor to store the jacobian subspace points.
            jacobian_subspace_points = torch.empty( ( num_level_set_points, num_dims, num_noisy_samples_per_level_set_point ), dtype = level_set_jacobian_subspaces.dtype, device = level_set_jacobian_subspaces.device )

            # Compute the number of lower and upper noisy samples per level set point.
            num_noisy_samples_per_level_set_point_lower = torch.ceil( num_noisy_samples_per_level_set_point/2 )
            num_noisy_samples_per_level_set_point_upper = num_noisy_samples_per_level_set_point - num_noisy_samples_per_level_set_point_lower

            num_samples = torch.tensor( [ num_noisy_samples_per_level_set_point_lower, num_noisy_samples_per_level_set_point_upper ], dtype = num_noisy_samples_per_level_set_point_lower.dtype, device = num_noisy_samples_per_level_set_point_lower.device ).int(  )

            for k1 in range( num_level_set_points ):
                for k2 in range( 2 ):

                    # Generate sample points in the jacobian subspaces.
                    jacobian_subspace_points_sample = self.generate_subspace_sample_points( torch.unsqueeze( level_set_jacobian_subspaces[ k1, ... ], dim = 0 ), distribution_widths[ k1, k2 ], num_samples[ k2 ], match_sign = True )

                    if k2 == 0:             # If this is the first set of samples...

                        # Take the diagonal of the jacobian subspace points.
                        jacobian_subspace_points[ k1, :, :num_samples[ 0 ] ] = jacobian_subspace_points_sample

                    else:                   # Otherwise... (i.e., if this is the second set of samples...)

                        # Take the diagonal of the jacobian subspace points.
                        jacobian_subspace_points[ k1, :, num_samples[ 0 ]: ] = jacobian_subspace_points_sample

            # Compute the noisy level set points.
            level_set_points_noisy = torch.unsqueeze( level_set_points, dim = -1 ) + jacobian_subspace_points

            # Restructure the noisy level set points.
            level_set_points_noisy = torch.squeeze( torch.vstack( torch.tensor_split( level_set_points_noisy, level_set_points_noisy.shape[ -1 ], dim = -1 ) ), dim = -1 )

            # Generate a tensor that stores the boundaries of the spatial domain in the same format as the noisy level set points.
            domain_boundaries = torch.repeat_interleave( torch.unsqueeze( torch.transpose( spatial_domain, dim0 = 0, dim1 = 1 ), dim = 0 ), level_set_points_noisy.shape[ 0 ], dim = 0 )

            # Ensure that any noisy level set points that are outside the domain truncated to be inside the domain.
            level_set_points_noisy[ level_set_points_noisy < domain_boundaries[ ..., 0 ] ] = domain_boundaries[ ..., 0 ][ level_set_points_noisy < domain_boundaries[ ..., 0 ] ]
            level_set_points_noisy[ level_set_points_noisy > domain_boundaries[ ..., 1 ] ] = domain_boundaries[ ..., 1 ][ level_set_points_noisy > domain_boundaries[ ..., 1 ] ]

        else:                                               # Otherwise...

            # Set the noisy level set points to be empty.
            level_set_points_noisy = level_set_points

        # DEBUGGING CODE

        # # # Retrieve the indexes associated with any of the noisy level set points that happen to be out of bounds.
        # # # indexes = torch.unique( torch.floor( torch.where( level_set_points_noisy[ :, 0 ] > spatial_domain[ 1, 0 ] )[ 0 ]/num_noisy_samples_per_level_set_point ) ).int(  )
        # # indexes_long = torch.where( level_set_points_noisy[ :, 0 ] > spatial_domain[ 1, 0 ] )[ 0 ].int(  )
        # # indexes_short = torch.unique( indexes_long % num_level_set_points ).int(  )

        # # level_set_points_oob = level_set_points[ indexes_short, : ]
        # # level_set_points_noisy_oob = level_set_points_noisy[ indexes_long, : ]
        # # level_set_jacobian_subspaces_oob = torch.squeeze( level_set_jacobian_subspaces[ indexes_short, : ], dim = -1 )
        # # distribution_widths_oob = distribution_widths[ indexes_short, : ]

        # spatial_domain_points = torch.tensor( [ [ spatial_domain[ 0, 0 ], spatial_domain[ 1, 0 ], spatial_domain[ 1, 0 ], spatial_domain[ 0, 0 ], spatial_domain[ 0, 0 ] ], [ spatial_domain[ 0, 1 ], spatial_domain[ 0, 1 ], spatial_domain[ 1, 1 ], spatial_domain[ 1, 1 ], spatial_domain[ 0, 1 ] ] ], dtype = spatial_domain.dtype, device = spatial_domain.device )

        # # indexes = torch.where( level_set_points_noisy[ :, 0 ] > spatial_domain[ 1, 0 ] )[ 0 ]/num_noisy_samples_per_level_set_point
        # import matplotlib.pyplot as plt
        # fig = plt.figure(  ); plt.xlabel( 'x0' ), plt.ylabel( 'x1' ), plt.title( 'Debugging Plot' )
        # plt.plot( spatial_domain_points[ 0, : ].detach(  ).cpu(  ).numpy(  ), spatial_domain_points[ 1, : ].detach(  ).cpu(  ).numpy(  ), '-k' )
        # plt.plot( level_set_points[ :, 0 ].detach(  ).cpu(  ).numpy(  ), level_set_points[ :, 1 ].detach(  ).cpu(  ).numpy(  ), '.r' )
        # plt.plot( level_set_points_noisy[ :, 0 ].detach(  ).cpu(  ).numpy(  ), level_set_points_noisy[ :, 1 ].detach(  ).cpu(  ).numpy(  ), '.b' )
        # plt.quiver( level_set_points[ :, 0 ].detach(  ).cpu(  ).numpy(  ), level_set_points[ :, 1 ].detach(  ).cpu(  ).numpy(  ), level_set_jacobian_subspaces[ :, 0, 0 ].detach(  ).cpu(  ).numpy(  ), level_set_jacobian_subspaces[ :, 1, 0 ].detach(  ).cpu(  ).numpy(  ), facecolor = 'm' )
        # # plt.plot( level_set_points_oob[ :, 0 ].detach(  ).cpu(  ).numpy(  ), level_set_points_oob[ :, 1 ].detach(  ).cpu(  ).numpy(  ), '.b' )
        # # plt.quiver( level_set_points_oob[ :, 0 ].detach(  ).cpu(  ).numpy(  ), level_set_points_oob[ :, 1 ].detach(  ).cpu(  ).numpy(  ), level_set_jacobian_subspaces_oob[ :, 0 ].detach(  ).cpu(  ).numpy(  ), level_set_jacobian_subspaces_oob[ :, 1 ].detach(  ).cpu(  ).numpy(  ), facecolor = 'b' )
        # # plt.plot( level_set_points_noisy_oob[ :, 0 ].detach(  ).cpu(  ).numpy(  ), level_set_points_noisy_oob[ :, 1 ].detach(  ).cpu(  ).numpy(  ), '.b' )
        # plt.savefig( r'./ann/closed_roa/save' + '/' + 'Debug.png' )

        # Return the noisy level set points.
        return level_set_points_noisy


    #%% ------------------------------------------------------------ QUERY FUNCTIONS ------------------------------------------------------------

    # Implement a function to determine whether a given grid is flat or expanded.
    def is_grid_flat( self, grid ):

        # Determine whether the given grid is flattened or expanded.
        if grid.dim(  ) > 3:                                        # If the number of grid dimensions is greater than three...

            # Set the flat flag to false.
            flat_flag = False

        elif grid.dim(  ) == 3:                                     # If the number of grid dimensions is equal to three...

            # Determine whether the grid should be classified as flat.
            if grid.shape[ 0 ] >= 10*grid.shape[ 1 ]:           # If the first dimension is much greater than the second dimension...

                # Set the flat flag to true.
                flat_flag = True

            else:                                               # Otherwise...

                # Set the flat flag to false.
                flat_flag = False
            
        elif grid.dim(  ) == 2:        # If the number of grid dimensions is greater than zero and less than equal to two...

            # Set the flat flag to true.
            flat_flag = True

        else:                                                       # Otherwise... ( i.e., the number of grid dimensions is not valid... )

            # Throw an error.
            raise ValueError( f'Invalid number of grid dimensions: {grid.dim(  )}' )

        # Return the flat flag.
        return flat_flag


    # Implement a function to compute the number of consecutively equal dimensions.
    def get_num_of_consecutive_dimensions( self, grid ):

        # Initialize the old dimension size.
        old_dim_size = grid.shape[ 0 ]

        # Define a flag variable.
        consistent_dimensions_flag = True

        # Initialize a counter variable.
        k = torch.tensor( 0, dtype = torch.uint8, device = grid.device )

        # Determine the number of consistent grid dimensions.
        while consistent_dimensions_flag and ( k < grid.dim(  ) ):              # While the grid dimensions are consistent and we have not yet examined all of the grid dimensions...

            # Set the new dim size.
            new_dim_size = grid.shape[ k ]

            # Determine whether the old and new dimension sizes match.
            if old_dim_size != new_dim_size:                    # If the old dimension size and the new dimension size do not match...
            
                # Set the consistent dimensions flag to false.
                consistent_dimensions_flag = False

            # Set the old dimension size to be the new dimension size.
            old_dim_size = new_dim_size

            # Advance the counter variable.
            k += 1

        # Compute the number of consecutive dimensions.
        num_consecutive_dims = k - 1

        # Return the number of consecutive dimensions.
        return num_consecutive_dims


    # Implement a function to determine whether a grid is expanded.
    def is_grid_expanded( self, grid ):

        # Determine whether the given grid is flattened or expanded.
        if grid.dim(  ) > 3:                                        # If the number of grid dimensions is greater than three...

            # Set the expanded flag to true.
            expanded_flag = True

        elif grid.dim(  ) == 3:                                 # If the number of grid dimensions is equal to three...

            # Determine whether the grid should be classified as flat.
            if grid.shape[ 0 ] >= 10*grid.shape[ 1 ]:           # If the first dimension is much greater than the second dimension...

                # Set the flat flag to true.
                expanded_flag = False

            else:                                               # Otherwise...

                # Set the flat flag to false.
                expanded_flag = True
            
        elif grid.dim(  ) == 2:        # If the number of grid dimensions is greater than zero and less than equal to two...

            # Set the flat flag to true.
            expanded_flag = False

        else:                                                       # Otherwise... ( i.e., the number of grid dimensions is not valid... )

            # Throw an error.
            raise ValueError( f'Invalid number of grid dimensions: {grid.dim(  )}' )

        # Return the expanded flag.
        return expanded_flag


    # Implement a function to determine the grid expanded/flat type.
    def get_grid_expanded_flat_type( self, grid ):

        # Determine the expanded/flat grid type.
        if self.is_grid_expanded( grid ):           # If the grid is expanded...

            # Set the grid type to be expanded.
            grid_type = 'expanded'

        elif self.is_grid_flat( grid ):             # If the grid is flat...

            # Set the grid type to be flat.
            grid_type = 'flat'

        else:                                       # Otherwise...

            # Throw an error.
            raise ValueError( f'Grid is neither expanded nor flat and is therefore invalid.' )

        # Return the expanded / flat grid type.
        return grid_type


    # Implement a function to determine where an expanded grid is temporal.
    def is_expanded_grid_temporal( self, grid ):

        # Determine whether the expanded grid is temporal.
        if ( grid.shape[ -1 ] == ( grid.dim(  ) - 1 ) ):             # If the last value is equal to its index...

            # Set the temporal grid flag to false.
            temporal_grid_flag = False

        elif ( grid.shape[ -2 ] == ( grid.dim(  ) - 2 ) ) or ( grid.shape[ -2 ] == 1 ):          # If the second to last value is equal to its index...

            # Set the temporal grid flag to true.
            temporal_grid_flag = True

        else:                                                       # Otherwise...

            # Set the temporal grid flag to false.
            temporal_grid_flag = False
            
        # Return the temporal grid flag.
        return temporal_grid_flag


    # Implement a function to determine whether a flattened grid is temporal.
    def is_flattened_grid_temporal( self, grid ):

        # Determine whether the flattened grid is temporal.
        if grid.dim(  ) == 3:                   # If there are three grid dimensions...

            # Set the temporal grid flag to true.
            temporal_grid_flag = True

        elif grid.dim (  ) == 2:                # If there are two grid dimensions...

            # Set the temporal grid flag to false.
            temporal_grid_flag = False

        else:                                   # Otherwise...

            # Throw an error.
            raise ValueError( 'Grid dimensions are not congruent with those expected of a flattened grid.' )

        # Return the temporal grid flag.
        return temporal_grid_flag


    # Implement a function to determine whether a grid is temporal.
    def is_grid_temporal( self, grid, grid_expanded_flat_type = None ):

        # Determine whether there is a grid to analyze.
        if grid != [  ]:                # If the grid is not empty...

            # Determine whether it is necessary to compute the grid type.
            if grid_expanded_flat_type is None:               # If the grid type was not provided...

                # Compute the grid type.
                grid_expanded_flat_type = self.get_grid_expanded_flat_type( grid )

            # Determine how to determine whether the grid is temporal.
            if grid_expanded_flat_type.lower(  ) == 'expanded':                 # If the grid expanded / flat type is expanded...

                # Detemine whether the grid is temporal.
                temporal_grid_flag = self.is_expanded_grid_temporal( grid )

            elif grid_expanded_flat_type.lower(  ) == 'flat':               # If the grid expanded / flat type is flat...

                # Detemine whether the grid is temporal.
                temporal_grid_flag = self.is_flattened_grid_temporal( grid )
                
            else:

                # Throw an error.
                raise ValueError( 'Grid is neither expanded nor flat and is therefore invalid.' )

        else:                               # Otherwise....

            # Set the temporal grid flag to false.
            temporal_grid_flag = False

        # Return the grid temporal flag.
        return temporal_grid_flag


    # Implement a function to determine the temporal type of a grid.
    def get_grid_temporal_type( self, grid, grid_expanded_flat_type = None ):

        # Determine whether it is necessary to compute the grid type.
        if grid_expanded_flat_type is None:               # If the grid type was not provided...

            # Compute the grid type.
            grid_expanded_flat_type = self.get_grid_expanded_flat_type( grid )

        # Determine the temporal grid type.
        if self.is_grid_temporal( grid, grid_expanded_flat_type ):          # If the grid is temporal...
        
            # Set the grid type to temporal.
            grid_type = 'temporal'

        else:                                                               # Otherwise...

            # Set the grid type to non-temporal.
            grid_type = 'nontemporal'

        # Return the grid type.
        return grid_type


    # Implement a function to determine the number of dimensions of a flattened or expanded grid.
    def get_number_of_dimensions( self, data ):

        # Determine how to retrieve the number of dimensions from a flattened or expanded grid.
        if torch.is_tensor( data ):                         # If the data is a tensor...

            # Set the number of dimensions to be the number of entries in the last dimension.
            num_dimensions = torch.tensor( data.shape[ -1 ], dtype = torch.uint8, device = data.device )

        elif isinstance( data, list ) or isinstance( data, tuple ):                      # If the data is a list or tuple...

            # Set the number of dimension to be the number of entries in the last dimension of each list entry.
            num_dimensions = [ torch.tensor( data[ k ].shape[ -1 ], dtype = torch.uint8, device = data[ k ].device ) for k in range( len( data ) ) ]

        else:                                               # Otherwise... (i.e., the data type is not recognized...)

            # Throw an error.
            raise ValueError( f'Invalid data object: {data}' )

        # Return the number of dimensions.
        return num_dimensions


    # Implement a function to retrieve the number of data sources.
    def get_number_of_sources( self, data, device = None ):

        # Determine whether to infer the device.
        if device is None:              # If no device was provided...

            # Infer the device from the given data.
            device = data.device

        # Determine how to retrieve the number of data sources.
        if torch.is_tensor( data ):                                                     # If the data is a tensor...

            # Set the number of data sources to one.
            num_sources = torch.tensor( 1, dtype = torch.uint8, device = device )

        elif isinstance( data, list ) or isinstance( data, tuple ):                      # If the data is a list or tuple...

            # Compute the number of data sources.
            num_sources = torch.tensor( len( data ), dtype = torch.uint8, device = device )

        else:                                                                           # Otherwise... ( i.e., the data type is not recognized... )

            # Throw an error.
            raise ValueError( f'Invalid data object: {data}' )

        # Return the number of data sources.
        return num_sources


    # Implement a function to determine whether a given point is in a batch.
    def is_point_in_batch( self, point, data, tolerance ):

        # Compute the distance from the point to each data point.
        distances = torch.linalg.norm( data - point, ord = 2, dim = 1 )

        # Determine whether the point is sufficiently close to any of the data points.
        close_enough_flag = any( distances < tolerance )

        # Return the close enough flag.
        return close_enough_flag


    # Implement a function to determine whether a given set of points are in a batch.
    def are_points_in_batch( self, points, data, tolerance ):

        # Retrieve the number of points.
        num_points = points.shape[ 0 ]

        # Preallocate an array to store the close enough flags.
        close_enough_flags = torch.empty( num_points, dtype = torch.bool, device = points.device )

        # Determine whether each of the points are in the data set.
        for k in range( num_points ):               # Iterate through each of the points...

            # Determine whether this point is in the batch.
            close_enough_flags[ k ] = self.is_point_in_batch( points[ k, : ], data, tolerance )

        # Return the close enough flags.
        return close_enough_flags


    # Implement a function to remove non-unique tensor entries up to some tolerance.
    def unique_tolerance( self, data, tolerance ):

        # Initialize a counter variable.
        k = torch.tensor( 0, dtype = torch.int64, device = data.device )

        # Remove the non-unique tensor entries.
        while k < data.numel(  ):               # While we have not yet checked every entry...

            # Retrieve this data entry.
            point = data[ k, : ]

            # Retrieve all of the data that is after this point.
            remaining_data = data[ ( k + 1 ):, : ]

            # Determine whether the current data point is unique.
            if self.is_point_in_batch( point, remaining_data, tolerance ):              # If this entry is non-unique up to some tolerance...

                # Remove this entry from the data set.
                data = torch.cat( ( data[ :k, : ], data[ ( k + 1 ):, : ] ), dim = 0 )

            else:                                                                       # Otherwise... ( i.e., if this entry is unique... )
            
                # Advance the counter.
                k += 1

        # Return the data set.
        return data


    # Implement a function to determine the points that are not included in the given data set up to some tolerance and would be unique from other points were they added to the given data.
    def are_points_in_adaptive_batch( self, points, data, tolerance ):

        # Retrieve the number of points.
        num_points = points.shape[ 0 ]

        # Preallocate an array to store the close enough flags.
        close_enough_flags = torch.empty( num_points, dtype = torch.bool, device = points.device )

        # Determine whether each of the points are in the data set.
        for k in range( num_points ):               # Iterate through each of the points...

            # Determine whether this point is in the batch.
            close_enough_flags[ k ] = self.is_point_in_batch( points[ k, : ], data, tolerance )

            # Determine whether to add the current point to the reference data set.
            if not close_enough_flags[ k ]:                 # If this point is not in the batch...

                # Add this point to the batch.
                data = torch.cat( ( data, torch.unsqueeze( points[ k, : ], dim = 0 ) ), dim = 0 )

        # Return the close enough flags.
        return close_enough_flags


    #%% ------------------------------------------------------------ SUBSTITUTION FUNCTIONS ------------------------------------------------------------

    # Implement a function to substitute a constant value into a specific subgrid index of a grid.
    def substitute_values_into_nontemporal_grid( self, grid, values, subgrid_indexes ):

        # Clone the grid.
        new_grid = torch.clone( grid )

        # Retrieve the number of values of interest.
        num_values = values.numel(  )

        # Substitute the specified value into the specified subgrid.
        for k in range( num_values ):              # Iterate through each of the values...

            # Substitute the specified value into this subgrid.
            new_grid[ ..., subgrid_indexes[ k ].item(  ) ] = values[ k ]*torch.ones_like( new_grid[ ..., subgrid_indexes[ k ].item(  ) ], dtype = torch.float32, device = grid.device )

        # Return the new grid.
        return new_grid


    # Implement a function to substitute a constant value into a specific subgrid index of a grid.
    def substitute_values_into_temporal_grid( self, grid, values, subgrid_indexes ):

        # Clone the grid.
        new_grid = torch.clone( grid )

        # Retrieve the number of values of interest.
        num_timesteps = grid.shape[ -1 ]
        num_values = values.numel(  )

        # Substitute the specified value into the specified subgrid.
        for k1 in range( num_timesteps ):           # Iterate through each timestep...
            for k2 in range( num_values ):              # Iterate through each of the values...

                # Substitute the specified value into this subgrid.
                new_grid[ ..., subgrid_indexes[ k2 ].item(  ), k1 ] = values[ k2 ]*torch.ones_like( new_grid[ ..., subgrid_indexes[ k2 ].item(  ), k1 ], dtype = torch.float32, device = grid.device )

        # Return the new grid.
        return new_grid


    # Implement a function to substitute a constant value into a specific subgrid index of a grid.
    def substitute_values_into_grid( self, grid, values, subgrid_indexes ):

        # Detetermine whether the grid is temporal.
        temporal_grid_flag = self.is_grid_temporal( grid, grid_expanded_flat_type = None )

        # Ensure that the values and subgrid indexes are compatible.
        assert self.validate_values_subgrid_indexes_compatibility( values, subgrid_indexes )

        # Ensure that the grid and subgrid indexes are compatible.
        assert self.validate_grid_subgrid_indexes_compatibility( grid, subgrid_indexes, temporal_grid_flag )

        # Determine how to substitute into the grid.
        if temporal_grid_flag:          # If this is a temporal grid...

            # Substitute values into the temporal grid.
            new_grid = self.substitute_values_into_temporal_grid( grid, values, subgrid_indexes )

        else:                           # Otherwise...

            # Substitute values into the nontemporal grid.
            new_grid = self.substitute_values_into_nontemporal_grid( grid, values, subgrid_indexes )

        # Return the new grid.
        return new_grid

