####################################################################################### 
# THIS SOURCE CODE IS PROPERTY OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. 
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND 
# CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION 
# ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN 
# LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE,
# CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL 
# PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641. 
####################################################################################### 


#%% ------------------------------------------------------------ PDE CLASS ------------------------------------------------------------

# This file implements a class for storing and managing PDE information.


#%% ------------------------------------------------------------ IMPORT LIBRARIES ------------------------------------------------------------

# Import standard libraries.
import torch

# Import custom libraries.
from domain_class import domain_class as domain_class
from initial_boundary_condition_class import initial_boundary_condition_class as initial_boundary_condition_class
from tensor_utilities_class import tensor_utilities_class as tensor_utilities_class
from plotting_utilities_class import plotting_utilities_class as plotting_utilities_class


#%% ------------------------------------------------------------ PDE CLASS ------------------------------------------------------------

# Implement the pde class.
class pde_class(  ):

    #%% ------------------------------------------------------------ CONSTRUCTOR ------------------------------------------------------------

    # Implement the class constructor.
    def __init__( self, name, type, domain, initial_boundary_conditions, residual_function, residual_code, flow_functions, device = 'cpu' ):

        # Create an instance of the tensor utilities class.
        self.tensor_utilities = tensor_utilities_class(  )

        # Create an instance of the plotting utilities class.
        self.plotting_utilities = plotting_utilities_class(  )

        # Store the device.
        self.device = device

        # Store the PDE information.
        self.name = self.validate_name( name )
        self.type = self.validate_type( type )

        # Store the PDE domain.
        self.domain = self.validate_domain( domain )

        # Store the PDE initial and boundary conditions.
        self.initial_boundary_conditions = self.validate_initial_boundary_conditions( initial_boundary_conditions )
        self.num_initial_boundary_conditions = torch.tensor( len( self.initial_boundary_conditions ), dtype = torch.uint8, device = self.device )

        # Store the residual function.
        self.residual_function = self.validate_residual_function( residual_function )
        self.residual_code = residual_code

        # Store the flow functions.
        self.flow_function = lambda s: torch.cat( [ flow_function( s ) for flow_function in flow_functions ], dim = 1 )


    #%% ------------------------------------------------------------ PREPROCESS FUNCTIONS ------------------------------------------------------------

    # Implement a function to preprocess the domain.
    def preprocess_domain( self, domain = None ):

        # Determine whether to use the stored domain.
        if domain is None:              # If the domain was not provided...

            # Set the domain to be the stored value.
            domain = self.domain

        # Return the domain.
        return domain


    # Implement a function to preprocess the initial-boundary conditions.
    def preprocess_initial_boundary_conditions( self, initial_boundary_conditions = None ):

        # Determine whether to use the stored initial-boundary conditions.
        if initial_boundary_conditions is None:             # If the initial-boundary conditions were not provided...

            # Set the initial-boundary conditions to be the stored value.
            initial_boundary_conditions = self.initial_boundary_conditions

        # Return the initial-boundary conditions.
        return initial_boundary_conditions


    # Implement a function to preprocess the temporal step size.
    def preprocess_temporal_step_size( self, dt ):

        # Preprocess the temporal step size.
        if dt is None:                              # If the temporal step size was not provided...

            # Set the default temporal step size.
            dt = torch.tensor( 1e-3, dtype = torch.float32, device = self.device )

        # Return the temporal step size.
        return dt


    # Implement a function to preprocess the initial state.
    def preprocess_initial_state( self, x0 ):

        # Preprocess the initial state.
        if x0 is None:                              # If the initial state was not provided...

            # Set the default initial state.
            x0 = torch.zeros( self.domain.num_spatial_dimensions, dtype = torch.float32, device = self.device )

        # Return the initial state.
        return x0


    # Implement a function to preprocess the temporal integration span.
    def preprocess_temporal_integration_span( self, tspan ):

        # Preprocess the temporal integration span.
        if tspan is None:                           # If the temporal integration span was not provided...

            # Set the default temporal integration span.
            tspan = torch.tensor( [ 0, 1 ], dtype = torch.float32, device = self.device )

        # Return the temporal integration span.
        return tspan


    # Implement a function to preprocess the flow function.
    def preprocess_flow_function( self, flow_function ):

        # Preprocess the flow function.
        if flow_function is None:                   # If the flow function was not provided...

            # Set the default flow function.
            flow_function = self.flow_function

        # Return the flow function.
        return flow_function


    #%% ------------------------------------------------------------ SETUP FUNCTIONS ------------------------------------------------------------

    # Implement a function to setup for flow integration.
    def setup_flow_integration( self, flow_function, tspan, x0, dt ):

        # Preprocess the temporal step size.
        dt = self.preprocess_temporal_step_size( dt )

        # Preprocess the initial state.
        x0 = self.preprocess_initial_state( x0 )

        # Preprocess the temporal integration span.
        tspan = self.preprocess_temporal_integration_span( tspan )

        # Preprocess the flow function.
        flow_function = self.preprocess_flow_function( flow_function )

        # Return the flow integration setup information.
        return flow_function, tspan, x0, dt


    #%% ------------------------------------------------------------ IS VALID FUNCTIONS ------------------------------------------------------------

    # Implement a function to validate the pde name.
    def is_name_valid( self, name ):

        # Determine whether the given name is valid.
        if isinstance( name, str ):                         # If the name is valid...

            # Set the valid flag to true.
            b_valid = True

        else:                                               # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    # Implement a function to validate the pde type.
    def is_type_valid( self, type ):

        # Determine whether the type is valid.
        if isinstance( type, str ) and ( ( type.lower(  ) == '' ) or ( type.lower(  ) == 'hyperbolic' ) or ( type.lower(  ) == 'parabolic' ) or ( type.lower(  ) == 'elliptic' ) or ( type.lower(  ) == 'first order' ) ):

            # Set the valid flag to true.
            b_valid = True

        else:                                           # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    # Implement a function to validate the pde domain.
    def is_domain_valid( self, domain ):

        # Determine whether the domain is valid.
        if isinstance( domain, domain_class ):                  # If the domain is valid...

            # Set the valid flag to true.
            b_valid = True

        else:                                                   # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    # Implement a function to validate the pde initial-boundary conditions.
    def is_initial_boundary_conditions_valid( self, initial_boundary_conditions ):

        # Determine whether the initial-boundary conditions are valid.
        if isinstance( initial_boundary_conditions, list ) and initial_boundary_conditions:                                         # If the initial boundary conditions variable is a list and is not empty...

            # Initialize the valid flag to true.
            b_valid = True

            # Initialize a loop counter variable.
            k = torch.tensor( 0, dtype = torch.int64, device = self.device )

            # Ensure that each member of the list is an initial-boundary condition object.
            while b_valid and ( k < len( initial_boundary_conditions ) ):                           # While we have not found an invalid initial-boundary condition and have not yet checked each of the initial-boundary condition...

                # Determine whether this initial boundary condition is valid.
                b_valid &= isinstance( initial_boundary_conditions[ k ], initial_boundary_condition_class )

                # Advance the loop counter.
                k += 1
        
        else:                                                                                                                   # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    # Implement a function to validate the residual function.
    def is_residual_function_valid( self, residual_function ):

        # Determine whether the residual function is valid.
        if callable( residual_function ):                                   # Determine whether the residual function is valid...
            
            # Set the valid flag to true.
            b_valid = True

        else:                                                                               # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    # Implement a function to validate the spatiotemporal pde order.
    def is_spatiotemporal_order_valid( self, spatiotemporal_order ):

        # Determine whether the given spatiotemporal order is valid.
        if torch.is_tensor( spatiotemporal_order ) and ( spatiotemporal_order.numel(  ) != 0 ) and ( spatiotemporal_order.dtype == torch.bool ):                  # If the spatiotemporal order is valid...

            # Set the valid flag to true.
            b_valid = True

        else:                                                                                                                                                       # Otherwise...

            # Set the valid flag to false.
            b_valid = False

        # Return the valid flag.
        return b_valid


    #%% ------------------------------------------------------------ SET FUNCTIONS ------------------------------------------------------------

    # Implement a function to set the name.
    def set_name( self, name, set_flag = True ):

        # Determine whether to set the name.
        if set_flag:                # If we want to set the name...

            # Set the name.
            self.name = name


    # Implement a function to set the type.
    def set_type( self, type, set_flag = True ):

        # Determine whether to set the type.
        if set_flag:                # If we want to set the type...

            # Set the type.
            self.type = type


    # Implement a function to set the domain.
    def set_domain( self, domain, set_flag = True ):

        # Determine whether to set the domain.
        if set_flag:            # If we want to set the domain...

            # Set the domain.
            self.domain = domain


    # Implement a function to set the initial-boundary conditions.
    def set_initial_boundary_conditions( self, initial_boundary_conditions, set_flag = True ):

        # Determine whether to set the initial-boundary conditions.
        if set_flag:            # If we want to set the initial-boundary conditions...

            # Set the initial-boundary conditions.
            self.initial_boundary_conditions = initial_boundary_conditions


    # Implement a function to set the residual function.
    def set_residual_function( self, residual_function, set_flag = True ):

        # Determine whether to set the residual function.
        if set_flag:            # If we want to set the residual function...

            # Set the residual function.
            self.residual_function = residual_function


    #%% ------------------------------------------------------------ VALIDATE FUNCTIONS ------------------------------------------------------------

    # Implement a function to set the pde name.
    def validate_name( self, name, set_flag = False ):

        # Determine whether to set the pde name.
        if not self.is_name_valid( name ):                          # If the pde name is not valid...

            # Throw an error.
            raise ValueError( f'Invalid name: {name}' )

        # Set the name (as required).
        self.set_name( name, set_flag )

        # Return the name.
        return name


    # Implement a function to set the pde type.
    def validate_type( self, type, set_flag = False ):

        # Determine whether to set the pde type.
        if not self.is_type_valid( type ):                          # If the pde type is not valid...

            # Throw an error.
            raise ValueError( f'Invalid type: {type}' )

        # Set the type (as required).
        self.set_type( type, set_flag )

        # Return the type.
        return type


    # Implement a function to set the pde domain.
    def validate_domain( self, domain, set_flag = False ):

        # Determine whether to set the pde domain.
        if not self.is_domain_valid( domain ):                          # If the pde domain is not valid...

            # Throw an error.
            raise ValueError( f'Invalid domain: {domain}' )

        # Set the domain (as required).
        self.set_domain( domain, set_flag )

        # Return the domain.
        return domain


    # Implement a function to set the pde initial-boundary conditions.
    def validate_initial_boundary_conditions( self, initial_boundary_conditions, set_flag = False ):

        # Determine whether to set the pde initial-boundary conditions.
        if not self.is_initial_boundary_conditions_valid( initial_boundary_conditions ):                          # If the pde initial-boundary conditions is not valid...

            # Throw an error.
            raise ValueError( f'Invalid initial-boundary conditions: {initial_boundary_conditions}' )

        # Set the initial-boundary conditions.
        self.set_initial_boundary_conditions( initial_boundary_conditions, set_flag )

        # Return the initial-boundary conditions.
        return initial_boundary_conditions


    # Implement a function to set the residual function.
    def validate_residual_function( self, residual_function, set_flag = False ):

        # Determine whether to set the pde domain.
        if not self.is_residual_function_valid( residual_function ):                          # If the pde residual function is not valid...

            # Throw an error.
            raise ValueError( f'Invalid residual function: {residual_function}' )

        # Set the residual function (as required).
        self.set_residual_function( residual_function, set_flag )

        # Return the residual function.
        return residual_function


    #%% ------------------------------------------------------------ COMPUTE FUNCTIONS ------------------------------------------------------------

    # Implement a function to compute a streamline of the flow over a finite time horizon.
    def integrate_flow( self, flow_function = None, tspan = None, x0 = None, dt = None ):

        # Setup for the flow integration.
        flow_function, tspan, x0, dt = self.setup_flow_integration( flow_function, tspan, x0, dt )

        # Construct the RK4 flow function.
        flow_function_RK4 = lambda t, x: flow_function( torch.cat( ( t, x ), dim = 1 ) )

        # Perform RK4 integration.
        x, dx = self.tensor_utilities.RK4_integration( flow_function_RK4, tspan, x0[ :, 1: ], dt )

        # Augment the flow states with the times.
        x = torch.cat( ( torch.unsqueeze( x0[ :, 0 ], dim = 1 ), x ), dim = 1 )

        # Return the flow state.
        return x, dx


    #%% ------------------------------------------------------------ PRINTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to print pde information.
    def print( self, num_dashes = 20, decoration_flag = True ):

        # Print a header.
        self.printing_utilities.print_header( 'PDE SUMMARY', num_dashes, decoration_flag )

        # Print the general pde information.
        print( 'General Information' )
        print( f'Device: {self.device}' )
        print( f'Name: {self.name}' )
        print( f'Type: {self.type}' )
        print( f'Domain: {self.domain}' )
        print( '\n' )

        # Print the initial-boundary condition information.
        print( 'Initial-Boundary Condition Information' )
        print( f'# of Initial-Boundary Conditions: {self.num_initial_boundary_conditions}' )
        print( f'Initial-Boundary Conditions: {self.initial_boundary_conditions}' )
        print( '\n' )

        # Print the residual information.
        print( 'Residual Information' )
        print( f'Residual Function: {self.residual_function}' )
        print( f'Residual Code: {self.residual_code}' )

        # Print a footer.
        self.printing_utilities.print_footer( num_dashes, decoration_flag )


    #%% ------------------------------------------------------------ PLOTTING FUNCTIONS ------------------------------------------------------------

    # Implement a function to plot the PDE domain.
    def plot_domain( self, domain = None, projection_dimensions = None, projection_values = None, level = 0, fig = None, domain_type = 'spatiotemporal', save_directory = r'.', as_surface = True, as_stream = True, as_contour = True, show_plot = False ):

        # Preprocess the domain.
        domain = self.preprocess_domain( domain )

        # Plot the PDE domain.
        figs, axes = domain.plot( projection_dimensions, projection_values, level, self.domain.dimension_labels, fig, domain_type, save_directory, as_surface, as_stream, as_contour, show_plot )

        # Return the figures and axes.
        return figs, axes


    # Implement a function to plot the PDE initial-boundary conditions.
    def plot_initial_boundary_conditions( self, num_points_per_dimension, num_timesteps, domain = None, initial_boundary_conditions = None, projection_dimensions = None, projection_values = None, level = 0, fig_targets = None, save_directory = r'.', as_surface = None, as_stream = True, as_contour = True, show_plot = False ):

        # Preprocess the domain.
        domain = self.preprocess_domain( domain )

        # Preprocess the initial-boundary conditions.
        initial_boundary_conditions = self.preprocess_initial_boundary_conditions( initial_boundary_conditions )

        # Compute the number of initial-boundary conditions.
        num_initial_boundary_conditions = torch.tensor( len( initial_boundary_conditions ), dtype = torch.uint8, device = self.device )

        # Determine whether to create a list of None values for figures.
        if fig_targets is None:                 # If no figure was provided...

            # Create a list of Nones.
            fig_targets = [ None ]*num_initial_boundary_conditions

        # Generate the input labels.
        input_labels = self.plotting_utilities.dimension_labels2axis_labels( domain.dimension_labels )

        # Generate the grid for plotting the initial and boundary conditions.
        grid = self.tensor_utilities.generate_grid( domain.spatiotemporal_domain, num_points_per_dimension )
        # grid = self.tensor_utilities.generate_grid( domain.spatiotemporal_domain, num_points_per_dimension, num_timesteps )

        # Define the tile dimensions.
        tile_dims = torch.ones( grid.dim(  ), dtype = torch.uint8, device = self.device )
        tile_dims[ -1 ] = num_timesteps

        # Ensure that there is an instance of the grid for each network timestep.
        grid = torch.tile( torch.unsqueeze( grid, dim = grid.dim(  ) ), dims = tile_dims.tolist(  ) )

        # Initialize empty lists for figures and axes.
        figs = [  ]
        axes = [  ]

        # Plot each of the initial-boundary conditions.
        for k in range( num_initial_boundary_conditions ):                     # Iterate through each of the initial-boundary conditions...

            # Create the title string.
            title_string = f'IBC{k}: {initial_boundary_conditions[ k ].name}'

            # Plot this initial-boundary condition.
            fig, ax = initial_boundary_conditions[ k ].plot( grid, initial_boundary_conditions[ k ].condition_functions, projection_dimensions, projection_values, level, fig_targets[ k ], input_labels, title_string, save_directory, as_surface, as_stream, as_contour, show_plot )

            # Append the figures and axes to the existing list.
            figs.append( fig )
            axes.append( ax )

        # Return the figures and axes.
        return figs, axes

