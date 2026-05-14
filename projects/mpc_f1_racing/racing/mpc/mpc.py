import casadi as ca
import numpy as np 


from racing.racetrack import RaceTrack
from racing.models import KinematicBicycle
from racing.models import VX_INDEX, VY_INDEX, N_INDEX, XI_INDEX, OMEGA_INDEX, STEERING_INDEX, THROTTLE_INDEX,STEERING_RATE_INDEX
from scipy.interpolate import interp1d


class NMPC:
    """
    Implementation of a Model Predictive Control (MPC) for a nonlinear racing car vehicle.
    """

    def __init__(self, car_model : str, 
                       racetrack : RaceTrack,
                       Q         : np.ndarray, 
                       R         : np.ndarray, 
                       N         : int, 
                       ds        : float = 0.5) -> None:
        
        """
        Initialize the MPC controller.
        :param car_model: The car model to be used in the MPC (e.g., "bugatti", "ferrai", "lamborghini", "tesla").
        :type car_model: str
        :param Q: State cost matrix (7-dimensional vector with the diagonal elements of the matrix Q).
        :type Q: np.ndarray
        :param R: Input cost matrix (2-dimensional vector with the diagonal elements of the matrix R).
        :type R: np.ndarray
        :param N: Prediction horizon (positive integer).
        :type N: int
        :param ds: Time step for the MPC (default is 0.5).
        :type ds: float
        :param racetrack: The racetrack object containing track information.
        :type racetrack: RaceTrack
        """

        ##################################################################################3
        # Input Checks
        ###################################################################################

        available_models = ["bugatti", "ferrai", "lamborghini", "tesla"]

        if car_model not in available_models:
            raise ValueError(f"Car model {car_model} is not available. Available models are: {available_models}")
        
        if len(Q) != 7 or len(R) != 2:
            raise ValueError("Q must be a 7-dimensional vector and R must be a 2-dimensional vector. Given Q: {}, R: {}".format(Q, R))

        if N <= 0 or not isinstance(N, int):
            raise ValueError("N must be a positive integer.")
        
        ###################################################################################
        # Initialization
        ###################################################################################
    
        
        self.horizon     = N                                                                         # Prediction horizon
        self.ds          = ds                                                                        # Time step for the MPC
        self.Q           = np.diag(Q)                                                                # State cost matrix
        self.R           = np.diag(R)                                                                # Input cost matrix
        self.lambda_T    = 1e1                                                                       # Terminal cost weight
        self.Qt          = self.Q * self.lambda_T                                                    # Terminal cost matrix
        self.look_ahead  = self.horizon * self.ds                                                    # Horizon of the MPC in meters
        self.model       = KinematicBicycle(car_model=car_model,track_width= racetrack.track_width)  # The vehicle model to be used in the MPC
        self.racetrack   = racetrack                                                                 # racetrack


        s_values    = np.linspace(0, racetrack.length, int(racetrack.length/self.ds) + 1)  # Curvilinear coordinates along the track
        kappa       = np.array([racetrack.curvature(s_i) for s_i in s_values])             # Curvature of the centerline

        s_extend = np.hstack((s_values, s_values[-1] + s_values[1:]))  # extend the s values to avoid interpolation errors at the end of the trajectory
        kappa     = np.hstack((kappa, kappa[1:]))  # repeat the curvature values to extend the interpolation

                
        self.u_ref_fun = None                                           # function u_ref(s) -> gives reference input for each curvilinear coordinate
        self.x_ref_fun = None                                           # function x_ref(s) -> gives reference state for each curvilinear coordinate
        self.k_ref_fun = interp1d(s_extend, kappa,  bounds_error=True)  # function k_ref(s) -> gives reference curvature for each curvilinear coordinate

        self.x_prev     = None  # Store the previous solution for warm start if needed
        self.u_prev     = None  # Store the previous solution for warm start if needed
        self.slack_prev = None  # Store the previous slack variables for warm start if needed
        self.lam_g_prev = None  # Store the previous dual variables for warm start if needed
        
        # scaling parameters
        self.x_scaling = self.model.x_scaling
        self.u_scaling = self.model.u_scaling
        self.was_initialized = False
        

    
    
    def _setup(self):
        # Setup the MPC problem here

        #################################################################################
        ######################  Setting MPC variables/parameters ########################
        #################################################################################
        
        self.opti = ca.Opti()  # Create an optimization problem
        
        # Create variables of the optimization problem 
        x_scaled     = self.opti.variable(self.model.n_states, self.horizon+1) # State variable
        u_scaled     = self.opti.variable(self.model.n_inputs, self.horizon)   # Control input variable
        x_var        = x_scaled * self.x_scaling[:,np.newaxis]                  # Scale the state variables
        u_var        = u_scaled * self.u_scaling[:,np.newaxis]                  # Control input variable
        
        slack_var = self.opti.variable(5,self.horizon+1)  # Slack variable for the state constraints

        # parameters of the optimizaiton problem
        kappa_par = self.opti.parameter(self.horizon+1)                          # Curvature parameter along the trajectory
        x_0       = self.opti.parameter(self.model.n_states)                  # Initial state of the vehicle
        x_ref     = self.opti.parameter(self.model.n_states, self.horizon+1)  # Reference trajectory states
        u_ref     = self.opti.parameter(self.model.n_inputs, self.horizon)    # Reference trajectory inputs 

        self.s_range_n_1 = np.cumsum(np.hstack((np.array([0.]),np.ones(self.horizon))))* self.ds
        self.s_range_n   = np.cumsum(np.hstack((np.array([0.]),np.ones(self.horizon-1))))* self.ds

    
        #################################################################################
        ######################  Setting MPC constraints  ################################
        #################################################################################
        alpha_progress = 1e-2
        # Weight for maximizing progress along the track
        cost = 0.
        for ii in range(self.horizon+1):
            # dynamic constraint
            kappa_i    = kappa_par[ii]
            x_i        = x_var[:, ii]   # Current state

            vx_i    = x_i[VX_INDEX]
            vy_i    = x_i[VY_INDEX]
            n_i     = x_i[N_INDEX]
            xi_i    = x_i[XI_INDEX]
            delta_i = x_i[STEERING_INDEX]
            omega_i = x_i[OMEGA_INDEX]

            # State constraints
            maximum_normal_acceleration = self.model.parameters.mu*self.model.gravity
            velocity_squared            = vx_i**2 + vy_i**2
            s_dot                       = (vx_i*ca.cos(xi_i) - vy_i*ca.sin(xi_i))/(1 - kappa_i*n_i)  # velocity along the centerline path
            
            # Progress maximization will be added to main cost function later

            # gx_1  =   
            # gx_2  =   
            # gx_3  =  
            # gx_4  =  
            # gx_5  =   
            # gx_6  =   
            gx_8 = n_i - (self.racetrack.track_width / 2.0)
            gx_9 = -n_i - (self.racetrack.track_width / 2.0)

            constraints_x = [gx_8, gx_9]
            self.opti.subject_to(ca.vertcat(*constraints_x) <= slack_var[ii])  # Add state constraints with slack variable
            
            # Input constraints
            if ii < self.horizon:

                u_i        = u_var[:, ii]
                throttle_i = u_i[THROTTLE_INDEX]
                steering_rate_i = u_i[STEERING_RATE_INDEX]
                
                # raise NotImplementedError("TODO: implement the state constraints")
                # gu_1 = u_t - 1, gu_2 = -u_t - 1, gu_3 = delta_dot - u_delta_max, gu_4 = -delta_dot - u_delta_max
                gu_1 = throttle_i - 1
                gu_2 = -throttle_i - 1
                gu_3 = steering_rate_i - self.model.parameters.max_steering_rate
                gu_4 = -steering_rate_i - self.model.parameters.max_steering_rate

                constraints_u = [gu_1, gu_2, gu_3, gu_4]
                self.opti.subject_to(ca.vertcat(*constraints_u) <= 0.)  # Add input constraints
 
                # Dynamic constraints
                x_next     = x_var[:, ii+1]
                self.opti.subject_to(x_next == self.model.step_ds(x_i, u_i , kappa_i, ds = self.ds) )   # Dynamic constraint: x_next = f(x_i, u_i, kappa_i, ds).


        # Initial state constrarint
        self.opti.subject_to(x_var[:, 0] == x_0)  # Initial state constraint: x_0 = x_var[:, 0]
        # Slack variable constraints
        self.opti.subject_to(ca.vec(slack_var) >= 0)  # Ensure that the slack variable is non-negative

        #################################################################################
        ######################  Setting MPC cost  #######################################
        #################################################################################
        
        cost = 0.  # Initialize cost function
        input_discount_factor = 1.  # Discount factor for the cost function
        alpha_progress = 0.1  # Weight for progress maximization
        
        # State tracking cost
        for ii in range(self.horizon+1):
            e = x_scaled[:, ii] - x_ref[:, ii]/self.x_scaling[:,np.newaxis]  # Error between the current state and the reference state
            tracking_state_cost = ca.dot(e, ca.mtimes(self.Q,e))  # Error between the current state and the reference state
            cost += tracking_state_cost*self.ds  # penalize the deviation from the reference trajectory states
        
        # Input tracking cost
        for ii in range(self.horizon):
            e = u_scaled[:, ii] - u_ref[:, ii]/self.u_scaling[:,np.newaxis]  # Error between the current input and the reference input
            input_cost = ca.dot(e, ca.mtimes(self.R, e))
            cost += input_cost * self.ds 

        # Progress maximization cost
        for ii in range(self.horizon+1):
            x_i = x_var[:, ii]
            vx_i = x_i[VX_INDEX]
            vy_i = x_i[VY_INDEX] 
            xi_i = x_i[XI_INDEX]
            v_tan = vx_i*ca.cos(xi_i) - vy_i*ca.sin(xi_i)  # tangential velocity
            cost += (-alpha_progress) * v_tan * self.ds   # maximize tangential velocity
        
        # Slack variable cost (minimal weight since no constraints use them yet)
        rho_slack = 1e-6  # Very small weight          
        for ii in range(self.horizon+1):
            e = slack_var[:, ii]
            cost += rho_slack*ca.dot(e, e)*self.ds

        # Terminal cost
        e = x_scaled[:, -1] - x_ref[:, -1]/self.x_scaling[:,np.newaxis]  # Error between the current state and the reference state
        terminal_cost = ca.dot(e, ca.mtimes(self.Qt, e))
        cost += terminal_cost

        #################################################################################
        ######################  Setting MPC solver  #####################################
        #################################################################################
        
  
        p_opts = {
                "qpsol"                : "osqp",
                "max_iter"             : 45,
                "hessian_approximation": "exact",
                "convexify_strategy"   : "regularize",
                "convexify_margin"     : 1e-3,
                "init_feasible"        : False,
                "c1"                   :1e-4,
                "beta"                 :0.8,
                "tol_du"               : 1e-2,
                "tol_pr"               : 1e-4,
                'gamma_0'              :0.1,
                "print_iteration": False, # set to true for debugging
                # QP options
                "qpsol_options.error_on_fail"    : 0,
                "qpsol_options.print_time"       : 0,
                "qpsol_options.verbose"          : 0,
                "qpsol_options.warm_start_primal": True,
                "qpsol_options.warm_start_dual"  : True,
                "qpsol_options.osqp.verbose"     : False,
                "qpsol_options.osqp.eps_abs"     : 1e-4,
                "qpsol_options.osqp.eps_rel"     : 1e-4,
            }
        

        self.opti.solver('sqpmethod',p_opts)  # Set the solver to SNOPT with the specified options
        self.opti.minimize(cost)
        
        # Now, convert the Opti object to a function
        self.mpc_controller = self.opti.to_function('MPCPlanner', [x_0 , x_ref, u_ref, kappa_par, x_scaled, u_scaled, slack_var],                   [x_var, u_var, slack_var], 
                                                                  ['initial_state', 'x_ref',"u_ref", "kappa_ref", 'x_guess', 'v_guess', 'slack_guess'], ['x_opt', 'u_opt', 'slack_opt'])

    
    def are_references_set(self) -> bool:
        """
        Check if the reference trajectory is set.
        :return: True if the reference trajectory is set, False otherwise.
        """
        return self.x_ref_fun is not None and self.u_ref_fun is not None 
    
    
    def set_reference_trajectory(self, x_ref : interp1d, u_ref : interp1d) :
        
        self.x_ref_fun = x_ref
        self.u_ref_fun = u_ref
    
    def compute_input(self, s_0 : float, x_0 : np.ndarray) :

        if self.x_ref_fun is None or self.u_ref_fun is None or self.k_ref_fun is None:
            raise ValueError("Reference trajectory not set. Please set the reference trajectory using set_reference_trajectory method.")
        
        # Get the reference trajectory from the curvilinear coordinate s0
        x_ref, u_ref, k_ref = self.get_reference_trajectory(s_0)

        if self.x_prev is not None and self.u_prev is not None:
            # Use the previous solution as a warm start

            u_n     = self.u_prev[:,-1]        
            x_n_1   = self.x_prev[:,-1]                                # take last state from last solution
            x_n_2   = self.model.step_ds(x_n_1, u_n , k_ref[-2], self.ds).full()  # Step the last state forward to get the next state guess using the last input and last state from the previous solution   
            

            self.x_prev[:,1] = x_0
            x_guess     = np.hstack((self.x_prev[:,1:],x_n_2))        # Add the next state as a guess for the last time step
            u_guess     = np.hstack((self.u_prev[:,1:],u_n[:,np.newaxis]))  # Add a zero input for the last time step
            slack_guess = self.slack_prev  # Use the previous slack variable solution as a guess

        else:
            # Initialize the guess for the optimization problem
            x_guess     = x_ref
            u_guess     = u_ref
            slack_guess = np.ones((5, self.horizon+1))*0.1                  # Initialize the slack variable guess to 0.1

        x_guess_scaled = x_guess / self.x_scaling[:,np.newaxis]  # Scale the state guess
        u_guess_scaled = u_guess / self.u_scaling[:,np.newaxis]  # Scale the input guess

        self.x_opt, self.u_opt, self.slack_opt = self.mpc_controller(x_0, x_ref, u_ref, k_ref, x_guess_scaled, u_guess_scaled, slack_guess)  # Call the MPC controller function

        self.x_opt = self.x_opt.full()  # Get the optimal state trajectory
        self.u_opt = self.u_opt.full()  # Get the optimal input trajectory

        optimal_input = self.u_opt[:,0]
    
        self.x_prev     = self.x_opt  # Store the optimal state trajectory for warm start
        self.u_prev     = self.u_opt  # Store the optimal input trajectory for warm start
        self.slack_prev = self.slack_opt  # Store the optimal slack variables for warm start

        return optimal_input  # Return the first input of the optimal trajectory plus the reference input at s_0
    
    def get_optimal_trajectory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the optimal trajectory from the MPC solution.
        :return: Tuple of state and input trajectories.
        """
        if self.x_opt is None or self.u_opt is None:
            raise ValueError("MPC solution not computed. Please call compute_input method first.")

        return self.x_opt, self.u_opt, self.slack_opt

    def get_reference_trajectory(self, s_0 :float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        x_ref_from_s0 = self.x_ref_fun(s_0 + self.s_range_n_1)
        u_ref_from_s0 = self.u_ref_fun(s_0 + self.s_range_n)
        k_ref_from_s0 = self.k_ref_fun(s_0 + self.s_range_n_1)


        return x_ref_from_s0, u_ref_from_s0, k_ref_from_s0 
    

    def s_dot(self,s:float , x: np.ndarray) -> float:
        """
        Compute the curvilinear coordinate s_dot from the state vector x.
        :param x: State vector
        :return: Curvilinear coordinate s_dot
        """
        return (x[VX_INDEX] * np.cos(x[XI_INDEX]) - x[VY_INDEX] * np.sin(x[XI_INDEX])) / (1 - x[N_INDEX] * self.k_ref_fun(s))
