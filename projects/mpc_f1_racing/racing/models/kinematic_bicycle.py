# Simple Implementation of the Bicycle Model
import numpy as np
import casadi as ca
import yaml
import os


###############################################
# constants definition for clear indexing
###############################################

# Name of the vehicles
BUGATTI     = "bugatti"
FERRARI     = "ferrari"
LAMBORGHINI = "lamborghini"
TESLA       = "tesla"

# Inertial states indices
HEADING_INDEX  = 0  # heading of the vehicle in the inertial frame
VX_INDEX       = 1  # velocity of the vehicle in the body frame
VY_INDEX       = 2  # velocity of the vehicle in the body frame
OMEGA_INDEX    = 3  # yaw rate of the vehicle in the inertial frame
STEERING_INDEX = 4  # steering angle of the vehicle in the body frame

# curvilinear states indices
N_INDEX        = 5  # normal distance from the centerline
XI_INDEX       = 6  # lateral displacement from the centerline

# CONTROL INPUTS INDICES
THROTTLE_INDEX = 0
STEERING_RATE_INDEX  = 1




class KinematicBicycle: 
    """
    Kinematic bicyle model in curvilinear coordinates.
    """

    def __init__(self, car_model : str, track_width: float , name= "KinematicBicycle" ) -> None:
        """
        Initialize the kinematic bicycle model.
        :param car_model: car model to use (bugatti, ferrari, lamborghini, tesla)
        :type car_model: str
        :param name: name of the model
        :type name: str
        :param track_width: width of the track (used for constraining the lateral position of the vehicle)
        :type track_width: float
        """

        ##############  general variables definition  ###################
        self.name       : str        = name         # name of the vehicle model
        self.car_model  : str        = car_model    # name of the car model
        self.parameters : Parameters = Parameters() # parameters of the vehicle model
        self.gravity    : float      = 9.81         # gravitational acceleration in m/s^2
        self.track_width: float      = track_width  # width of the track (used for constraining the lateral position of the vehicle)
        
        # load parameters of the vehicle model
        current_path    : str        = os.path.dirname(os.path.abspath(__file__))
        self.parameters.from_file(f"{current_path}/../car_models/{self.car_model}.yaml")

        f_max : float = self.parameters.max_drive_force
        f_min : float = -self.parameters.max_break_force

        self.zero_force_throttle         : float = -2 * f_min / (f_max - f_min) - 1  # if the throttle is bounded between -1 and 1 then this is the throttle value that gives zero driving force
        self.maximum_normal_acceleration : float = self.parameters.mu*self.gravity   # maximum centripertal acceleration

        self.n_states   : int = 7  # number of states in the model
        self.n_inputs   : int = 2  # number of control inputs in the model
        

        #########################################################################################
        # Variables definition
        #########################################################################################

        ##############  states  ###################
        # scaled inertial states
        self.heading_  = ca.SX.sym("heading") # heading in the inertial frame (a.k.a yaw angle)
        self.vx_       = ca.SX.sym("vx")      # longitudial velocity of the vehicle in the body frame of the vehicle.
        self.vy_       = ca.SX.sym("vy")      # lateral velocity of the vehicle in the body frame of the vehicle.
        self.omega_    = ca.SX.sym("omega")   # heading rate (yaw rate) of the vehicle in the inertial frame
        self.delta_    = ca.SX.sym("delta")   # steering angle of the vehicle in the body frame of the vehicle [-max_delta, max_delta]
        self.n_        = ca.SX.sym("n")       # lateral distance from centerline
        self.xi_       = ca.SX.sym("xi")      # angle between centerline and the vehicle axis
        
        ########## control inputs #################
        self.throttle_    = ca.SX.sym("Fx")         # acceleration throttle [-1,1] with -1 being maximum break and 1 being maximum acceleration  -> conversion -f_min + (f_max - f_min) * (throttle+1)/2 
        self.delta_dot_   = ca.SX.sym("delta_dot")  # steering angle derivative [-max_delta, max_delta]

        ########## stacked states/ inputs and curvature parameter #################
        self.x_     = ca.vertcat(self.heading_, self.vx_, self.vy_, self.omega_, self.delta_, self.n_, self.xi_)  # states
        self.u_     = ca.vertcat(self.throttle_, self.delta_dot_)                                                 # control inputs
        self.kappa_ = ca.SX.sym("kappa")                                                                          # curvature of the path

        #######################################
        f_max               = self.parameters.max_drive_force
        f_min               = -self.parameters.max_break_force
        self.driving_force_ = f_min + (f_max - f_min) * (self.throttle_ + 1) / 2  # convert throttle to driving force

        # scaling for each coordinate (velocity scaling is the most important one)
        self.heading_scaling = 2*np.pi
        self.vx_scaling      = self.parameters.max_velocity
        self.vy_scaling      = self.parameters.max_velocity
        self.omega_scaling   = self.parameters.max_velocity * np.tan(self.parameters.max_steering_angle) / self.parameters.L  # omega = v / R = v* tan(delta) / L
        self.delta_scaling   = 1.0
        self.n_scaling       = track_width/2.0
        self.xi_scaling      = 2*np.pi

        self.delta_dot_scaling = self.parameters.max_steering_rate


        self.x_scaling  = np.array([self.heading_scaling,self.vx_scaling,self.vy_scaling,self.omega_scaling,self.delta_scaling,self.n_scaling, self.xi_scaling])
        self.u_scaling  = np.array([1., self.delta_dot_scaling])


        ##########################################################################################
        # Dynamics of the vehicle model
        ##########################################################################################
        # TODO:fill here
        # Equations according to (4)
        heading_dot  = self.omega_
        vx_dot       = (1/self.parameters.mass) * (self.driving_force_ - 0.5 * (self.vx_**2 + self.vy_**2) * self.parameters.Cd * self.parameters.Aref * self.parameters.air_density) # assume closing parantheis at the end of the line
        vy_dot       = (self.delta_dot_  * self.vx_ + vx_dot * self.delta_) * self.parameters.l_r / (self.parameters.l_f + self.parameters.l_r)
        omega_dot    = (self.delta_dot_  * self.vx_ + vx_dot * self.delta_) * 1 / (self.parameters.l_f + self.parameters.l_r)
        delta_dot    = self.delta_dot_  # I'm unsure about this one, but cant find any better option for u_delta

        # Equations according to (3)
        s_dot        = (self.vx_ * ca.cos(self.xi_) - self.vy_ * ca.sin(self.xi_))/(1 - self.n_ * self.kappa_)
        n_dot        = self.vx_ * ca.sin(self.xi_) + self.vy_ * ca.cos(self.xi_)
        xi_dot       = self.omega_ - s_dot * self.kappa_
        

        # create dynamics function
        self.dyn_dt          = ca.vertcat(heading_dot, vx_dot, vy_dot, omega_dot, delta_dot, n_dot, xi_dot) # the order of the states is important!
        self.dyn_ds          = ca.vertcat(heading_dot, vx_dot, vy_dot, omega_dot, delta_dot, n_dot, xi_dot)*1./s_dot # the order of the states is important!
        self.dyn_fun_dt      = ca.Function("dynamics_dt", [self.x_, self.u_, self.kappa_], [self.dyn_dt])      
        self.dyn_fun_ds      = ca.Function("dynamics_ds", [self.x_, self.u_, self.kappa_], [self.dyn_ds])     

        # create constraint functions
        velocity_squared = self.vx_**2 + self.vy_**2
        a_c              = (velocity_squared * ca.tan(self.delta_)/self.parameters.L)  # centripetal acceleration

        self.centripetal_acceleration_  = ca.Function("curvature_constraint"    ,[self.x_,self.kappa_], [a_c])
        self.velocity_squared_          = ca.Function("max_velocity_constraint" ,[self.x_,self.kappa_], [velocity_squared])



    def dynamics_ds(self, x, u, kappa) -> ca.SX: 
        return self.dyn_fun_ds(x, u, kappa)  # Get the dynamics function for the model
    
    
    def dynamics_dt(self, x, u, kappa) -> ca.SX:
        return self.dyn_fun_dt(x, u, kappa)
    
    # Functions applied to define constraints on the system for optimization
    def centripetal_acceleration(self, x, kappa) -> ca.SX:
        """
        Get the centripetal acceleration of the model.
        :param x: current state
        :param kappa: curvature of the path
        :return: centripetal acceleration
        """
        return self.centripetal_acceleration_(x, kappa)
    
    def velocity_squared(self, x, kappa) -> ca.SX:
        """
        Get the squared velocity of the model.
        :param x: current state
        :param kappa: curvature of the path
        :return: squared velocity
        """
        return self.velocity_squared_(x, kappa)
    
    def steering_angle(self, x, kappa) -> ca.SX:
        """
        Get the steering angle of the model.
        :param x: current state
        :param kappa: curvature of the path
        :return: steering angle
        """
        return x[STEERING_INDEX]
    
    def throttle(self, u, kappa) -> ca.SX:
        """
        Get the throttle of the model.
        :param x: current state
        :param kappa: curvature of the path
        :return: throttle
        """
        return u[THROTTLE_INDEX]
    
    def steering_rate(self, u, kappa) -> ca.SX:
        """
        Get the steering rate of the model.
        :param u: control inputs
        :param kappa: curvature of the path
        :return: steering rate
        """
        return u[STEERING_RATE_INDEX]

    
    def get_state_constraints_function(self):
        """
        Get the constraints of the model expressed as a vector g(x, u, kappa) <= 0.
        :param x: current state
        :param u: control inputs
        :param kappa: curvature of the path
        :return: constraints
        """

        maximum_normal_acceleration = self.parameters.mu*self.gravity
        velocity_squared            = self.vx_**2 + self.vy_**2
        s_dot                       = (self.vx_*ca.cos(self.xi_) - self.vy_*ca.sin(self.xi_))/(1 - self.kappa_*self.n_)  # velocity along the centerline path
        max_omega                   = self.parameters.max_velocity * np.tan(self.parameters.max_steering_angle) / self.parameters.L



        # TODO: fill here the constraints
        # Constraints according to (6)
        lateral_acceleration = self.omega_ * ca.sqrt(velocity_squared + 1e-9)

        gx_1  = velocity_squared - self.parameters.max_velocity**2
        gx_2  = lateral_acceleration - maximum_normal_acceleration
        gx_3  = -lateral_acceleration - maximum_normal_acceleration
        gx_4  = self.delta_ - self.parameters.max_steering_angle
        gx_5  = -self.delta_ - self.parameters.max_steering_angle
        gx_6  = self.xi_ - self.parameters.max_xi
        gx_7  = -self.xi_ - self.parameters.max_xi 
        gx_8  = self.n_ - self.track_width/2
        gx_9  = -self.n_ - self.track_width/2
        gx_10 = -s_dot + 1

        constraints = [gx_1/(self.parameters.max_velocity**2),
                       gx_2/maximum_normal_acceleration,
                       gx_3/maximum_normal_acceleration,
                       gx_4/self.parameters.max_steering_angle,
                       gx_5/self.parameters.max_steering_angle,
                       gx_6/(self.parameters.max_xi),
                       gx_7/(self.parameters.max_xi),
                       gx_8/(self.track_width),
                       gx_9/(self.track_width),
                       gx_10/(self.parameters.max_velocity)]


        return ca.Function("constraints", [self.x_, self.kappa_], [ca.vertcat(*constraints)])  # g(x,kappa) <= 0
    
    def get_input_constraints_function(self):
        """ Get the constraints of the model expressed as a vector g(x, u, kappa) <= 0.
        :param x: current state
        :param u: control inputs
        :param kappa: curvature of the path
        :return: constraints
        """
        
        constraints = [
            self.throttle_ - 1                                                                      , # throttle <= 1
            -self.throttle_ - 1                                                                     , # throttle >= -1
            (self.delta_dot_ - self.parameters.max_steering_rate) / self.parameters.max_steering_rate, # steering rate <= max_steering_rate
            (-self.delta_dot_ - self.parameters.max_steering_rate) / self.parameters.max_steering_rate, # steering rate >= -max_steering_rate
        ]

        return ca.Function("input_constraints", [self.u_, self.kappa_], [ca.vertcat(*constraints)])  # g(u,kappa) <= 0

    
    
    def omega_dot_max(self)-> None:
        """
        Get the maximum value of omega_dot.
        :return: maximum value of omega_dot
        """
        return np.tan(self.parameters.max_steering_angle) * (self.parameters.max_drive_force/self.parameters.mass) * 1 / self.parameters.L

    
    def s_dot(self, x, kappa):
        """
        Get the velocity along the centerline path.
        :param x: current state
        :param kappa: curvature of the path
        :return: velocity along the centerline path
        """
        return (x[VX_INDEX]*ca.cos(x[XI_INDEX]) - x[VY_INDEX]*ca.sin(x[XI_INDEX]))/(1 - kappa*x[N_INDEX])
    
    
    def step_ds(self, x, u, kappa, ds=0.1) -> ca.SX | np.ndarray:
        """
        Step the model forward in the ds coordinate using forward euler model.
        :param x: current state
        :param u: control inputs
        :param kappa: curvature of the path
        :param ds: curvilinear coordinate step size
        :type x: np.ndarray
        :return: next state
        """

        # rk4 integration

        k1 = self.dynamics_ds(x                , u, kappa)
        k2 = self.dynamics_ds(x + 0.5 * ds * k1, u, kappa)
        k3 = self.dynamics_ds(x + 0.5 * ds * k2, u, kappa)
        k4 = self.dynamics_ds(x + ds * k3      , u, kappa)

        return  (k1 + 2 * k2 + 2 * k3 + k4) * ds / 6.0  + x # rk4 step
    
    
    def step_dt(self, x, u, kappa, dt=0.1) -> ca.SX | np.ndarray:
        """
        Step the model forward in the ds coordinate using forward euler model.
        :param x: current state
        :param u: control inputs
        :param kappa: curvature of the path
        :param dt: time step size
        :type x: np.ndarray
        :return: next state
        """

        # rk4 integration
        k1 = self.dynamics_dt(x, u, kappa)
        k2 = self.dynamics_dt(x + 0.5 * dt * k1, u, kappa)
        k3 = self.dynamics_dt(x + 0.5 * dt * k2, u, kappa)
        k4 = self.dynamics_dt(x + dt * k3, u, kappa)

        return (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6.0 + x  # rk4 step

  

    

    def force_to_throttle(self, force: float) -> float:
        """
        Convert a force to a throttle value.
        :param force: force to convert
        :return: throttle value in [-1, 1]
        """
        f_max = self.parameters.max_drive_force
        f_min = -self.parameters.max_break_force
        
        return np.clip(2 * (force - f_min) / (f_max - f_min) - 1, -1, 1)
    


class Parameters:
    """
    Parameters of the vehicle model.
    """
    def __init__(self,) :

        self.mass                           :float = 0. # mass of the car
        self.inertia                        :float = 0. # inertia of the car (z-axis inertia)
        self.l_f                            :float = 0. # distance from the center of mass to the front axle
        self.l_r                            :float = 0. # distance from the center of mass to the rear axle
        self.L                              :float = 0. # length of the car
        self.h_com                          :float = 0. # height of the center of mass
        self.roll_moment_distribution       :float = 0. # roll moment distribution % [0,1]
        self.wheel_to_center_line_distance  :float = 0. # distance from the wheel to the center line
        self.max_drive_force                :float = 0. # maximum drive force
        self.max_break_force                :float = 0. # maximum break force
        self.max_steering_angle             :float = 0. # maximum steering angle
        self.max_velocity                   :float = 0. # maximum velocity of the car
        self.max_steering_rate              :float = 0. # maximum steering rate (steering angle derivative)
        self.I_z                            :float = 0. # yaw moment of inertia
        self.max_xi                         :float = np.pi/2 # maximum heading angle

        self.Cd                             :float = 0. # drag coefficient
        self.Aref                           :float = 0. # reference area
        self.air_density                    :float = 0. # air density
        

        # simple tyre mode : F_y = = D sin(C arctan(B alpha))
        self.pacejka_B                      :float = 0. # stiffness factor
        self.pacejka_C                      :float = 0. #
        self.pacejka_D_front                :float = 0. # N (mu * Fz_front)
        self.pacejka_D_rear                 :float = 0. # N (mu * Fz_rear)
        self.mu                             :float = 0.


    def from_file(self, file_path):
        """
        Load parameters from a YAML file.
        :param file_path: Path to the YAML file.
        """
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
            for key, value in params.items():
                setattr(self, key, value)

    def __repr__(self):
        return f"mass={self.mass},\n inertia={self.inertia},\n l_f={self.l_f},\n l_r={self.l_r},\n L={self.L},\n h_com={self.h_com},\n roll_moment_distribution={self.roll_moment_distribution},\n wheel_to_center_line_distance={self.wheel_to_center_line_distance},\n max_drive_force={self.max_drive_force},\n max_break_force={self.max_break_force},\n max_steering_angle={self.max_steering_angle}"

    def __str__(self):
        return self.__repr__()

