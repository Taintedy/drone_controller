from matplotlib.pyplot import axis
import numpy as np
import do_mpc
import rospy


class PID:

    def __init__(self, P, I, D, error_lim) -> None:
        self.P = P
        self.I = I
        self.D = D
        self.error_lim = error_lim
        self.prev_error = 0
        self.error_sum = 0

    
    def get_control(self, current_pose_axis, goal_pose_axis, dt):

        error = goal_pose_axis - current_pose_axis
        derror = (error - self.prev_error) / dt
        self.error_sum += error * dt

        if abs(self.error_sum) > self.error_lim:
            self.error_sum = np.sign(self.error_sum) * self.error_lim

        self.prev_error = error

        return self.P * error + self.I * self.error_sum + self.D * derror


def get_vel(current_pose, goal_pose, max_speed):
    return max_speed * (goal_pose - current_pose) / np.linalg.norm((goal_pose - current_pose))




class MPC:
    def __init__(self, n_horizon=20, t_step=0.1, n_robust=0, r_term=[100, 100, 100], max_vel=[2.5, 2.5, 2.5]) -> None:

        # saving hyperparameters
        self.n_horizon = n_horizon
        self.t_step = t_step
        self.n_robust = n_robust
        self.max_vel = max_vel
        self.r_term = r_term
        self.prev_pose = np.array([0, 0, 0])
        self.prev_time = rospy.get_time()
        self.current_goal = self.prev_pose

        # Creating the system model (kinematic model) 
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # Setting up state variables
        x = model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
        y = model.set_variable(var_type='_x', var_name='y', shape=(1, 1))
        z = model.set_variable(var_type='_x', var_name='z', shape=(1, 1))

        Vx = model.set_variable(var_type='_x', var_name='Vx', shape=(1, 1))
        Vy = model.set_variable(var_type='_x', var_name='Vy', shape=(1, 1))
        Vz = model.set_variable(var_type='_x', var_name='Vz', shape=(1, 1))
        
        # Setting up control variables
        Vx_set = model.set_variable(var_type='_u', var_name='Vx_set')
        Vy_set = model.set_variable(var_type='_u', var_name='Vy_set')
        Vz_set = model.set_variable(var_type='_u', var_name='Vz_set')

        # Setting up time varying variables
        x_ref = model.set_variable(var_type='_tvp', var_name='x_ref')
        y_ref = model.set_variable(var_type='_tvp', var_name='y_ref')
        z_ref = model.set_variable(var_type='_tvp', var_name='z_ref')

        # right hand side equations
        model.set_rhs('x', Vx)
        model.set_rhs('y', Vy)
        model.set_rhs('z', Vz)
        model.set_rhs('Vx', Vx_set - Vx)
        model.set_rhs('Vy',  Vy_set - Vy)
        model.set_rhs('Vz', Vz_set - Vz)

        model.setup()

        self.model = model

        # Creating the MPC
        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            'n_horizon': self.n_horizon,
            't_step': self.t_step,
            'n_robust': self.n_horizon,
            'store_full_solution': True,
        }
        mpc.set_param(**setup_mpc)

        # Cost function
        lterm = (self.model.tvp['x_ref'] - self.model.x['x']) ** 2 + (
                        self.model.tvp['y_ref'] - self.model.x['y']) ** 2 + (
                                self.model.tvp['z_ref'] - self.model.x['z']) ** 2

        mterm = lterm
        mpc.set_objective(mterm=mterm, lterm=lterm)

        # Cost of action
        mpc.set_rterm(
            Vx_set=self.r_term[0],
            Vy_set=self.r_term[1],
            Vz_set=self.r_term[2],
        )

        # Setting speed bounds
        mpc.bounds['lower', '_u', 'Vx_set'] = -self.max_vel[0]  # [m/s]
        mpc.bounds['lower', '_u', 'Vy_set'] = -self.max_vel[1]  # [m/s]
        mpc.bounds['lower', '_u', 'Vz_set'] = -self.max_vel[2]  # [m/s]

        mpc.bounds['upper', '_u', 'Vx_set'] = self.max_vel[0]  # [m/s]
        mpc.bounds['upper', '_u', 'Vy_set'] = self.max_vel[1]  # [m/s]
        mpc.bounds['upper', '_u', 'Vz_set'] = self.max_vel[2]  # [m/s]


        # Setting the function of time varying variables
        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            for k in range(setup_mpc['n_horizon'] + 1):
                tvp_template['_tvp', k, 'x_ref'] = self.current_goal[0]
                tvp_template['_tvp', k, 'y_ref'] = self.current_goal[1]
                tvp_template['_tvp', k, 'z_ref'] = self.current_goal[2]
            return tvp_template

        
        mpc.set_tvp_fun(tvp_fun)

        mpc.setup()
        x0 = np.array([0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        mpc.x0 = x0

        mpc.set_initial_guess()
        self.mpc = mpc

    
    def get_control(self, current_pose, current_goal):
        """
        Gets the control vector from MPC

        param current_pose: numpy array of the position of the robot [x, y, z]
        param current_goal: numpy array of the goal position [x, y, z]

        return control: the control vector [Vx, Vy, Vz]
        """
        state = np.zeros(6).reshape(6, 1)
        state[:3] = current_pose.reshape(3, 1)
        current_time = rospy.get_time()
        dt = current_time - self.prev_time
        state[3:] = ((current_pose - self.prev_pose) / dt).reshape(3, 1)
        self.current_goal = current_goal
        control = self.mpc.make_step(state)
        self.prev_pose = current_pose
        self.prev_time = current_time
        return control

def get_lookahead_dist(current_pose, path, max_dist, min_dist):
    """
    Calculates the threshold to switch from one goal point to the other

    param current_pose: numpy array of the position of the robot [x, y, z]
    param path: array of positions, representing the trajectory [[x1, y1, z1], [x2, y2, z2], ...]
    param max_dist: the maximum threshold value
    param min_dist: the minimum threshold value
    """
    crosstrack_error = min(np.linalg.norm(path-current_pose, axis=1))
    dist = (max_dist - min_dist) * np.exp(-0.5 * crosstrack_error) + min_dist
    return dist