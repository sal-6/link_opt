
import numpy as np
import matplotlib.pyplot as plt

def state_dynamics(t, state, controller, L, m, g):
    print(f"t: {t}")
    
    # get current position and velocity
    _q = state[0:2] # joint angles
    _q_dot = state[2:4] # joint rates
    
    # compute basic trig values potentially used in dynamics
    s1 = np.sin(_q[0])
    s2 = np.sin(_q[1])
    c1 = np.cos(_q[0])
    c2 = np.cos(_q[1])
    s12 = np.sin(_q[0] + _q[1])
    c12 = np.cos(_q[0] + _q[1])
    
    # compute the mass matrix
    M = np.array([[0.0, 0.0], [0.0, 0.0]])
    
    M[0,0] = L[1]** 2 * m[1] + 2 * L[0] * L[1] * m[1] * c2 + L[0] ** 2 * (m[0] + m[1])
    M[0,1] = L[1] ** 2 * m[1] + L[0] * L[1] * m[1] * c2
    M[1,0] = M[0,1]
    M[1,1] = L[1] ** 2 * m[1]
    
    # compute the coriolis matrix
    V = np.array([[0.0], [0.0]])
    
    V[0, 0] = - m[1] * L[0] * L[1] * s2 * _q_dot[1] ** 2 - 2 * m[1] * L[0] * L[1] * s2 * _q_dot[0] * _q_dot[1]
    V[1, 0] = m[1] * L[0] * L[1] * s2 * _q_dot[0] ** 2
    
    # compute the gravity vector
    G = np.array([[0.0], [0.0]])
    
    G[0, 0] = m[1] * L[1] * g * c12 + (m[0] + m[1]) * L[0] * g * c1
    G[1, 0] = m[1] * L[1] * g * c12
    
    # compute the control input
    tau = controller(t, state)
    
    # compute q_ddot as M^-1 * (tau - V - G)
    _q_ddot = np.linalg.inv(M) @ (tau - V - G)
    
    state_der = np.array([_q_dot[0], _q_dot[1], _q_ddot[0][0], _q_ddot[1][0]])

    # return the derivative of the state
    return state_der


def forward_kinematics(thetas, L):
    
    # [endx, endy, joint1x, joint1y]
    pos = [0, 0, 0, 0]
    
    c1 = np.cos(thetas[0])
    s1 = np.sin(thetas[0])
    c12 = np.cos(thetas[0] + thetas[1])
    s12 = np.sin(thetas[0] + thetas[1])
    
    pos[0] = L[0] * c1 + L[1] * c12
    pos[1] = L[0] * s1 + L[1] * s12
    
    pos[2] = L[0] * c1
    pos[3] = L[0] * s1
    
    return pos


def no_input_controller(t, state):
    u = np.array([[0.0], [0.0]])
    return u

def constant_input_controller(k):
    def controller(t, state):
        u = np.array([[k], [k]])
        return u
    return controller

def create_alpha_beta_controller(q_goal, q_dot_goal, q_dot_dot_goal, L_model, m_model, g, Kp, Kv):
    def controller(t, state):
        
        # control of form: tau = alpha * f' + beta
        # alpha = M(q)
        # beta = V(q, q_dot) + G(q)
        # f' = q_dot_dot_goal + Kp * (q_goal - q) + Kv * (q_dot_goal - q_dot)
        # total tau = M(q) * (q_dot_dot_goal + Kp * (q_goal - q) + Kv * (q_dot_goal - q_dot)) + V(q, q_dot) + G(q)
        
        # get current position and velocity
        _q = state[0:2]
        _q_dot = state[2:4]
        
        # compute basic trig values potentially used in dynamics
        s1 = np.sin(_q[0])
        s2 = np.sin(_q[1])
        c1 = np.cos(_q[0])
        c2 = np.cos(_q[1])
        s12 = np.sin(_q[0] + _q[1])
        c12 = np.cos(_q[0] + _q[1])
        
        m = m_model
        L = L_model
        
        # compute the mass matrix
        M = np.array([[0.0, 0.0], [0.0, 0.0]])
        
        M[0,0] = L[1]** 2 * m[1] + 2 * L[0] * L[1] * m[1] * c2 + L[0] ** 2 * (m[0] + m[1])
        M[0,1] = L[1] ** 2 * m[1] + L[0] * L[1] * m[1] * c2
        M[1,0] = M[0,1]
        M[1,1] = L[1] ** 2 * m[1]
        
        # compute the coriolis matrix
        V = np.array([[0.0], [0.0]])
        
        V[0, 0] = - m[1] * L[0] * L[1] * s2 * _q_dot[1] ** 2 - 2 * m[1] * L[0] * L[1] * s2 * _q_dot[0] * _q_dot[1]
        V[1, 0] = m[1] * L[0] * L[1] * s2 * _q_dot[0] ** 2
        
        # compute the gravity vector
        G = np.array([[0.0], [0.0]])
        
        G[0, 0] = m[1] * L[1] * g * c12 + (m[0] + m[1]) * L[0] * g * c1
        G[1, 0] = m[1] * L[1] * g * c12
        
        # as column vectors for math shenanigans
        _q = np.array([[state[0]], [state[1]]])
        _q_dot = np.array([[state[2]], [state[3]]])
        
        # compute f'
        f_prime = q_dot_dot_goal + Kp @ (q_goal - _q) + Kv @ (q_dot_goal - _q_dot)
        
        # compute tau
        tau = M @ f_prime + V + G
        
        print(f"tau: {tau}")
        
        return tau
    return controller


def calculate_angular_overshoot_percentage(times, joint_angles, link_lengths, forward_kinematics, goal_joint_angles):
    achieved_end_points = []
    goal_end_points = []

    # Calculate achieved end points for each set of joint angles
    for i in range(len(times)):
        end_point = forward_kinematics(joint_angles[i], link_lengths)[:2]
        achieved_end_points.append(end_point)
    
    # Calculate goal end points based on the goal joint angles
    for goal_angles in goal_joint_angles:
        goal_end_point = forward_kinematics(goal_angles, link_lengths)[:2]
        goal_end_points.append(goal_end_point)

    # Calculate the difference between achieved and goal end points
    total_angular_difference = 0
    for i in range(len(achieved_end_points)):
        diff = np.linalg.norm(achieved_end_points[i] - goal_end_points[i])
        total_angular_difference += diff

    # Calculate the total distance between achieved and goal end points
    total_goal_distance = 0
    for i in range(len(goal_end_points) - 1):
        distance = np.linalg.norm(goal_end_points[i + 1] - goal_end_points[i])
        total_goal_distance += distance

    # Calculate the angular overshoot percentage
    angular_overshoot_percentage = (total_angular_difference / total_goal_distance) * 100

    return angular_overshoot_percentage


def calculate_angular_error(t, states, q_goal, L):

    goal_point = forward_kinematics([q_goal[0][0], q_goal[1][0]], L)
    goal_point = np.array([[goal_point[0]], [goal_point[1]]])

    goal_angle = np.arctan2(goal_point[1], goal_point[0])

    error_over_time = []

    for i in range(len(t)):
        curr_state = states[:, i][:2]
        curr_fwd = forward_kinematics(curr_state, L)

        curr_fwd = np.array([[curr_fwd[0]], [curr_fwd[1]]])
        curr_angle = np.arctan2(curr_fwd[1], curr_fwd[0])

        curr_error = goal_angle - curr_angle

        error_over_time.append(curr_error)

    
    plt.figure()
    plt.plot(t, error_over_time)
    plt.show()

    return error_over_time
    

def calculate_raw_end_effector_error(t, states, q_goal, L):

    goal_point = forward_kinematics([q_goal[0][0], q_goal[1][0]], L)
    goal_point = np.array([[goal_point[0]], [goal_point[1]]])

    error_over_time = []

    for i in range(len(t)):
        curr_state = states[:, i][:2]
        curr_fwd = forward_kinematics(curr_state, L)

        curr_fwd = np.array([[curr_fwd[0]], [curr_fwd[1]]])
        curr_error = goal_point - curr_fwd

        error_mag = np.sqrt(curr_error[0][0] ** 2 + curr_error[1][0] ** 2)

        error_over_time.append(error_mag)

    #plt.figure()
    #plt.plot(t, error_over_time)
    #plt.show()

    return error_over_time


def angular_rise_overshoot(t, ang_err_arr):

    init_value = ang_err_arr[0]

    # check if it overshot
    overshot = False
    for i in range(len(t) - 1):
        if ang_err_arr[i] * ang_err_arr[i+1] <= 0:
            overshot = True

    if not overshot:
        return 0
    
    overshot_mag = abs(min(ang_err_arr))

    percent_overshoot = overshot_mag[0] / init_value[0] * 100

    return percent_overshoot


def angular_rise_time(t, ang_err_arr):
    # calculate how long it takes to get within 5 percent of the target angular point
    
    init_value = ang_err_arr[0]
    
    threshold = init_value * 0.05

    for i in range(len(t)):
        if ang_err_arr[i] < threshold:
            return t[i]

    # never makes it to threshold
    return 50


def raw_final_error(t, end_eff_err_arr):

    return end_eff_err_arr[-1]



