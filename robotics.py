
import numpy as np

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
        
        return tau
    return controller