
from scipy.integrate import solve_ivp
import numpy as np

from utilis import plot_joint_ends


state_history = []

def state_dynamics(t, state, L, m, g, q_goal, q_dot_goal, q_dot_dot_goal, Kp, Kd, Ki):
    print(f"t: {t}")
    
    global state_history
    
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
    
    # pid controller
    error = q_goal - _q
    error_dot = _q_dot
    
    # compute the control input
    tau = Kp @ error + Kd @ error_dot
    
    # compute q_ddot as M^-1 * (tau - V - G)
    _q_ddot = np.linalg.inv(M) @ (tau - V - G)
    
    state_der = np.array([_q_dot[0], _q_dot[1], _q_ddot[0][0], _q_ddot[1][0]])

    # return the derivative of the state
    return state_der


def main():
    
    L = np.array([1.0, 1.0])
    m = np.array([1.0, 1.0])
    g = 9.81
    
    init_state = np.array([0.0, 0.0, 0.0, 0.0])
    
    t_span = (0.0, 2.0)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    
    time_interval = t_eval[1] - t_eval[0] * 1000
    
    tol = 10**-13
        
    Kp = np.array([[15, 0], [0, 15]])
    Kd = np.array([[1, 0], [0, 1]])
    Ki = np.array([[0, 0], [0, 0]])
    
    q_goal = np.array([[np.pi/2], [0]])
    q_dot_goal = np.array([[0], [0]])
    q_dot_dot_goal = np.array([[0], [0]])
        
    sol = solve_ivp(state_dynamics, t_span, init_state, method='RK45', t_eval=t_eval, atol=tol, rtol=tol, args=(L, m, g, q_goal, q_dot_goal, q_dot_dot_goal, Kp, Kd, Ki))
    
    plot_joint_ends(sol.t, sol.y, L, plot_interval=50)

    
if __name__ == "__main__":
    main()