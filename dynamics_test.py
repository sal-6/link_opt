from scipy.integrate import solve_ivp
import numpy as np

from robotics import state_dynamics, no_input_controller, constant_input_controller, create_alpha_beta_controller
from utilis import plot_end_position, plot_end_position_animation2, plot_joint_ends


def main():
    
    L = np.array([1.0, 1.0])
    m = np.array([1.0, 1.0])
    g = 9.81
    
    init_state = np.array([-np.pi/2, np.pi/4, 0.0, 0.0])
    
    t_span = (0.0, 10.0)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    
    time_interval = t_eval[1] - t_eval[0] * 1000
    
    tol = 10**-13
        
    # alpha beta controller
    q_goal = np.array([[np.pi/2], [0]])
    q_dot_goal = np.array([[0], [0]])
    q_dot_dot_goal = np.array([[0], [0]])
    
    Kp = np.array([[100, 0], [0, 2]])
    Kv = np.array([[40, 0], [0, 50]])
    
    controller = create_alpha_beta_controller(q_goal, q_dot_goal, q_dot_dot_goal, L, m, g, Kp, Kv)
    
    sol = solve_ivp(state_dynamics, t_span, init_state, method='RK45', t_eval=t_eval, atol=tol, rtol=tol, args=(controller, L, m, g))
    
    plot_joint_ends(sol.t, sol.y, L, plot_interval=50)

    
if __name__ == "__main__":
    main()