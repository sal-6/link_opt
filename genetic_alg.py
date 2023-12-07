
## https://medium.com/@AnasBrital98/genetic-algorithm-explained-76dfbc5de85d used as a conceptual reference

from bitstring import BitArray
import numpy as np
from scipy.integrate import solve_ivp

from robotics import *
from utilis import plot_end_position, plot_end_position_animation2, plot_joint_ends
import time

MAX_GAIN = 50
POPULATION_SIZE = 24
NUM_DROPOUT = 16
NUM_GENERATIONS = 100

WEIGHT_ANG_RISE = 30
WEIGHT_ANG_OVERSHOOT = 2
WEIGHT_RAW_ERR = 25

def decode(chromosome):
    # encode a gain
    # binary represents a N-bit unsigned int
    # divide by max allowable gain for value

    # chromosme is a sequence of 4 10-bit unsigned ints
    # the order is Kp1, Kp2, Kv1, Kv2
    
    # Kp1
    Kp1 = BitArray(bin=chromosome[0:10]).uint / 1023 * MAX_GAIN
    
    # Kp2
    Kp2 = BitArray(bin=chromosome[10:20]).uint / 1023 * MAX_GAIN
    
    # Kv1
    Kv1 = BitArray(bin=chromosome[20:30]).uint / 1023 * MAX_GAIN
    
    # Kv2
    Kv2 = BitArray(bin=chromosome[30:40]).uint / 1023 * MAX_GAIN
    
    return Kp1, Kp2, Kv1, Kv2


def fitness_function(t, states, q_goal, L):
    
    ang_err = calculate_angular_error(t, states, q_goal, L)
    pos_err = calculate_raw_end_effector_error(t, states, q_goal, L)
    
    ang_overshoot = angular_rise_overshoot(t, ang_err)
    ang_rise_time = angular_rise_time(t, ang_err)
    raw_err = raw_final_error(t, pos_err)

    #print(f"Overshoot: {ang_overshoot}")
    #print(f"Rise Time: {ang_rise_time}")
    #print(f"Error: {raw_err}")
    
    # normalize metrics and compute weighted sum
    # want to maximize fitness
    
    # lower overshoot is better
    # expect worst case to be 100% 
    # + 1 to account for 0% overshoot
    term_1 = 100 / (ang_overshoot + 1)
    
    # lower rise time is better
    # expect worst case to be 10 seconds
    term_2 = 10 / (ang_rise_time + 1)
    
    # lower error is better
    # expect worst case to be 2 meters 
    # + 1 to account for 0 meters
    term_3 = 2 / (raw_err + 1)    
    
    fitness = WEIGHT_ANG_OVERSHOOT * term_1 + WEIGHT_ANG_RISE * term_2 + WEIGHT_RAW_ERR * term_3
    
    return fitness
    
    
    
    
def run_simulation(chromosome):
    # decode 
    # run dynamics for decoded gains
    # get metrics
    # perform weighted sum
    
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
    
    
    Kp1, Kp2, Kv1, Kv2 = decode(chromosome)
    Kp = np.array([[Kp1, 0], [0, Kp2]])
    Kv = np.array([[Kv1, 0], [0, Kv2]])
    
    controller = create_alpha_beta_controller(q_goal, q_dot_goal, q_dot_dot_goal, L, m, g, Kp, Kv)
    
    sol = solve_ivp(state_dynamics, t_span, init_state, method='RK45', t_eval=t_eval, atol=tol, rtol=tol, args=(controller, L, m, g))
    
    #plot_joint_ends(sol.t, sol.y, L, plot_interval=50)

    return sol
    

def genetic_alg():
    
    q_goal = np.array([[np.pi/2], [0]])
    q_dot_goal = np.array([[0], [0]])
    q_dot_dot_goal = np.array([[0], [0]])
    
    L = np.array([1.0, 1.0])
    m = np.array([1.0, 1.0])
    g = 9.81
    
    # initialize population with random chromosomes   
    population = []
    
    for i in range(POPULATION_SIZE):
        chromosome = ""
        for j in range(40):
            chromosome += str(np.random.randint(0, 2))
        
        population.append(chromosome)
        
    # run simulation for each chromosome
    # compute fitness for each chromosome
    # select top 8 chromosomes
    # perform crossover

    try:
        for gen in range(NUM_GENERATIONS):
            s = time.time()
            print(f"Generation {gen}")
            
            # run simulation for each chromosome
            fitness = []
            for chromosome in population:
                sol = run_simulation(chromosome)
                fitness.append(fitness_function(sol.t, sol.y, q_goal, L))
            
            # select top 8 chromosomes
            top_8 = np.argsort(fitness)[-8:]
            
            new_population = []
            
            # keep top 8 chromosomes
            for i in top_8:
                new_population.append(population[i])
            
            # perform crossover with top 8 chromosomes for remaining 16 chromosomes
            for i in range(POPULATION_SIZE - NUM_DROPOUT):
                # select 2 random chromosomes from top 8
                parent_1 = np.random.randint(0, 8)
                parent_2 = np.random.randint(0, 8)
                
                # select random crossover point
                crossover_point = np.random.randint(0, 40)
                
                # perform crossover
                child = new_population[parent_1][:crossover_point] + new_population[parent_2][crossover_point:]
                
                # chance of mutation
                if np.random.randint(0, 100) < 20:
                    # select random mutation point
                    mutation_point = np.random.randint(0, 40)
                    
                    # flip bit at mutation point
                    child = child[:mutation_point] + str((int(child[mutation_point]) + 1) % 2) + child[mutation_point + 1:]
                
                # add to new population
                new_population.append(child)
                
            # replace old population with new population
            population = new_population
            
            print(f"\t Current top fitness: {fitness[top_8[-1]]}")
            Kp1, Kp2, Kv1, Kv2 = decode(population[top_8[-1]])
            print(f"\t Kp1: {Kp1}")
            print(f"\t Kp2: {Kp2}")
            print(f"\t Kv1: {Kv1}")
            print(f"\t Kv2: {Kv2}")
            print(f"\t Generation took: {time.time() - s} seconds")       
            
    except KeyboardInterrupt:
        print(f"Exited at generation {gen}")
        
        # current best chromosome
        top_8 = np.argsort(fitness)[-8:]
        best_chromosome = population[top_8[-1]]
        
        # decode
        Kp1, Kp2, Kv1, Kv2 = decode(best_chromosome)
        
        print(f"Kp1: {Kp1}")
        print(f"Kp2: {Kp2}")
        print(f"Kv1: {Kv1}")
        print(f"Kv2: {Kv2}")
    
    
    
if __name__ == "__main__":
    genetic_alg()