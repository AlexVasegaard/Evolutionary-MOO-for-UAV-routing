
###############################################################################
##################### CA2X2 a prioir vs a posterior analysis ! ################
###############################################################################

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import plotly.express as px
from plotly.offline import plot
import pandas as pd
import seaborn as sns
from collections import Counter


class Node:
    def __init__(self, x, y, priority=0):
        self.x = x
        self.y = y
        self.priority = priority  # New: Priority label


def get_risk(node1, node2):
    #node1 = depot
    #node2 = nodes[path[0]]
    #"""Return the average risk between two nodes."""
    x_values = np.linspace(node1.x, node2.x, 100).astype(int)
    y_values = np.linspace(node1.y, node2.y, 100).astype(int)
    risks = [risk_map[x, y] for x, y in zip(x_values, y_values)]
    return np.max(risks) #could also be np.mean()


def plot_initial_scenario(nodes, depot, risk_map, name):
    plt.figure(figsize=(10,10))
    
    # Display the risk map as a background
    #plt.imshow(risk_map, cmap='hot_r', extent=[0, 100, 0, 100], origin='lower')
    plt.imshow(risk_map, cmap='terrain', extent=[0, 100, 0, 100], origin='lower')
    plt.colorbar(label='Elevation Level')
    
    # Plot depot
    plt.scatter(depot.x, depot.y, color='blue', s=100, label='Depot', edgecolors='black')

    # Placeholder for legend
    plt.scatter([], [], color='yellow', label='Priority 1')
    plt.scatter([], [], color='red', label='Priority 0')
    
    # Plot nodes with annotations based on priority
    for idx, node in enumerate(nodes):
        color = 'yellow' if node.priority == 1 else 'red'
        plt.scatter(node.x, node.y, color=color)
        plt.text(node.x + 1, node.y + 1, str(idx), ha='right', va='bottom', fontsize=9, weight='bold')

    #plt.colorbar(label='Risk Level')
    plt.legend()
    plt.title("Initial Problem Scenario")
    if name != None:
        plt.savefig('experiment' + name + 'scenario.png')
    plt.show()
    

def distance(node1, node2):
    #node1=depot, node2=nodes[path[0]]
    return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)


def evaluate(path, nodes, deviations, depot):
    #for the sorted functionality: path=population[0][0], deviations=population[0][1]
    total_distance = distance(depot, nodes[path[0]])
    total_risk = get_risk(depot, nodes[path[0]])
    total_priority_time = 0
    time_elapsed = 0

    for i in range(len(path) - 1):
        adjusted_midpoint = get_adjusted_midpoint(nodes[path[i]], nodes[path[i+1]], deviations[i])
        
        total_distance += (distance(nodes[path[i]], adjusted_midpoint) + 
                           distance(adjusted_midpoint, nodes[path[i+1]]))
        total_risk += (get_risk(nodes[path[i]], adjusted_midpoint) +
                       get_risk(adjusted_midpoint, nodes[path[i+1]]))
        
        time_elapsed += total_distance #assume time to be equivalent with distance, i.e. constant speed
        if nodes[path[i]].priority == 1:
            total_priority_time += time_elapsed

    total_distance += distance(nodes[path[-1]], depot)
    total_risk += get_risk(nodes[path[-1]], depot)

    # Weighted sum of objectives for simplicity
    weighted_sum = total_distance + 100*total_risk + 0.1*total_priority_time
    return weighted_sum


def initialize_population(pop_size, num_nodes):
    population = []
    for _ in range(pop_size):
        path = list(range(num_nodes))
        random.shuffle(path)
        deviations = np.random.uniform(0, 1, num_nodes-1)
        population.append((path, deviations))
    return population


def select(population, nodes, num_mates, depot):
    sorted_population = sorted(population, key = lambda x : evaluate(x[0], nodes, x[1], depot))
    return sorted_population[:num_mates]


# Crossover function
def crossover(parent1, parent2):
    path1, deviations1 = parent1[:2]
    path2, deviations2 = parent2[:2]
    cut = random.randint(1, len(path1)-2)
    #OX1
    child_path = path1[:cut] + [item for item in path2 if item not in path1[:cut]]
    if random.random() < 0.5:
        child_deviations = np.mean([deviations1, deviations2], axis=0)
    elif random.random() < 0.55:
        child_deviations = deviations1
    else: #random.random() < 0.7:
        child_deviations = deviations2   
    return (child_path, child_deviations)


# Mutation function
def mutate(individual):
    #individual = child
    
    mutation_type = np.random.randint(0,3)
    path, deviations = individual
    
    if mutation_type == 0: #mutate both
        i, j = random.sample(range(len(path)), 2)
        path[i], path[j] = path[j], path[i]
        index = random.randint(0, len(deviations)-1)
        deviations[index] = np.random.uniform(0, 1)
    
    elif mutation_type == 1:
        i, j = random.sample(range(len(path)), 2)
        path[i], path[j] = path[j], path[i]
        index = random.randint(0, len(deviations)-1)
        deviations[index] = min(max(deviations[index] + np.random.normal(0, 0.25),0),1)
    
    elif mutation_type == 2:
        index = random.randint(0, len(deviations)-1)
        deviations[index] = min(max(deviations[index] + np.random.normal(0, 0.25),0),1)
    
    return (path, deviations)


def get_adjusted_midpoint(node1, node2, deviation):
    #midpoint = Node((node1.x + node2.x) / 2, (node1.y + node2.y) / 2)
    #deviation_x = (node2.x - node1.x) * deviation * 0.5
    #deviation_y = (node2.y - node1.y) * deviation * 0.5

    #return Node(midpoint.x + deviation_x, midpoint.y + deviation_y)
    #xm = min(max(0, node1.x + deviation*(node2.x-node1.x)),100)
    #ym = min(max(0, node2.y - (deviation)*(node2.y-node1.y)), 100)
    
    #amp = get_adjusted_midpoint(Node(0,0), Node(0,1), 0)
    #print(amp.x, amp.y)
    
    mid = np.array([(node1.x + node2.x)/2, (node1.y + node2.y)/2])
    turn_v = np.array([mid[0]-node1.x , mid[1]-node1.y])
    
    #new extrema points. note 90 degree turn [x,y] -> [y,-x] while 270 degree turn --> [-y,x]
    e1 = np.array([mid[0] + turn_v[1], mid[1]-turn_v[0]])
    e2 = np.array([mid[0] - turn_v[1], mid[1]+turn_v[0]])
    
    xmp = min(max(0, e1[0] + deviation*(e2[0]-e1[0])),99)
    ymp = min(max(0, e1[1] + deviation*(e2[1]-e1[1])), 99)
    return Node(xmp, ymp)


def genetic_algorithm(nodes, depot, initial_pop, pop_size=100, num_mates=20, num_generations=100, mutation_rate=0.4):
    if initial_pop == None:
        population = initialize_population(pop_size, len(nodes))
    else:
        population = list(initial_pop)
        
    best_path = None
    best_distance = float('inf')
    best_deviations = None
    generations_obj_evolution = list()
    
    generations_obj_evolution2 = np.zeros((num_generations, 3)) #3 for the three objectives
    
    
    for generation in range(num_generations):
        mates = select(population, nodes, num_mates, depot)
        new_population = list(mates)

        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(mates, 2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)

        #combine parent and child population
        population.extend(new_population)

        population = select(population, nodes, pop_size, depot)
        
        best_candidate = population[0] #select(population, nodes, 1, depot)[0]
        best_candidate_distance = evaluate(best_candidate[0], nodes, best_candidate[1], depot)
        if best_candidate_distance < best_distance:
            best_distance = best_candidate_distance
            best_path = best_candidate[0]
            best_deviations = best_candidate[1]
        
        generations_obj_evolution.append(best_candidate_distance)
        multi_obj_vals = np.array(evaluate2(best_path, nodes, best_deviations, depot))
        generations_obj_evolution2[generation, :] = multi_obj_vals
        print(f"Generation {generation+1}, Best obj val: {best_distance}, multi-obj vals: {multi_obj_vals}")
        

        #todo:
            ##compute all three obj value for entire population of solutions in generation
            

    return best_path, best_deviations, generations_obj_evolution, generations_obj_evolution2


#### begin NSGA-II specific code

def evaluate2(path, nodes, deviations, depot):
    # Compute your objectives here
    #for the sorted functionality: path=population[0][0], deviations=population[0][1]
    total_distance = distance(depot, nodes[path[0]])
    total_risk = get_risk(depot, nodes[path[0]])
    total_priority_dist = 0
    # time_elapsed = 0

    for i in range(len(path) - 1):
        adjusted_midpoint = get_adjusted_midpoint(nodes[path[i]], nodes[path[i+1]], deviations[i])
        
        total_distance += (distance(nodes[path[i]], adjusted_midpoint) + 
                           distance(adjusted_midpoint, nodes[path[i+1]]))
        total_risk += (get_risk(nodes[path[i]], adjusted_midpoint) +
                       get_risk(adjusted_midpoint, nodes[path[i+1]]))
        
        #assume time to be equivalent with distance, i.e. constant speed
        if nodes[path[i]].priority == 1:
            total_priority_dist += total_distance



    total_distance += distance(nodes[path[-1]], depot)
    total_risk += get_risk(nodes[path[-1]], depot)

    # Weighted sum of objectives for simplicity
    #weighted_sum = total_distance + 100*total_risk + 0.1*total_priority_time
    return [total_distance, total_risk, total_priority_dist] #/np.sum([nodes[i].priority for i in range(len(nodes))])]


def select2(population, nodes, num_mates, depot):
    sorted_population = sorted(population, key=lambda x: evaluate2(x[0], nodes, x[1], depot))
    return sorted_population[:num_mates]


def dominated(p, q, nodes, depot, population):
    """
    Check if ind1 is dominated by ind2.
    """
    #ind1, ind2 = population[p], population[q]
    if len(population[p]) == 2:
        population[p] = tuple(list(population[p]) + list(evaluate2(population[p][0], nodes, population[p][1], depot)))
    
    if len(population[q]) == 2:
        population[q] = tuple(list(population[q]) + list(evaluate2(population[q][0], nodes, population[q][1], depot)))
    
    Fs1 = list(population[p][2:])
    Fs2 = list(population[q][2:])
    
    less_than = np.any([o1 < o2 for o1, o2 in zip(Fs1, Fs2)])
    less_than_equal = np.all([o1 <= o2 for o1, o2 in zip(Fs1, Fs2)])
    return less_than and less_than_equal

def fast_non_dominated_sort(population, nodes, depot):
    """
    Perform non-dominated sort on the population.
    """
    fronts = [[]]
    S = [[] for _ in range(len(population))]
    n = [0 for _ in range(len(population))]
    rank = [[] for _ in range(len(population))]
    
    for p in range(len(population)):
        S[p] = []
        n[p] = 0
        for q in range(len(population)):
            if dominated(p, q, nodes, depot, population):
                S[p].append(q)
            elif dominated(q, p, nodes, depot, population):
                n[p] += 1
        if n[p] == 0:
            rank[p].append(1) 
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1 #not 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    del fronts[-1]
    return fronts,rank

def crowding_distance_assignment(front, objectives):
    """
    Assign crowding distance for individuals in the same front.
    """
    distance = [0 for _ in range(len(front))]
    for obj_index in range(len(objectives[0])):
        front_enumerated = sorted(enumerate(front), key=lambda i: objectives[i[0]][obj_index])
        #front = [a[1] for a in front2]
        distance[0], distance[-1] = float('inf'), float('inf')
        f_min = objectives[front_enumerated[0][0]][obj_index]
        f_max = objectives[front_enumerated[-1][0]][obj_index]
        for i in range(1, len(front_enumerated) - 1):
            if f_max - f_min == 0:
                distance[i] = float('inf')
            else:
                distance[i] += (objectives[front_enumerated[i + 1][0]][obj_index] - objectives[front_enumerated[i - 1][0]][obj_index]) / (f_max - f_min)
    return distance


def NSGAII(nodes, depot, initial_pop, pop_size=100, num_mates=20, num_generations=100, mutation_rate=0.4):
    if initial_pop == None:
        population = initialize_population(pop_size, len(nodes))
    else:
        population = list(initial_pop)
        
    # best_path = None
    # best_deviations = None
    # generations_obj_evolution = list()
    generations_poulation_performance = np.zeros((num_generations, pop_size, 3)) #3 for the three objectives
    
    for generation in range(num_generations):
        mates = select(population, nodes, num_mates, depot)  #selecting based on weighting!!!
        new_population = list()

        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(mates, 2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)

        #combine parent and child population
        population.extend(new_population)

        # Non-dominated sorting
        fronts,rank = fast_non_dominated_sort(population, nodes, depot)

        # Assign crowding distance
        new_population = []
        for f in range(len(fronts)):
            objectives = [list(population[ind][2:]) for ind in fronts[f]]
            distances = crowding_distance_assignment(fronts[f], objectives)
            
            
            
            # Sort based on rank and distance
            front_info = list()
            for i in range(len(fronts[f])):
                front_info.append([fronts[f][i], rank[fronts[f][i]], -distances[i]])
                
            front_sorted = sorted(front_info, key=lambda i: (i[1], i[2]))
            front = [i[0] for i in front_sorted]

            # front = sorted(fronts[f], key=lambda i: (rank[i], -distances[i]))

            new_population.extend([population[i] for i in front[:pop_size - len(new_population)]])
            if len(new_population) == pop_size:
                break
            if len(new_population) >= pop_size:
                new_population = new_population[:pop_size]
                
        population = new_population
        
        # best_candidate = select(population, nodes, 1, depot)[0]
        # best_candidate_distance = evaluate(best_candidate[0], nodes, best_candidate[1], depot)
        # if best_candidate_distance < best_distance:
        #     best_distance = best_candidate_distance
        #     best_path = best_candidate[0]
        #     best_deviations = best_candidate[1]
        
        #avg_performance
        perf_gen = np.array([population[i][2:] for i in range(len(population))])
        avg_perf = list(np.mean(perf_gen, axis=0))
        generations_poulation_performance[generation, :,:] = perf_gen 
        
        #generations_obj_evolution.append(best_candidate_distance)
        print(f"Generation {generation+1}, avg performance: {avg_perf}")
    
    best_paths = list()
    best_deviations = list()
    obj_vals = list()
    for i in range(len(population)):
        best_paths.append(population[i][0])
        best_deviations.append(population[i][1])
        obj_vals.append(population[i][2:])

    return best_paths, best_deviations, obj_vals, generations_poulation_performance
    

#########


def generate_scenario(num_targets = 20, num_centers = 100, risk_spread = [1,10], depot_pos = [50,50], name = None):
    # Generate a random risk map
    dim_x, dim_y = 100, 100
    
    # Number of risk centers elevation
    #num_centers = 100
    
    # Generate random risk center locations
    centers_x = np.random.randint(0, dim_x, num_centers)
    centers_y = np.random.randint(0, dim_y, num_centers)
    
    # Risk spread for each center
    risk_spreads = np.random.uniform(risk_spread[0], risk_spread[1], num_centers)
    
    # Initialize risk map
    risk_map = np.zeros((dim_x, dim_y))
    
    # Populate the risk map
    for x in range(dim_x):
        for y in range(dim_y):
            for cx, cy, spread in zip(centers_x, centers_y, risk_spreads):
                # Gaussian spread of risk
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                risk = np.exp(-dist**2 / (2 * spread**2))
                risk_map[x, y] += risk
    
    # Normalize the risk to be between 0 and 1
    risk_map /= risk_map.max()
    
    # Nodes and Depot initialization
    depot = Node(depot_pos[0], depot_pos[1])
    nodes = [Node(random.uniform(0, 100), random.uniform(0, 100), random.choice([0, 1])) for _ in range(num_targets)]
    
    #[nodes[i].priority for i in range(0,len(nodes))]
    
    # Execute the function
    plot_initial_scenario(nodes, depot, risk_map, name)
    
    return(nodes, depot, risk_map)


#nodes = [Node(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]
#depot = Node(50, 50)  # Assuming the depot is at the center, but you can adjust this.


# print("Best Path:", best_path)
# print("Best Deviations:", best_deviations)






















def plot_route(plotting_GA, best_path, best_deviation, obj_val, best_paths, 
               best_deviations, obj_vals, nodes, risk_map, depot,
               sol_number = 0):
    #run first time
    #p=-1
    # p=9
    
    #plotting_GA = False
    
    
    
    #run iteratively
    #p +=1
    
    p = sol_number
    if plotting_GA == True:
        best_path_p = best_path
        best_deviation_p = best_deviation
        #obj_val_p = obj_val
    else:
        best_path_p = best_paths[p]
        best_deviation_p = best_deviations[p]
        obj_val_p = obj_vals[p]
    
    # Plotting solution 
    plt.figure(figsize=(10,10))
    
    # Display the risk map as a background
    plt.imshow(risk_map, cmap='terrain', extent=[0, 100, 0, 100], origin='lower')
    
    # Node paths
    path_x = [depot.x, nodes[best_path_p[0]].x]  # Start with depot and first node
    path_y = [depot.y, nodes[best_path_p[0]].y]
    
    # Placeholder for legend
    plt.scatter([], [], color='yellow', label='Priority 1')
    plt.scatter([], [], color='red', label='Priority 0')
    plt.scatter([], [], color='green', label='Midpoint')
    
    
    for i in range(len(best_path) - 1):
        adjusted_midpoint = get_adjusted_midpoint(nodes[best_path_p[i]], nodes[best_path_p[i+1]], best_deviation_p[i])
        path_x.extend([adjusted_midpoint.x, nodes[best_path_p[i+1]].x])
        path_y.extend([adjusted_midpoint.y, nodes[best_path_p[i+1]].y])
    
    path_x.append(depot.x)  # returning to depot
    path_y.append(depot.y)  # returning to depot
    
    plt.scatter(depot.x, depot.y, color='blue', s=100, label='Depot')
    plt.plot(path_x, path_y, color='cyan', linewidth=1)
    
    # Annotate the nodes based on priority
    for idx, node in enumerate(nodes):
        color = 'yellow' if node.priority == 1 else 'red'
        plt.scatter(node.x, node.y, color=color)
        plt.text(node.x, node.y, str(idx), ha='right', va='bottom')
    
    # Annotate midpoints
    for i in range(len(best_path) - 1):
        adjusted_midpoint = get_adjusted_midpoint(nodes[best_path_p[i]], nodes[best_path_p[i+1]], best_deviation_p[i])
        plt.scatter(adjusted_midpoint.x, adjusted_midpoint.y, color='green', s=50)  # plotting the midpoint
        plt.text(adjusted_midpoint.x, adjusted_midpoint.y, 'M'+str(i), ha='right', va='bottom')  # M for midpoint
    
    if plotting_GA == True:
        obj_val3 = evaluate2(best_path_p, nodes, best_deviation_p, depot)
        plt.title('(dist, elevation, priority) ' +  str(np.round(np.array(obj_val3),3)))
    else:
        plt.title('Sol#:' + str(p) + ': (dist, elevation, priority) ' +  str(np.round(np.array(obj_val_p),3)))
    
    plt.colorbar(label='Elevation Level')
    plt.legend()
    plt.show()


#nodes[best_path]






# PLotting evolution of objective value in solution search 
def plot_objective_evolution(obj_vals):
    """
    Plots the evolution of the objective value.

    Args:
    - objective_values (list): List of objective values over the generations.
    """
    
    if type(obj_vals[0]) == tuple:
        objective_values = np.array(obj_vals) @ np.array([1, 100, 0.1]) #np.mean(obj_vals, axis=0)
    else:
        objective_values = np.copy(obj_vals)
    
    plt.figure(figsize=(10,6))
    plt.plot(objective_values, marker='o', linestyle='-')
    plt.title("Evolution of Objective Value")
    plt.xlabel("Generation")
    plt.ylabel("Objective Value")
    plt.grid(True)
    plt.show()


#rather than making a single step genetic algorithm with both the order of node visits and the deviation, then make an iterative approach where order is first identified, then the midpoint, and then the midpoints of the midpoints

#modify the operators to better tune the deviations

#add a midpoint between depot and the first and last nodes




















### 

def is_dominated(s1, s2):
    """Check if solution s1 is dominated by solution s2."""
    return all(a >= b for a, b in zip(s1, s2)) and any(a > b for a, b in zip(s1, s2))

def identify_pareto_front(metrics):
    """Identify the indices of the Pareto front solutions."""
    pareto_indices = []
    for i, solution in enumerate(metrics):
        if not any(is_dominated(solution, other_solution) for j, other_solution in enumerate(metrics) if i != j):
            pareto_indices.append(i)
    return pareto_indices

def plot_pareto_front(metrics, name):
    pareto_indices = identify_pareto_front(metrics)
    pareto_front = np.zeros((len(metrics),1))
    pareto_front[pareto_indices,0] = 1
    pareto_metrics0 = np.concatenate((metrics, pareto_front), axis = 1)
    pareto_metrics  = np.concatenate((pareto_metrics0, np.array([range(len(metrics))]).T), axis = 1)
    
    metrics_pd = pd.DataFrame(data = pareto_metrics)
    metrics_pd.columns = ['Total distance', 'Total Elevation', 'Avg priority 1 dist', 'pareto', 'index']
    
    fig = px.scatter_3d(metrics_pd, x='Total distance', y='Total Elevation', z='Avg priority 1 dist',
              color='pareto', text= 'index')
    
    plot(fig, filename = str(name) + 'result.html')
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot all solutions with transparency
    # ax.scatter(metrics[:, 0], metrics[:, 1], metrics[:, 2], alpha=0.5, color='blue')
    
    # # Plot Pareto front solutions in red
    # ax.scatter(pareto_metrics[:, 0], pareto_metrics[:, 1], pareto_metrics[:, 2], color='red')

    # ax.set_xlabel('Total distance')
    # ax.set_ylabel('Total Risk')
    # ax.set_zlabel('Avg priority 1 dist')
    # plt.title('Pareto Front (in red)')
    # plt.show()

# Sample data
# metrics = np.array([
#     [5, 3, 2],
#     [4, 7, 1],
#     [3, 5, 6],
#     # ... add more metrics
# ])



def generate_3d_interactive(obj_vals, gen_obj_val, name):
    #metrics = np.array([population[i][2:] for i in range(len(population))])
    metrics = np.array([obj_vals[i] for i in range(len(obj_vals))])
    
    #plot_pareto_front(metrics)
    
    #np.array([range(len(metrics))]).T.shape
    
    #add GA sol
    metrics2 = np.concatenate((metrics, np.reshape(gen_obj_val[-1,:], (1,3))))
    plot_pareto_front(metrics2, name)
    










#Behaviour of domination between a priori and a posterior
def plot_domination_evolution(gen_domination_number):
    """
    Plots the evolution of the objective value.

    Args:
    - objective_values (list): List of objective values over the generations.
    """
    
    plt.figure(figsize=(10,6))
    plt.plot(np.array(gen_domination_number)/100, marker='o', linestyle='-')
    plt.title("Evolution of dominated a posterior solutions")
    plt.xlabel("Generation of comparison")
    plt.ylabel("Ratio of population dominated")
    plt.grid(True)
    plt.show()


def dominated_evolution(gen_obj_vals, gen_obj_val):
    """
    Counts the number of NSGA-II solutions in the population of each generation that is dominated by the solutions in the equivalent generation of the GA.

    """
    num_generations = 100
    pop_size = 100
    gen_domination_number = list()
    for gen in range(num_generations):
        dominated_number = 0
        for i in range(pop_size):
            if is_dominated(gen_obj_vals[gen][i], gen_obj_val[gen]):
                dominated_number += 1
        gen_domination_number.append(dominated_number)
    
    plot_domination_evolution(gen_domination_number)
    return gen_domination_number

def final_domination(gen_obj_vals, gen_obj_val):
    """
    Checks which scenario the last generation is in. Either:
        - GA solution dominates more than zero solutions in the pool of solutions produced by the NSGA-II
        - GA solution dominates no one, and is not dominated by any of the NSGA-II solutions
        - GA solution is dominated by at least one solution from the NSGA-II pool of solutions.

    Args:
    - generations of objective_values (array): List of objective values over the generations for NSGA-II.
    - generations of objective_values (array): List of objective values over the generations for GA.
    """
    nsga_dominated_sols = 0
    ga_dominated_sols = 0
    result = None
    
    for i in range(gen_obj_vals.shape[1]):
        #if nsga sols is dominated counter
        if is_dominated(gen_obj_vals[-1,i,:], gen_obj_val[-1,:]):
                nsga_dominated_sols += 1
        #if ga sols is dominated counter
        if is_dominated(gen_obj_val[-1,:], gen_obj_vals[-1,i,:]):
            ga_dominated_sols += 1
    
    if nsga_dominated_sols > 0 and ga_dominated_sols == 0:
        result = 'GA dominates NSGA-II'

    if nsga_dominated_sols == 0 and ga_dominated_sols == 0:
        result = 'GA on pareto front'

    if nsga_dominated_sols == 0 and ga_dominated_sols > 0:
        result = 'NSGA-II dominates GA'
        
    return result
    

def scenario_historgram(strings):
    """
    plots the count of appearance for scenarios the last generation is in. Either:
        - GA solution dominates more than zero solutions in the pool of solutions produced by the NSGA-II
        - GA solution dominates no one, and is not dominated by any of the NSGA-II solutions
        - GA solution is dominated by at least one solution from the NSGA-II pool of solutions.

    Args:
    - generations of objective_values (array): List of objective values over the generations for NSGA-II.
    - generations of objective_values (array): List of objective values over the generations for GA.
    """# Sample list of strings
    #strings = ["apple", "banana", "apple", "orange", "banana", "apple", "grape", "orange", "grape"]
     
    # Count occurrences of each string
    string_counts = Counter(strings)
    
    # Extract data for plotting
    labels = list(string_counts.keys())
    values = list(string_counts.values())
    
    # Create a bar chart
    plt.bar(labels, values, color='blue', alpha=0.7)
    plt.xlabel('Scenarios')
    plt.ylabel('Counts')
    plt.title('Histogram of Scenario Appearances')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()


def seedsetter(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
###################### RUN #######################

#### todo:
#make plots into functions
#create a loop for multiple scenario to domination identifications
#create a timeline boxplot of results to indicate spread of domination


seedsetter(112)






#scenario setup
nodes, depot, risk_map = generate_scenario()

#solution approach_setup
pop_size = 100
initial_pop = initialize_population(pop_size, len(nodes))
#GA
best_path, best_deviation, obj_val, gen_obj_val = genetic_algorithm(nodes, depot, initial_pop, num_generations=100)
#NSGA-II
best_paths, best_deviations, obj_vals, gen_obj_vals = NSGAII(nodes, depot, initial_pop)


#plot route
plot_route(plotting_GA = True, best_path = best_path, best_deviation=best_deviation, 
           obj_val=obj_val, best_paths=best_paths, best_deviations=best_deviations, 
           obj_vals=obj_vals, nodes=nodes, risk_map=risk_map, depot=depot, sol_number = 0)

plot_route(plotting_GA = False, best_path = best_path, best_deviation=best_deviation, 
           obj_val=obj_val, best_paths=best_paths, best_deviations=best_deviations, 
           obj_vals=obj_vals, nodes=nodes, risk_map=risk_map, depot=depot, sol_number = 0)

plot_route(plotting_GA = False, best_path = best_path, best_deviation=best_deviation, 
           obj_val=obj_val, best_paths=best_paths, best_deviations=best_deviations, 
           obj_vals=obj_vals, nodes=nodes, risk_map=risk_map, depot=depot, sol_number = 50)

#plot obj evolution overall
plot_objective_evolution(obj_val) #only GA
plot_objective_evolution(gen_obj_val) #only GA all three objectives
plot_objective_evolution(np.mean(gen_obj_vals, axis=1)) #only NSGA all three objectives


#plot aggregated pareto front interactively
generate_3d_interactive(obj_vals, gen_obj_val, name='first ')


#plot evolution of domination between a priori and a posterior
dominated_evolution(gen_obj_vals, gen_obj_val)


#which scenario are we in?
final_domination(gen_obj_vals, gen_obj_val)







############# LOOP FOR TESTING DOMINANCE ##########################
seedsetter(42)
Number_of_experiments = 100
num_generations = 100
exp_dom_results = np.zeros((Number_of_experiments, num_generations))
scenario_results = list()
for i in range(Number_of_experiments):
    #scenario setup
    nodes, depot, risk_map = generate_scenario(name=str(i))

    #solution approach_setup
    pop_size = 100
    initial_pop = initialize_population(pop_size, len(nodes))
    #NSGA-II
    best_paths, best_deviations, obj_vals, gen_obj_vals = NSGAII(nodes, depot, initial_pop, num_generations=100)
    #GA
    best_path, best_deviation, obj_val, gen_obj_val = genetic_algorithm(nodes, depot, initial_pop, num_generations=100)


    # #plot route
    # plot_route(plotting_GA = True, best_path = best_path, best_deviation=best_deviation, 
    #            obj_val=obj_val, best_paths=best_paths, best_deviations=best_deviations, 
    #            obj_vals=obj_vals, nodes=nodes, risk_map=risk_map, depot=depot, sol_number = 0)

    # plot_route(plotting_GA = False, best_path = best_path, best_deviation=best_deviation, 
    #            obj_val=obj_val, best_paths=best_paths, best_deviations=best_deviations, 
    #            obj_vals=obj_vals, nodes=nodes, risk_map=risk_map, depot=depot, sol_number = 0)

    # plot_route(plotting_GA = False, best_path = best_path, best_deviation=best_deviation, 
    #            obj_val=obj_val, best_paths=best_paths, best_deviations=best_deviations, 
    #            obj_vals=obj_vals, nodes=nodes, risk_map=risk_map, depot=depot, sol_number = 50)

    #plot obj evolution overall
    # plot_objective_evolution(obj_val) #only GA
    # plot_objective_evolution(gen_obj_val) #only GA all three objectives
    # plot_objective_evolution(np.mean(gen_obj_vals, axis=1)) #only NSGA all three objectives


    #plot aggregated pareto front interactively
    generate_3d_interactive(obj_vals, gen_obj_val, name = 'experiment' + str(i))


    #plot evolution of domination between a priori and a posterior
    dom_res_i = dominated_evolution(gen_obj_vals, gen_obj_val)
    
    exp_dom_results[i, :] = dom_res_i
    
    #scenario counter: whether GA sol is dominated, whether GA sol is another place on the pareto front, or whether it dominates a suite of NSGA-II solutions
    scenario_results.append(final_domination(gen_obj_vals, gen_obj_val))


#create boxplots of the time series of difference
box_plot_df = pd.DataFrame({str(gen) : exp_dom_results[:,gen] for gen in range(num_generations) if (gen % 5) == 0})
box_plot_df_melted = pd.melt(box_plot_df)
box_plot_df_melted.columns = ['Generations', 'Ratio in pct']
#box_plot_df_melted.head()


#create seaborn boxplots by group
sns.boxplot(x='Generations', y='Ratio in pct', data=box_plot_df_melted)

scenario_historgram(scenario_results)



































import imageio

data1 = np.copy(gen_obj_val)
data2 = np.copy(gen_obj_vals)


filenames = []

for gen in range(100):
    # Extract and plot the Pareto front for this generation
    data = np.concatenate((np.reshape(data1[gen][0], (1,3)), data2[gen]), axis = 0)
    
    pareto_indices = identify_pareto_front(data)
    pareto_front = np.zeros((len(data),1))
    pareto_front[pareto_indices,0] = 1
    pareto_metrics0 = np.concatenate((data, pareto_front), axis = 1)
    pareto_metrics  = np.concatenate((pareto_metrics0, np.array([range(len(data))]).T), axis = 1)
    
    metrics_pd = pd.DataFrame(data = pareto_metrics)
    metrics_pd.columns = ['Total distance', 'Total Elevation', 'Avg priority 1 dist', 'pareto', 'index']
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pareto_data[:, 0], pareto_data[:, 1], pareto_data[:, 2], c=pareto_data[:, 2], cmap='viridis', s=50)
    ax.set_title(f"Generation {gen}")
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.set_zlabel("Objective 3")
    ax.set_xlim([0, 1])  # Change according to your data range
    ax.set_ylim([0, 1])  # Change according to your data range
    ax.set_zlim([0, 1])  # Change according to your data range
    filename = f"pareto_gen_{gen}.png"
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()

# Create gif
with imageio.get_writer('pareto_evolution.gif', mode='I', duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove the individual images (optional)
for filename in filenames:
    os.remove(filename)


