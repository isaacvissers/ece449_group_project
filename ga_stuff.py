import json
from random import randint, random
import time
from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
import numpy as np
import pygad
from my_controler_ga import MyControllerGA
from test_controller import TestController
from scott_dick_controller import ScottDickController
# from my_controller_ga import MyControllerGA
from graphics_both import GraphicsBoth


def create_chromosome():
    pos_x = randint(250, 450)
    pos_y = randint(200, 350)
    heading = randint(45, 90)
    speed = sorted([randint(0, 220) for _ in range(5)])
    bullet_time = sorted([random() * 0.1, random() * 0.1])
    thrust = sorted([randint(0, 260) for _ in range(6)])

    return [pos_x, pos_y, heading] + speed + bullet_time + thrust

def fitness(ga, solution, index):
    chromosome = solution
    my_test_scenario = Scenario(name='Test Scenario',
                                num_asteroids=5,
                                ship_states=[
                                    {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1},
                                    # {'position': (600, 400), 'angle': 90, 'lives': 3, 'team': 2},
                                ],
                                map_size=(1000, 800),
                                time_limit=60,
                                ammo_limit_multiplier=0,
                                stop_if_no_ammo=False)
    game_settings = {'perf_tracker': True,
                     'graphics_type': GraphicsType.Tkinter,
                     'realtime_multiplier': 1,
                     'graphics_obj': None}
    # game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
    game = TrainerEnvironment(settings=game_settings) # Use this for max-speed, no-graphics simulation
    pre = time.perf_counter()
    score, perf_data = game.run(scenario=my_test_scenario, controllers=[MyControllerGA(chromosome), ])
    print('Scenario eval time: ' + str(time.perf_counter() - pre))
    print(score.stop_reason)
    print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
    print('Deaths: ' + str([team.deaths for team in score.teams]))
    print('Accuracy: ' + str([team.accuracy for team in score.teams]))
    print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
    print('Evaluated frames: ' + str([controller.eval_frames for controller in score.final_controllers]))
    asteroids = score.teams[0].asteroids_hit
    if asteroids == 200:
        asteroids += score.teams[0].accuracy
    return asteroids

def initial_population():
    """
    Creates the initial population using the custom chromosome creation method.
    """
    return np.array([create_chromosome() for _ in range(8)])

# def crossover(parents, offspring_size, ga):
#     parent1 = parents[0].tolist()
#     parent2 = parents[1].tolist()
#     crossover_point = randint(1, len(parent1) - 2)
#     child1 = parent1[:crossover_point] + parent2[crossover_point:]
#     child2 = parent2[:crossover_point] + parent1[crossover_point:]
#
#     # Ensure sorted sections stay sorted
#     child1[3:8] = sorted(child1[3:8])  # speed
#     child1[8:10] = sorted(child1[8:10])  # bullet_time
#     child1[10:] = sorted(child1[10:])  # thrust
#
#     child2[3:8] = sorted(child2[3:8])  # speed
#     child2[8:10] = sorted(child2[8:10])  # bullet_time
#     child2[10:] = sorted(child2[10:])  # thrust
#     if random() < 0.5: return np.array(child1)
#     else: return np.array(child2)

def crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)  # Initialize an empty array for offspring

    for i in range(offspring_size[0]):  # Generate each offspring
        # Select parents for crossover
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]
        parent1 = parents[parent1_idx].tolist()
        parent2 = parents[parent2_idx].tolist()

        # Perform single-point crossover
        crossover_point = randint(1, len(parent1) - 2)
        child = parent1[:crossover_point] + parent2[crossover_point:]

        # Ensure sorted sections stay sorted
        child[3:8] = sorted(child[3:8])  # speed
        child[8:10] = sorted(child[8:10])  # bullet_time
        child[10:] = sorted(child[10:])  # thrust

        # Add the child to the offspring array
        offspring[i] = np.array(child)

    return offspring

def mutate(chromosome, ga):
    offspring = np.empty(chromosome.shape)
    # chromosome = chromosome.tolist()
    chromosome = chromosome.tolist()[0]
    mutation_rate = 0.1
    # TODO make sure all genes stay in range
    for i in range(len(chromosome)):
        if random() < mutation_rate:
            if i == 0:
                chromosome[i] = randint(250, 450)
            elif i == 1:
                chromosome[i] = randint(200, 350)
            elif i == 2:
                chromosome[i] = randint(45, 90)
            elif 3 <= i < 8:  # Sorted `speed`
                chromosome[i] = randint(0, 220)
                chromosome[3:8] = sorted(chromosome[3:8])
            elif 8 <= i < 10:  # Sorted `bullet_time`
                chromosome[i] = random() * 0.1
                chromosome[8:10] = sorted(chromosome[8:10])
            elif i >= 10:  # Sorted `thrust`
                chromosome[i] = randint(0, 260)
                chromosome[10:] = sorted(chromosome[10:])

    # Re-sort sorted sections
    chromosome[3:8] = sorted(chromosome[3:8])  # speed
    chromosome[8:10] = sorted(chromosome[8:10])  # bullet_time
    chromosome[10:] = sorted(chromosome[10:])  # thrust
    offspring[0] = np.array(chromosome)
    return offspring

ga_instance = pygad.GA(
    num_generations=2,
    num_parents_mating=2,
    fitness_func=fitness,
    initial_population=initial_population(),
    num_genes=16,
    gene_space=None,  # Disabled as we handle gene creation manually
    parent_selection_type="rank",
    crossover_type=crossover,
    mutation_type=mutate,
    mutation_percent_genes=20,
)

ga_instance.run()

# Display the best solution
best_solution, best_solution_fitness, _ = ga_instance.best_solution()
print(f"Best solution: {best_solution}")
print(f"Fitness of the best solution: {best_solution_fitness}")
try:
    with open("best_results", "r") as file:
        best_prev_fitness = json.load(file)["fitness"]
# Load from file here, otherwise use default values
except:
    best_prev_fitness = 0
if best_prev_fitness < best_solution_fitness:
    data = {
        "fitness": float(best_solution_fitness),
        "solution": best_solution.tolist()
    }
    with open("best_results", "w") as file:
        json.dump(data, file)
# Plot the fitness progress
ga_instance.plot_fitness()