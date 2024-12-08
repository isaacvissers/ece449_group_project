from random import randint, random
import time
from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
import numpy as np
import pygad
from my_controler_ga import MyControllerGA
from my_new_controller import MyNewController
from test_controller import TestController
from scott_dick_controller import ScottDickController
# from my_controller_ga import MyControllerGA
from graphics_both import GraphicsBoth


def create_chromosome():
    pos_x = randint(250, 450)
    pos_y = randint(200, 350)
    heading = randint(45, 90)
    speed = [randint(0, 220) for _ in range(5)]
    bullet_time = [random() * 0.1, random() * 0.1]
    thrust = [randint(0, 260) for _ in range(6)]

    return [pos_x, pos_y, heading, speed, bullet_time, thrust]

def fitness(chromosome):
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
    return [team.asteroids_hit for team in score.teams][0]

def initial_population():
    """
    Creates the initial population using the custom chromosome creation method.
    """
    return np.array([create_chromosome() for _ in range(10)])

ga_instance = pygad.GA(
    num_generations=5,
    num_parents_mating=2,
    fitness_func=fitness,
    initial_population=initial_population(),
    num_genes=16,
    gene_space=None,  # Disabled as we handle gene creation manually
    parent_selection_type="rank",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=20
)

ga_instance.run()

# Display the best solution
best_solution, best_solution_fitness, _ = ga_instance.best_solution()
print(f"Best solution: {best_solution}")
print(f"Fitness of the best solution: {best_solution_fitness}")

# Plot the fitness progress
ga_instance.plot_fitness()