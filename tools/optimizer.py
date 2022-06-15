import os
import random
import subprocess
import glob
import multiprocessing

import numpy

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

#from scoop import futures

NUMBER_OF_PARAMETERS = 7
EXPECTED_X_FOOT_COORDINATE = 2.5
EXPECTED_Y_FOOT_COORDINATE = 0.0 

CONFIG_FILE = """[simulation]
formulation=homography
N = 15
m = 2
T = 0.1
double_support_lenght = 2
single_support_lenght = 7
iterations = 250

[qp]
alpha = {}
betah11 = {}
betah13 = {}
betah33 = {}
gamma = {}
eta = {}
kappa = {}

# camera only (x, y)

[reference]
orientation = 0.0:250
total_points = 1
camera_position0_x = 2.5
camera_position0_y = 0.0
orientation0 = 0.0
p00 = 12.07,2.12726896,1.714
p01 = 12.07,2.12726896,1.414
p02 = 12.07,-2.12726896,1.714
p03 = 12.07,-2.12726896,1.414

[initial_values]
orientation = 0.0
foot_x_position = 0.0
foot_y_position = 0.0

[camera]
fx = 391.5937
fy = 391.5937
u0 = 0
v0 = 0
"""

class SimulationError(Exception):
    pass

def create_config_file(values):
    """ """
    alpha, betah11, betah13, betah33, gamma, eta, kappa = values
    with open('config.ini', 'w') as file_handle:
        file_handle.write(CONFIG_FILE.format(alpha, betah11, betah13, betah33, gamma, eta, kappa))
    return 'config.ini'


def run_simulation(config_file):
    pid = subprocess.Popen(['./main', config_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, __ = pid.communicate()
    
    if pid.returncode:
        raise SimulationError()

    # search for line 'results will be saved to: <output file>.txt'
    for line in stdout.splitlines():
        if 'results will be saved to:' in line:
            __, log_file = line.split(':')
            return log_file.strip()

    raise SimulationError()

    

def parse_results(log_file):
    """ """
    is_foot_position = lambda l: '[foot_position]' in l
    is_objective_function = lambda l: 'objective_function' in l

    def _parse_foot_position(entry):
        """expected line: ... [foot_position]: iteration=250, x=2.5, y=0.110142, orientation=0"""
        __, raw_data = entry.split(']: ')
        __, x_coordinate, y_coordinate, __ = raw_data.split(', ')
        __, x_coordinate = x_coordinate.strip().split('=')
        __, y_coordinate = y_coordinate.strip().split('=')
        return float(x_coordinate), float(y_coordinate)

    x_foot_positions = []
    y_foot_positions = []
    objective_function = None
    with open(log_file, 'r') as file_handle:
        for line in file_handle.readlines():
            if is_foot_position(line):
                x_coordinate, y_coordinate = _parse_foot_position(line)
                x_foot_positions.append(x_coordinate)
                y_foot_positions.append(y_coordinate)
            elif is_objective_function(line):
                __, objective_function = line.split(': ')
                objective_function = float(objective_function.strip())

    return x_foot_positions, y_foot_positions, objective_function
    

def cleanup():
    os.remove('config.ini')
    for file_to_remove in glob.glob('*log*.txt'):
        os.remove(file_to_remove)


def evaluate(individual):
    """should return tuple, single objective fitness is a special case of
    multi-objective"""
    config_file = create_config_file(individual)

    try:
        log_file = run_simulation(config_file)
    except SimulationError:
        return (1 << 64),    # return big number
    
    x_foot_positions, y_foot_positions, objective_function = parse_results(log_file)
    cleanup()

    x_foot_mean = numpy.mean(x_foot_positions[-15:])
    x_foot_std = numpy.std(x_foot_positions[-15:])

    y_foot_mean = numpy.mean(y_foot_positions[-15:])
    y_foot_std = numpy.std(y_foot_positions[-15:])

    is_within_x_margin = lambda x: (x < 1.02 * EXPECTED_X_FOOT_COORDINATE) and (x > 0.98 * EXPECTED_X_FOOT_COORDINATE)
    is_within_y_margin = lambda y: (y < 1.02 * EXPECTED_Y_FOOT_COORDINATE) and (y > 0.98 * EXPECTED_Y_FOOT_COORDINATE)

    if is_within_x_margin(x_foot_mean) and is_within_y_margin(y_foot_mean):
        fitness = (x_foot_std / x_foot_mean + y_foot_std / y_foot_std) * objective_function
    elif is_within_x_margin(x_foot_mean):
        fitness = 100 * abs(EXPECTED_Y_FOOT_COORDINATE - y_foot_mean) * objective_function
    elif is_within_y_margin(y_foot_mean):
        fitness = 100 * abs(EXPECTED_X_FOOT_COORDINATE - x_foot_mean) * objective_function
    else:
        fitness = 1 << 32
    
    return (fitness,)


# classes
creator.create('FitnessMinimize', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMinimize)


# single individuals
toolbox = base.Toolbox()
toolbox.register('attr_float', random.uniform, 1e-7, 1e-2)
toolbox.register('individual', tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=NUMBER_OF_PARAMETERS)

# population
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# (must have) operators
toolbox.register("mate", tools.cxTwoPoints)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# parallelize ?
#pool = multiprocessing.Pool()
#toolbox.register("map", pool.map)
#toolbox.register("map", futures.map)


def main():
    """ """
    population = toolbox.population(n=50)
    hall_of_fame = tools.HallOfFame(10)

    # cxpb - The probability of mating two individuals.
    # mutpb - The probability of mutating an individual.
    # ngen - The number of generation.
    final_population, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2,
                                                ngen=350, halloffame=hall_of_fame, verbose=True)

    best = tools.selBest(final_population, 1)
    best = best[0]
    create_config_file(best)


if __name__ == '__main__':
    main()
