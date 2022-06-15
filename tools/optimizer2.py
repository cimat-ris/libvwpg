import os
import random
import subprocess
import glob
import tempfile
import subprocess
import multiprocessing

import numpy

from deap import base
from deap import creator
from deap import tools
from deap import algorithms


NUMBER_OF_PARAMETERS = 7
EXPECTED_X_FOOT_COORDINATE = 2.5
EXPECTED_Y_FOOT_COORDINATE = 0.0
FOOT_WIDTH = 0.058
FOOT_LENGTH = 0.1372

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


class SimulationRunner(object):

    def __init__(self, gains):
        self._gains = gains
        self._config_file = tempfile.NamedTemporaryFile(suffix='.ini')
        self.create_config_file(self._gains, self._config_file.name)
        self._log_file = None
        self._x_foot_positions = []
        self._y_foot_positions = []
        self._objective_function = None
        self._farthest_zmp = (-1e100, -1e-100)
        self._farthest_foot = (-1e100, -1e-100)


    @staticmethod
    def create_config_file(values, config_file):
        """ """
        alpha, betah11, betah13, betah33, gamma, eta, kappa = values
        with open(config_file, 'w') as file_handle:
            file_handle.write(CONFIG_FILE.format(alpha, betah11, betah13, betah33, gamma, eta, kappa))


    def run_simulation(self):
        pid = subprocess.Popen(['./main', self._config_file.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, __ = pid.communicate()
        
        if pid.returncode:
            raise SimulationError()

        # search for line 'results will be saved to: <output file>.txt'
        for line in stdout.splitlines():
            if 'results will be saved to:' in line:
                __, log_file = line.split(':')
                self._log_file = log_file.strip()
                self._parse_results()
                return

        raise SimulationError()

    @staticmethod
    def _unpack_data(*packed_data):
        unpacked_data = []
        for entry in packed_data:
            try:
                entry = float(entry)
            except ValueError:
                pass
            unpacked_data.append(entry)
        return unpacked_data


    @staticmethod
    def _parse_raw_log_entry(entry):
        """expected line format: ... [tag]: entry0=value0, entry1=value1, ..."""
        __, raw_data = entry.split(']: ')
        all_elements = []
        for element in raw_data.split(', '):
            __, value = element.split('=')
            all_elements.append(value.strip())
        return SimulationRunner._unpack_data(*all_elements)


    def _parse_results(self):
        """ """
        is_foot_position = lambda l: '[foot_position]' in l
        is_zmp_position = lambda l: '[zmp_position]' in l
        is_objective_function = lambda l: '[objective_function]' in l
        index = None

        with open(self._log_file, 'r') as file_handle:
            for line in file_handle.readlines():
                if is_foot_position(line):
                    __, x_coordinate, y_coordinate, __ = self._parse_raw_log_entry(line.strip())
                    self._x_foot_positions.append(x_coordinate)
                    self._y_foot_positions.append(y_coordinate)
                elif is_objective_function(line):
                    __, objective_function = line.split(': ')
                    self._objective_function = float(objective_function.strip())
                elif is_zmp_position(line):
                    iteration, x_coordinate, y_coordinate = self._parse_raw_log_entry(line.strip())
                    index = int(iteration)
                    __, y_farthest_zmp = self._farthest_zmp
                    if y_coordinate > y_farthest_zmp:
                        self._farthest_zmp = x_coordinate, y_coordinate

        self._farthest_foot = self._x_foot_positions[index], self._y_foot_positions[index]


    def cleanup(self):
        self._config_file.close()


    def get_average_final_foot_position(self, final_steps=15):
        average_x_position = numpy.mean(self._x_foot_positions[-final_steps:])
        average_y_position = numpy.mean(self._y_foot_positions[-final_steps:])
        return average_x_position, average_y_position
        

    def get_std_final_foot_position(self, final_steps=15):
        std_x_position = numpy.std(self._x_foot_positions[-final_steps:])
        std_y_position = numpy.std(self._y_foot_positions[-final_steps:])
        return std_x_position, std_y_position

    def get_max_zmp_position(self):
        return self._farthest_zmp 


    def get_max_foot_position(self):
        return self._farthest_foot 

    def get_cost_function_value(self):
        return self._objective_function


def evaluate(individual):
    """should return tuple, single objective fitness is a special case of
    multi-objective"""
    
    def _zmp_puishment(foot_coordinates, zmp_coordinates):
        x_foot, y_foot = foot_coordinates
        x_zmp, y_zmp = zmp_coordinates

        x_delta = FOOT_LENGTH / 2.0
        y_delta = FOOT_WIDTH / 2.0
        is_within_x_margin = (x_zmp < x_foot + x_delta) and (x_zmp > x_foot - x_delta)
        is_within_y_margin = (y_zmp < y_foot + y_delta) and (y_zmp > y_foot - y_delta)

        if is_within_x_margin and is_within_y_margin:
            return 0.0
        if is_within_x_margin and not is_within_y_margin:
            return 100000 * abs((y_foot + y_delta) - y_zmp)
        if not is_within_x_margin and is_within_y_margin:
            return 100000 * abs((x_foot + x_delta) - x_zmp)
        return (1 << 40)

    simulation_runner = SimulationRunner(individual)
    try:
        simulation_runner.run_simulation() 
    except SimulationError:
        return (1 << 64),    # return big number
    simulation_runner.cleanup()

    is_within_margin = lambda x, lower_bondary, upper_boundary: (x < upper_boundary) and (x > lower_bondary)
    fitness = 0.0

    # foot final position
    x_foot_mean, y_foot_mean = simulation_runner.get_average_final_foot_position()
    x_foot_std, y_foot_std = simulation_runner.get_std_final_foot_position()
    objective_function = simulation_runner.get_cost_function_value()
    objective_function = 0.0

    if (is_within_margin(x_foot_mean, 0.98*EXPECTED_X_FOOT_COORDINATE, 1.02*EXPECTED_X_FOOT_COORDINATE) and
        is_within_margin(y_foot_mean, 0.98*EXPECTED_Y_FOOT_COORDINATE, 1.02*EXPECTED_Y_FOOT_COORDINATE)):
        fitness += 1.0e5 * (x_foot_std / x_foot_mean + y_foot_std / y_foot_std) * objective_function
    elif is_within_margin(x_foot_mean, 0.98*EXPECTED_X_FOOT_COORDINATE, 1.02*EXPECTED_X_FOOT_COORDINATE):
        fitness += 1.0e6 * abs(EXPECTED_Y_FOOT_COORDINATE - y_foot_mean) * objective_function
    elif is_within_margin(y_foot_mean, 0.98*EXPECTED_Y_FOOT_COORDINATE, 1.02*EXPECTED_Y_FOOT_COORDINATE):
        fitness += 1.0e8 * abs(EXPECTED_X_FOOT_COORDINATE - x_foot_mean) * objective_function
    else:
        fitness += 1 << 32

    # zmp
    x_max_zmp, y_max_zmp = simulation_runner.get_max_zmp_position()
    x_max_foot, y_max_foot = simulation_runner.get_max_foot_position()
    x_delta, y_delta = (FOOT_LENGTH / 2.0, FOOT_WIDTH / 2.0)

    if (is_within_margin(x_max_zmp, x_max_foot - x_delta, x_max_foot + x_delta) and
        is_within_margin(y_max_zmp, y_max_foot - y_delta, y_max_foot + y_delta)):
        fitness -= 1.0e0
    elif is_within_margin(x_max_zmp, x_max_foot - x_delta, x_max_foot + x_delta):
        fitness += 50 * abs((y_max_foot + y_delta) - y_max_zmp)
    elif is_within_margin(y_max_zmp, y_max_foot - y_delta, y_max_foot + y_delta):
        fitness += 10 * abs((x_max_foot + x_delta) - x_max_zmp)
    else:
        fitness += float(1 << 40)

    return (fitness,)


def main():
    """ """
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
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # parallel ?
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    population = toolbox.population(n=80)
    hall_of_fame = tools.HallOfFame(5)

    # cxpb - The probability of mating two individuals.
    # mutpb - The probability of mutating an individual.
    # ngen - The number of generation.
    final_population, log = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2,
                                                ngen=500, halloffame=hall_of_fame, verbose=True)

    best = tools.selBest(final_population, 1)
    best = best[0]
    SimulationRunner.create_config_file(best, 'best.ini')
    
    # final clean up
    for file_to_remove in glob.glob('*log*.txt'):
        os.remove(file_to_remove)

if __name__ == '__main__':
    main()
