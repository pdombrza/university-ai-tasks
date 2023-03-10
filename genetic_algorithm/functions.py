import numpy as np
import random
from copy import deepcopy


class Chromosome:
    def __init__(self, genome):
        self.genome = genome
        self.size = len(genome)

    def set_value(self, val):
        self.value = val

    def set_probability(self, probablity):
        self.probability = probablity

    def set_genome(self, new_genome):
        self.genome = new_genome

    def count_ones(self):
        return self.genome.count(1)


def initialize_population(genome_size, population_size):
    return [Chromosome(random.choices((0, 1), k=genome_size)) for _ in range(population_size)]


def find_best(population):
    best = max(population, key=lambda x: x.value)
    return best


def evaluate(q_func, population):
    evaluated_population = population
    for individual in evaluated_population:
        individual.set_value(q_func(individual))
    return evaluated_population


def roulette_selection(evaluated_population):
    adjusted_population = evaluated_population
    for individual in adjusted_population: # add + C to all individuals' values to allow roulette wheel selection
        individual.value += 1200

    adjusted_population_values = [i.value for i in adjusted_population]
    total_sum = sum(adjusted_population_values)

    for i in adjusted_population: # extra
        i.set_probability(i.value/total_sum)

    probabilities = [individual.value/total_sum for individual in adjusted_population]

    for ind in adjusted_population: # readjust values to originals
        ind.value -= 1200

    temp_population = np.random.choice(adjusted_population, len(adjusted_population), p=probabilities)
    return temp_population


def cross(first_chromosone, second_chromosone):
    pivot = random.randint(0, first_chromosone.size-1)
    new_first = deepcopy(first_chromosone)
    new_second = deepcopy(second_chromosone)
    new_first.genome[pivot:] = second_chromosone.genome[pivot:]
    new_second.genome[pivot:] = first_chromosone.genome[pivot:]
    return new_first, new_second


def cross_population(population, crossing_probability):
    crossed_population = []
    i = 0
    while True:
        if np.random.random() < crossing_probability:
            new_first, new_second = cross(population[i], population[i+1])
        else:
            new_first, new_second = population[i], population[i+1]
        crossed_population.append(new_first)
        crossed_population.append(new_second)
        i += 2
        if i >= len(population):
            return crossed_population


def mutate(individual, mutation_probability):
    new_individual = deepcopy(individual)
    for i, bit in enumerate(new_individual.genome):
        if np.random.random() < mutation_probability:
                bit = bit ^ 1
                new_individual.genome[i] = bit
    return new_individual


def mutate_population(population, mutation_probability):
    mutated_population = []
    for individual in population:
        mutated_individual = mutate(individual, mutation_probability)
        mutated_population.append(mutated_individual)
    return mutated_population


def q_func(individual):
    current_velocity = 0 # starting velocity = 0
    current_height = 200
    fuel_count = individual.count_ones()
    rocket_mass = 200 + fuel_count

    for gene in individual.genome:
        current_acceleration = -0.09 # base acceleration is always -0.09 due to gravity
        if gene:
            rocket_mass -= 1
            current_acceleration += 45/rocket_mass # if engines are on, acceleration increases
        current_velocity += current_acceleration # add current acceleration to velocity
        current_height += current_velocity # move to the next height

        if current_height < 0:
            profit = -1000 - fuel_count
            return profit

        if current_height < 2 and abs(current_velocity) < 2:
            profit = 2000 - fuel_count
            return profit

    profit = - fuel_count
    return profit
