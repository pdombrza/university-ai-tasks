from abc import ABC, abstractmethod
from functions import *


class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters

    @abstractmethod
    def get_parameters(self) -> dict:
        """Returns a dictionary of hyperparameters"""
        return self.parameters

    @abstractmethod
    def solve(self, problem, pop0, *args, **kwargs):
        """
        A method that solves the given problem for given initial solutions.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        """
        pass


class Solution(Solver):
    def __init__(self, parameters: dict) -> None:
        super().__init__(parameters)

    def get_parameters(self) -> dict:
        return self.parameters

    def solve(self):
        q_func = self.parameters["eval function"]
        genome_size = self.parameters["genome size"]
        population_size = self.parameters["population size"]
        crossing_probability = self.parameters["crossing probability"]
        mutation_probability = self.parameters["mutation probability"]
        t = 0  # current iteration
        initial_population = initialize_population(genome_size, population_size)
        evaluated_population = evaluate(q_func, initial_population)
        best = find_best(evaluated_population) # has best value and genome
        while t < self.parameters["t_max"]:
            temp_population = roulette_selection(evaluated_population)
            crossed_population = cross_population(temp_population, crossing_probability)
            mutated_population = mutate_population(crossed_population, mutation_probability)
            new_evaluated_population = evaluate(q_func, mutated_population)
            new_best = find_best(new_evaluated_population)
            if new_best.value >= best.value:
                best = new_best
            evaluated_population = new_evaluated_population
            t += 1
        return best


def main():
    parameters = {
        "t_max": 1000,
        "population size": 100,
        "genome size": 100,
        "eval function": q_func,
        "crossing probability": 0.85,
        "mutation probability": 0.05
    }

    solve = Solution(parameters)
    best_one = solve.solve()
    print(best_one.value)


if __name__ == "__main__":
    main()
