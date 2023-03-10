from abc import ABC, abstractmethod
import numpy as np
from math import exp
from plot import *


class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        return self.parameters

    @abstractmethod
    def solve(self, problem, x0, *args, **kwargs):
        """
        A method that solves the given problem for given initial solution.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        """
        pass


class Solution(Solver):
    def __init__(self, parameters: dict) -> None:
        super().__init__(parameters)

    def get_parameters(self):
        return self.parameters

    def solve(self):
        epsilon = self.parameters["epsilon"]
        point = self.parameters["starting point"]
        steps = [self.parameters["starting point"]]
        max_step = self.parameters["max step"]
        step = self.parameters["beta"]
        grad = self.parameters["gradient"]
        count = -1
        distances = []
        while True:
            point = np.subtract(point, step * grad(point))
            steps.append(point)
            count += 1
            dist = np.sqrt(np.sum((point - steps[count])**2, axis=0))
            distances.append(dist)
            if dist <= epsilon or count+3 > max_step:
                break
        return steps


def f(x):
    return 1/4 * x**4


def f_grad(x):
    return x**3


def g(point):
    x1 = point[0]
    x2 = point[1]
    exp1 = exp(-x1**2 - x2**2)
    exp2 = exp(-(x1 + 1.5)**2 - (x2 - 2)**2)
    return 2 - exp1 - 0.5*exp2


def g_grad(point):
    x1 = point[0]
    x2 = point[1]
    exp1 = exp(-x1**2 - x2**2)
    exp2 = exp(-(x1 + 1.5)**2 - (x2 - 2)**2)
    return np.array([
        2*x1*exp1 + (x1 + 1.5)*exp2,
        2*x2*exp1 + (x2 - 2)*exp2
    ])


def main():
    parameters_g = {
        "epsilon": 1e-8,
        "starting point": np.array((1, 1)),
        "beta": 0.5,
        "max step": 5000,
        "gradient": g_grad
    }
    parameters_f = {
        "epsilon": 1e-8,
        "starting point": 2,
        "beta": 0.1,
        "max step": 20000,
        "gradient": f_grad
    }
    grad_descent_g = Solution(parameters_g)
    points_g = grad_descent_g.solve()
    print(points_g)
    g_contour_plot(points_g, grad_descent_g.get_parameters()["beta"], 'g_plot.png', save=False)
    print(grad_descent_g.get_parameters())
    grad_descent_f = Solution(parameters_f)
    points_f = grad_descent_f.solve()
    print(grad_descent_f.get_parameters())
    f_plot(points_f, grad_descent_f.get_parameters()["beta"], 'f_plot.png', save=False)



if __name__ == "__main__":
    main()
