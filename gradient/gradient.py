import numpy as np
from math import exp
from plot import *


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


def gradient_descent(func, grad, starting_point, step, max_step=5000):
    epsilon = 1e-8
    point = starting_point
    steps = [starting_point]
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


def main():
    starting_point_g = np.random.uniform(low=-3, high=3, size=2)
    test_start = 1
    test_start2 = np.array((1, 1))
    beta_f = 0.2
    beta_g = 0.25
    f_points = gradient_descent(f, f_grad, test_start, beta_f)
    g_points = gradient_descent(g, g_grad, test_start2, beta_g) # best - 0.5 , 1
    print(g_points)
    print(f_points)


if __name__ == "__main__":
    main()
