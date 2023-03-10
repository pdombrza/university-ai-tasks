import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as ticker
import numpy as np


def g(x, y):
    return 2 - np.exp(-x**2 - y**2) - 0.5*np.exp(-(x + 1.5)**2 - (y - 2)**2)


def f(x):
    return 1/4 * x**4


def g_surface_plot(points, beta_value, plot_name, save):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(-5, 5, 0.005)
    Y = np.arange(-5, 5, 0.005)
    X, Y = np.meshgrid(X, Y)
    Z = g(X, Y)
    x1 = [point[0] for point in points]
    y1 = [point[1] for point in points]
    z = [g(a, b) for a, b in zip(x1, y1)]

    surf = ax.plot_surface(X, Y, Z, cmap=cm.spring,
                        linewidth=0, antialiased=False, alpha=0.3)
    ax.scatter3D(x1, y1, z, color='black', antialiased=False, s=10, label=f'steps: {len(points)}\nbeta value: {beta_value}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.24, 1.08))
    ax.set_zlim(0.01, 3.01)
    ax.zaxis.set_major_locator(LinearLocator(10))

    ax.zaxis.set_major_formatter('{x:.02f}')

    fig.colorbar(surf, shrink=0.4, aspect=5)
    if save:
        plt.savefig(plot_name)
    else:
        plt.show()



def g_contour_plot(points, beta_value, plot_name, save):
    X = np.arange(-4, 4, 0.005)
    Y = np.arange(-2, 4, 0.005)
    X, Y = np.meshgrid(X, Y)
    Z = g(X, Y)

    x1 = [point[0] for point in points]
    y1 = [point[1] for point in points]
    plt.contour(X, Y, Z, levels=10, cmap=cm.spring)
    plt.colorbar()
    plt.scatter(x1, y1, color='red',s=10, label=f'steps: {len(points)}\nbeta value: {beta_value}')
    plt.legend()
    if save:
        plt.savefig(plot_name)
    else:
        plt.show()


def f_plot(points, beta_value, plot_name, save):
    x = np.arange(-5, 5, 0.01)
    y = f(x)
    y1 = [f(point) for point in points]
    fig, ax = plt.subplots()
    ax.scatter(points, y1, color="red", s=10, label=f'steps: {len(points)}\nbeta value: {beta_value}')
    ax.plot(x, y)
    ax.legend()
    ax.set(xlabel="x", ylabel="y", title="f(x)", xticks=np.arange(-3, 4), yticks=np.arange(1, 6), xlim=(-4, 4), ylim=(0, 6))
    ax.grid()
    if save:
        plt.savefig(plot_name)
    else:
        plt.show()
