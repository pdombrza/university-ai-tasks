import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

def plot(x: list, y: list):
    fig, ax = plt.subplots()
    colors = list(mcolors.TABLEAU_COLORS.keys())[:2]
    ax.bar(x=x, height=y, color=colors)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Max depth")
    ax.set_title("Accuracy by max depth")
    plt.savefig("acc_depth.png")


def main():
    x = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    y = [0.49563663849378137, 0.719508005222291, 0.720263863121006, 0.7311207311207312, 0.7354497354497355, 0.7341441627155912, 0.7318078746650175, 0.7301587301587301, 0.7258984401841545, 0.7235621521335807, 0.71854600426029]
    plot(x, y)

if __name__ == "__main__":
    main()