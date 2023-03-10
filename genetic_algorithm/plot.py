from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot(df, path):
    sns.set_theme(style="ticks", palette="pastel")
    boxplot = sns.boxplot(x="variable", y="value", data=pd.melt(df))
    boxplot.set(xlabel="mutation probability", ylabel="max value", title="Mutation probability vs Max value")
    plt.savefig(path)


def read_file(path):
    with open(path, 'r') as fh:
        lines = fh.readlines()
    return lines


def parse_data(raw, start, end, amount, columns, data_len):
    data = []
    s = start
    e = end
    for i in range(amount):
        data1 = []
        for j in range(s, e):
            data1.append(int(raw[j][:4]))
        data.append(data1)
        s += data_len
        e += data_len
    data_arr = np.array(data)
    df = pd.DataFrame(data_arr.transpose(), columns=columns)
    return df


def main():
    lines = read_file("mutation_results.txt")
    s = 1
    e = 26
    df = parse_data(lines, s, e, amount=7, columns=['0', '0.001', '0.01', '0.05', '0.1', '0.25', '0.5'], data_len=26)
    plot(df, 'boxplot.png')


if __name__ == "__main__":
    main()
