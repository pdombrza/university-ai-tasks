import json
import numpy as np
from matplotlib import pyplot as plt

from parameters import *


def read_file(fname):
    with open(fname, 'r') as fh:
        lines = json.load(fh)
    return lines


def get_avg_rewards(data):
    return data["avg_rewards"]


def get_training_rewards(data):
    train_rewards_temp = data["training_rewards"]
    return [x for sub in train_rewards_temp for x in sub]


def get_training_episode_lenghts(data):
    episode_lenghts = data["training_episode_lengths"]
    return [x for sub in episode_lenghts for x in sub]

def plot_rewards(data, savename):
    fig, ax = plt.subplots()
    ax.set_title("Average evaluation rewards: ")
    ax.plot(range(len(data)), data)
    plt.savefig(savename)


def plot(data, title, savename):
    fig, ax = plt.subplots()
    ax.set_title(title)
    moving_average = np.convolve(np.array(data), np.ones(20), mode="valid") / 20
    ax.plot(range(len(moving_average)), moving_average)
    plt.savefig(savename)


def main():
    data = read_file("results_epsilon.json")
    avg_rewards = get_avg_rewards(data)
    train_rewards = get_training_rewards(data)
    episode_lenghts = get_training_episode_lenghts(data)
    plot_rewards(avg_rewards, "plots/avg_rewards_discount2.png")
    plot(train_rewards, "Training reward: ", "plots/train_rewards_discount2.png")
    plot(episode_lenghts, "Episode length: ", "plots/episode_lenghts_discount2.png")


if __name__ == "__main__":
    main()