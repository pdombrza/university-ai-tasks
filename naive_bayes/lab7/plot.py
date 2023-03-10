from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')


def plot(data, title, color=False):
    fig, ax = plt.subplots()
    ax.set_title(title)
    if color:
        ax.plot(range(len(data)), data, color='orange')
    else:
        ax.plot(range(len(data)), data)
    plt.ylabel("accuracy")
    plt.xlabel("iterations")
    plt.ylim(0.5, 0.8)
    plt.savefig("kfold_reduced.png")


def main():
    result_tts = [0.7239755394672103, 0.7236943839179025, 0.7172980951711534, 0.7254516061010754, 0.7227103394953258]
    result_cross = [0.6942739547981803, 0.6722159251912723, 0.6362880310485827, 0.6077100583683996, 0.5758842657110624]
    plot(result_cross, "K-fold cross validation accuracy with reduced noise", True)
    pass


if __name__ == "__main__":
    main()