import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

def plot(data):
    lineplot = sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    lineplot.set(xlabel="learn rate", ylabel="accuracy", title="Accuracy vs learn rate")
    plt.savefig("accuracy_learn_rate2.png")

def main():
    # cols = ['3 layers', '4 layers']
    columns3 = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5]
    # iter3 = [0.10317460317460317, 0.8835978835978836, 0.8677248677248677, 0.9232804232804233, 0.8412698412698413, 0.9047619047619048, 0.9391534391534392]
    # iter4 = [0.082010582010582, 0.082010582010582, 0.8465608465608465, 0.9365079365079365, 0.9391534391534392, 0.9206349206349206, 0.9603174603174603]
    # data_dict = {cols[0]: iter3, cols[1]: iter4}
    iter4_dif = [0.1164021164021164, 0.9417989417989417, 0.9761904761904762, 0.9682539682539683, 0.9735449735449735, 0.9814814814814815, 0.9788359788359788]
    learn4_dif = [0.15608465608465608, 0.164021164021164, 0.20105820105820105, 0.9497354497354498, 0.9656084656084656, 0.9894179894179894, 0.9761904761904762, 0.9312169312169312, 0.5661375661375662]
    data_dict = {'4 layers': learn4_dif}
    df = pd.DataFrame(data=data_dict)
    df = df.set_axis(columns3)
    print(df)
    plot(df)


if __name__ == "__main__":
    main()