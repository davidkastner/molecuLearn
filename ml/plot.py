"""General plotting functions."""

import matplotlib.pyplot as plt
import seaborn as sn
from statistics import mean

def plot_data(df_charge, df_dist, mimos):
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('distances')
    ax[1].set_title('charges')
    for mimo in mimos:
        avg_dist = [mean(row[1]) for row in df_dist[mimo].iterrows()]
        avg_charge = [mean(row[1]) for row in df_charge[mimo].iterrows()]
        ax[0].plot(avg_dist, label=mimo)
        ax[1].plot(avg_charge, label=mimo)
    ax[0].legend(loc='upper left')
    ax[1].legend(loc='upper left')
    fig.tight_layout()
    plt.show()

def plot_confusion_matrices(cms, mimos):
    features = ['dist', 'charge']
    for feature in features:
        plt.figure()
        sn.heatmap(cms[feature], annot=True, cmap='coolwarm', fmt='d', cbar=False)
        plt.title(f'Confusion Matrix for {feature}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    plt.show()

