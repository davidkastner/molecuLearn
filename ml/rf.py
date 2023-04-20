import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sn
from statistics import mean

def load_data(mimos, data_loc):
    """
    Load data from CSV files for each mimo in the given list.

    Parameters
    ----------
    mimos : list of str
        List of mimo names.
    data_loc : str
        The location of the (e.g, /home/kastner/packages/molecuLearn/ml/data)

    Returns
    -------
    df_charge : dict
        Dictionary with mimo names as keys and charge data as values in pandas DataFrames.
    df_dist : dict
        Dictionary with mimo names as keys and distance data as values in pandas DataFrames.
    """
    
    df_charge = {}
    df_dist = {}
    
    # Iterate through each mimo in the list
    for mimo in mimos:
        # Load charge and distance data from CSV files and store them in dictionaries
        df_charge[mimo] = pd.read_csv(f'{data_loc}/{mimo}_charges.csv')
        df_dist[mimo] = pd.read_csv(f'{data_loc}/{mimo}_pairwise_distances.csv')
        
    return df_charge, df_dist

def preprocess_data(df_charge, df_dist, mimos, test_frac=0.8):
    """
    Preprocess data for training and testing by splitting it based on the given test fraction.

    Parameters
    ----------
    df_charge : dict
        Dictionary with mimo names as keys and charge data as values in pandas DataFrames.
    df_dist : dict
        Dictionary with mimo names as keys and distance data as values in pandas DataFrames.
    mimos : list of str
        List of mimo names.
    test_frac : float, optional, default: 0.8
        Fraction of data to use for training (the remaining data will be used for testing).

    Returns
    -------
    data_split : dict
        Dictionary containing the training and testing data for distance and charge features.
    """
    
    class_assignment = {'mc6': 0, 'mc6s': 1, 'mc6sa': 2}
    features = ['dist', 'charge']
    
    # Create dictionaries for X (feature data) and y (class labels)
    X = {'dist': {mimo: np.array(df_dist[mimo]) for mimo in mimos},
         'charge': {mimo: np.array(df_charge[mimo]) for mimo in mimos}}
    y = {'dist': {},
         'charge': {}}
    
    # Assign class labels for each mimo based on the class_assignment dictionary
    for mimo in mimos:
        y_aux = np.zeros((df_dist[mimo].shape[0], 3))
        y_aux[:, class_assignment[mimo]] = 1
        y['dist'][mimo] = y_aux
        
        y_aux = np.zeros((df_charge[mimo].shape[0], 3))
        y_aux[:, class_assignment[mimo]] = 1
        y['charge'][mimo] = y_aux

    data_split = {}
    
    # Split data into training and testing sets based on the test_frac parameter
    for feature in features:
        n_cutoff = int(test_frac * X[feature]['mc6'].shape[0])
        
        data_split[feature] = {'X_train': np.vstack([X[feature][mimo][0:n_cutoff, :] for mimo in mimos]),
                               'X_test': np.vstack([X[feature][mimo][n_cutoff:, :] for mimo in mimos]),
                               'y_train': np.vstack([y[feature][mimo][0:n_cutoff, :] for mimo in mimos]),
                               'y_test': np.vstack([y[feature][mimo][n_cutoff:, :] for mimo in mimos])}

    return data_split

def train_random_forest(data_split, n_trees, max_depth):
    """
    Train random forest classifiers for the distance and charge features.

    Parameters
    ----------
    data_split : dict
        Dictionary containing the training and testing data for distance and charge features.
    n_trees : int
        Number of trees in the random forest.
    max_depth : int
        Maximum depth of the trees in the random forest.

    Returns
    -------
    rf_cls : dict
        Dictionary containing trained random forest classifiers for distance and charge features.
    """
    
    rf_cls = {}
    features = ['dist', 'charge']
    
    # Train random forest classifiers for each feature
    for feature in features:
        rf_cls[feature] = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth)
        rf_cls[feature].fit(data_split[feature]['X_train'], data_split[feature]['y_train'])

    return rf_cls

def evaluate(rf_cls, data_split, mimos):
    """
    Evaluate the random forest classifiers and return confusion matrices for both features.

    Parameters
    ----------
    rf_cls : dict
        Dictionary containing trained random forest classifiers for distance and charge features.
    data_split : dict
        Dictionary containing the training and testing data for distance and charge features.
    mimos : list
        List of MIMO types, e.g. ['mc6', 'mc6s', 'mc6sa']

    Returns
    -------
    cms : dict
        Dictionary containing confusion matrices for distance and charge features.
    """

    features = ['dist', 'charge']
    y_pred = {}
    cms = {}
    
    for feature in features:
        y_pred[feature] = rf_cls[feature].predict(data_split[feature]['X_test'])
        
        # Replace the lin_idx() function with its implementation
        true_labels = [data_split[feature]['y_test'][i, :].argmax() for i in range(data_split[feature]['y_test'].shape[0])]
        pred_labels = [y_pred[feature][i, :].argmax() for i in range(y_pred[feature].shape[0])]
        
        cm = confusion_matrix(pred_labels, true_labels)
        cms[feature] = pd.DataFrame(cm, mimos, mimos)

    return cms

def plot_data(df_charge, df_dist, mimos):
    """
    Plot the average charge and distance data for the given MIMO types.

    Parameters
    ----------
    df_charge : dict
        Dictionary of DataFrames containing charge data for each MIMO type.
    df_dist : dict
        Dictionary of DataFrames containing distance data for each MIMO type.
    mimos : list
        List of MIMO types, e.g. ['mc6', 'mc6s', 'mc6sa']

    """

    # Create a 1x2 subplot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Set titles for each subplot
    ax[0].set_title('distances')
    ax[1].set_title('charges')

    # Loop through each MIMO type and plot average distances and charges
    for mimo in mimos:
        avg_dist = [mean(row[1]) for row in df_dist[mimo].iterrows()]
        avg_charge = [mean(row[1]) for row in df_charge[mimo].iterrows()]
        ax[0].plot(avg_dist, label=mimo)
        ax[1].plot(avg_charge, label=mimo)

    # Add legends to each subplot
    ax[0].legend(loc='upper left')
    ax[1].legend(loc='upper left')

    # Apply tight layout and show the plot
    fig.tight_layout()
    plt.savefig("rf_data.png", bbox_inches="tight", format="png", dpi=300)
    plt.close()

def plot_confusion_matrices(cms, mimos):
    """
    Plot confusion matrices for distance and charge features.

    Parameters
    ----------
    cms : dict
        Dictionary containing confusion matrices for distance and charge features.
    mimos : list
        List of MIMO types, e.g. ['mc6', 'mc6s', 'mc6sa']

    """

    # Define the features for which confusion matrices will be plotted
    features = ['dist', 'charge']

    # Create a 1x2 subplot for the confusion matrices
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    # Loop through each feature and plot its confusion matrix
    for i, feature in enumerate(features):
        
        # Create a heatmap for the confusion matrix
        sn.heatmap(cms[feature], annot=True, cmap='inferno', fmt='d', cbar=False, ax=axs[i])
        
        # Set title, xlabel, and ylabel for the heatmap
        axs[i].set_title(f'Confusion Matrix for {feature}', fontweight='bold')
        axs[i].set_xlabel('Predicted', fontweight='bold')
        axs[i].set_ylabel('True', fontweight='bold')

    # Apply tight layout and save the plotted confusion matrices
    fig.tight_layout()
    plt.savefig("rf_cm.png", bbox_inches="tight", format="png", dpi=300)
    plt.close()

def format_plots() -> None:
    """
    General plotting parameters for the Kulik Lab.

    """
    font = {"family": "sans-serif", "weight": "bold", "size": 10}
    plt.rc("font", **font)
    plt.rcParams["xtick.major.pad"] = 5
    plt.rcParams["ytick.major.pad"] = 5
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.size"] = 7
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.size"] = 7
    plt.rcParams["ytick.major.width"] = 2
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["svg.fonttype"] = "none"


if __name__ == "__main__":
    # Get datasets
    format_plots()
    mimos = ['mc6', 'mc6s', 'mc6sa']
    data_loc = input("   > Where are your data files located? ")
    df_charge, df_dist = load_data(mimos, data_loc)
    plot_data(df_charge, df_dist, mimos)

    # Preprocess the data and split into train and test sets
    data_split = preprocess_data(df_charge, df_dist, mimos)

    # Train a random forest classifier for each feature
    rf_cls = train_random_forest(data_split, n_trees=200, max_depth=50)

    # Evaluate classifiers and plot confusion matrices
    cms = evaluate(rf_cls, data_split, mimos)
    plot_confusion_matrices(cms, mimos)