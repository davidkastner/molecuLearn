import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sn
from statistics import mean

def load_data(mimos):
    """
    Load data from CSV files for each mimo in the given list.

    Parameters
    ----------
    mimos : list of str
        List of mimo names.

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
        df_charge[mimo] = pd.read_csv(f'data/{mimo}_charges.csv')
        df_dist[mimo] = pd.read_csv(f'data/{mimo}_pairwise_distances.csv')
        
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

    Returns
    -------
    None
    """

    # Create a 1x2 subplot
    fig, ax = plt.subplots(1, 2)

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
    plt.show()

def plot_confusion_matrices(cms, mimos):
    """
    Plot confusion matrices for distance and charge features.

    Parameters
    ----------
    cms : dict
        Dictionary containing confusion matrices for distance and charge features.
    mimos : list
        List of MIMO types, e.g. ['mc6', 'mc6s', 'mc6sa']

    Returns
    -------
    None
    """

    # Define the features for which confusion matrices will be plotted
    features = ['dist', 'charge']

    # Loop through each feature and plot its confusion matrix
    for feature in features:
        plt.figure()
        
        # Create a heatmap for the confusion matrix
        sn.heatmap(cms[feature], annot=True, cmap='coolwarm', fmt='d', cbar=False)
        
        # Set title, xlabel, and ylabel for the heatmap
        plt.title(f'Confusion Matrix for {feature}')
        plt.xlabel('Predicted')
        plt.ylabel('True')

    # Show the plotted confusion matrices
    plt.show()

if __name__ == "__main__":
    mimos = ['mc6', 'mc6s', 'mc6sa']
    df_charge, df_dist = load_data(mimos)
    plot_data(df_charge, df_dist, mimos)

    # Preprocess the data and split into train and test sets
    data_split = preprocess_data(df_charge, df_dist, mimos)

    # Train a random forest classifier for each feature
    rf_cls = train_random_forest(data_split, n_trees=200, max_depth=50)

    # Evaluate classifiers and plot confusion matrices
    cms = evaluate(rf_cls, data_split, mimos)
    plot_confusion_matrices(cms, mimos)