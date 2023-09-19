"""Functions for the random forest classifier."""

import numpy as np
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.utils import shuffle
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sn
from statistics import mean
from itertools import cycle
import shap
np.set_printoptions(threshold=sys.maxsize)


def load_data(mimos, data_loc):
    """
    Load data from CSV files for each mimo in the given list.

    Parameters
    ----------
    mimos : list of str
        List of mimo names
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

    # Would you like the pairwise charges included as features?
    include_pairwise_charges = False

    # Iterate through each mimo in the list
    for mimo in mimos:
        # Load charge data from CSV file and store in dictionary
        df_charge[mimo] = pd.read_csv(f"{data_loc}/{mimo}_charge_esp.csv")
        df_charge[mimo] = df_charge[mimo].drop(columns=["replicate"])

        # Load distance data from CSV file and store in dictionary
        df_dist[mimo] = pd.read_csv(f"{data_loc}/{mimo}_pairwise_distance.csv")
        df_dist[mimo] = df_dist[mimo].drop(columns=["replicate"])

        if include_pairwise_charges:
            # Read in {mimo}_charges_pairwise_multiply.csv
            df_pairwise_multiply = pd.read_csv(f"{data_loc}/{mimo}_charges_pairwise_multiply.csv")

            # Combine it column-wise with the {mimo}_charge_esp.csv dataframe
            df_charge[mimo] = pd.concat([df_charge[mimo], df_pairwise_multiply], axis=1)

    return df_charge, df_dist


def preprocess_data(df_charge, df_dist, mimos, data_split_type, test_frac=0.8):
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
    data_split_type : int
        Integer value of 1 or 2 to pick the type of data split.
    test_frac : float, optional, default: 0.8
        Fraction of data to use for training (the remaining data will be used for testing).

    Returns
    -------
    data_split : dict
        Dictionary containing the training and testing data for distance and charge features.
    df_dist : dict
        Revised dictionary with mimo names as keys and distance data as values in pandas DataFrames.
    df_charge : dict
        Revised dictionary with mimo names as keys and charge data as values in pandas DataFrames.

    Notes
    -----
    In data_split_type, 1 corresponds to splitting each trajectory into train/test then stitching together the
    train/test sets from each trajectory together to get an overall train/test set. The
    splitting within each trajectory is based on the provided fractional parameter.
    2 corresponds to splitting the entire dataset such that the first set of trajectories belong to
    the train set, and the second set of trajectories belong to the test set. The splitting of the
    trajectories is based on the provided fractional parameter.

    """

    class_assignment = {"mc6": 0, "mc6s": 1, "mc6sa": 2}
    features = ["dist", "charge"]

    # Create dictionaries for X (feature data) and y (class labels)
    X = {
        "dist": {mimo: np.array(df_dist[mimo]) for mimo in mimos},
        "charge": {mimo: np.array(df_charge[mimo]) for mimo in mimos},
    }
    y = {"dist": {}, "charge": {}}

    # Assign class labels for each mimo based on the class_assignment dictionary
    for mimo in mimos:
        y_aux = np.full((df_dist[mimo].shape[0],), class_assignment[mimo], dtype=int)
        y["dist"][mimo] = y_aux

        y_aux = np.full((df_charge[mimo].shape[0],), class_assignment[mimo], dtype=int)
        y["charge"][mimo] = y_aux

    data_split = {}
    if data_split_type == 1:
        X_train = {
            "dist": {mimo: np.empty((0, df_dist[mimo].shape[1])) for mimo in mimos},
            "charge": {mimo: np.empty((0, df_charge[mimo].shape[1])) for mimo in mimos},
        }

        X_test = {
            "dist": {mimo: np.empty((0, df_dist[mimo].shape[1])) for mimo in mimos},
            "charge": {mimo: np.empty((0, df_charge[mimo].shape[1])) for mimo in mimos},
        }

        y_train = {
            "dist": {mimo: np.empty((0,), dtype=int) for mimo in mimos},
            "charge": {mimo: np.empty((0,), dtype=int) for mimo in mimos},
        }

        y_test = {
            "dist": {mimo: np.empty((0,), dtype=int) for mimo in mimos},
            "charge": {mimo: np.empty((0,), dtype=int) for mimo in mimos},
        }

        # Split data into training and testing sets based on the test_frac parameters and normalize data.
        for feature in features:
            # Split data
            test_cutoff = int(test_frac * (X[feature]["mc6"].shape[0] / 8))
            count = 0
            while count < int(X[feature]["mc6"].shape[0]):
                for mimo in mimos:
                    X_train[feature][mimo] = np.vstack(
                        (
                            X_train[feature][mimo],
                            X[feature][mimo][count : count + test_cutoff, :],
                        )
                    )
                    X_test[feature][mimo] = np.vstack(
                        (
                            X_test[feature][mimo],
                            X[feature][mimo][
                                count
                                + test_cutoff : count
                                + int(X[feature][mimo].shape[0] / 8),
                                :,
                            ],
                        )
                    )
                    y_train[feature][mimo] = np.concatenate(
                        (
                            y_train[feature][mimo],
                            y[feature][mimo][count : count + test_cutoff],
                        )
                    )
                    y_test[feature][mimo] = np.concatenate(
                        (
                            y_test[feature][mimo],
                            y[feature][mimo][
                                count
                                + test_cutoff : count
                                + int(y[feature][mimo].shape[0] / 8)
                            ],
                        )
                    )
                count += int(X[feature]["mc6"].shape[0] / 8)

            data_split[feature] = {
                "X_train": np.vstack([X_train[feature][mimo] for mimo in mimos]),
                "X_test": np.vstack([X_test[feature][mimo] for mimo in mimos]),
                "y_train": np.concatenate([y_train[feature][mimo] for mimo in mimos]),
                "y_test": np.concatenate([y_test[feature][mimo] for mimo in mimos]),
            }

            # Normalize data
            x_scaler = StandardScaler()
            x_scaler.fit(data_split[feature]["X_train"])
            data_split[feature]["X_train"] = x_scaler.transform(
                data_split[feature]["X_train"]
            )
            data_split[feature]["X_test"] = x_scaler.transform(
                data_split[feature]["X_test"]
            )

            # Shuffle data splits while ensuring correspondence between features and labels.
            data_split[feature]["X_train"], data_split[feature]["y_train"] = shuffle(
                data_split[feature]["X_train"],
                data_split[feature]["y_train"],
                random_state=42,
            )
            data_split[feature]["X_test"], data_split[feature]["y_test"] = shuffle(
                data_split[feature]["X_test"],
                data_split[feature]["y_test"],
                random_state=42,
            )

    elif data_split_type == 2:
        for feature in features:
            test_cutoff = int(test_frac * X[feature]["mc6"].shape[0])

            data_split[feature] = {
                "X_train": np.vstack(
                    [X[feature][mimo][0:test_cutoff, :] for mimo in mimos]
                ),
                "X_test": np.vstack(
                    [X[feature][mimo][test_cutoff:, :] for mimo in mimos]
                ),
                "y_train": np.concatenate(
                    [y[feature][mimo][0:test_cutoff] for mimo in mimos]
                ),
                "y_test": np.concatenate(
                    [y[feature][mimo][test_cutoff:] for mimo in mimos]
                ),
            }

            # Normalize data
            x_scaler = StandardScaler()
            x_scaler.fit(data_split[feature]["X_train"])
            data_split[feature]["X_train"] = x_scaler.transform(
                data_split[feature]["X_train"]
            )
            data_split[feature]["X_test"] = x_scaler.transform(
                data_split[feature]["X_test"]
            )

            # Shuffle data splits while ensuring correspondence between features and labels.
            data_split[feature]["X_train"], data_split[feature]["y_train"] = shuffle(
                data_split[feature]["X_train"],
                data_split[feature]["y_train"],
                random_state=42,
            )
            data_split[feature]["X_test"], data_split[feature]["y_test"] = shuffle(
                data_split[feature]["X_test"],
                data_split[feature]["y_test"],
                random_state=42,
            )

    return data_split, df_dist, df_charge


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
    features = ["dist", "charge"]

    # Train random forest classifiers for each feature
    for feature in features:
        rf_cls[feature] = RandomForestClassifier(
            n_estimators=n_trees, max_depth=max_depth
        )
        rf_cls[feature].fit(
            data_split[feature]["X_train"], data_split[feature]["y_train"]
        )

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
    y_true : dict
        Dictionary containing 1D-array test data ground truth labels for distance and charge features.
    y_pred_proba : dict
        Softmax probs dict (2D-array, Ncolumns = number of classes) of the predicted labels for distance and charge features.

    """
    features = ["dist", "charge"]
    y_pred_proba = {}
    y_true = {}
    cms = {}
    for feature in features:
        y_pred_proba[feature] = rf_cls[feature].predict_proba(
            data_split[feature]["X_test"]
        )
        y_pred = rf_cls[feature].predict(data_split[feature]["X_test"])

        true_labels = data_split[feature]["y_test"]
        y_true[feature] = true_labels

        cm = confusion_matrix(y_pred, true_labels)
        cms[feature] = pd.DataFrame(cm, mimos, mimos)

    return cms, y_true, y_pred_proba


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
    ax[0].set_title("distances")
    ax[1].set_title("charges")

    # Loop through each MIMO type and plot average distances and charges
    for mimo in mimos:
        avg_dist = [mean(row[1]) for row in df_dist[mimo].iterrows()]
        avg_charge = [mean(row[1]) for row in df_charge[mimo].iterrows()]
        ax[0].plot(avg_dist, label=mimo)
        ax[1].plot(avg_charge, label=mimo)

    # Add legends to each subplot
    ax[0].legend(loc="upper left")
    ax[1].legend(loc="upper left")

    # Apply tight layout and show the plot
    fig.tight_layout()
    plt.savefig("rf_data.png", bbox_inches="tight", format="png", dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, mimos):
    """
    Plot the ROC curve for the test data of the charge and distance features.

    Parameters
    ----------
    y_true : dict
        Dictionary containing 1D-array test data ground truth labels for distance and charge features.
    y_pred_proba : dict
        Softmax probs dict (2D-array, Ncolumns = number of classes) of the predicted labels for distance and charge features.
    mimos : list
        List of MIMO types, e.g. ['mc6', 'mc6s', 'mc6sa']

    """
    features = ["dist", "charge"]

    lb = {}
    for feature in features:
        lb[feature] = LabelBinarizer()
        lb[feature].fit(y_true[feature])

    # Loop through each feature and plot ROC curve
    for feature in features:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for j in range(len(mimos)):
            y_true_j = lb[feature].transform(y_true[feature])[:, j]
            fpr[j], tpr[j], _ = roc_curve(y_true_j, y_pred_proba[feature][:, j])
            roc_auc[j] = auc(fpr[j], tpr[j])

        plt.figure()
        colors = cycle(["red", "blue", "green"])
        for j, color in zip(range(len(mimos)), colors):
            plt.plot(
                fpr[j],
                tpr[j],
                color=color,
                lw=2,
                label="ROC curve (area = %0.2f) for class %s" % (roc_auc[j], mimos[j]),
            )
        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("false positive rate", weight="bold")
        plt.ylabel("true positive rate", weight="bold")
        plt.title(
            "Multi-class classification ROC for %s features" % feature, weight="bold"
        )
        plt.legend(loc="best")
        plt.savefig(
            "rf_roc_" + feature + ".png", bbox_inches="tight", format="png", dpi=300
        )
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
    features = ["dist", "charge"]

    # Create a 1x2 subplot for the confusion matrices
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    # Loop through each feature and plot its confusion matrix
    for i, feature in enumerate(features):
        # Create a heatmap for the confusion matrix
        sn.heatmap(
            cms[feature], annot=True, cmap="inferno", fmt="d", cbar=False, ax=axs[i]
        )

        # Set title, xlabel, and ylabel for the heatmap
        axs[i].set_title(f"Confusion Matrix for {feature}", fontweight="bold")
        axs[i].set_xlabel("Predicted", fontweight="bold")
        axs[i].set_ylabel("True", fontweight="bold")

    # Apply tight layout and save the plotted confusion matrices
    fig.tight_layout()
    plt.savefig("rf_cm.png", bbox_inches="tight", format="png", dpi=300)
    plt.close()


def shap_analysis(rf_cls, data_split, df_dist, df_charge, mimos):
    """
    Plot SHAP dot plots for each mimichrome.

    Plots are generated for both charge and distance features.
    Identifies contribution of each feature to the prediction for a specific instance.

    Parameters
    ----------
    rf_cls : dict
        Dictionary containing trained RF classifiers for distance and charge features
    data_split : dict
        Dictionary containing the training and testing data for distance and charge features.
    df_dist : dict
        Dictionary of DataFrames containing distance data for each MIMO type.
    df_charge : dict
        Dictionary of DataFrames containing charge data for each MIMO type.
    mimos : list
        List of MIMO types, e.g. ['mc6', 'mc6s', 'mc6sa']
    
    """
    features = ["dist", "charge"]

    df = {"dist": df_dist, "charge": df_charge}

    shap_values = {}
    test = {}
    for i, feature in enumerate(features):
        # Load in a random batch from the test dataloader and interpret predictions for 156 data points
        test_features = data_split[feature]["X_test"]
        # Define the first 100 datapoints as the background used as reference when calculating SHAP values
        background = test_features[:100]
        test[feature] = test_features[-156:]  # Same number of test data points as MLP
        explainer = shap.TreeExplainer(rf_cls[feature], data=background)
        shap_values[feature] = explainer.shap_values(test[feature])

    # For each mimichrome, plot the SHAP values as dot plots
    for i in range(len(mimos)):
        # Each mimichrome has two datasets: charges and features
        fig, axs = plt.subplots(1, 2, figsize=(20, 5))
        for j, ax in enumerate(axs):
            plt.sca(ax)
            shap.summary_plot(
                shap_values[features[j]][i],
                test[features[j]],
                feature_names=df[features[j]][mimos[i]].columns.to_list(),
                show=False,
            )
            axs[j].set_title(f"{features[j]}, {mimos[i]}", fontweight="bold")
        plt.savefig(
            f"rf_shap_{mimos[i]}.png", bbox_inches="tight", format="png", dpi=300
        )
        plt.close()

    # Get the summary SHAP plots that combine feature importance for all classes
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for j, feature in enumerate(features):
        plt.sca(axs[j])
        shap.summary_plot(
            shap_values[feature],
            test[feature],
            feature_names=df[feature]["mc6"].columns.to_list(),
            plot_type="bar",
            show=False,
            plot_size=(15, 5),
            class_names=mimos,
        )
        axs[j].set_title(f"{feature}", fontweight="bold")
    plt.savefig(f"rf_shap_combined.png", bbox_inches="tight", format="png", dpi=300)
    plt.close()


def plot_gini_importance(rf_cls, df_dist, df_charge):
    """
    Plot Gini importance bar plots for the top 20 features for each feature type.

    Parameters
    ----------
    rf_cls : dict
        Dictionary containing trained RF classifiers for distance and charge features
    df_dist : dict
        Dictionary of DataFrames containing distance data for each MIMO type.
    df_charge : dict
        Dictionary of DataFrames containing charge data for each MIMO type.

    """
    features = ["dist", "charge"]

    df = {"dist": df_dist, "charge": df_charge}

    gini_importance = {}
    top_gini_importance = {}
    top_feature_names = {}
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    for i, feature in enumerate(features):
        # Obtain importances from the trained model attribute and sort
        gini_importance[feature] = rf_cls[feature].feature_importances_
        sorted_indices = np.argsort(gini_importance[feature])[::-1]
        top_indices = sorted_indices[:20]
        top_gini_importance[feature] = gini_importance[feature][top_indices]
        # Get corresponding feature names
        all_feature_names = df[feature][
            "mc6"
        ].columns.to_list()  # All mimo classes havev same feature labels
        top_feature_names[feature] = [all_feature_names[j] for j in top_indices]

        axs[i].bar(
            range(len(top_gini_importance[feature])),
            top_gini_importance[feature],
            tick_label=top_feature_names[feature],
            color="red",
        )
        axs[i].set_xlabel("features", weight="bold")
        axs[i].set_ylabel("Gini importance", weight="bold")
        axs[i].set_title(
            f"top 20 Gini importances for all classes for {feature} feature",
            weight="bold",
        )
        axs[i].tick_params(axis="x", rotation=90)
    plt.savefig(f"rf_gini.png", bbox_inches="tight", format="png", dpi=300)
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

def rf_analysis():
    # Get datasets
    format_plots()
    mimos = ["mc6", "mc6s", "mc6sa"]
    data_loc = input("   > Where are your data files located? ")
    df_charge, df_dist = load_data(mimos, data_loc)
    plot_data(df_charge, df_dist, mimos)

    # Preprocess the data and split into train and test sets
    data_split, df_dist, df_charge = preprocess_data(df_charge, df_dist, mimos, 1)

    # Train a random forest classifier for each feature
    rf_cls = train_random_forest(data_split, n_trees=200, max_depth=50)

    # Evaluate classifiers and plot confusion matrices, roc curves, and SHAP dot plots and Gini importance bar plots
    cms, y_true, y_pred_proba = evaluate(rf_cls, data_split, mimos)
    plot_roc_curve(y_true, y_pred_proba, mimos)
    plot_confusion_matrices(cms, mimos)
    shap_analysis(rf_cls, data_split, df_dist, df_charge, mimos)
    plot_gini_importance(rf_cls, df_dist, df_charge)


if __name__ == "__main__":
    rf_analysis()