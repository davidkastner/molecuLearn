"""Functions for the multi-layer perceptron classifier."""

import os
import shap
import torch
import optuna
import math
import shutil
import argparse
import numpy as np
import pandas as pd
import seaborn as sn
from itertools import cycle
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelBinarizer


def load_data(mimos, include_esp, data_loc):
    """
    Load data from CSV files for each mimo in the given list.

    Parameters
    ----------
    mimos : list[str]
        List of mimo names.
    data_loc : str
        The location of the (e.g, /home/kastner/packages/molecuLearn/ml/data)

    Returns
    -------
    df_charge : dict
        Dict with mimo names as keys and charge data as values
    df_dist : dict
        Dict with mimo names as keys and distance data as values
    """

    df_charge = {}
    df_dist = {}

    # Iterate through each mimo in the list
    for mimo in mimos:
        # Load charge data from CSV file and store in dictionary
        df_charge[mimo] = pd.read_csv(f"{data_loc}/{mimo}_charge_esp.csv")
        df_charge[mimo] = df_charge[mimo].drop(columns=["replicate"])
        
        # Option to include the ESP features
        include_esp = include_esp.strip().lower()
        if include_esp not in ['t', 'true', True]:
            df_charge[mimo].drop(columns=["upper", "lower"], inplace=True)

        # Load distance data from CSV file and store in dictionary
        df_dist[mimo] = pd.read_csv(f"{data_loc}/{mimo}_pairwise_distance.csv")
        df_dist[mimo] = df_dist[mimo].drop(columns=["replicate"])

    return df_charge, df_dist


def gradient_step(model, dataloader, optimizer, device):
    """
    A function to train on the entire dataset for one epoch.

    Parameters
    ----------
        model : torch.nn.Module
            The model
        dataloader : torch.utils.data.DataLoader
            DataLoader object for the train data
        optimizer : torch.optim.Optimizer(())
            optimizer object to interface gradient calculation and optimization
        device : str
            The device (usually 'cuda:0' for GPU or 'cpu' for CPU)

    Returns
    -------
        loss : float
            Loss averaged over all the batches
    """

    epoch_loss = []
    model.train()  # Set model to training mode

    for batch in dataloader:
        X, y = batch
        X = X.to(device)
        y = y.to(device)

        # train your model on each batch here
        y_pred = model(X)

        loss = torch.nn.functional.cross_entropy(y_pred, y)
        epoch_loss.append(loss.item())

        # run backpropagation given the loss you defined
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = np.array(epoch_loss).mean()

    return loss


def validate(model, dataloader, device):
    """
    A function to validate on the validation dataset for one epoch.

    Parameters
    ----------
        model : torch.nn.Module
            The model
        dataloader : torch.utils.data.DataLoader
            DataLoader object for the validation data
        device : str
            Your device (usually 'cuda:0' for GPU or 'cpu' for CPU)

    Returns
    -------
        loss : float
            Loss averaged over all the batches
    """

    val_loss = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            # validate your model on each batch here
            y_pred = model(X)
            loss = torch.nn.functional.cross_entropy(y_pred.squeeze(), y)
            val_loss.append(loss.item())

    loss = np.array(val_loss).mean()

    return loss


def train(feature, layers, lr, n_epochs, l2, train_dataloader, val_dataloader, device):
    """
    A function to train and validate the model over all epochs.

    Parameters
    ----------
    layers : dict
        Dict containing model architecture for distance and charge features
    lr : float
        Step size for adjusting parameters given computed error gradient
    n_epochs: int
        Number of epochs over training and validation sets
    train_dataloader : dict of torch.utils.data.DataLoader
        Dict of DataLoader objects for distance and charge training data
    val_dataloader : dict[torch.utils.data.DataLoader]
        Dict containing DataLoader object for distance and charge validation
    device : str
        Your device (usually 'cuda:0' for GPU or 'cpu' for CPU)

    Returns
    -------
    mlp_cls : dict
        Dict containing trained MLP models for distance and charge features
    train_loss_per_epoch : dict
        Dict containing training loss as a function of epoch number
    val_loss_per_epoch : dict
        Dictionary containing validation loss as a function of epoch number
    """

    mlp_cls = {}
    train_loss_per_epoch = {}
    val_loss_per_epoch = {}

    # Train MLP classifiers for each feature
    print("> Training MLP for " + feature + " features:\n")
    print("+-------+------------+----------+")
    print("| Epoch | Train-loss | Val-loss |")
    print("+-------+------------+----------+")

    val_losses = []
    train_losses = []
    model = MimoMLP(layers[feature]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    for epoch in range(n_epochs):
        # Train model on training data
        epoch_loss = gradient_step(
            model, train_dataloader[feature], optimizer, device=device
        )

        # Validate model on validation data
        val_loss = validate(model, val_dataloader[feature], device=device)

        # Record train and loss performance
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        print(f" {epoch:.4f}     {epoch_loss:.4f}      {val_loss:.4f}")

    mlp_cls[feature] = model
    train_loss_per_epoch[feature] = train_losses
    val_loss_per_epoch[feature] = val_losses

    return mlp_cls, train_loss_per_epoch, val_loss_per_epoch


def evaluate_model(feature, mlp_cls, test_dataloader, device, mimos):
    """
    A function to evaluate the model on test data.

    Parameters
    ----------
    mlp_cls : dict of torch.nn.Module
        Dict containing trained MLP classifiers
    test_dataloader (dict of torch.utils.data.DataLoader):
        Dict containing DataLoader object for the test data
    device : str
        Your device (usually 'cuda:0' for GPU or 'cpu' for CPU)
    mimos : list
        List of MIMO types, e.g. ['mc6', mc6s', 'mc6sa']

    Returns
    -------
    test loss : dict
        Dict containing average test loss
    y_true : dict
        Dict containing test data ground truth labels
    y_pred_proba : dict
        Dict containing softmax probabilities of the predicted labels
    y_pred : dict
        Dict containing prediction labels
    cms : dict
        Dict containing confusion matrices
    """

    y_pred_proba = {}
    y_pred = {}
    y_true = {}
    test_loss = {}
    cms = {}

    mlp_cls[feature].eval()
    y_true_feature_specific = np.empty((0, 3))
    y_pred_proba_feature_specific = np.empty((0, 3))
    y_pred_feature_specific = np.empty(0)
    losses = []
    with torch.no_grad():
        for batch in test_dataloader[feature]:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            logits = mlp_cls[feature](X)
            losses.append(torch.nn.functional.cross_entropy(logits, y))

            y_true_feature_specific = np.vstack(
                (y_true_feature_specific, y.detach().cpu().numpy())
            )
            y_proba = (
                torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            )
            y_pred_proba_feature_specific = np.vstack(
                (y_pred_proba_feature_specific, y_proba)
            )
            y_pred_feature_specific = np.hstack(
                (
                    y_pred_feature_specific,
                    logits.argmax(dim=1).detach().cpu().numpy(),
                )
            )

        y_pred_proba[feature] = y_pred_proba_feature_specific
        y_pred[feature] = y_pred_feature_specific
        y_true[feature] = y_true_feature_specific
        print(f"   > Mean test loss for {feature} MLP model: {np.array(losses).mean():.4f}")
        test_loss[feature] = np.array(losses).mean()

        y_true_feature_specific = np.argmax(y_true_feature_specific, axis=1)
        cm = confusion_matrix(y_true_feature_specific, y_pred_feature_specific)
        cms[feature] = pd.DataFrame(cm, mimos, mimos)

    return test_loss, y_true, y_pred_proba, y_pred, cms


class MimoMLP(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def preprocess_data(
    df_charge, df_dist, mimos, data_split_type, val_frac=0.6, test_frac=0.8
):
    """
    Split train and test based on the given test and validation fractions.

    Parameters
    ----------
    df_charge : dict
        Dict with mimo names as keys and charge data as values
    df_dist : dict
        Dictionary with mimo names as keys and distance data as values
    mimos : list of str
        List of mimo names.
    data_split_type : int
        Integers 1 (each traj as train/val/test) or 2 (split the entire dataset)
    val_frac : float, optional, default: 0.6
        Fraction of data to use for training (rest for val and test)
    test_frac : float, optional, default: 0.8
        Fraction of data to use for train and val (rest for testing)

    Returns
    -------
    data_split : dict
        Dict containing the train and test data for distance and charge features
    df_charge : dict
        Revised dict with mimo names as keys and charge data as values
    df_dist : dict
        Revised dict with mimo names as keys and distance data as values
    """

    class_assignment = {"mc6": 0, "mc6s": 1, "mc6sa": 2}
    features = ["dist", "charge"]

    # Create dictionaries for X (feature data) and y (class labels)
    X = {
        "dist": {mimo: np.array(df_dist[mimo]) for mimo in mimos},
        "charge": {mimo: np.array(df_charge[mimo]) for mimo in mimos},
    }

    y = {"dist": {}, "charge": {}}

    # Assign class labels for each mimo based on the class_assignment dict
    for mimo in mimos:
        y_aux = np.zeros((df_dist[mimo].shape[0], 3))
        y_aux[:, class_assignment[mimo]] = 1
        y["dist"][mimo] = y_aux

        y_aux = np.zeros((df_charge[mimo].shape[0], 3))
        y_aux[:, class_assignment[mimo]] = 1
        y["charge"][mimo] = y_aux

    data_split = {}
    if data_split_type == 1:
        X_train = {
            "dist": {mimo: np.empty((0, df_dist[mimo].shape[1])) for mimo in mimos},
            "charge": {mimo: np.empty((0, df_charge[mimo].shape[1])) for mimo in mimos},
        }

        X_val = {
            "dist": {mimo: np.empty((0, df_dist[mimo].shape[1])) for mimo in mimos},
            "charge": {mimo: np.empty((0, df_charge[mimo].shape[1])) for mimo in mimos},
        }

        X_test = {
            "dist": {mimo: np.empty((0, df_dist[mimo].shape[1])) for mimo in mimos},
            "charge": {mimo: np.empty((0, df_charge[mimo].shape[1])) for mimo in mimos},
        }

        y_train = {
            "dist": {mimo: np.empty((0, len(mimos))) for mimo in mimos},
            "charge": {mimo: np.empty((0, len(mimos))) for mimo in mimos},
        }

        y_val = {
            "dist": {mimo: np.empty((0, len(mimos))) for mimo in mimos},
            "charge": {mimo: np.empty((0, len(mimos))) for mimo in mimos},
        }

        y_test = {
            "dist": {mimo: np.empty((0, len(mimos))) for mimo in mimos},
            "charge": {mimo: np.empty((0, len(mimos))) for mimo in mimos},
        }
        # Split into train, val and test sets based on the val_frac and test_frac parameters
        for feature in features:
            # Split
            val_cutoff = int(val_frac * (X[feature]["mc6"].shape[0] / 8))
            test_cutoff = int(test_frac * (X[feature]["mc6"].shape[0] / 8))
            count = 0
            while count < int(X[feature]["mc6"].shape[0]):
                for mimo in mimos:
                    X_train[feature][mimo] = np.vstack(
                        (
                            X_train[feature][mimo],
                            X[feature][mimo][count : count + val_cutoff, :],
                        )
                    )
                    X_val[feature][mimo] = np.vstack(
                        (
                            X_val[feature][mimo],
                            X[feature][mimo][
                                count + val_cutoff : count + test_cutoff, :
                            ],
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
                    y_train[feature][mimo] = np.vstack(
                        (
                            y_train[feature][mimo],
                            y[feature][mimo][count : count + val_cutoff, :],
                        )
                    )
                    y_val[feature][mimo] = np.vstack(
                        (
                            y_val[feature][mimo],
                            y[feature][mimo][
                                count + val_cutoff : count + test_cutoff, :
                            ],
                        )
                    )
                    y_test[feature][mimo] = np.vstack(
                        (
                            y_test[feature][mimo],
                            y[feature][mimo][
                                count
                                + test_cutoff : count
                                + int(y[feature][mimo].shape[0] / 8),
                                :,
                            ],
                        )
                    )
                count += int(X[feature]["mc6"].shape[0] / 8)

            data_split[feature] = {
                "X_train": np.vstack([X_train[feature][mimo] for mimo in mimos]),
                "X_val": np.vstack([X_val[feature][mimo] for mimo in mimos]),
                "X_test": np.vstack([X_test[feature][mimo] for mimo in mimos]),
                "y_train": np.vstack([y_train[feature][mimo] for mimo in mimos]),
                "y_val": np.vstack([y_val[feature][mimo] for mimo in mimos]),
                "y_test": np.vstack([y_test[feature][mimo] for mimo in mimos]),
            }

            # Normalize
            x_scaler = StandardScaler()
            x_scaler.fit(data_split[feature]["X_train"])
            data_split[feature]["X_train"] = x_scaler.transform(
                data_split[feature]["X_train"]
            )
            data_split[feature]["X_val"] = x_scaler.transform(
                data_split[feature]["X_val"]
            )
            data_split[feature]["X_test"] = x_scaler.transform(
                data_split[feature]["X_test"]
            )

    elif data_split_type == 2:
        for feature in features:
            val_cutoff = int(val_frac * X[feature]["mc6"].shape[0])
            test_cutoff = int(test_frac * X[feature]["mc6"].shape[0])

            data_split[feature] = {
                "X_train": np.vstack(
                    [X[feature][mimo][0:val_cutoff, :] for mimo in mimos]
                ),
                "X_val": np.vstack(
                    [X[feature][mimo][val_cutoff:test_cutoff, :] for mimo in mimos]
                ),
                "X_test": np.vstack(
                    [X[feature][mimo][test_cutoff:, :] for mimo in mimos]
                ),
                "y_train": np.vstack(
                    [y[feature][mimo][0:val_cutoff, :] for mimo in mimos]
                ),
                "y_val": np.vstack(
                    [y[feature][mimo][val_cutoff:test_cutoff, :] for mimo in mimos]
                ),
                "y_test": np.vstack(
                    [y[feature][mimo][test_cutoff:, :] for mimo in mimos]
                ),
            }

            # Normalize data
            x_scaler = StandardScaler()
            x_scaler.fit(data_split[feature]["X_train"])
            data_split[feature]["X_train"] = x_scaler.transform(
                data_split[feature]["X_train"]
            )
            data_split[feature]["X_val"] = x_scaler.transform(
                data_split[feature]["X_val"]
            )
            data_split[feature]["X_test"] = x_scaler.transform(
                data_split[feature]["X_test"]
            )

    return data_split, df_dist, df_charge


def build_dataloaders(data_split):
    """
    A function to build the DataLoaders from the data split.

    Parameters
    ----------
    data split : dict
        Dict containing the train and testg data all features.

    Returns
    -------
    train_loader : dict
        Dict containing DataLoader object for the train data for all features
    val_loader : dict
        Dict containing DataLoader object for the val data for all features
    test_loader : dict
        Dict containing DataLoader object for the test data for all features
    """

    train_sets = {
        feature: MDDataset(
            data_split[feature]["X_train"], data_split[feature]["y_train"]
        )
        for feature in data_split.keys()
    }
    val_sets = {
        feature: MDDataset(data_split[feature]["X_val"], data_split[feature]["y_val"])
        for feature in data_split.keys()
    }
    test_sets = {
        feature: MDDataset(data_split[feature]["X_test"], data_split[feature]["y_test"])
        for feature in data_split.keys()
    }
    train_loader = {
        feature: torch.utils.data.DataLoader(
            train_sets[feature], batch_size=256, shuffle=True
        )
        for feature in data_split.keys()
    }
    val_loader = {
        feature: torch.utils.data.DataLoader(
            val_sets[feature], batch_size=256, shuffle=True
        )
        for feature in data_split.keys()
    }
    test_loader = {
        feature: torch.utils.data.DataLoader(
            test_sets[feature], batch_size=256, shuffle=True
        )
        for feature in data_split.keys()
    }

    return train_loader, val_loader, test_loader


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
    extensions = ["svg", "png"]
    for ext in extensions:
        plt.savefig(f"mlp_data.{ext}", bbox_inches="tight", format=ext, dpi=300)


def plot_train_val_losses(train_loss_per_epoch, val_loss_per_epoch):
    """
    Plot the train and validation losses as a function of epoch number.

    Parameters
    ----------
    train_loss_per_epoch : dict
        Dict of np.arrays containing train losses per epoch
    val_loss_per_epoch : dict
        Dict of np.arrays containing val losses per epoch
    """

    features = ["dist", "charge"]
    # Create a 1x2 subplot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Loop through features and plot the train and val losses on the same axes
    for i, feature in enumerate(features):
        ax[i].plot(train_loss_per_epoch[feature], color="red", label="training loss")
        ax[i].plot(val_loss_per_epoch[feature], color="blue", label="validation loss")
        ax[i].set_xlabel("epoch", weight="bold")
        ax[i].set_ylabel("loss", weight="bold")
        ax[i].set_title(f"{feature}", weight="bold")
        ax[i].legend(loc="best")

    # Apply tight layout and show the plot
    fig.tight_layout()
    extensions = ["svg", "png"]
    for ext in extensions:
        plt.savefig(f"mlp_loss_v_epoch.{ext}", bbox_inches="tight", format=ext, dpi=300)


def plot_roc_curve(y_true, y_pred_proba, mimos, data_set_type):
    """
    Plot the ROC curve for the test data of the charge and distance features.

    Parameters
    ----------
    y_true : dict
        Dict[np.arrays] containing ground truth labels of test data
    y_pred_proba : dict
        Dict[np.arrays] containing softmaxed probability predictions of the test
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
                label="%s ROC (AUC = %0.2f)" % (mimos[j], roc_auc[j]),
            )
        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.axis("square")
        plt.xlabel("false positive rate", weight="bold")
        plt.ylabel("true positive rate", weight="bold")
        plt.title(
            f"{data_set_type}_multi-class classification ROC for %s features" % feature, weight="bold"
        )
        plt.legend(loc="best")
        extensions = ["svg", "png"]
        for ext in extensions:
            plt.savefig(f"mlp_{data_set_type}_roc_" + feature + f".{ext}", bbox_inches="tight", format=ext, dpi=300)


def plot_confusion_matrices(cms, mimos):
    """
    Plot confusion matrices for distance and charge features.

    Parameters
    ----------
    cms : dict
        Dict containing confusion matrices for distance and charge features.
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
    extensions = ["svg", "png"]
    for ext in extensions:
        plt.savefig(f"mlp_cm.{ext}", bbox_inches="tight", format=ext, dpi=300)


def shap_analysis(mlp_cls, train_loader, test_loader, val_loader, df_dist, df_charge, mimos):
    """
    Plot SHAP dot plots for each mimichrome to identify importance

    Parameters
    ----------
    mlp_cls : dict
        Dict containing trained MLP classifiers for distance and charge features
    train_loader : dict
        Dict containing DataLoader object for the train data for distance
        and charge features
    test_loader : dict
        Dict containing DataLoader object for the test data for distance
        and charge features
    val_loader : dict
        Dict containing DataLoader object for the val data for distance
        and charge features
    df_dist : dict
        Dict of DataFrames containing distance data for each MIMO type.
    df_charge : dict
        Dict of DataFrames containing charge data for each MIMO type.
    mimos : list
        List of MIMO types, e.g. ['mc6', 'mc6s', 'mc6sa']
    """

    features = ["dist", "charge"]

    df = {"dist": df_dist, "charge": df_charge}

    shap_values = {}
    test = {}
    for i, feature in enumerate(features):
        # Take in the provided data loader(s) and retrieve the entire dataset
        # For now, function only takes in train_loader, but in the future, we
        # can add val_loader and test_loader, iterate through each loader and
        # append to all_data
        all_data = None
        for data, _ in train_loader[feature]:
            if all_data is None:
                all_data = data
            else:
                all_data = torch.cat((all_data, data), dim = 0)
        
        for data, _ in test_loader[feature]:
            all_data = torch.cat((all_data, data), dim = 0)

        for data, _ in val_loader[feature]:
            all_data = torch.cat((all_data, data), dim = 0)
        
        print(f"This is the shape of the entire dataset {all_data.shape}")
        
        # The background is a subset of all_data. It takes evenly spaced points
        # from all_data to recover a background comprising 100 data points.
        spacing = math.ceil(all_data.shape[0] / 100)
        background = all_data[::spacing]
        print(f"This is the shape of the background {background.shape}")
        # Use all_data to calculate SHAP values
        test[feature] = all_data
        # Deep Explainer:
        explainer = shap.DeepExplainer(mlp_cls[feature], background)
        shap_values[feature] = explainer.shap_values(test[feature])
        # Print out mean of absolute shap values for each charge/mimochrome
        # combination
        if feature == "charge":
            charge_mean_shap_vals = []
            for mimo_arr in shap_values[feature]:
                mimo_arr = np.transpose(mimo_arr)
                mimo_mean_shap_arr = []
                for charge_feature in mimo_arr:
                    mimo_mean_shap_arr.append(np.mean(np.abs(charge_feature)))
                charge_mean_shap_vals.append(mimo_mean_shap_arr)

            charge_mean_shap_vals = np.transpose(np.array(charge_mean_shap_vals))
            row_labels = df[feature][mimos[0]].columns.to_list()
            column_labels = mimos
            header_row = "," + ",".join(column_labels)
            charge_mean_shap_vals_with_labels = [f"{row_label}," + ",".join(map(str, row)) for row_label, row in zip(row_labels, charge_mean_shap_vals)]
            shap_values_csv_content = "\n".join([header_row] + charge_mean_shap_vals_with_labels)
            with open('mlp_charge_mean_abs_shap_values.csv', 'w') as file:
                file.write(shap_values_csv_content)

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
        extensions = ["svg", "png"]
        for ext in extensions:
            plt.savefig(f"mlp_shap_{mimos[i]}.{ext}", bbox_inches="tight", format=ext, dpi=300)


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
    ext = "png"
    file_name = "mlp_shap_combined"
    extensions = ["svg", "png"]
    for ext in extensions:
        plt.savefig(f"{file_name}.{ext}", bbox_inches="tight", format=ext, dpi=300)


class MDDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(np.array(X))  # store X as a pytorch Tensor
        self.y = torch.Tensor(np.array(y))  # store y as a pytorch Tensor
        self.len = len(self.X)  # number of samples in the data

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

def create_layers(input_size, n_neurons):
    return (torch.nn.Linear(input_size, n_neurons), torch.nn.ReLU(), 
            torch.nn.Linear(n_neurons, n_neurons), torch.nn.ReLU(), 
            torch.nn.Linear(n_neurons, n_neurons), torch.nn.ReLU(), 
            torch.nn.Linear(n_neurons, 3))


def run_mlp(data_split_type, include_esp, n_epochs, hyperparams):

    # Get datasets
    format_plots()
    mimos = ["mc6", "mc6s", "mc6sa"]
    data_loc = os.getcwd()
    df_charge, df_dist = load_data(mimos, include_esp, data_loc)
    plot_data(df_charge, df_dist, mimos)

    # Preprocess the data and split into train, validation, and test sets
    data_split, df_dist, df_charge = preprocess_data(df_charge, df_dist, mimos, data_split_type)
    # data_split, df_dist, df_charge = preprocess_data(df_charge, df_dist, mimos, data_split_type, val_frac=0.75, test_frac=0.875)

    # Build the train, validation, and test dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(data_split)

    # Get input sizes for each dataset and build model architectures
    n_dist = data_split['dist']['X_train'].shape[1]
    n_charge = data_split['charge']['X_train'].shape[1]
    
    # Distance hyperparameters
    lr_dist = hyperparams[0]["lr"]
    l2_dist = hyperparams[0]["l2"]
    n_neurons_dist = hyperparams[0]["n_neurons"]
    layers_dist = {'dist': create_layers(n_dist, n_neurons_dist)}
    mlp_cls_dist, train_loss_per_epoch_dist, val_loss_per_epoch_dist = train("dist", layers_dist, lr_dist, n_epochs, l2_dist, train_loader, val_loader, 'cpu')

    # Charge hyperparameters
    lr_charge = hyperparams[1]["lr"]
    l2_charge = hyperparams[1]["l2"]
    n_neurons_charge = hyperparams[1]["n_neurons"]
    layers_charge = {'charge': create_layers(n_charge, n_neurons_charge)}
    mlp_cls_charge, train_loss_per_epoch_charge, val_loss_per_epoch_charge = train("charge", layers_charge, lr_charge, n_epochs, l2_charge, train_loader, val_loader, 'cpu')

    # Save models
    torch.save(mlp_cls_dist['dist'].state_dict(), 'mlp_cls_dist.pth')
    torch.save(mlp_cls_charge['charge'].state_dict(), 'mlp_cls_charge.pth')


    # Evaluate model on the test data
    mlp_cls = {**mlp_cls_dist, **mlp_cls_charge}
    train_loss_per_epoch = {**train_loss_per_epoch_dist, **train_loss_per_epoch_charge}
    val_loss_per_epoch = {**val_loss_per_epoch_dist, **val_loss_per_epoch_charge}
    plot_train_val_losses(train_loss_per_epoch, val_loss_per_epoch)
    test_loss, y_true_dist, y_pred_proba_dist, y_pred, cms_dist = evaluate_model("dist", mlp_cls, test_loader, 'cpu', mimos)
    test_loss, y_true_charge, y_pred_proba_charge, y_pred, cms_charge = evaluate_model("charge", mlp_cls, test_loader, 'cpu', mimos)
    # Combine values back together
    y_true = {**y_true_dist, **y_true_charge}
    y_pred_proba = {**y_pred_proba_dist, **y_pred_proba_charge}
    cms = {**cms_dist, **cms_charge}

    # Plot ROC-AUC curves, confusion matrices and SHAP dot plots
    data_set_type = "Test"
    plot_roc_curve(y_true, y_pred_proba, mimos, data_set_type)
    plot_confusion_matrices(cms, mimos)
    shap_analysis(mlp_cls, train_loader, test_loader, val_loader, df_dist, df_charge, mimos)

    # Evaluate the model on the training data
    train_loss, y_true_train_dist, y_pred_proba_train_dist, y_pred_train, cms_train_dist = evaluate_model("dist", mlp_cls, train_loader, 'cpu', mimos)
    train_loss, y_true_train_charge, y_pred_proba_train_charge, y_pred_train, cms_train_charge = evaluate_model("charge", mlp_cls, train_loader, 'cpu', mimos)
    y_true_train = {**y_true_train_dist, **y_true_train_charge}
    y_pred_proba_train = {**y_pred_proba_train_dist, **y_pred_proba_train_charge}

    # Plot ROC-AUC curves for training data
    data_set_type = "Train"
    plot_roc_curve(y_true_train, y_pred_proba_train, mimos, data_set_type)


    # Clean up the newly generated files
    mlp_dir = "MLP"
    # Create the "rf/" directory if it doesn't exist
    if not os.path.exists(mlp_dir):
        os.makedirs(mlp_dir)

    # Move all files starting with "rf_" into the "rf/" directory
    for file in os.listdir():
        if file.startswith("mlp_"):
            shutil.move(file, os.path.join(mlp_dir, file))


def train_with_hyperparameters(trial, feature, train_loader, val_loader, n_dist, n_charge):
    # Hyperparameters
    n_epochs = 200
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)  
    l2 = trial.suggest_float('l2', 1e-6, 1e-2, log=True)  
    n_layers = trial.suggest_int('n_layers', 2, 4)  # Number of hidden layers
    n_neurons = trial.suggest_int('n_neurons', 32, 256)  # Neurons per layer
    
    n_input = {'dist': n_dist, 'charge': n_charge}
    layers_list = [torch.nn.Linear(n_input[feature], n_neurons), torch.nn.ReLU()]
    for _ in range(n_layers):
        layers_list.extend([torch.nn.Linear(n_neurons, n_neurons), torch.nn.ReLU()])
    layers_list.append(torch.nn.Linear(n_neurons, 3))  # Output layer

    layers = {feature: tuple(layers_list)}

    mlp_cls, train_loss_per_epoch, val_loss_per_epoch = train(feature, layers, lr, n_epochs, l2, train_loader, val_loader, 'cpu')

    return val_loss_per_epoch[feature][-1]


def optuna_mlp(data_split_type, include_esp, n_trials, out_name):
    # Get datasets
    features = ["dist", "charge"]
    mimos = ["mc6", "mc6s", "mc6sa"]
    data_loc = os.getcwd()
    df_charge, df_dist = load_data(mimos, include_esp, data_loc)

    # Preprocess the data and split into train, validation, and test sets
    data_split, df_dist, df_charge = preprocess_data(df_charge, df_dist, mimos, data_split_type)
    # data_split, df_dist, df_charge = preprocess_data(df_charge, df_dist, mimos, data_split_type, val_frac=0.75, test_frac=0.875)

    # Build the train, validation, and test dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(data_split)

    # Get input sizes for each dataset and build model architectures
    n_dist = data_split['dist']['X_train'].shape[1]
    n_charge = data_split['charge']['X_train'].shape[1]

    filename = f"mlp_hyperopt_{out_name}.txt"
    with open(filename, 'w') as file:
        for feature in features:
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: train_with_hyperparameters(trial, feature, train_loader, val_loader, n_dist, n_charge), n_trials)
            
            best_params = study.best_params
            best_loss = study.best_value

            print(f"Feature: {feature}", file=file)
            print(f"Best parameters: {best_params}", file=file)
            print(f"Best validation loss: {best_loss}\n", file=file)


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
    # Setting up argument parser
    import argparse
    
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('--n_trials', type=int, default=500,
                        help='The number of trials for optuna_mlp')
    parser.add_argument('--data_split_type', type=int, default=1, 
                        help='Type of data split to use. Default is 1.')
    parser.add_argument('--include_esp', type=str, default='False',
                        help='Whether to include ESP features.')
    parser.add_argument('--out_name', type=str, default='MLP',
                        help='Adds distinguishing extension to out file.')
    args = parser.parse_args()

    optuna_mlp(args.data_split_type, args.include_esp, args.n_trials, args.out_name)
