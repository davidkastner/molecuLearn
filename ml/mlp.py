"""Functions for the multi-layer perceptron classifier."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.utils import shuffle
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sn
from statistics import mean
import torch
from itertools import cycle
import shap

def load_data(mimos, data_loc):
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
        # Load charge and distance data from CSV files and store them in dictionaries
        df_charge[mimo] = pd.read_csv(f"{data_loc}/{mimo}_esp.csv")
        df_charge[mimo] = df_charge[mimo].drop(columns=["replicate"])
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


def train(layers, learning_rate, n_epochs, train_dataloader, val_dataloader, device):
    """
    A function to train and validate the model over all epochs.

    Parameters
    ----------
    layers : dict
        Dict containing model architecture for distance and charge features
    learning_rate : float
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
    features = ["dist", "charge"]

    # Train MLP classifiers for each feature
    for feature in features:
        print("Training MLP for " + feature + " features:")
        print("epoch", "train-loss", "validation-loss")
        val_losses = []
        train_losses = []
        model = MimoMLP(layers[feature]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
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

            print(epoch, epoch_loss, val_loss)

        mlp_cls[feature] = model
        train_loss_per_epoch[feature] = train_losses
        val_loss_per_epoch[feature] = val_losses

    return mlp_cls, train_loss_per_epoch, val_loss_per_epoch


def evaluate_model(mlp_cls, test_dataloader, device, mimos):
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
    features = ["dist", "charge"]
    y_pred_proba = {}
    y_pred = {}
    y_true = {}
    test_loss = {}
    cms = {}

    for feature in features:
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
        print(f"Mean test loss for {feature} MLP model: {np.array(losses).mean():.4f}")
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
        Int 1 (each trajectory into train/val/test) or 2 (split the entire)
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

    # Drop distances between Glu3, Aib20, or Aib23
    for mimo in mimos:
        if mimo == "mc6":
            df_dist[mimo] = df_dist[mimo].loc[
                :, ~df_dist[mimo].columns.str.contains("GLU3|GLN20|SER23")
            ]
        elif mimo == "mc6s":
            df_dist[mimo] = df_dist[mimo].loc[
                :, ~df_dist[mimo].columns.str.contains("LEU3|GLN20|SER23")
            ]
        elif mimo == "mc6sa":
            df_dist[mimo] = df_dist[mimo].loc[
                :, ~df_dist[mimo].columns.str.contains("LEU3|AIB20|AIB23")
            ]

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
    plt.savefig("mlp_data.png", bbox_inches="tight", format="png", dpi=300)
    plt.show()


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
    plt.savefig("mlp_loss_v_epoch.png", bbox_inches="tight", format="png", dpi=300)
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, mimos):
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
            "mlp_roc_" + feature + ".png", bbox_inches="tight", format="png", dpi=300
        )
        plt.show()


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
    plt.savefig("mlp_cm.png", bbox_inches="tight", format="png", dpi=300)
    plt.show()


def shap_analysis(mlp_cls, test_loader, df_dist, df_charge, mimos):
    """
    Plot SHAP dot plots for each mimichrome to identify importance

    Parameters
    ----------
    mlp_cls : dict
        Dict containing trained MLP classifiers for distance and charge features
    test_loader : dict
        Dict containing DataLoader object for the test data for distance
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
        # Load in a random batch from the test dataloader and interpret predictions for 156 data points
        batch = next(iter(test_loader[feature]))
        data, _ = batch
        # Define the first 100 datapoints as the background used as reference when calculating SHAP values
        background = data[:100]
        test[feature] = data[100:]
        explainer = shap.DeepExplainer(mlp_cls[feature], background)
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
            f"mlp_shap_{mimos[i]}.png", bbox_inches="tight", format="png", dpi=300
        )
        plt.show()

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
    plt.savefig(f"{file_name}.{ext}", bbox_inches="tight", format=ext, dpi=300)
    plt.show()


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


class MDDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(np.array(X))  # store X as a pytorch Tensor
        self.y = torch.Tensor(np.array(y))  # store y as a pytorch Tensor
        self.len = len(self.X)  # number of samples in the data

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


if __name__ == "__main__":
    # Get datasets
    format_plots()
    mimos = ["mc6", "mc6s", "mc6sa"]
    data_loc = input("   > Where are your data files located? ")
    df_charge, df_dist = load_data(mimos, data_loc)
    plot_data(df_charge, df_dist, mimos)

    # Preprocess the data and split into train, validation, and test sets
    data_split, df_dist, df_charge = preprocess_data(df_charge, df_dist, mimos, 1)

    # Build the train, validation, and test dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(data_split)

    # Get input sizes for each dataset and build model architectures
    n_dist = data_split["dist"]["X_train"].shape[1]
    n_charge = data_split["charge"]["X_train"].shape[1]
    layers = {
        "dist": (
            torch.nn.Linear(n_dist, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3),
        ),
        "charge": (
            torch.nn.Linear(n_charge, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3),
        ),
    }

    # Train model on training and validation data
    mlp_cls, train_loss_per_epoch, val_loss_per_epoch = train(
        layers, 1e-3, 100, train_loader, val_loader, "cpu"
    )
    plot_train_val_losses(train_loss_per_epoch, val_loss_per_epoch)
    # Evaluate model on test data
    test_loss, y_true, y_pred_proba, y_pred, cms = evaluate_model(
        mlp_cls, test_loader, "cpu", mimos
    )
    # Plot ROC-AUC curves, confusion matrices and SHAP dot plots
    plot_roc_curve(y_true, y_pred_proba, mimos)
    plot_confusion_matrices(cms, mimos)
    shap_analysis(mlp_cls, test_loader, df_dist, df_charge, mimos)
