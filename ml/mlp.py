import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sn
from statistics import mean
import torch
from pathlib import Path

def gradient_step(model, dataloader, optimizer, device):
    
    '''
    A function train on the entire dataset for one epoch .
    
    Args: 
        model (torch.nn.Module): your model from before 
        dataloader (torch.utils.data.DataLoader): DataLoader object for the train data
        optimizer (torch.optim.Optimizer(()): optimizer object to interface gradient calculation and optimization 
        device (str): Your device (usually 'cuda:0' for your GPU)
        
    Returns: 
        float: loss averaged over all the batches 
    
    '''

    epoch_loss = []
    model.train() # Set model to training mode 
    
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

    return np.array(epoch_loss).mean()


def validate(model, dataloader, device):
    
    '''
    A function validate on the validation dataset for one epoch .
    
    Args: 
        model (torch.nn.Module): your model for before 
        dataloader (torch.utils.data.DataLoader): DataLoader object for the validation data
        device (str): Your device (usually 'cuda:0' for your GPU)
        
    Returns: 
        float: loss averaged over all the batches 
    
    '''
    
    val_loss = []
    model.eval() # Set model to evaluation mode 
    with torch.no_grad():    
        for batch in dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            
            # validate your model on each batch here 
            y_pred = model(X)

            loss = torch.nn.functional.cross_entropy(y_pred.squeeze(), y) # fill in loss here
            val_loss.append(loss.item())
            
    return np.array(val_loss).mean()

def train(layers, learning_rate, n_epochs, train_dataloader, val_dataloader, device):
    '''
    A function validate on the validation dataset for one epoch .
    
    Args: 
        layers: model architecture
        dataloader (torch.utils.data.DataLoader): DataLoader object for the validation data
        device (str): Your device (usually 'cuda:0' for your GPU)
        
    Returns: 
        float: loss averaged over all the batches 
    
    '''

    print("epoch", "train loss", "validation loss")
    val_loss_curve = []
    train_loss_curve = []
    model = MimoMLP(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for epoch in range(n_epochs):
        
        # Compute train your model on training data
        epoch_loss = gradient_step(model, train_dataloader, optimizer,  device=device)
        
        # Validate your on validation data 
        # no validation for now
        val_loss = 0 #validate(model, val_dataloader, device=device) 
        
        # Record train and loss performance 
        train_loss_curve.append(epoch_loss)
        val_loss_curve.append(val_loss)
        
        print(epoch, epoch_loss, val_loss)
    return model

def evaluate_model(model, loader, device):
    model.eval() 
    y_true = np.empty((0,3))
    y_pred_soft = np.empty((0,3))
    y_pred_hard = np.empty(0)
    losses = []
    with torch.no_grad():    
        for batch in loader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            losses.append(torch.nn.functional.cross_entropy(logits, y)) 
            
            y_true = np.vstack((y_true, y.detach().cpu().numpy()))
            y_soft = torch.nn.functional.softmax(logits).detach().cpu().numpy()
            y_pred_soft = np.vstack((y_pred_soft, y_soft))
            y_pred_hard = np.hstack((y_pred_hard, logits.argmax(dim=1).detach().cpu().numpy()))
            
    return np.array(losses).mean(), y_true, y_pred_soft, y_pred_hard
    
            
            
    

class MimoMLP(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

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
        df_charge[mimo] = pd.read_csv(f"{data_loc}/{mimo}_charges.csv")
        df_dist[mimo] = pd.read_csv(f"{data_loc}/{mimo}_pairwise_distances.csv")

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
        y_aux = np.zeros((df_dist[mimo].shape[0], 3))
        y_aux[:, class_assignment[mimo]] = 1
        y["dist"][mimo] = y_aux

        y_aux = np.zeros((df_charge[mimo].shape[0], 3))
        y_aux[:, class_assignment[mimo]] = 1
        y["charge"][mimo] = y_aux

    data_split = {}

    # Split data into training and testing sets based on the test_frac parameter
    for feature in features:
        n_cutoff = int(test_frac * X[feature]["mc6"].shape[0])

        data_split[feature] = {
            "X_train": np.vstack([X[feature][mimo][0:n_cutoff, :] for mimo in mimos]),
            "X_test": np.vstack([X[feature][mimo][n_cutoff:, :] for mimo in mimos]),
            "y_train": np.vstack([y[feature][mimo][0:n_cutoff, :] for mimo in mimos]),
            "y_test": np.vstack([y[feature][mimo][n_cutoff:, :] for mimo in mimos]),
        }

    return data_split


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
        self.len=len(self.X)                # number of samples in the data 

    def __getitem__(self, index):
        return self.X[index], self.y[index] 

    def __len__(self):
        return self.len

if __name__ == "__main__":
    # Get datasets
    format_plots()
    mimos = ["mc6", "mc6s", "mc6sa"]
    data_loc = str(Path(__file__).resolve().parent) + "/data" #input("   > Where are your data files located? ")
    df_charge, df_dist = load_data(mimos, data_loc)
    plot_data(df_charge, df_dist, mimos)

    # Preprocess the data and split into train and test sets
    data_split = preprocess_data(df_charge, df_dist, mimos)

    train_sets = {feature : MDDataset(data_split[feature]['X_train'], data_split[feature]['y_train']) for feature in data_split.keys()}
    test_sets = {feature : MDDataset(data_split[feature]['X_test'], data_split[feature]['y_test']) for feature in data_split.keys()}
    train_loaders = {feature : torch.utils.data.DataLoader(train_sets[feature], batch_size=256) for feature in data_split.keys()}
    test_loaders = {feature : torch.utils.data.DataLoader(test_sets[feature], batch_size=256) for feature in data_split.keys()}
    
    # dist 
    n_dist = len(train_sets['dist'][0][0])
    n_label = len(train_sets['dist'][0][1])
    layers = (torch.nn.Linear(n_dist, 128), torch.nn.ReLU(), 
              torch.nn.Linear(128, 128), torch.nn.ReLU(), 
              torch.nn.Linear(128, 128), torch.nn.ReLU(), 
              torch.nn.Linear(128, 3))
    
    dist_mlp_model=train(layers, 1e-3, 100, train_loaders['dist'], None, 'cpu')
    
    test_loss, y_true, y_pred_soft, y_pred_hard = evaluate_model(dist_mlp_model, test_loaders['dist'], 'cpu')
    
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sn
from statistics import mean
import torch

def gradient_step(model, dataloader, optimizer, device):
    
    '''
    A function train on the entire dataset for one epoch .
    
    Args: 
        model (torch.nn.Module): your model from before 
        dataloader (torch.utils.data.DataLoader): DataLoader object for the train data
        optimizer (torch.optim.Optimizer(()): optimizer object to interface gradient calculation and optimization 
        device (str): Your device (usually 'cuda:0' for your GPU)
        
    Returns: 
        float: loss averaged over all the batches 
    
    '''

    epoch_loss = []
    model.train() # Set model to training mode 
    
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

    return np.array(epoch_loss).mean()


def validate(model, dataloader, device):
    
    '''
    A function validate on the validation dataset for one epoch .
    
    Args: 
        model (torch.nn.Module): your model for before 
        dataloader (torch.utils.data.DataLoader): DataLoader object for the validation data
        device (str): Your device (usually 'cuda:0' for your GPU)
        
    Returns: 
        float: loss averaged over all the batches 
    
    '''
    
    val_loss = []
    model.eval() # Set model to evaluation mode 
    with torch.no_grad():    
        for batch in dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            
            # validate your model on each batch here 
            y_pred = model(X)

            loss = torch.nn.functional.cross_entropy(y_pred.squeeze(), y) # fill in loss here
            val_loss.append(loss.item())
            
    return np.array(val_loss).mean()

def train(layers, learning_rate, n_epochs, train_dataloader, val_dataloader, device):
    '''
    A function validate on the validation dataset for one epoch .
    
    Args: 
        layers: model architecture
        dataloader (torch.utils.data.DataLoader): DataLoader object for the validation data
        device (str): Your device (usually 'cuda:0' for your GPU)
        
    Returns: 
        float: loss averaged over all the batches 
    
    '''

    print("epoch", "train loss", "validation loss")
    val_loss_curve = []
    train_loss_curve = []
    model = MimoMLP(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for epoch in range(n_epochs):
        
        # Compute train your model on training data
        epoch_loss = gradient_step(model, train_dataloader, optimizer,  device=device)
        
        # Validate your on validation data 
        # no validation for now
        val_loss = 0 #validate(model, val_dataloader, device=device) 
        
        # Record train and loss performance 
        train_loss_curve.append(epoch_loss)
        val_loss_curve.append(val_loss)
        
        print(epoch, epoch_loss, val_loss)
    return model

def evaluate_model(model, loader, device):
    model.eval() 
    y_true = np.empty((0,3))
    y_pred_soft = np.empty((0,3))
    y_pred_hard = np.empty(0)
    losses = []
    with torch.no_grad():    
        for batch in loader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            losses.append(torch.nn.functional.cross_entropy(logits, y)) 
            
            y_true = np.vstack((y_true, y.detach().cpu().numpy()))
            y_soft = torch.nn.functional.softmax(logits).detach().cpu().numpy()
            y_pred_soft = np.vstack((y_pred_soft, y_soft))
            y_pred_hard = np.hstack((y_pred_hard, logits.argmax(dim=1).detach().cpu().numpy()))
            
    return np.array(losses).mean(), y_true, y_pred_soft, y_pred_hard
    
            
            
    

class MimoMLP(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

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
        df_charge[mimo] = pd.read_csv(f"{data_loc}/{mimo}_charges.csv")
        df_dist[mimo] = pd.read_csv(f"{data_loc}/{mimo}_pairwise_distances.csv")

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
        y_aux = np.zeros((df_dist[mimo].shape[0], 3))
        y_aux[:, class_assignment[mimo]] = 1
        y["dist"][mimo] = y_aux

        y_aux = np.zeros((df_charge[mimo].shape[0], 3))
        y_aux[:, class_assignment[mimo]] = 1
        y["charge"][mimo] = y_aux

    data_split = {}

    # Split data into training and testing sets based on the test_frac parameter
    for feature in features:
        n_cutoff = int(test_frac * X[feature]["mc6"].shape[0])

        data_split[feature] = {
            "X_train": np.vstack([X[feature][mimo][0:n_cutoff, :] for mimo in mimos]),
            "X_test": np.vstack([X[feature][mimo][n_cutoff:, :] for mimo in mimos]),
            "y_train": np.vstack([y[feature][mimo][0:n_cutoff, :] for mimo in mimos]),
            "y_test": np.vstack([y[feature][mimo][n_cutoff:, :] for mimo in mimos]),
        }

    return data_split


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
        self.len=len(self.X)                # number of samples in the data 

    def __getitem__(self, index):
        return self.X[index], self.y[index] 

    def __len__(self):
        return self.len

if __name__ == "__main__":
    # Get datasets
    format_plots()
    mimos = ["mc6", "mc6s", "mc6sa"]
    data_loc = "molecuLearn/ml/data" #__file__#input("   > Where are your data files located? ")
    df_charge, df_dist = load_data(mimos, data_loc)
    plot_data(df_charge, df_dist, mimos)

    # Preprocess the data and split into train and test sets
    data_split = preprocess_data(df_charge, df_dist, mimos)

    train_sets = {feature : MDDataset(data_split[feature]['X_train'], data_split[feature]['y_train']) for feature in data_split.keys()}
    test_sets = {feature : MDDataset(data_split[feature]['X_test'], data_split[feature]['y_test']) for feature in data_split.keys()}
    train_loaders = {feature : torch.utils.data.DataLoader(train_sets[feature], batch_size=32) for feature in data_split.keys()}
    test_loaders = {feature : torch.utils.data.DataLoader(test_sets[feature], batch_size=32) for feature in data_split.keys()}
    
    # dist 
    n_dist = len(train_sets['dist'][0][0])
    n_label = len(train_sets['dist'][0][1])
    layers = (torch.nn.Linear(n_dist, 128), torch.nn.ReLU(), 
              torch.nn.Linear(128, 128), torch.nn.ReLU(), 
              torch.nn.Linear(128, 128), torch.nn.ReLU(), 
              torch.nn.Linear(128, 3))
    
    dist_mlp_model=train(layers, 1e-3, 100, train_loaders['dist'], None, 'cpu')
    
    test_loss, y_true, y_pred_soft, y_pred_hard = evaluate_model(dist_mlp_model, test_loaders['dist'], 'cpu')
    
