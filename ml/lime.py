#from mlp import load_data
import numpy as np
import torch

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from itertools import combinations
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression, Lasso
from tqdm import tqdm

def evaluate_model(model, inputs, output="logits"):
    if isinstance(model, torch.nn.Module):
        # For PyTorch model
        model.eval()
        with torch.no_grad():
            inputs = torch.Tensor(inputs)
            logits = model(inputs)
            if output == "logits":
                return torch.nn.functional.log_softmax(logits, dim=-1).numpy()
            else:
                return torch.nn.functional.softmax(logits, dim=-1).numpy()
        
    elif isinstance(model, RandomForestClassifier):
        # For scikit-learn random forest model
        if output == "logits":
            return np.log(model.predict(inputs))
        else:
            return model.predict(inputs)

    else:
        raise ValueError("Invalid model type. Supported types: PyTorch nn.Module, scikit-learn RandomForestClassifier")


def perturb_data(x, default_val, n):
    """_summary_

    Args:
        x (1d-np.array): feature vector
        default (float): default value for perturbation
        n (int): number of features perturbed
        
    Outputs:
        x_pert (2d-np.array): perturbed feature vectors stacked into matrix (row-wise) 
        x_bin (2d-np.array): binary representation of perturbed entries
    """
    combs = list(combinations(range(len(x)),n))
    m = len(combs)
    x_pert = np.zeros((m, len(x)))
    x_bin = np.zeros((m, len(x)))
    for (i, idcs) in enumerate(combs):
        x_pert[i,:] = x
        x_pert[i,idcs] = default_val
        x_bin[i,idcs] = 1 
        
    return x_pert, x_bin


def lime(perturb_data, data, model, lin_model, n_important = 1):
    """_summary_
    
    Args: 
        perturb_data => fxn: x -> perturbed data point (sets entries to zero, mean, etc...)
        data => 2d-np.array -> data[i,:] = ith data point
        model => full order model (MLP or RF)
        lin_model (sklearn.linear_model)
    """
    ws = []
    important_features = []
    important_features_per_label = []
    n_frames = data.shape[0]
    for i in tqdm(range(n_frames)):
        x = data[i,:]
        label = evaluate_model(model, x).argmax()
        x_pert, x_bin = perturb_data(x)
        if isinstance(lin_model, LogisticRegression):
            y_pert = evaluate_model(model, x_pert, output="probs").argmax(axis=1)
        else:
            y_pert = evaluate_model(model, x_pert, output="logits")
        
        reg = lin_model.fit(x_bin, y_pert)
        ws.append(reg.coef_)
        feature_importance = ws[-1][label,:]
        important_features.append(np.argpartition(feature_importance, n_important-1)[0:n_important])
        
    return important_features, ws

