#from mlp import load_data
import numpy as np
import torch

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from itertools import combinations

def evaluate_model(model, inputs):
    if isinstance(model, torch.nn.Module):
        # For PyTorch model
        model.eval()
        with torch.no_grad():
            return np.log(model(inputs).numpy())
    elif isinstance(model, RandomForestClassifier):
        # For scikit-learn random forest model
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


def lime(perturb_data, data, model, lin_model):
    """_summary_
    
    Args: 
        perturb_data => fxn: x -> perturbed data point (sets entries to zero, mean, etc...)
        data => 2d-np.array -> data[i,:] = ith data point
        model => full order model (MLP or RF)
        lin_model (sklearn.linear_model)
    """
    ws = []
    for i in range(data.shape[0]):
        x = data[i,:]
        x_pert, x_bin = perturb_data(x)
        y_pert = evaluate_model(model, x_pert)
        reg = lin_model.fit(x_bin, y_pert)
        ws.append(reg.coef_)
        
    return ws

