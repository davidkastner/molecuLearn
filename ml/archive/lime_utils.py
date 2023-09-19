"""Functions used by line in RF and MLP."""

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import lime
import numpy as np
import torch
import sklearn


def lime_analysis(data, model, class_names, feature_names):
    """
    Perform the LIME analysis.

    Parameters
    ----------
    data : np.array
        Input data, each row is a featurization of the associated data point.
    model :
        classifier that takes in feature vector or matrix and returns logits for
        the associated features
    class_names : list
        List of strings carrying the class names in the same order as the
        model outputs is encoded.
    feature_names : list of strings
        names of all model features

    Returns
    -------
    important_features: list
        List of feature importance ranking for each frame.
        Each entry is a list of tuples (feature, importance score) ranked in
        descending order according to abs value of the importance score.
    avg_important_features: list
        List of importance scores computed by averaging the absolute value
        of lime importance scores over all data points
    avg_important_features: Dict
        Dictionary containing the average importances of different features but
        for separetly for each label as predicted by the model.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        data,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
    )
    y_preds = model(data).argmax(axis=1)
    important_features = extract_imporant_features(explainer, model, data)

    return (
        important_features,
        y_preds,
        get_avg_importance(important_features),
        get_avg_importance_by_label(important_features, class_names, y_preds),
    )


def extract_imporant_features(explainer, model, data):
    """
    Extract important features.

    Parameters
    ----------
    explainer :
        lime explainer class object for the given model and data (see https://github.com/marcotcr/lime)
    model :
        classifier that takes in feature vector or matrix and returns logits for
        the associated features
    data : np.array
        Input data, each row is a featurization of the associated data point.

    Returns
    -------
    important_features: list
        List of feature importance ranking for each frame.
        Each entry is a list of features ranked in descending order according
        to their importance.

    """
    important_features = []
    n_features = data.shape[1]
    for i in tqdm(range(data.shape[0])):
        x = data[i, :]
        exp = explainer.explain_instance(x, model, num_features=n_features)
        important_features.append(exp.as_map()[1])
    return important_features


def get_score(scores, idx, trafo=lambda x: x):
    """
    Get scores.

    Parameters
    ----------
    scores : list
        List of (feature, importance score) tuples.
    idx : int
        Index of feature whose score shall be extracted.

    trafo : function, optional
        Transformation that shall be applied upon extraction, often abs value.
        The default is lambda x : x.

    Returns
    -------
    list
        extracted and transformed scores for given feature

    """
    return [trafo(s[1]) for s in scores if s[0] == idx]


def get_avg_importance(important_features):
    """
    Get the average importance.

    Parameters
    ----------
    important_features :  list
        List of (feature, importance score) tuples.

    Returns
    -------
    list
        Mean importance score for each feature obtained by averaging
        the absolute value over all data points.

    """
    n_features = len(important_features[0])
    return [
        np.mean([get_score(f, i, lambda x: abs(x)) for f in important_features])
        for i in range(n_features)
    ]


def get_avg_importance_by_label(important_features, class_names, y_preds):
    """
    Get the average importance by label.

    Parameters
    ----------
    important_features :  list
        List of lists carrying (feature, importance score) tuples for each data
        point.
    class_names : List
        List of strings carrying the class labels in an order consistent with
        y_preds.
    y_preds : list/np.array
        List that carries the model prediction in form of an int index for the
        respective data points that important_features were computed for.

    Returns
    -------
    avg_importance : Dict
        For each class, this dictionary carries the mean importance score for each feature
        obtained by averaging the absolute value of importance scores over all
        data points that were labelled to belong to this class by the model.

    """
    avg_importance = {}
    n_features = len(important_features[0])
    for k, label in enumerate(class_names):
        scores = [
            np.mean(
                [
                    get_score(f, i, lambda x: abs(x))
                    for (j, f) in enumerate(important_features)
                    if y_preds[j] == k
                ]
            )
            for i in range(n_features)
        ]
        avg_importance[label] = scores
    return avg_importance


def get_nth_mif_by_label(important_features, y_preds, n, n_labels):
    """
    Get the important features with labels.

    Parameters
    ----------
    important_features : list
        List of lists carrying (feature, importance score) tuples for each data
        point.
    y_preds : list/np.array
        List that carries the model prediction in form of an int index for the
        respective data points that important_features were computed for.
    n : int
        Index of which rank of features shall be extracted.

    Returns
    -------
    features_by_label : dict
        For each class, this dictionary contains the label of the nth most important
        feature for all data points that were labelled to belong to this class
        by the model.

    """
    features_by_label = [
        [f[n][0] for (i, f) in enumerate(important_features) if y_preds[i] == k]
        for k in range(n_labels)
    ]
    return features_by_label


def get_nth_mif(important_features, n):
    """
    Get the nth important feature.

    Parameters
    ----------
    important_features : list
        List of lists carrying (feature, importance score) tuples for each data
        point.
    y_preds : list/np.array
        List that carries the model prediction in form of an int index for the
        respective data points that important_features were computed for.
    n : int
        Index of which rank of features shall be extracted.

    Returns
    -------
    list
        List of the nth most important feature for all data points.

    """
    return [f[n][0] for f in important_features]


def plot_hists(n_max, important_features, class_names, y_preds, **kwargs):
    """
    Plotting function to generate histogram plot.

    Parameters
    ----------
    n_max : Int
        A plot is generated for the first n_max most important features.
    important_features : list
        List of lists carrying (feature, importance score) tuples for each data
        point.
    class_names : list
        list of class names
    y_preds : list/np.array
        List that carries the model prediction in form of an int index for the
        respective data points that important_features were computed for.
    **kwargs :
        savepath - save path but without file extension.
        bin_labels - save labels in case the individual bins shall be labelled

    Returns
    -------
    Creates two figures. Each containing n_max different histograms.
    The nth plot (from the top) shows the frequency with which a given feature
    was the nth most important for classification.
    The figures are distinct as one shows the histograms without resolution
    of predicted labels and the other one with.

    """
    savepath = kwargs["savepath"] if "savepath" in kwargs.keys() else None
    bin_labels = kwargs["bin_labels"] if "bin_labels" in kwargs.keys() else None

    n_features = len(important_features[0])
    n_labels = len(class_names)
    fig, axs = plt.subplots(n_max)
    fig.set_figheight(n_max * 2)
    fig.set_figwidth(5)
    for i in range(n_max):
        features = get_nth_mif_by_label(important_features, y_preds, i, n_labels)
        bins = np.linspace(-0.5, n_features - 0.5, n_features + 1)
        axs[i].hist(features, bins=bins, label=class_names, stacked=True)
        if i == n_max - 1:
            if bin_labels != None:
                axs[i].set_xticks(range(n_features))
                axs[i].set_xticklabels(bin_labels, rotation=90)
        else:
            if bin_labels != None:
                axs[i].set_xticks(range(n_features))
                axs[i].set_xticklabels(["" for i in range(n_features)])
        axs[i].set_ylabel("number of frames")
        if i == 0:
            axs[i].legend()

    fig.tight_layout()
    if savepath != None:
        fig.savefig(savepath + "_importance_by_frame_and_label.png")

    fig, axs = plt.subplots(n_max)
    fig.set_figheight(n_max * 2)
    fig.set_figwidth(5)
    for i in range(n_max):
        features = get_nth_mif(important_features, i)
        bins = np.linspace(-0.5, n_features - 0.5, n_features + 1)
        axs[i].hist(features, bins=bins)
        if i == n_max - 1:
            if bin_labels != None:
                axs[i].set_xticks(range(n_features))
                axs[i].set_xticklabels(bin_labels, rotation=90)
        else:
            if bin_labels != None:
                axs[i].set_xticks(range(n_features))
                axs[i].set_xticklabels(["" for i in range(n_features)])
        axs[i].set_ylabel("number of frames")

    fig.tight_layout()
    if savepath != None:
        fig.savefig(savepath + "_importance_by_frame.png")

    pass


def plot_importance_ranking(avg_scores, feature_names, n_max, **kwargs):
    """
    Plotting function for the importance.

    Parameters
    ----------
    avg_scores : list
        List of average importance rankings for each feature (in order of the features)
    feature_names : list
        List of feature names
    n_max : Int
        Number of features
    **kwargs :
        savepath - save path but without file extension.

    Returns
    -------
    Creates a single figure. The figure shows an hbar plot where each bar corresponds
    to the importance the n_max most important features in descending order

    """
    savepath = kwargs["savepath"] if "savepath" in kwargs.keys() else None

    rankings = np.argsort(-np.array(avg_scores))[0:n_max]

    fig, ax = plt.subplots()
    ax.set_xlabel("importance score")
    ax.barh(
        np.linspace(n_max - 1, 0, n_max),
        [avg_scores[r] for r in rankings],
        tick_label=[feature_names[r] for r in rankings],
    )

    fig.tight_layout()
    if savepath != None:
        fig.savefig(savepath + "_avg_importance.png")


def plot_importance_ranking_by_label(
    avg_scores_by_label, feature_names, class_names, n_max, stacked=False, **kwargs
):
    """
    Plotting function for the importance.

    Parameters
    ----------
    avg_scores_by_label : Dict
        Dictionary that carries the a list of avg importance scores for each class
        (in order of the feature names)
    feature_names : list
        List of feature names.
    class_names : list
        List of class names
    n_max : int
        The n_max most important features are plotted
    stacked : TYPE, optional
        If stacked = False, the bars for the different features are shown
        next to eachother. If stacked = True they are stacked on top of
        eachother and the bar is scaled to avg length.
        The default is False.
    **kwargs :
        savepath - save path but without file extension.

    Returns
    -------
    Creates a single figure. The figure shows an hbar plot where each bar corresponds
    to the importance the n_max most important features for each class in descending order.

    """
    savepath = kwargs["savepath"] if "savepath" in kwargs.keys() else None

    if stacked:
        rankings = np.argsort(
            sum(-np.array(avg_scores_by_label[label]) for label in class_names)
        )[0:n_max]
        fig, ax = plt.subplots()
        ax.set_xlabel("importance score")
        n_label = len(class_names)
        offset = np.zeros(n_max)
        for label in class_names:
            ranked_scores = np.array(
                [avg_scores_by_label[label][r] / n_label for r in rankings]
            )
            ax.barh(
                np.linspace(n_max - 1, 0, n_max),
                ranked_scores,
                left=offset,
                tick_label=[feature_names[r] for r in rankings],
                label=label,
            )
            offset += ranked_scores
        ax.legend()
        fig.tight_layout()
        if savepath != None:
            fig.savefig(savepath + "_avg_importance_by_label_stacked.png")
    else:
        local_rankings = {
            label: np.argsort(-np.array(avg_scores_by_label[label]))[0:n_max]
            for label in class_names
        }
        candidates = np.unique(
            np.hstack([local_rankings[label] for label in class_names])
        )
        order = np.argsort(
            [
                np.array(
                    [-avg_scores_by_label[label][r] for label in class_names]
                ).min()
                for r in candidates
            ]
        )[0:n_max]
        rankings = [candidates[r] for r in order]
        fig, ax = plt.subplots()
        ax.set_xlabel("importance score")

        y_range = np.linspace(n_max - 1, 0, n_max)

        width = 0.8
        sub_width = width / len(class_names)
        y_range += sub_width * (len(class_names) - 1) / 2
        for label in class_names:
            ranked_scores = np.array([avg_scores_by_label[label][r] for r in rankings])
            ax.barh(y_range, ranked_scores, height=sub_width, label=label)
            y_range -= sub_width

        ax.set_yticks(np.linspace(n_max - 1, 0, n_max))
        ax.set_yticklabels([feature_names[r] for r in rankings])
        ax.legend()
        fig.tight_layout()
        if savepath != None:
            fig.savefig(savepath + "_avg_importance_by_label_staggered.png")


def evaluate_model(model, inputs):
    """
    A wrapper to enable seamless use for both RF and MLP classifiers.

    """
    if isinstance(model, torch.nn.Module):
        # For PyTorch model
        model.eval()
        with torch.no_grad():
            inputs = torch.Tensor(inputs)
            logits = model(inputs)
            return torch.nn.functional.softmax(logits, dim=-1).numpy()

    elif isinstance(model, RandomForestClassifier):
        # For scikit-learn random forest model
        return model.predict_proba(inputs)

    else:
        raise ValueError(
            "Invalid model type. Supported types: PyTorch nn.Module, scikit-learn RandomForestClassifier"
        )
