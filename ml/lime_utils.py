from tqdm import tqdm
import matplotlib.pyplot as plt
import lime 
import numpy as np

def lime_analysis(data, model, class_names, feature_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(data, feature_names=feature_names, 
                                                       class_names=class_names, 
                                                       discretize_continuous=True)
    y_preds = model(data).argmax(axis=1)
    important_features = extract_imporant_features(explainer, model, data)
    
    return important_features,\
           y_preds, \
           get_avg_importance(important_features), \
           get_avg_importance_by_label(important_features, class_names, y_preds)

def extract_imporant_features(explainer, model, data):
    important_features = []
    n_features = data.shape[1]
    for i in tqdm(range(data.shape[0])):
        x = data[i,:]
        exp = explainer.explain_instance(x, model, num_features=n_features)
        important_features.append(exp.as_map()[1])
    return important_features

def get_score(scores, idx, trafo = lambda x : x):
    return [trafo(s[1]) for s in scores if s[0] == idx]

def get_avg_importance(important_features):
    n_features = len(important_features[0])
    return [np.mean([get_score(f, i, lambda x : abs(x)) for f in important_features]) for i in range(n_features)]
    
def get_avg_importance_by_label(important_features, class_names, y_preds):
    avg_importance = {}
    n_features = len(important_features[0])
    for (k, label) in enumerate(class_names):
        scores = [np.mean([get_score(f, i, lambda x : abs(x)) for (j,f) in enumerate(important_features) if y_preds[j] == k]) 
                  for i in range(n_features)]
        avg_importance[label] = scores
    return avg_importance

def get_nth_mif_by_label(important_features, y_preds, n):
    features_by_label = [[f[n][0] for (i,f) in enumerate(important_features) if y_preds[i] == k] 
                         for k in range(y_preds.argmax()+1)] 
    return features_by_label

def get_nth_mif(important_features, n):
    return [f[n][0] for f in important_features]

def plot_hists(n_max, important_features, class_names, y_preds, **kwargs):
    savepath = kwargs['savepath'] if 'savepath' in kwargs.keys() else None
    bin_labels = kwargs['bin_labels'] if 'bin_labels' in kwargs.keys() else None
    
    n_features = len(important_features[0])
    fig, axs = plt.subplots(n_max)
    fig.set_figheight(n_max*3)
    fig.set_figwidth(5)
    for i in range(n_max):
        features = get_nth_mif_by_label(important_features, y_preds, i)
        bins = np.linspace(-0.5, n_features-0.5, n_features+1)
        axs[i].hist(features, bins=bins, label = class_names, stacked=True) 
        if i == n_max - 1:
            if bin_labels != None:
                axs[i].set_xticks(range(n_features))
                axs[i].set_xticklabels(bin_labels, rotation = 90)
        else:
            if bin_labels != None:
                axs[i].set_xticks(range(n_features))
                axs[i].set_xticklabels(["" for i in range(n_features)])
        axs[i].set_ylabel("number of frames")
        if i == 0:
            axs[i].legend()
            
    fig.tight_layout()
    if savepath != None:
        fig.savefig(savepath+"_importance_by_frame_and_label.png")
        
    fig, axs = plt.subplots(n_max)
    fig.set_figheight(n_max*3)
    fig.set_figwidth(5)
    for i in range(n_max):
        features = get_nth_mif(important_features, i)
        bins = np.linspace(-0.5, n_features-0.5, n_features+1)
        axs[i].hist(features, bins=bins) 
        if i == n_max - 1:
            if bin_labels != None:
                axs[i].set_xticks(range(n_features))
                axs[i].set_xticklabels(bin_labels, rotation = 90)
        else:
            if bin_labels != None:
                axs[i].set_xticks(range(n_features))
                axs[i].set_xticklabels(["" for i in range(n_features)])
        axs[i].set_ylabel("number of frames")
    
    fig.tight_layout()
    if savepath != None:
        fig.savefig(savepath+"_importance_by_frame.png")
        
    pass

def plot_importance_ranking(avg_scores, feature_names, n_max, **kwargs):
    savepath = kwargs['savepath'] if 'savepath' in kwargs.keys() else None
    
    rankings = np.argsort(-np.array(avg_scores))[0:n_max]

    fig, ax = plt.subplots()
    ax.set_xlabel("importance score")
    ax.barh(np.linspace(n_max-1, 0, n_max), [avg_scores[r] for r in rankings],
            tick_label = [feature_names[r] for r in rankings])
    
    fig.tight_layout()
    if savepath != None:
        fig.savefig(savepath+"_avg_importance.png")
        
def plot_importance_ranking_by_label(avg_scores_by_label, feature_names, class_names, n_max, stacked = False, **kwargs):
    savepath = kwargs['savepath'] if 'savepath' in kwargs.keys() else None
    
    
    if stacked:
        rankings = np.argsort(sum(-np.array(avg_scores_by_label[label]) for label in class_names))[0:n_max]
        fig, ax = plt.subplots()
        ax.set_xlabel("importance score")
        n_label = len(class_names)
        offset = np.zeros(n_max)
        for label in class_names:
            ranked_scores = np.array([avg_scores_by_label[label][r]/n_label for r in rankings])
            ax.barh(np.linspace(n_max-1, 0, n_max), ranked_scores,
                    left = offset,
                    tick_label = [feature_names[r] for r in rankings], label = label)
            offset += ranked_scores 
        ax.legend()
    else:
       local_rankings = {label : np.argsort(-np.array(avg_scores_by_label[label]))[0:n_max] for label in class_names}
       candidates = np.unique(np.hstack([local_rankings[label] for label in class_names]))
       order = np.argsort([np.array([-avg_scores_by_label[label][r] for label in class_names]).min() for r in candidates])[0:n_max]
       rankings = [candidates[r] for r in order]
       fig, ax = plt.subplots()
       ax.set_xlabel("importance score")
       
       y_range = np.linspace(n_max-1, 0, n_max)
       
       width = 0.8
       sub_width = width/len(class_names)
       y_range += sub_width*(len(class_names) - 1)/2
       for label in class_names:
           ranked_scores = np.array([avg_scores_by_label[label][r] for r in rankings])
           ax.barh(y_range, ranked_scores, height=sub_width, label = label)
           y_range -= sub_width
           print(y_range)
       ax.set_yticks(np.linspace(n_max-1, 0, n_max))
       ax.set_yticklabels([feature_names[r] for r in rankings])
       ax.legend()          
        
        
       fig.tight_layout()
       if savepath != None:
           fig.savefig(savepath+"_avg_importance_by_label.png")
    