from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np

mimos = ['mc6', 'mc6s', 'mc6sa']
df_charge = {}
df_dist = {}
for mimo in mimos:
    df_charge[mimo] = pd.read_csv('data/'+mimo+'_charges.csv')
    df_dist[mimo] = pd.read_csv('data/'+mimo+'_pairwise_distances.csv')

fig, ax = plt.subplots(1,2)
ax[0].set_title('distances')
ax[1].set_title('charges')
for mimo in mimos:
    avg_dist = [mean(row[1]) for row in df_dist[mimo].iterrows()]
    avg_charge = [mean(row[1]) for row in df_charge[mimo].iterrows()]
    ax[0].plot(avg_dist, label = mimo)
    ax[1].plot(avg_charge, label = mimo)
ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')
fig.tight_layout()
fig


class_assignment = {'mc6' : 0, 'mc6s' : 1, 'mc6sa' : 2}
features = ['dist', 'charge']
X = {'dist' : {mimo :np.array(df_dist[mimo]) for mimo in mimos},
     'charge' : {mimo : np.array(df_charge[mimo]) for mimo in mimos}}
y = {'dist' : {},
     'charge' : {}}
for mimo in mimos:
        y_aux = np.zeros((df_dist[mimo].shape[0], 3))
        y_aux[:, class_assignment[mimo]] = 1
        y['dist'][mimo] = y_aux
        y_aux = np.zeros((df_charge[mimo].shape[0], 3))
        y_aux[:, class_assignment[mimo]] = 1
        y['charge'][mimo] = y_aux

test_frac = 0.8
X_test, y_test = {}, {}
X_train, y_train = {}, {}
for feature in features:
    n_cutoff = int(test_frac*X[feature]['mc6'].shape[0])
    X_train[feature] = np.vstack(X[feature][mimo][0:n_cutoff, :] for mimo in mimos)
    X_test[feature] = np.vstack(X[feature][mimo][n_cutoff:, :] for mimo in mimos)
    y_train[feature] = np.vstack(y[feature][mimo][0:n_cutoff, :] for mimo in mimos)
    y_test[feature] = np.vstack(y[feature][mimo][n_cutoff:, :] for mimo in mimos)
    
n_trees = 200
max_depth = 50
rf_cls = {}
for feature in features:
    rf_cls[feature] = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth)
    rf_cls[feature].fit(X_train[feature], y_train[feature])

def lin_idx(y):
    return [y[i,:].argmax() for i in range(y.shape[0])]

y_pred = {}
cms = {}
for feature in features:
    y_pred[feature] = rf_cls[feature].predict(X_test[feature])
    cm = confusion_matrix(lin_idx(y_pred[feature]),lin_idx(y_test[feature]))
    cms[feature] = pd.DataFrame(cm, mimos, mimos)

plt.figure()
sn.heatmap(cms['dist'], annot=True)
fig

plt.figure()
sn.heatmap(cms['charge'], annot=True)
fig
