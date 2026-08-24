#%% 
import pandas as pd
import networkx as nx
import os
import numpy as np
from utils import load, dump
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from graph_utils import explore_graph_features
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from training_utils import cv_split, process_label_info

RANDOM_STATE = 42

def equally_sample_groups(X,y,group, total_samples = 0.2):
    
    total_samples = int(total_samples * len(X))
    group = pd.Series(group, index=X.index, name='group')
    Xtemp = pd.concat([X, group], axis=1)
    unique_groups = Xtemp['group'].unique()
    per_group_samples = int(total_samples / len(unique_groups))
    #group_indices = Xtemp[Xtemp['group'].isin(unique_groups)].index
    # Get the classes corresponding to the unique groups
    Xsample = pd.DataFrame()
    ysample = pd.DataFrame(columns=['valence_bin','arousal_bin'])
    for ugroup in unique_groups:
        groupData = Xtemp[Xtemp['group'] == ugroup].index
        group_classes = y[groupData]
        samples, _, sample_classes, _ = train_test_split(groupData, group_classes[:,0], stratify=group_classes[:,0], train_size=per_group_samples,random_state=RANDOM_STATE)
        Xsample = pd.concat([Xsample, Xtemp.loc[samples]])
        ysample = pd.concat([ysample, pd.DataFrame(y[samples], columns=['valence_bin','arousal_bin'])])

    groupSample = Xsample['group']
    indexSample = Xsample.index
    groupSample.reset_index(drop=True, inplace=True)
    Xsample = Xsample.drop(columns='group')
    Xsample.reset_index(drop=True, inplace=True)
    return Xsample, ysample, groupSample, indexSample
    


PATH_FILES = os.path.dirname(os.getcwd())
PATH_PLOTS = os.path.join(PATH_FILES,'plots')
PATH_INTERMEDIATE = os.path.join(PATH_FILES,'intermediate_ass3')
PATH_PREPROCESSED = os.path.join(PATH_FILES,'data_preprocessing') 
if not os.path.exists(PATH_PREPROCESSED):
    os.makedirs(PATH_PREPROCESSED)
data_subset = 'ScheduledResponseValid'#,'ScheduledResponseExcluded','VoluntaryResponse','all']
use_data = 'all' #Options: sensor, phone, all
cv_mode = 'kfold' #Options: hold_out, kfold, leave_one_out
train_size = 0.60
val_size = 0.15
test_size = 0.25
regularization_strength = 15e-3 #C, determines number of selected features w LinearSVC - the bigger the more
label = 'emo_quad'  #valence - 14e-3
"Feature selector"
SELECT_SVC = SelectFromModel(
    estimator=LinearSVC(
        penalty='l1',
        loss='squared_hinge',
        dual=False,
        tol=1e-3,
        C=regularization_strength,
        max_iter=5000,
        random_state=RANDOM_STATE
    ),
    threshold=1e-5
)

"Load data"
X, y, group, t, _ = load(os.path.join(PATH_INTERMEDIATE, use_data + '_' + data_subset +'_all_labels.pkl'))

X,y,group,old_index = process_label_info(X,y,group,label)

"Select target class, needed in Kfold CV"
y_target = y[label]

train_inds, val_inds, test_inds = cv_split(X,y_target,group,train_size,val_size,data_subset = 'ScheduledResponseValid',cv_mode = 'cross_val')



