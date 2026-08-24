import os
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import dump, load
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

RANDOM_STATE = 42
PATH_FILES = os.path.dirname(os.getcwd())
PATH_INTERMEDIATE = os.path.join(PATH_FILES,'intermediate_ass4')#'intermediate_investigation') 

def cv_split(X,y,group,selector,train_size,val_size,data_subset = 'ScheduledResponseValid',cv_mode = 'hold_out',subgraph_sampling = None,K_train = None):

    if cv_mode == 'kfold':
        "Create the folds"
        sgkf = StratifiedGroupKFold(n_splits=5)
        n_splits = sgkf.get_n_splits(X,y,group)
        for _, (train_index, test_index) in enumerate(sgkf.split(X,y, group)):
            group_train, group_test = group.iloc[train_index], group.iloc[test_index]
            unique_total_groups = np.unique(group)
            "First split the training set into train and validation sets"
            unique_groups = np.unique(group_train)
            val_num = int(val_size * len(unique_total_groups))
            val_subjects = np.random.choice(unique_groups, size = val_num, replace=False)
            train_subjects = unique_groups[~np.isin(unique_groups, val_subjects)]

            if subgraph_sampling == 'random_both':
                "For semi-supervised analysis"
                "At each fold iteration, a number of labeled training participants (9) are sampled from the training set"
                labeled_train_subjects = np.random.choice(train_subjects,size = K_train,replace=False)
                unlabeled_train_subjects = np.setdiff1d(train_subjects,labeled_train_subjects)
                "Sets indices"
                train_inds = group_train[group_train.isin(train_subjects)].index
                labeled_train_inds = group_train[group_train.isin(labeled_train_subjects)].index
                unlabeled_train_inds = group_train[group_train.isin(unlabeled_train_subjects)].index
                val_inds = group_train[group_train.isin(val_subjects)].index
                test_inds = group_test.index

                "Feature-wise normalization with training set"
                X_train, X_val, X_test, C_cat, C_num = _normalize_featurewise(X,train_inds,val_inds,test_inds)
                
                X_labeled_train = X_train.loc[labeled_train_inds]
                X_unlabeled_train = X_train.loc[unlabeled_train_inds]
                "Feature selection on labeled training set"
                X_sel = _select_features(X_labeled_train, X_val, X_test,y,selector,labeled_train_inds,val_inds,test_inds,C_cat,C_num,X_train2 = X_unlabeled_train)
                X_labeled_train = X_sel.loc[labeled_train_inds]
                X_unlabeled_train = X_sel.loc[unlabeled_train_inds]
                X_val = X_sel.loc[val_inds]
                X_test = X_sel.loc[test_inds]

                yield X_sel,X_labeled_train,X_unlabeled_train,X_val,X_test,labeled_train_inds, unlabeled_train_inds, val_inds, test_inds, n_splits

            elif subgraph_sampling is None:
                
                "Sets indices"
                train_inds = group_train[group_train.isin(train_subjects)].index
                val_inds = group_train[group_train.isin(val_subjects)].index
                test_inds = group_test.index

                "Feature-wise normalization with training set"
                X_train, X_val, X_test, C_cat, C_num = _normalize_featurewise(X,train_inds,val_inds,test_inds)

                "Feature selection on training set - Reduce to 50 features"
                X_sel = _select_features(X_train, X_val, X_test,y,selector,train_inds,val_inds,test_inds,C_cat,C_num)
                
                X_train = X_sel.loc[train_inds]
                X_val = X_sel.loc[val_inds]
                X_test = X_sel.loc[test_inds]

                yield X_sel,X_train,X_val,X_test,train_inds, val_inds, test_inds, n_splits

    elif cv_mode == 'hold_out':
        "Create train and test set indices for subjects"
        n_splits = 1
        hold_out_CV_fname = os.path.join(PATH_INTERMEDIATE, data_subset+'_split_sets_whole.pkl')
        if os.path.exists(hold_out_CV_fname):
            "Load existing split"
            split_groups = load(hold_out_CV_fname)
            train_subjects = split_groups['train']
            val_subjects = split_groups['val']
            test_subjects = split_groups['test']
        else:
            "Or create a new one"
            unique_groups = np.unique(group)
            train_num = int(train_size * len(unique_groups))
            train_subjects = np.random.choice(unique_groups, size = train_num, replace=False)
            rest_subjects = unique_groups[~np.isin(unique_groups, train_subjects)]
            val_num = int(val_size * len(unique_groups))
            val_subjects = np.random.choice(rest_subjects, size = val_num, replace=False)
            test_subjects = rest_subjects[~np.isin(rest_subjects, val_subjects)]
            split_groups = {'train': train_subjects, 'val': val_subjects, 'test': test_subjects}
            dump(split_groups, hold_out_CV_fname)

        if subgraph_sampling == 'random_both':
            "For semi-supervised analysis"
            "At each fold iteration, a number of labeled training participants (9) are sampled from the training set"
            labeled_train_subjects = np.random.choice(train_subjects,size = K_train,replace=False)
            unlabeled_train_subjects = np.setdiff1d(train_subjects,labeled_train_subjects)
            "Sets indices"
            train_inds = group_train[group_train.isin(train_subjects)].index
            labeled_train_inds = group_train[group_train.isin(labeled_train_subjects)].index
            unlabeled_train_inds = group_train[group_train.isin(unlabeled_train_subjects)].index
            val_inds = group_train[group_train.isin(val_subjects)].index
            test_inds = group_test.index

            "Feature-wise normalization with training set"
            X_train, X_val, X_test, C_cat, C_num = _normalize_featurewise(X,train_inds,val_inds,test_inds)
            
            X_labeled_train = X_train.loc[labeled_train_inds]
            X_unlabeled_train = X_train.loc[unlabeled_train_inds]
            "Feature selection on labeled training set"
            X_sel = _select_features(X_labeled_train, X_val, X_test,y,selector,labeled_train_inds,val_inds,test_inds,C_cat,C_num,X_train2 = X_unlabeled_train)
            X_labeled_train = X_sel.loc[labeled_train_inds]
            X_unlabeled_train = X_sel.loc[unlabeled_train_inds]
            X_val = X_sel.loc[val_inds]
            X_test = X_sel.loc[test_inds]

            yield X_sel,X_labeled_train,X_unlabeled_train,X_val,X_test,labeled_train_inds, unlabeled_train_inds, val_inds, test_inds, n_splits

        elif subgraph_sampling is None:

            "Sets' indices"
            train_inds = group[group.isin(train_subjects)].index
            val_inds = group[group.isin(val_subjects)].index
            test_inds = group[group.isin(test_subjects)].index

            "Feature-wise normalization with training set"
            X_train, X_val, X_test, C_cat, C_num = _normalize_featurewise(X,train_inds,val_inds,test_inds)

            "Feature selection on training set - Reduce to 50 features"
            X_sel = _select_features(X_train, X_val, X_test,y,selector,train_inds,val_inds,test_inds,C_cat,C_num)
            
            X_train = X_sel.loc[train_inds]
            X_val = X_sel.loc[val_inds]
            X_test = X_sel.loc[test_inds]

            yield X_sel,X_train,X_val,X_test,train_inds, val_inds, test_inds, n_splits

    elif cv_mode == 'leave_one_out':
        "Create the group-wise folds"
        logo = LeaveOneGroupOut()
        n_splits = logo.get_n_splits(X,groups=group)
        for train_index_temp, test_index in logo.split(X,groups=group):
            group_train, group_test = group.iloc[train_index_temp], group.iloc[test_index]
            X_train_temp = X.iloc[train_index_temp]
            unique_total_groups = np.unique(group)
            "Split the training set into train and validation sets"
            unique_groups = np.unique(group_train)
            train_val_inds = group_train[group_train.isin(unique_groups)].index
            y_train_val = y.iloc[train_val_inds] #contains both train and val
            "Stratify based on the train_val_inds and respect the groups"
            #val_num = int(val_size * len(unique_total_groups))
            #val_subjects = np.random.choice(unique_groups, size = val_num, replace=False)
            sgkf = StratifiedGroupKFold(n_splits=10)
            for _, (train_index, val_index) in enumerate(sgkf.split(X_train_temp, y_train_val, group_train)):
                train_inds = train_val_inds[train_index]
                y_train = y_train_val.loc[train_inds]
                val_inds = train_val_inds[val_index]
                y_val = y_train_val.loc[val_inds]
                break

            train_subjects = group_train[train_inds].unique()
            val_subjects = group_train[val_inds].unique()
            #OR: train_subjects = unique_groups[~np.isin(unique_groups, val_subjects)]

            "Sets' indices"
            train_inds = group_train[group_train.isin(train_subjects)].index
            val_inds = group_train[group_train.isin(val_subjects)].index
            test_inds = group_test.index
            
            if subgraph_sampling == 'random_both':
                "For semi-supervised analysis"
                "At each fold iteration, a number of labeled training participants (9) are sampled from the training set"
                labeled_train_subjects = np.random.choice(train_subjects,size = K_train,replace=False)
                unlabeled_train_subjects = np.setdiff1d(train_subjects,labeled_train_subjects)
                "New train sets' indices"
                labeled_train_inds = group_train[group_train.isin(labeled_train_subjects)].index
                unlabeled_train_inds = group_train[group_train.isin(unlabeled_train_subjects)].index

                "Feature-wise normalization with training set"
                X_train, X_val, X_test, C_cat, C_num = _normalize_featurewise(X,train_inds,val_inds,test_inds)
                
                X_labeled_train = X_train.loc[labeled_train_inds]
                X_unlabeled_train = X_train.loc[unlabeled_train_inds]
                "Feature selection on labeled training set"
                X_sel = _select_features(X_labeled_train, X_val, X_test,y,selector,labeled_train_inds,val_inds,test_inds,C_cat,C_num,X_train2 = X_unlabeled_train)
                X_labeled_train = X_sel.loc[labeled_train_inds]
                X_unlabeled_train = X_sel.loc[unlabeled_train_inds]
                X_val = X_sel.loc[val_inds]
                X_test = X_sel.loc[test_inds]

                yield X_sel,X_labeled_train,X_unlabeled_train,X_val,X_test,labeled_train_inds, unlabeled_train_inds, val_inds, test_inds, n_splits

            elif subgraph_sampling == 'only_labeled':
                "For supervised analysis with labeled sampling only"
                labeled_train_subjects = np.random.choice(train_subjects,size = K_train,replace=False)
                labeled_train_inds = group_train[group_train.isin(labeled_train_subjects)].index
                "Feature-wise normalization with training set"
                X_labeled_train, X_val, X_test, C_cat, C_num = _normalize_featurewise(X,labeled_train_inds,val_inds,test_inds)
                "Feature selection on labeled training set"
                X_sel = _select_features(X_labeled_train, X_val, X_test,y,selector,labeled_train_inds,val_inds,test_inds,C_cat,C_num,X_train2 = None)
                X_labeled_train = X_sel.loc[labeled_train_inds]
                X_val = X_sel.loc[val_inds]
                X_test = X_sel.loc[test_inds]

                yield X_sel,X_labeled_train,X_val,X_test,labeled_train_inds, val_inds, test_inds, n_splits


            elif subgraph_sampling is None:
                
                "Feature-wise normalization with training set"
                X_train, X_val, X_test, C_cat, C_num = _normalize_featurewise(X,train_inds,val_inds,test_inds)

                "Feature selection on training set - Reduce to 50 features"
                X_sel = _select_features(X_train, X_val, X_test,y,selector,train_inds,val_inds,test_inds,C_cat,C_num)
                
                X_train = X_sel.loc[train_inds]
                X_val = X_sel.loc[val_inds]
                X_test = X_sel.loc[test_inds]

                yield X_sel,X_train,X_val,X_test,train_inds, val_inds, test_inds,n_splits


def process_label_info(X,y,group,label):
    "Process label information"
    y = pd.DataFrame(y, columns=['valence_bin','arousal_bin','emo_quad'])
    group = pd.Series(group, index=X.index, name='group')
    old_index = X.index

    lab = LabelEncoder()
    colV = lab.fit_transform(y['valence_bin'])
    colA = lab.fit_transform(y['arousal_bin'])
    colQ = lab.fit_transform(y['emo_quad'])
    y['valence_bin'] = colV
    y['arousal_bin'] = colA
    y['emo_quad'] = colQ
    #sample_data = {'X': X, 'y': y, 'group': group, 'old_index': old_index}
    #if not os.path.exists(os.path.join(PATH_PREPROCESSED, use_data + '_' + data_subset +'_whole.pkl')):
    #    dump(sample_data, os.path.join(PATH_PREPROCESSED, use_data + '_' + data_subset +'_whole.pkl'))

    #print(f'Preparing graph for {use_data} data, {data_subset} subset and label {label}')

    "Drop label history features for the other label"
    if label == 'valence_bin':
        X = X.drop(columns=['ESM#LIK#H06#arousal_bin',
                'ESM#LIK#H12#arousal_bin','ESM#LIK#H24#arousal_bin'])
    elif label == 'arousal_bin':
        X = X.drop(columns=['ESM#LIK#H06#valence_bin',
                'ESM#LIK#H12#valence_bin','ESM#LIK#H24#valence_bin'])
    elif label == 'emo_quad':
        X = X.drop(columns=['ESM#LIK#H06#valence_bin','ESM#LIK#H06#arousal_bin',
                'ESM#LIK#H12#valence_bin','ESM#LIK#H12#arousal_bin',
                'ESM#LIK#H24#valence_bin','ESM#LIK#H24#arousal_bin'])

    return X,y,group,old_index


def _normalize_featurewise(X,train_inds,val_inds,test_inds):
    "Feature normalization on numerical features - train set"
    
    X_train = X.loc[train_inds]
    X_val = X.loc[val_inds]
    X_test = X.loc[test_inds]

    "Separate categorical from numerical features"
    cats = X.columns[X.dtypes == bool]
    C_cat = np.asarray(sorted(cats))
    C_num = np.asarray(sorted(X.columns[~X.columns.isin(C_cat)]))
    
    "Normalize numerical features with train set normalization"
    X_N_train = X_train[C_num].values
    X_C_train = X_train[C_cat]
    X_N_val = X_val[C_num].values
    X_C_val = X_val[C_cat]
    X_N_test = X_test[C_num].values
    X_C_test = X_test[C_cat]

    scaler_train = StandardScaler().fit(X_N_train) 
    X_N_train_sc = scaler_train.transform(X_N_train)
    X_N_val_sc = scaler_train.transform(X_N_val)
    X_N_test_sc = scaler_train.transform(X_N_test)

    X_N_train_sc = pd.DataFrame(X_N_train_sc, columns = C_num,index = train_inds)
    X_N_val_sc = pd.DataFrame(X_N_val_sc, columns = C_num, index = val_inds)
    X_N_test_sc = pd.DataFrame(X_N_test_sc, columns = C_num, index = test_inds)

    "Merge with categorical again"
    X_S_train = pd.concat([X_C_train, X_N_train_sc], axis=1)
    X_S_val = pd.concat([X_C_val, X_N_val_sc], axis=1)
    X_S_test = pd.concat([X_C_test, X_N_test_sc], axis=1)

    return X_S_train, X_S_val, X_S_test, C_cat, C_num


def _select_features(X_train,X_val,X_test,y,selector,train_inds,val_inds,test_inds,C_cat,C_num,X_train2 = None):

    y_train = y.loc[train_inds]
    "Feature selection on training set - Reduce to 50 features"
    if not isinstance(y, np.ndarray):
        target = y_train.values

    C = np.asarray(X_train.columns)
    print(f'Original number of features: {len(C)} #Cat. = {len(C_cat)}; # Num. = {len(C_num)}' )
    M = selector.fit(X=X_train.values, y=target).get_support()
    C_sel = C[M]
    C_cat = C_cat[np.isin(C_cat, C_sel)]
    C_num = C_num[np.isin(C_num, C_sel)]

    #Reduce all sets based on these selected features
    X_N_train_sel = X_train[C_num]
    X_C_train_sel = X_train[C_cat]
    X_N_val_sel = X_val[C_num]
    X_C_val_sel = X_val[C_cat]
    X_N_test_sel = X_test[C_num]
    X_C_test_sel = X_test[C_cat]
    if X_train2 is not None:
        X_N_train2_sel = X_train2[C_num]
        X_C_train2_sel = X_train2[C_cat]

    if X_train2 is not None:
        X_N = pd.concat([X_N_train_sel, X_N_train2_sel,X_N_val_sel, X_N_test_sel],axis = 0)
        X_N = X_N.sort_index()
        X_C = pd.concat([X_C_train_sel, X_C_train2_sel,X_C_val_sel, X_C_test_sel],axis = 0)
        X_C = X_C.sort_index()
    else:
        X_N = pd.concat([X_N_train_sel, X_N_val_sel, X_N_test_sel],axis = 0)
        X_N = X_N.sort_index()
        X_C = pd.concat([X_C_train_sel, X_C_val_sel, X_C_test_sel],axis = 0)
        X_C = X_C.sort_index()

    X_sel = pd.concat([X_C, X_N], axis=1)

    print(f'# Selected features: {len(C_sel)} #Cat. = {len(C_cat)}; # Num. = {len(C_num)}' )
    
    "Feature importance could be taken from C_sel, C_cat, C_num"

    return X_sel


def train(model, data, optimizer, criterion,device,lambda1 = 0.01):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    x = data.x.to(device, dtype = torch.float)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    if hasattr(model, 'convs') or hasattr(model, 'conv1'):
        #out = model(x, edge_index) #data.x, data.edge_index
        out = model(x,edge_index,task = 'classif')
    #elif hasattr(model,'lin1'):
    #    out = model(x)  #data.x / Perform a single forward pass.
    #Cross-entropy loss
    Lce = criterion(out[data.train_mask], y[data.train_mask])  #data.y / Compute the loss solely based on the training nodes.
    if hasattr(data,'val_mask'):
        unlabeled_probs = out[data.val_mask]
        pseudolabels = unlabeled_probs.argmax(dim=1)
        #Compute entropy regularization loss
        Len = lambda1*criterion(unlabeled_probs,pseudolabels)
    else:
        Len = 0
        pseudolabels = None
    
    loss = Lce + Len
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss,pseudolabels
    
def train_SSL(model,data,data_aug,optimizer,criterion,device,ssl_task,lambda1 = 0.01,lambda2 = 0.1):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    x = data.x.to(device, dtype = torch.float)
    x_aug = data_aug.x.to(device, dtype = torch.float)
    y = data.y.to(device)
    y_aug = data_aug.y.to(device)
    edge_index = data.edge_index.to(device)
    edge_index_aug = data_aug.edge_index.to(device)

    num_nodes = x.shape[0]
    #if hasattr(model, 'convs') or hasattr(model, 'conv1'):
    #    out = model(x, edge_index) #data.x, data.edge_index
    #elif hasattr(model,'lin1'):
    #    out = model(x)  #data.x / Perform a single forward pass.
    
    "Get model outputs - Order does not matter since weights will be updated later"
    out_classif = model(x,edge_index,mode = 'train',task = 'classif')
    out_ssl_aug= model(x_aug,edge_index_aug,mode = 'train',task = 'ssl')
    out_ssl_raw = model(x,edge_index,mode = 'train',task = 'ssl')

    "Calculate losses"
    #1.Supervised and semi-supervised loss
    Lce = criterion(out_classif[data.train_mask], y[data.train_mask])  #data.y / Compute the loss solely based on the training nodes.
    if hasattr(data,'val_mask'):
        unlabeled_probs = out_classif[data.val_mask]
        pseudolabels = unlabeled_probs.argmax(dim=1)
        #Compute entropy regularization loss
        Len = criterion(unlabeled_probs,pseudolabels)        
    else:
        #If semi-supervised loss is not used
        Len = 0
        pseudolabels = None
    
    #2. SSL loss
    "Calc loss based on SSL task"
    if ssl_task == 'denoising' or ssl_task == 'feature_masking' or ssl_task == 'edge_removing' or ssl_task == 'edge_perturbation':
        Lssl = (1/num_nodes)*torch.linalg.matrix_norm(out_ssl_aug-out_ssl_raw)
    elif ssl_task == 'node_masking':
        "Find the masked nodes"
        x_m = x_aug.clone()
        x_m = x_m.to('cpu').detach().numpy()
        is_zero = np.all(x_m == 0,axis = 1)
        num_masked_nodes = np.sum(is_zero)
        node_mask = np.where(is_zero == True)[0]
        "Find the embeddings of the masked nodes"
        masked_embeddings_aug = out_ssl_aug[node_mask,:]
        masked_embeddings_raw = out_ssl_raw[node_mask,:]
        "Evaluate loss only on masked embeddings"
        Lssl = (1/num_masked_nodes)*torch.linalg.matrix_norm(masked_embeddings_aug-masked_embeddings_raw)
    #elif ssl_task == 'node_dropping' or ssl_task == 'edge_perturbation':
    #    pass


    loss = Lce + lambda1*Len + lambda2*Lssl
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
        
    return loss, pseudolabels


def test(mask,model,data,criterion,device):
    model.eval()

    with torch.no_grad():
        x = data.x.to(device, dtype = torch.float)
        y = data.y.to(device).long()
        edge_index = data.edge_index.to(device)
        if hasattr(model, 'convs') or hasattr(model, 'conv1'):
            out = model(x, edge_index,task='classif') #data.edge_index
        #elif hasattr(model,'lin1'):
        #    out = model(x)
        test_loss = criterion(out[mask], y[mask])  # Compute the loss solely based on the training nodes.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[mask] == y[mask]  # Check against ground-truth labels.
    test_false = pred[mask] != y[mask]
    test_acc = int(test_correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    y = y.to('cpu').numpy()
    pred = pred.to('cpu').numpy()
    "In case of multi-class"
    classes_ = np.linspace(0,data.num_classes-1,data.num_classes,dtype=int)
    testR = _metrics(y[mask],pred[mask],pos_label=data.num_classes - 1,classes=classes_)
    if testR['acc'] != test_acc:
        print('Accuracy mismatch')
    #testR['acc'] = test_acc
    testR['loss'] = test_loss.item()
    #Calculate and print confusion matrix
    cm = confusion_matrix(y[mask], pred[mask])
    #plt.figure(figsize=(10,7))
    #sns.heatmap(cm, annot=True,fmt='d')
    #plt.xlabel('Predicted')
    #plt.ylabel('Actual')
    #plt.tight_layout()
    #plt.show()
    testR['confMat'] = cm

    "Set model back to train mode"
    model.train()

    return testR

def _metrics(y_true,y_pred,pos_label,classes):
    R = {}
    n_classes = len(classes)
    pre, rec, f1, _ = precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, pos_label=pos_label, average='macro', zero_division=0
    )
    R['acc'] = accuracy_score(y_true=y_true, y_pred=y_pred)
    R[f'pre_macro'] = pre
    R[f'rec_macro'] = rec
    R[f'f1_macro'] = f1
        
    for c in classes:
        pre, rec, f1, _ = precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, pos_label=c, average='binary', zero_division=0
        )
        R[f'pre_{c}'] = pre
        R[f'rec_{c}'] = rec
        R[f'f1_{c}'] = f1
    return R

def get_class_weights(y_target):
    class_distribution = np.bincount(y_target)
    class_weights = np.zeros(len(class_distribution))
    minority_class = np.argmin(class_distribution)
    class_weights[minority_class] = 1.0
    for i in range(len(class_distribution)):
        if i != minority_class:
            class_weights[i] = class_distribution[minority_class]/class_distribution[i]
    return class_weights, class_distribution


def mean_confidence_interval(data, confidence=0.95):
    """
    Description: Calculates x% mean confidence interval of the data provided as input.

    Parameters
    ----------
    data : array-like vector or matrix. Columnwise calculation of CI
    confidence : float, 0-1, optional. Level of confidence. The default is 0.95.

    Returns
    -------
    m : float, estimated value (mean).
    m-h: lower CI limit
    m+h: upper CI limit
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a,axis = 0), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


class WarmStartCosineAnnealingLR(_LRScheduler):
    "Implements a custom learning rate scheduler that combines warm start and cosine annealing"
    def __init__(self, optimizer, cosine_epochs, warm_start_epochs = 10,cosine_cycles = 1, eta_min=1e-6, last_epoch=-1):
        self.warm_start_epochs = warm_start_epochs
        self.cosine_epochs = cosine_epochs
        self.cosine_cycles = cosine_cycles
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        super(WarmStartCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warm_start_epochs:
            # Warm start: linearly increase learning rate
            return [base_lr * self.last_epoch / self.warm_start_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max = self.cosine_epochs/self.cosine_cycles, eta_min = self.eta_min, last_epoch = self.last_epoch - self.warm_start_epochs)
            return cosine_scheduler.get_lr()



