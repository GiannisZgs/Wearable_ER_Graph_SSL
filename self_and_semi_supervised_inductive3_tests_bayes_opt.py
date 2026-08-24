#%%
"Sample a small number of labeled and unlabeled nodes to construct graph during training"
"only these nodes are used for graph construction"
"Evaluation is performed by adding new nodes during inference"
"Considered semi-supervised"

import networkx as nx
import os
import numpy as np
from scipy import stats
from utils import load, dump
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import gc
from torch_geometric.utils import from_networkx
import torch_geometric.transforms as T
import torch
from models import MLP,GCN,GCN_basic, GAT_basic, SAGE_basic, count_parameters, GCN_test,GAT_test, GCN_basic_projection
from training_utils import train,test, process_label_info, cv_split, mean_confidence_interval, get_class_weights, WarmStartCosineAnnealingLR, train_SSL
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from construct_graph_utils import construct_from_np_adjacency, compute_adjacency, get_graph_masks, inductive_val
import time
import seaborn as sns
from datetime import datetime 
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import transforms as graph_transforms
import GCL.augmentors as A

RANDOM_STATE = 42   
PATH_FILES = os.path.dirname(os.getcwd())
PATH_INTERMEDIATE = os.path.join(PATH_FILES,'intermediate_ass4')#'intermediate_investigation') 
PATH_DIR_LOGS_JSON = os.path.join(PATH_FILES,'logs_bayes_opt_self_semi_sup_induct3')
if not os.path.exists(PATH_DIR_LOGS_JSON):
    os.makedirs(PATH_DIR_LOGS_JSON)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
num_workers = 4
print("Number of workers:", num_workers)
fp16_precision = True
disable_cuda = False
use_gpus = 1
"If testing on other subsets, modify logging filenames to include subset name"
subset = 'ScheduledResponseValid'#,'ScheduledResponseExcluded','VoluntaryResponse','all']
if subset == 'ScheduledResponseValid':
    subjects = 47
use_data = 'all' #Options: sensor, phone, all
learning = 'inductive' #'transductive' or 'inductive'
sup_mode = 'semi_sup' # 'fully-sup' / 'semi-sup' / 'self-sup'
#render_unlabeled = 0.8 #Percentage (/1) of unlabeled nodes to render
use_labels = True # True or False
subgraph_sampling = 'random_both' # / random_strat_labeled / random_both / random_strat_both (without replacement)
label = 'arousal_bin'
run_until = 9 #run until fold number (including that fold)
early_break = True
run_new = True

"Training set sampling parameters"
K_train_percentage = 0.2 #Percentage of training participants se"lected to be labeled from training set
labeled_use_percentage = 1
unlabeled_use_percentage = 0.5
K_train = int(np.floor(K_train_percentage*subjects)) #20% - Number of training participants selected to be labeled from training set
if K_train % 2 == 0:
    Ls_train = int(labeled_use_percentage*K_train)
    Us_train = int(unlabeled_use_percentage*K_train)
elif K_train % 2 == 1:
    Ls_train = int(np.floor(labeled_use_percentage*K_train)) #participants in training set with labels
    Us_train = int(np.ceil(unlabeled_use_percentage*K_train)) #participants in training set without labels
#assert K_train == Ls_train + Us_train

regularization_strength = 15e-3 #C, determines number of selected features w LinearSVC - the bigger the more
#valence - 14e-3
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

#num_features = 260 #arousal:260 / valence:258
construction_method = 'nearest_farthest_neighbors' #Options: corr, nearest_neighbors,nearest_farthest_neighbors
distance_metric = 'cosine' #Options: euclidean, cosine,manhattan
#archs = ['GCN_basic','GAT_basic','SAGE_basic']#,'MLP']
arch = 'GCN_basic_proj'
cv_mode = 'leave_one_out' #Options: hold_out, kfold, leave_one_out

#cv_mode == 'leave_one_out':
n_splits = 47

train_size = 0.60
val_size = 0.15
test_size = 0.25

"Self-supervised hyperparameters"
#masking_percentage = 0.15 #Percentage of features/nodes to mask
#feature_noise_percentage = 0.05 #Percentage of features to add noise to
#node_dropping_prob = 0.1 #Probability of dropping a node
#perturb_edge_prob = 0.1 #Probability of perturbing an edge (add or remove)
#lambda2 = 0.1 #Regularization parameter for self-supervised loss
#aug_probability = 0.1

augmentation = 'node_masking' #Options: denoising, feature_masking, node_masking, edge_perturbation

#if augmentation == 'denoising':
#    graph_aug = graph_transforms.Denoising(p=aug_probability, mu=0.0, sigma=1.0)
#elif augmentation == 'feature_masking':
#    graph_aug = graph_transforms.Completion(p=aug_probability,masking = 'feature')
#elif augmentation == 'node_masking':
#    graph_aug = graph_transforms.Completion(p=aug_probability,masking = 'node')
#elif augmentation == 'edge_perturbation':
#    graph_aug = A.RandomChoice([A.EdgeAdding(pe=aug_probability),
#                                A.EdgeRemoving(pe=aug_probability)],
#                                num_choices=1)
#elif augmentation == 'shuffling':
#    pass
#elif augmentation == 'node_dropping':
#    pass
#    graph_aug = A.NodeDropping(pn=node_dropping_prob)

"Optimizer, Scheduler and Training Parameters"
#lambda1 = 0.01 #Regularization parameter for semi-supervised pseudo-labeling loss
#label_smoothing = 0.1
#lambda1 = 0.37
#label_smoothing = 0.0
#K_train_percentage = 0.1
#att_heads = 4
#agg_func = 'max' #Options: mean, max
"Non-tunable"
nearest_neighbors = 2
if construction_method == 'nearest_neighbors':
    nearest_neighbors += 1
farthest_neighbors = 1
activation = 'tanh'

"Currently fixed"
if label == 'valence_bin':
    use_lnorm = False
    weights_loss = 'non_weighted' #Options: 'weighted_loss', 'non_weighted'
    hidden_channels = 96 
    dropout = 0.4
    lr = 0.0055
    num_layers = 3

elif label == 'arousal_bin':
    use_lnorm = True
    weights_loss = 'non_weighted'
    hidden_channels = 96
    dropout = 0.5
    lr = 0.0055
    num_layers = 3

epochs = 200
eta_min = 1e-6
weight_decay = 1e-5
scheduler = 'constant' #Options: cosine, step, plateau, warmCosine, constant
if scheduler == 'cosine':
    cosine_cycles = 2
    cosine_epochs = epochs
    eta_min = 1e-6
elif scheduler == 'warmCosine':
    warm_start_epochs = 10
    cosine_epochs = epochs - warm_start_epochs
    eta_min = 1e-6
elif scheduler == 'step':
    step_size = 15
    reduction_factor = 0.1
elif scheduler == 'plateau':
    reduction_factor = 0.2
    min_lr = 1e-6 #eta_min
    patience = 10 #epochs to wait before reducing lr
elif scheduler == 'constant':
    pass

PATH_LOGS_JSON = os.path.join(PATH_DIR_LOGS_JSON,'logs_'+label+'_'+augmentation)

def bayes_opt_cv(lambda1,lambda2,label_smoothing,aug_probability):
    #def bayes_opt_cv(lr,epochs,num_layers,hidden_channels):
    "Measure execution time"
    start_time = time.time()

    if augmentation == 'denoising':
        graph_aug = graph_transforms.Denoising(p=aug_probability, mu=0.0, sigma=1.0)
    elif augmentation == 'feature_masking':
        graph_aug = graph_transforms.Completion(p=aug_probability,masking = 'feature')
    elif augmentation == 'node_masking':
        graph_aug = graph_transforms.Completion(p=aug_probability,masking = 'node')
    elif augmentation == 'edge_perturbation':
        graph_aug = A.RandomChoice([A.EdgeAdding(pe=aug_probability),
                                    A.EdgeRemoving(pe=aug_probability)],
                                    num_choices=1)

    "Load data"
    X, y, group, t, _ = load(os.path.join(PATH_INTERMEDIATE, use_data + '_' + subset +'_all_labels.pkl'))

    "Process label info - Encode labels and drop label-related features"
    X,y,group,old_index = process_label_info(X,y,group,label)

    if run_until is not None and early_break:
        n_splits = run_until + 1

    "Output of the function to be optimized"
    best_val_losses = []
    best_val_F1s = []
    "Cross-validation loop"
    for i,(X_sel,X_labeled_train,X_unlabeled_train,X_val,X_test,labeled_train_inds, unlabeled_train_inds, val_inds, test_inds, _) in enumerate(cv_split(X,y[label],group,SELECT_SVC,train_size,val_size,data_subset = subset,cv_mode = cv_mode,subgraph_sampling = subgraph_sampling,K_train = K_train)):
        if early_break and i > run_until:
            break
        "Select target class, needed in Kfold CV"
        y_target = y[label].values
        
        y_pseudo = np.ones((X_sel.shape[0],epochs))*(-1) 
        labeled_train_groups = group[labeled_train_inds]
        unlabeled_train_groups = group[unlabeled_train_inds]
        labeled_train_groups_unique = np.unique(labeled_train_groups)
        unlabeled_train_groups_unique = np.unique(unlabeled_train_groups)
        
        #print(f'Cross-validation mode: {cv_mode} - fold {i+1} of {n_splits} for {arch}, label = {label}')
        
        num_features = X_sel.shape[1]
        num_classes = len(np.unique(y_target))
        y_target = torch.from_numpy(y_target)
                
        "Initialize model instance"
        if arch == 'GCN_basic':
            model = GCN_basic(num_layers = num_layers,activation = activation,dropout = dropout,
                hidden_channels=hidden_channels,input_size=num_features,output_size=num_classes,use_lnorm = use_lnorm).to(device)
        elif arch == 'GCN_basic_proj':
            model = GCN_basic_projection(num_layers = num_layers,activation = activation,dropout = dropout,
                hidden_channels=hidden_channels,input_size=num_features,output_size=num_classes,use_lnorm = use_lnorm).to(device)
        elif arch == 'GCN_test':
            model = GCN_test(num_layers = num_layers,activation = activation,dropout = dropout,
                hidden_channels=hidden_channels,input_size=num_features,output_size=num_classes,use_lnorm = use_lnorm).to(device)
        elif arch == 'GAT_basic': 
            model = GAT_basic(num_layers = num_layers,activation = activation,dropout = dropout,
                hidden_channels=hidden_channels,input_size=num_features,output_size=num_classes,heads=att_heads,use_lnorm = use_lnorm).to(device)
        elif arch == 'SAGE_basic':   
            model = SAGE_basic(num_layers = num_layers,activation = activation,dropout = dropout,
                aggregation = agg_func,hidden_channels=hidden_channels,input_size=num_features,output_size=num_classes,use_lnorm = use_lnorm).to(device)

        n_params = count_parameters(model)
        #print(f'Initializing {arch} with {n_params} trainable parameters')

        "Parallelize if possible"
        if torch.cuda.device_count() > 1 and use_gpus > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            model = torch.nn.DataParallel(model)
        elif torch.cuda.device_count == 1 and use_gpus == 1:
            print("Using", torch.cuda.device_count(), "GPU")
        elif torch.cuda.device_count == 1 and use_gpus > 1:
            raise ValueError("use_gpus > 1: Only 1 GPU available, but you asked for more than one")

        "Define training parameters"
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Define optimizer.
        if scheduler == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs/cosine_cycles, 
                            eta_min=eta_min,last_epoch=-1)
        elif scheduler == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=reduction_factor,
                            last_epoch=-1, verbose = True)
        elif scheduler == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=reduction_factor, 
                            patience=patience, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=eta_min, 
                            eps=1e-08, verbose=True)
        elif scheduler == 'warmCosine':
            lr_scheduler = WarmStartCosineAnnealingLR(optimizer, cosine_epochs=cosine_epochs, warm_start_epochs=warm_start_epochs,
                            cosine_cycles=cosine_cycles, eta_min=eta_min,last_epoch=-1)
        elif scheduler == 'constant':
            pass


        "Train the model"
        min_val_loss = 1000.0
        max_val_F1 = 0.0
        val_losses = []
        val_F1s = []
        train_losses = []
        Us_groups_memory = []
        for epoch in range(1, epochs + 1):
            Ls_groups_train = np.random.choice(labeled_train_groups_unique,size = Ls_train,replace=False)
            "The below ensures sampling without replacement for unlabeled"
            available_Us_groups = np.setdiff1d(unlabeled_train_groups_unique,Us_groups_memory)
            if len(available_Us_groups) < Us_train:
                Us_groups_train = available_Us_groups
                try:
                    Us_groups_train2 = np.random.choice(Us_groups_memory,size = Us_train-len(Us_groups_train),replace=False)
                except:
                    "In this case it means the number of requested unlabeled subjects is larger than the number of available subjects"
                    if Us_train > len(unlabeled_train_groups_unique) and Us_groups_memory == []:
                        Us_groups_train2 = []
                    else:
                        raise ValueError("Number of requested unlabeled subjects is larger than the number of available subjects")
                Us_groups_train = np.hstack((Us_groups_train,Us_groups_train2))
                Us_groups_memory = []
                for g in Us_groups_train2:
                    Us_groups_memory.append(g)
            elif len(available_Us_groups) == Us_train:
                Us_groups_train = available_Us_groups
            else:
                Us_groups_train = np.random.choice(available_Us_groups,size = Us_train,replace=False)
                for g in Us_groups_train:
                    Us_groups_memory.append(g)
            
            Ls_train_inds = group[group.isin(Ls_groups_train)].index
            Us_train_inds = group[group.isin(Us_groups_train)].index

            "Get the sampled training data"
            Xs_L = X_sel.iloc[Ls_train_inds]
            Xs_U = X_sel.iloc[Us_train_inds]
            X_train = np.vstack((Xs_L,Xs_U))

            "Prepare train masks"
            Ls_train_inds_formask = np.linspace(0,len(Ls_train_inds)-1,len(Ls_train_inds))
            Us_train_inds_formask = np.linspace(len(Ls_train_inds),len(Ls_train_inds)+len(Us_train_inds)-1,len(Us_train_inds))

            A_train = compute_adjacency(X_train,y_target[Ls_train_inds],construction_method,distance_metric,use_labels = use_labels,
                nearest_neighbors = nearest_neighbors,farthest_neighbors = farthest_neighbors,sup_mode = sup_mode,X_labels = Xs_L,X_no_labels = Xs_U)
                        # End the timer

            "Construct training graph from adjacency matrix"
            G_train = construct_from_np_adjacency(A_train,X_train)
            data = from_networkx(G_train) #this command costs the most time ~0.15 s
            
            "Get labels"
            y_target_pseudo = torch.tensor(np.ones(len(Us_train_inds),)*(2), dtype=torch.long)
            y_target_labeled = y_target[Ls_train_inds].clone().detach() #torch.tensor(y_target[Ls_train_inds], dtype=torch.long)
            y_train_comp = torch.cat((y_target_labeled,y_target_pseudo),0)
            y_target_train = y_train_comp.clone().detach()#torch.tensor(y_train_comp, dtype=torch.long)
            data.num_classes = len(np.unique(y_target))
            data.x = data.embedding
            data.y = y_target_train
            
            "Get target class distribution based on the labeled samples"
            if weights_loss == 'weighted':
                class_weights, class_distribution = get_class_weights(y_target_labeled)
            elif weights_loss == 'non_weighted':
                class_weights = np.array([1.0,1.0])#np.array([0.47,1.0])
            class_weights = torch.FloatTensor(class_weights).to(device)

            "Define training criterion"
            if weights_loss == 'weighted':
                pass
                #print('Using weighted loss with weights ',class_weights)
            if label_smoothing > 0.0:
                pass
                #print('Using label smoothing with epsilon ',label_smoothing)
            criterion = torch.nn.CrossEntropyLoss(weight = class_weights,label_smoothing=label_smoothing)  # Define loss criterion.

            "Get labels"
            Ls_train_labels = y_target[Ls_train_inds]
            val_labels = y_target[val_inds]
            test_labels = y_target[test_inds]
            #y_target = torch.tensor(y_target, dtype=torch.long)

            "Convert data labels to tensors"
            data.x = data.x.clone().detach()#torch.tensor(data.x, dtype=torch.float)
            data.num_features = len(data.x[0])
            Ls_train_labels = Ls_train_labels.clone().detach()#torch.tensor(Ls_train_labels)
            #Us_train_labels can not be defined
            val_labels = val_labels.clone().detach()#torch.tensor(val_labels)
            test_labels = test_labels.clone().detach() #torch.tensor(test_labels)
            
            "For training the mask has to be given now, for validation after training"
            data = get_graph_masks(data,train_inds = Ls_train_inds_formask,val_inds = Us_train_inds_formask)

            "Also construct the augmented graph"
            if augmentation == 'node_dropping' or augmentation == 'edge_perturbation':
                x, edge_index, edge_weights = graph_aug(x=data.x,edge_index = data.edge_index,edge_weight = data.weight)
                data_aug = data.clone()
                data_aug.x = x
                data_aug.edge_index = edge_index
                data_aug.weight = edge_weights
            else:
                data_aug = graph_aug(data)
                data_aug.num_features = data.num_features
                data_aug.train_mask = data.train_mask
                data_aug.val_mask = data.val_mask
                data_aug.num_classes = data.num_classes
                data_aug.num_nodes = data.num_nodes
                data_aug.weight = data.weight

            #loss,pseudolabels = train(model, data, optimizer, criterion,device,lambda1 = lambda1)
            loss,pseudolabels = train_SSL(model,data,data_aug,optimizer,criterion,device,augmentation,lambda1=lambda1,lambda2=lambda2)
            y_train_new = torch.cat((y_target[Ls_train_inds],pseudolabels.cpu()),0)

            data_val,_ = inductive_val(A_train,X_val,X_train,y_target,y_train_new,val_labels,construction_method,distance_metric,mode = 'val',use_labels = True,nearest_neighbors = nearest_neighbors,farthest_neighbors = farthest_neighbors)
            valR = test(data_val.val_mask,model,data_val,criterion,device)

            val_acc = valR['acc']; val_loss = valR['loss']
            val_F1 = valR['f1_macro']
            val_F1s.append(val_F1)
            val_losses.append(val_loss)
            train_losses.append(loss.detach().cpu().numpy())
            #print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}')
            #print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
            "Check if validation loss is decreasing"
            """if epoch > 10:
                runningLoss = np.mean(val_losses[-10:])
                if runningLoss < min_val_loss:
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
            """
            if epoch > 5:
                if max_val_F1 < val_F1:
                    max_val_F1 = val_F1

            if scheduler == 'plateau':
                lr_scheduler.step(val_loss)
            elif scheduler == 'constant':
                pass
            else:
                lr_scheduler.step()
        
        best_val_losses.append(min_val_loss)
        best_val_F1s.append(max_val_F1)

    print('Finished this iteration of the optimization')   
    # End the timer
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    "Monitoring metric for optimization"
    #return -(np.mean(best_val_losses))
    return np.mean(best_val_F1s)

def bayes_opt_cv_for_discrete(lambda1,lambda2,label_smoothing,aug_probability):
    num_layers_d = round(num_layers)
    hidden_channels_d = round(hidden_channels)
    weights_loss_b = round(weights_loss)
    if weights_loss_b == 0:
        weights_loss_b = 'non_weighted'
    elif weights_loss_b == 1:
        weights_loss_b = 'weighted'
    use_lnorm_b = round(use_lnorm)
    if use_lnorm_b == 0:
        use_lnorm_b = False
    elif use_lnorm_b == 1:
        use_lnorm_b = True
    return bayes_opt_cv(lambda1,dropout,weights_loss_b,lr, num_layers_d,hidden_channels_d,use_lnorm_b)


if run_new is True:
    "Bounded region of parameter space"
    pbounds = {'lambda1': (0.01, 0.5), 'lambda2':(0.01,0.5),
               'label_smoothing': (0.0, 0.5), 'aug_probability': (0.05, 0.3)}
    
    optimizer = BayesianOptimization(
        f=bayes_opt_cv,
        pbounds=pbounds,
        random_state=1,
        verbose = 2,
    )
    #bayes_opt_cv_for_discrete
    "Create logger"
    logger = JSONLogger(path=PATH_LOGS_JSON)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=3,
        n_iter=15
    )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    
    print(optimizer.max)

else:
    print("Not running new optimization")
    "Load previous progress from logger"
    pbounds = {'lambda1': (0.01, 0.5), 'lambda2':(0.1,0.5),
               'label_smoothing': (0.0, 0.5), 'aug_probability': (0.05, 0.3)}

    optimizer = BayesianOptimization(
        f=bayes_opt_cv,
        pbounds=pbounds,
        random_state=1,
        verbose = 2,
    )
    
    load_logs(optimizer, logs=[PATH_LOGS_JSON + '.json'])

    print("Optimizer is already aware of {} points.".format(len(optimizer.space)))

    "Open logger"
    logger = JSONLogger(path=PATH_LOGS_JSON)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=0,
        n_iter=15
    )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)

# %%
