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
import transforms as graph_transforms
import GCL.augmentors as A

def visualize(h, color,save_name):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="tab20b")
    if not os.path.exists(save_name):
        plt.savefig(save_name)

def plot_graph(g, y,legend_labels,save_graph_name = None):
    color_map = plt.cm.get_cmap('Set1', len(np.unique(y)))

    plt.figure(figsize=(9, 7))
    nx.draw_spring(g, node_size=30, arrows=False,width = 0.2, node_color=y,cmap=color_map)
    handles = [mpatches.Patch(color=color_map(i), label=legend_labels[i]) for i in range(len(np.unique(y)))]
    plt.legend(handles=handles)
    if save_graph_name is not None and not os.path.exists(save_graph_name):
        plt.savefig(save_graph_name)
    #else:
    #plt.show() 

RANDOM_STATE = 42   
now = datetime.now()
now_str = now.strftime("%d_%m_%Y__%H_%M_%S")
PATH_FILES = os.path.dirname(os.getcwd())
PATH_INTERMEDIATE = os.path.join(PATH_FILES,'intermediate_ass4')#'intermediate_investigation') 
PATH_RESULTS = os.path.join(PATH_FILES,'results_self_sup_induct_ablate_probs_'+now_str)
PATH_LOGS = os.path.join(PATH_FILES,'logs_self_sup_induct_ablate_probs_'+now_str)
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)
if not os.path.exists(PATH_LOGS):
    os.makedirs(PATH_LOGS)
PATH_TRAINING_GRAPHS = os.path.join(PATH_RESULTS,'graphs','training_graphs')
if not os.path.exists(PATH_TRAINING_GRAPHS):
    os.makedirs(PATH_TRAINING_GRAPHS)
PATH_TEST_GRAPHS = os.path.join(PATH_RESULTS,'graphs','test_graphs')
if not os.path.exists(PATH_TEST_GRAPHS):
    os.makedirs(PATH_TEST_GRAPHS)

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
sup_mode = 'self_sup' # 'fully-sup' / 'semi-sup' / 'self-sup'
#render_unlabeled = 0.8 #Percentage (/1) of unlabeled nodes to render
use_labels = True # True or False
subgraph_sampling = 'random_both' # / random_strat_labeled / random_both / random_strat_both (without replacement)
label = 'valence_bin'
if label == 'arousal_bin':
    K_train_percentage = 0.25
elif label == 'valence_bin':
    K_train_percentage = 0.2

"Training set sampling parameters"
K_train = int(np.floor(K_train_percentage*subjects)) #20% - Number of training participants selected to be labeled from training set
labeled_use_percentage = 1
unlabeled_use_percentage = 0.5
if K_train % 2 == 0:
    Ls_train = int(labeled_use_percentage*K_train)
    Us_train = int(unlabeled_use_percentage*K_train)
elif K_train % 2 == 1:
    Ls_train = int(np.floor(labeled_use_percentage*K_train)) #participants in training set with labels
    Us_train = int(np.ceil(unlabeled_use_percentage*K_train)) #participants in training set without labels
#assert K_train == Ls_train + Us_train
#labels = ['arousal_bin','valence_bin']#,'emo_quad']
run_until = 9 #run until fold number (including that fold)
early_break = False
save_loss_plot = False
plot_step = 5 #plot training graph every plot_step epochs
plot_number = 5 #number of times to plot training graphs
save_train_graph = True
save_test_graph = True
plot_times = 0 #for plotting training graphs - only for the first fold is enough
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
if cv_mode == 'hold_out':
    n_splits = 1
elif cv_mode == 'kfold':
    n_splits = 5
elif cv_mode == 'leave_one_out':
    n_splits = 47

rand_iter = 1
if cv_mode == 'hold_out':
    rand_iter = 1
train_size = 0.60
val_size = 0.15
test_size = 0.25

"Hyperparameters"
"Non-tunable"
nearest_neighbors = 2
if construction_method == 'nearest_neighbors':
    nearest_neighbors += 1
farthest_neighbors = 1
activation = 'tanh'
lr = 0.0055
hidden_channels = 96
weights_loss = 'non_weighted'
epochs = 200
label_smoothing = 0.1
num_layers = 3
lambda1 = 0.0#Regularization parameter for semi-supervised pseudo-labeling loss
dropout = 0.5
use_lnorm = False


"Optimizer, Scheduler Parameters"
eta_min = 1e-6
weight_decay = 1e-5
scheduler = 'constant' #Options: cosine, step, plateau, warmCosine, constant
if scheduler == 'cosine':
    cosine_cycles = 4
    cosine_epochs = epochs
    eta_min = 1e-6
elif scheduler == 'warmCosine':
    warm_start_epochs = 10
    cosine_cycles = 4
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

"Self-supervised hyperparameters"
aug_prob_vec = [0.05,0.1,0.15,0.20,0.25]
lambda2 = 0.2 #Regularization parameter for self-supervised loss
augmentation = 'edge_removing' #Options: c, shuffling, completion, mix


for aug_prob in aug_prob_vec: 

    "Measure execution time"
    start_time = time.time()

    if augmentation == 'denoising':
        graph_aug = graph_transforms.Denoising(p=aug_prob, mu=0.0, sigma=1.0)
    elif augmentation == 'feature_masking':
        graph_aug = graph_transforms.Completion(p=aug_prob,masking = 'feature')
    elif augmentation == 'node_masking':
        graph_aug = graph_transforms.Completion(p=aug_prob,masking = 'node')
    elif augmentation == 'edge_removing':
        graph_aug = A.EdgeRemoving(pe=aug_prob)
        #graph_aug = A.RandomChoice([A.EdgeAdding(pe=aug_prob),
        #                            A.EdgeRemoving(pe=aug_prob)],
        #                            num_choices=1)
    elif augmentation == 'shuffling':
        pass
    elif augmentation == 'node_dropping':
        pass
        #graph_aug = A.NodeDropping(pn=node_dropping_prob)

    "Create logging directory"
    log_dir = os.path.join(PATH_LOGS,cv_mode + '_'+ str(aug_prob) + '_augm_prob_' + augmentation)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    "Create results directory"
    results_dir = os.path.join(PATH_RESULTS,cv_mode + '_'+ str(aug_prob) + '_augm_prob_' + augmentation)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


    "Load data"
    X, y, group, t, _ = load(os.path.join(PATH_INTERMEDIATE, use_data + '_' + subset +'_all_labels.pkl'))

    "Process label info - Encode labels and drop label-related features"
    X,y,group,old_index = process_label_info(X,y,group,label)

    "Select target class, needed in Kfold CV"
    y_target = y[label].values

    if run_until is not None and early_break:
        n_splits = run_until + 1

    "Initialize storing variables"
    if cv_mode == 'hold_out':
        test_acc_total = []
        test_f1_total = []
        test_f1low_total = []
        test_f1high_total = []
        test_pre_macro_total = []
        test_rec_macro_total = []
        test_prelow_total = []
        test_reclow_total = []
        test_prehigh_total = []
        test_rechigh_total = []
    else:
        test_acc_total = np.zeros((rand_iter,n_splits))
        test_f1_total = np.zeros((rand_iter,n_splits))
        test_f1low_total = np.zeros((rand_iter,n_splits))
        test_f1high_total = np.zeros((rand_iter,n_splits))
        test_pre_macro_total = np.zeros((rand_iter,n_splits))
        test_rec_macro_total = np.zeros((rand_iter,n_splits))
        test_prelow_total = np.zeros((rand_iter,n_splits))
        test_reclow_total = np.zeros((rand_iter,n_splits))
        test_prehigh_total = np.zeros((rand_iter,n_splits))
        test_rechigh_total = np.zeros((rand_iter,n_splits))
    test_cm_total = []


    "Cross-validation loop"
    for i,(X_sel,X_labeled_train,X_unlabeled_train,X_val,X_test,labeled_train_inds, unlabeled_train_inds, val_inds, test_inds, _) in enumerate(cv_split(X,y[label],group,SELECT_SVC,train_size,val_size,data_subset = subset,cv_mode = cv_mode,subgraph_sampling = subgraph_sampling,K_train = K_train)):
        if early_break and i > run_until:
            break

        "Select target class, needed in Kfold CV"
        y_target = y[label].values

        #y_pseudo = np.ones((X_sel.shape[0],epochs))*(-1) 
        labeled_train_groups = group[labeled_train_inds]
        unlabeled_train_groups = group[unlabeled_train_inds]
        labeled_train_groups_unique = np.unique(labeled_train_groups)
        unlabeled_train_groups_unique = np.unique(unlabeled_train_groups)
        "We have to save pseudo-labels for the unlabeled subjects after each epoch"
        "Now we have to get the labels for the labeled training set - make a labeled training mask"
        
        "Log fold"
        log_dir_fold = os.path.join(log_dir,'fold_'+str(i+1))
        if not os.path.exists(log_dir_fold):
            os.makedirs(log_dir_fold)
        
        print(f'Augmentation Probability: {aug_prob} / Cross-validation mode: {cv_mode} - fold {i+1} of {n_splits} for {arch}, label = {label}')
        
        num_features = X_sel.shape[1]
        num_classes = len(np.unique(y_target))
        y_target = torch.from_numpy(y_target)
                
        "Initialize shared encoder instance"
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
        print(f'Initializing {arch} with {n_params} trainable parameters')

        "Parallelize if possible"
        if torch.cuda.device_count() > 1 and use_gpus > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            model = torch.nn.DataParallel(model)
        elif torch.cuda.device_count == 1 and use_gpus == 1:
            print("Using", torch.cuda.device_count(), "GPU")
        elif torch.cuda.device_count == 1 and use_gpus > 1:
            raise ValueError("use_gpus > 1: Only 1 GPU available, but you asked for more than one")

        "Visualize embeddings before training"
        save_emb_before_dir = os.path.join(results_dir,'untrained_embeddings','fold_'+str(i+1))
        if not os.path.exists(save_emb_before_dir):
            os.makedirs(save_emb_before_dir)
        save_emb_after_dir = os.path.join(results_dir,'trained_embeddings','fold_'+str(i+1))
        if not os.path.exists(save_emb_after_dir):
            os.makedirs(save_emb_after_dir)

        save_emb_before = os.path.join(save_emb_before_dir, r''+arch + '_' + label +  '_' + 'fold' + str(i+1) +  '_loss_' + weights_loss+'_untrained_embedding.png')
        save_emb_after = os.path.join(save_emb_after_dir,r''+ arch + '_' + label +  '_' + 'fold' + str(i+1) + '_loss_' + weights_loss+'_trained_embedding.png')

        """
        model.eval()
        with torch.no_grad():
            x = data.x.to(device, dtype = torch.float)
            edge_index = data.edge_index.to(device)
            #if arch == 'MLP':
            #    out = model(x) #data.x)
            #else:
            out = model(x, edge_index,mode = 'visualize') #data.x
            visualize(out, data.y,save_emb_before)
        model.train()
        """
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
        
        "Check if it works for hold_out - Adjust saves"

        "Train the model"
        min_val_loss = 1000.0
        val_losses = []
        train_losses = []
        Us_groups_memory = []
        for epoch in range(1, epochs + 1):
            "Subgraph sampling"
            "At each fold iteration, a number of labeled training participants (9) are sampled from the training set"
            "At each epoch during training: randomly sample Ns_train subjects from training set"
            "Ns_train = Ls_train + Us_train"
            "Ns_train = 9, Ls_train = 4, Us_train = 5"
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
            #train_mask = np.zeros(X_train.shape[0],dtype = bool)
            #Ls_train_inds_formask = train_mask.copy()
            #Ls_train_inds_formask[:len(Ls_train_inds)] = True
            Ls_train_inds_formask = np.linspace(0,len(Ls_train_inds)-1,len(Ls_train_inds))
            Us_train_inds_formask = np.linspace(len(Ls_train_inds),len(Ls_train_inds)+len(Us_train_inds)-1,len(Us_train_inds))

            #Here instead of compute, get from already calculated adjacency matrix
            #We were here
            A_train = compute_adjacency(X_train,y_target[Ls_train_inds],construction_method,distance_metric,use_labels = use_labels,
                nearest_neighbors = nearest_neighbors,farthest_neighbors = farthest_neighbors,sup_mode = sup_mode,X_labels = Xs_L,X_no_labels = Xs_U)
                        # End the timer

            "Construct training graph from adjacency matrix"
            G_train = construct_from_np_adjacency(A_train,X_train)
            data = from_networkx(G_train) #this command costs the most time ~0.15 s           

            "Get labels"
            y_target_pseudo = torch.tensor(np.ones(len(Us_train_inds),)*(2), dtype=torch.long)
            y_target_labeled = y_target[Ls_train_inds].clone().detach()
            y_train_comp = torch.cat((y_target_labeled,y_target_pseudo),0)
            y_target_train = y_train_comp.clone().detach()
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
                print('Using weighted loss with weights ',class_weights)
            if label_smoothing > 0.0:
                print('Using label smoothing with epsilon ',label_smoothing)
            criterion = torch.nn.CrossEntropyLoss(weight = class_weights,label_smoothing=label_smoothing)  # Define loss criterion.

            if epoch % plot_step == 0 and plot_times <= plot_number:
                if save_train_graph: 
                    save_graph_name = 'subsampled_training_graph_' + label
                    save_graph_dir = os.path.join(PATH_TRAINING_GRAPHS,cv_mode + '_'+ str(aug_prob) + '_augm_prob_' + augmentation)
                    if not os.path.exists(save_graph_dir):
                        os.makedirs(save_graph_dir)
                    save_graph_name = os.path.join(save_graph_dir,save_graph_name + '_fold_'+str(i+1) + '_epoch_' + str(epoch) +'.png')
                "Plot the graph"
                legend_labels = ['low_'+label,'high_'+label,'unlabeled_'+label]
                y_plot = np.hstack((y_target_labeled.numpy(),y_target_pseudo.numpy()))
                plot_graph(G_train,y_plot,legend_labels,save_graph_name = save_graph_name)
                plot_times += 1

            "Get labels"
            Ls_train_labels = y_target[Ls_train_inds]
            val_labels = y_target[val_inds]
            test_labels = y_target[test_inds]
            #y_target = torch.tensor(y_target, dtype=torch.long)

            "Convert data labels to tensors"
            data.x = data.x.clone().detach()
            data.num_features = len(data.x[0])
            Ls_train_labels = Ls_train_labels.clone().detach()
            #Us_train_labels can not be defined
            val_labels = val_labels.clone().detach()
            test_labels = test_labels.clone().detach()
            
            "For training the mask has to be given now, for validation after training"
            data = get_graph_masks(data,train_inds = Ls_train_inds_formask)#,val_inds = Us_train_inds_formask)

            "Also construct the augmented graph"
            if augmentation == 'node_dropping' or augmentation == 'edge_removing' or augmentation == 'edge_perturbation':
                x, edge_index, edge_weights = graph_aug(x=data.x,edge_index = data.edge_index,edge_weight = data.weight)
                data_aug = data.clone()
                data_aug.x = x
                data_aug.edge_index = edge_index
                data_aug.weight = edge_weights
            else:
                data_aug = graph_aug(data)
                data_aug.num_features = data.num_features
                data_aug.train_mask = data.train_mask
                #data_aug.val_mask = data.val_mask
                data_aug.num_classes = data.num_classes
                data_aug.num_nodes = data.num_nodes
                data_aug.weight = data.weight

            #loss,pseudolabels = train(model, data, optimizer, criterion,device,lambda1 = lambda1)
            loss,_ = train_SSL(model,data,data_aug,optimizer,criterion,device,augmentation,lambda1=lambda1,lambda2=lambda2)
            y_train_new = y_target[Ls_train_inds].clone().detach()
            "Add a placeholder y_unlabeled vector for the unlabeled subjects"
            y_unlabeled = torch.tensor(np.ones(len(Us_train_inds),)*(-1), dtype=torch.long)
            y_train_new = torch.cat((y_train_new,y_unlabeled),0)

            data_val,_ = inductive_val(A_train,X_val,X_train,y_target,y_train_new,val_labels,construction_method,distance_metric,mode = 'val',use_labels = True,nearest_neighbors = nearest_neighbors,farthest_neighbors = farthest_neighbors)
            valR = test(data_val.val_mask,model,data_val,criterion,device)

            val_acc = valR['acc']; val_loss = valR['loss']
            val_losses.append(val_loss)
            train_losses.append(loss.detach().cpu().numpy())
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
            "Check if validation loss is decreasing"
            if epoch > 5:
                #runningLoss = np.mean(val_losses[-10:])
                #if runningLoss < min_val_loss:
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    best_epoch = epoch
                    print('Saving model...')
                    torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, 
                        os.path.join(log_dir_fold,'{}_{}_E{}_fold_{}_loss_{}_best.pt'.format(
                                        arch,label, str(best_epoch),str(i+1),weights_loss)))
                    if learning == 'transductive':
                        testR = test(data.test_mask,model,data,criterion,device)
                    elif learning == 'inductive':
                        data_test,_ = inductive_val(A_train,X_test,X_train,y_target,y_train_new,test_labels,construction_method,distance_metric,mode = 'test',use_labels = True,nearest_neighbors = nearest_neighbors,farthest_neighbors = farthest_neighbors)
                        testR = test(data_test.test_mask,model,data_test,criterion,device)
                    test_acc = testR['acc']; test_f1 = testR['f1_macro']
                    test_f1low = testR['f1_0']; test_f1high = testR['f1_1']
                    #print(f'Test Accuracy: {test_acc:.4f}')
                    #print(f'Test F1 macro: {test_f1:.4f}')
                    #print(f'Test F1 Low: {test_f1low:.4f}')
                    #print(f'Test F1 High: {test_f1high:.4f}')

            if scheduler == 'plateau':
                lr_scheduler.step(val_loss)
            elif scheduler == 'constant':
                pass
            else:
                lr_scheduler.step()
            try:
                print(f'Learning rate: {lr_scheduler.get_last_lr()}')
            except:
                pass

        "Plot and save training and validation losses"
        plt.figure(figsize=(10,7))
        plt.plot(train_losses,label = 'Train Loss')
        plt.plot(val_losses,label = 'Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        if save_loss_plot:
            save_loss_dir = os.path.join(results_dir,'loss_plots','fold_'+str(i+1))
            if not os.path.exists(save_loss_dir):
                os.makedirs(save_loss_dir)
            save_loss_plot = os.path.join(save_loss_dir,arch + '_' + label +  '_' + 'fold' + str(i+1) + '_loss_' + weights_loss+'_loss_plot.png')
            plt.savefig(save_loss_plot)
        else:
            plt.show()

        "Evaluate on test set with best model"
        best_pt = torch.load(os.path.join(log_dir_fold,'{}_{}_E{}_fold_{}_loss_{}_best.pt'.format(
                                    arch,label, str(best_epoch),str(i+1),weights_loss)))           
        state_dict = best_pt['model_state_dict']
        "Create new instance of best model"
        if arch == 'GCN_basic': 
            model = GCN_basic(num_layers = num_layers,activation = activation,dropout = dropout,
                hidden_channels=hidden_channels,input_size=data.num_features,output_size=data.num_classes,use_lnorm = use_lnorm).to(device)
        elif arch == 'GCN_basic_proj':
            model = GCN_basic_projection(num_layers = num_layers,activation = activation,dropout = dropout,
                hidden_channels=hidden_channels,input_size=num_features,output_size=num_classes,use_lnorm = use_lnorm).to(device)
        elif arch == 'GCN_test':
            model = GCN_test(num_layers = num_layers,activation = activation,dropout = dropout,
                hidden_channels=hidden_channels,input_size=num_features,output_size=num_classes,use_lnorm = use_lnorm).to(device)
        elif arch == 'GAT_basic': 
            model = GAT_basic(num_layers = num_layers,activation = activation,dropout = dropout,
                hidden_channels=hidden_channels,input_size=data.num_features,output_size=data.num_classes,heads=att_heads,use_lnorm = use_lnorm).to(device)
        elif arch == 'SAGE_basic':   
            model = SAGE_basic(num_layers = num_layers,activation = activation,dropout = dropout,
                aggregation = agg_func,hidden_channels=hidden_channels,input_size=data.num_features,output_size=data.num_classes,use_lnorm = use_lnorm).to(device)

        "Transfer the learned weights"
        model.load_state_dict(state_dict) 

        "Sample again labeled/unlabeled data"
        all_groups_train_S = np.random.choice(labeled_train_groups_unique,size = K_train,replace = False)
        all_train_inds = group[group.isin(all_groups_train_S)].index
        #labeled_train_inds is calculated from before
            
        "Get the sampled training data"
        X_train_inf = X_sel.iloc[all_train_inds]
        
        y_train_inf = y_target[all_train_inds]    
        
        y_plot = y_train_inf.clone() 

        A_train_inf = compute_adjacency(X_train_inf,y_train_inf,construction_method,distance_metric,use_labels = True,
                nearest_neighbors = nearest_neighbors,farthest_neighbors = farthest_neighbors,sup_mode = None,X_labels = None,X_no_labels = None)
        
        data_test,G_test = inductive_val(A_train_inf,X_test,X_train_inf,y_target,y_train_inf,test_labels,construction_method,distance_metric,mode = 'test',use_labels = True,nearest_neighbors = nearest_neighbors,farthest_neighbors = farthest_neighbors)
        testR = test(data_test.test_mask,model,data_test,criterion,device)
        y_test_plot = data_test.y[-len(test_labels):].detach().numpy()
        y_test_plot[y_test_plot == 0] = 2
        y_test_plot[y_test_plot == 1] = 3
        y_plot = np.concatenate((y_plot,y_test_plot),axis = 0)

        "Plot test graph"
        "See the labels for the plot - y_plot"
        if save_test_graph: 
            save_graph_name = 'test_graph_' + label
            save_graph_dir = os.path.join(PATH_TEST_GRAPHS,cv_mode + '_'+ str(aug_prob) + '_augm_prob_'+ augmentation)
            if not os.path.exists(save_graph_dir):
                os.makedirs(save_graph_dir)
            save_graph_name = os.path.join(save_graph_dir,save_graph_name + '_fold_' + str(i+1) + '.png')
            legend_labels = ['low_'+label,'high_'+label,'test_low_'+label,'test_high_'+label]
            plot_graph(G_test,y_plot,legend_labels,save_graph_name = save_graph_name)

        "Get test metrics"
        test_acc_final = testR['acc']; test_f1 = testR['f1_macro']
        test_f1low = testR['f1_0']; test_f1high = testR['f1_1']
        test_pre_macro = testR['pre_macro']; test_rec_macro = testR['rec_macro']
        test_pre_low = testR['pre_0']; test_rec_low = testR['rec_0']
        test_pre_high = testR['pre_1']; test_rec_high = testR['rec_1']
        print(f'Test Accuracy: {test_acc_final:.4f}')
        print(f'Test F1 macro: {test_f1:.4f}')
        print(f'Test F1 Low: {test_f1low:.4f}')
        print(f'Test F1 High: {test_f1high:.4f}')

        "Append group results to total results"
        if cv_mode == 'hold_out':
            test_acc_total.append(test_acc_final)
            test_f1_total.append(test_f1)
            test_f1low_total.append(test_f1low)
            test_f1high_total.append(test_f1high)
            test_pre_macro_total.append(testR['pre_macro'])
            test_rec_macro_total.append(testR['rec_macro'])
            test_prelow_total.append(testR['pre_0'])
            test_reclow_total.append(testR['rec_0'])  
            test_prehigh_total.append(testR['pre_1'])
            test_rechigh_total.append(testR['rec_1'])
        else:
            test_acc_total[0,i] = test_acc_final
            test_f1_total[0,i] = test_f1
            test_f1low_total[0,i] = test_f1low
            test_f1high_total[0,i] = test_f1high
            test_pre_macro_total[0,i] = testR['pre_macro']
            test_rec_macro_total[0,i] = testR['rec_macro']
            test_prelow_total[0,i] = testR['pre_0']
            test_reclow_total[0,i] = testR['rec_0']   
            test_prehigh_total[0,i] = testR['pre_1']
            test_rechigh_total[0,i] = testR['rec_1']
        test_cm_total.append(testR['confMat'])

        "Save evaluation results"
        save_dir = os.path.join(results_dir,'eval','fold_'+str(i+1))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_eval_results = os.path.join(save_dir, arch + '_' + label + '_' + 'fold' + str(i+1) + '_loss_' + weights_loss+ '_test_results.pkl')
        if not os.path.exists(save_eval_results):
            dump(testR,save_eval_results)

        "Visualize embeddings after training"
        model.eval()
        x = data.x.to(device, dtype = torch.float)
        edge_index = data.edge_index.to(device)
        #if arch == 'MLP':
        #    out = model(x) #data.x
        #else:
        out = model(x, edge_index,mode = 'visualize') #data.
        visualize(out, data.y, save_emb_after)

    if not cv_mode == 'hold_out':
        "Shuffle within train/val split to get more robust results"
        mean_acc = np.zeros(n_splits,)
        mean_acc_low = np.zeros(n_splits,)
        mean_acc_high = np.zeros(n_splits,)
        mean_f1 = np.zeros(n_splits,)
        mean_f1_low = np.zeros(n_splits,)
        mean_f1_high = np.zeros(n_splits,)
        mean_f1low = np.zeros(n_splits,)
        mean_f1low_low = np.zeros(n_splits,)
        mean_f1low_high = np.zeros(n_splits,)
        mean_f1high = np.zeros(n_splits,)
        mean_f1high_low = np.zeros(n_splits,)
        mean_f1high_high = np.zeros(n_splits,)
        for k in range(n_splits):
            mean_acc[k],mean_acc_low[k],mean_acc_high[k] = mean_confidence_interval(
                        test_acc_total[:,k], confidence=0.95)
            mean_f1[k],mean_f1_low[k],mean_f1_high[k] = mean_confidence_interval(
                        test_f1_total[:,k], confidence=0.95)
            mean_f1low[k],mean_f1low_low[k],mean_f1low_high[k] = mean_confidence_interval(
                        test_f1low_total[:,k], confidence=0.95)
            mean_f1high[k],mean_f1high_low[k],mean_f1high_high[k] = mean_confidence_interval(
                        test_f1high_total[:,k], confidence=0.95)

        print('Average Fold Test Accuracy: ',mean_acc)
        print('Average Fold Test F1 macro: ',mean_f1)
        print('Average Fold Test F1 Low: ',mean_f1low)
        print('Average Fold Test F1 High: ',mean_f1high)
        mm_acc = np.mean(mean_acc)
        mm_f1 = np.mean(mean_f1)
        mm_f1low = np.mean(mean_f1low)
        mm_f1high = np.mean(mean_f1high)
        print(f'Average Overall Test Accuracy: {mm_acc:.4f}')
        print(f'Average Overall Test F1 macro: {mm_f1:.4f}')
        print(f'Average Overall Test F1 Low: {mm_f1low:.4f}')
        print(f'Average Overall Test F1 High: {mm_f1high:.4f}')

        save_dir = os.path.join(results_dir,'eval','overall')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_eval_results1 = os.path.join(save_dir, arch + '_' + label + '_loss_' + weights_loss+'_overall_test_results.pkl')
        save_eval_results2 = os.path.join(save_dir, arch + '_' + label + '_loss_' + weights_loss+'_per_fold_test_results.pkl')
        if not os.path.exists(save_eval_results1):
            dump(testR,save_eval_results1)
        if not os.path.exists(save_eval_results2):
            perFoldResults = {'test_acc':test_acc_total,'test_f1':test_f1_total,'test_f1low':test_f1low_total,
                            'test_f1high':test_f1high_total,'test_pre':test_pre_macro_total,'test_rec':test_rec_macro_total,
                            'test_prelow':test_prelow_total,'test_reclow':test_reclow_total,'test_prehigh':test_prehigh_total,
                            'test_rechigh':test_rechigh_total,
                            'test_cm':test_cm_total}
            dump(perFoldResults,save_eval_results2)


    # End the timer
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

# %%
#See last run's results

keyss = ['test_acc','test_f1','test_f1low','test_f1high']#perFoldResults.keys()
for key in keyss:
    if key == 'test_cm':
        continue
    test_metric = perFoldResults[key].squeeze()
    if test_metric.ndim == 1:
        mean_metric,mean_metric_low,mean_metric_high = mean_confidence_interval(
                        test_metric, confidence=0.95)
        
    else:
        mean_metric = np.zeros(n_splits,)
        mean_metric_low = np.zeros(n_splits,)
        mean_metric_high = np.zeros(n_splits,)
        for k in range(n_splits):
            mean_acc[k],mean_metric_low[k],mean_metric_high[k] = mean_confidence_interval(
                        test_metric[:,k], confidence=0.95)

    print(f'Average Fold {key}: ',mean_metric)
    print(f'Average Fold {key} Low CI: ',mean_metric_low)
    print(f'Average Fold {key} High CI: ',mean_metric_high)

#Calculate overall confusion matrix
cm = perFoldResults['test_cm']
cm_stacked = np.stack(cm,axis = 2)
cum_cm = np.sum(cm_stacked,axis = 2)
norm_cum_cm = np.sum(cm_stacked,axis = 2)/np.sum(np.sum(np.sum(cm_stacked,axis = 2)))
plt.figure(figsize=(10,7))
sns.heatmap(cum_cm, annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()



