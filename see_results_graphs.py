#%%
import os
from utils import load, dump
import numpy as np
from training_utils import mean_confidence_interval

PATH_FILES = os.path.dirname(os.getcwd())
PATH_RESULTS = os.path.join(PATH_FILES,'results_self_semi_sup_induct3_24_05_2024__02_10_24','leave_one_out_edge_removing','eval')
#mode = 'leave_one_out_3_labeled_subjects'
label = 'arousal_bin'#['valence_bin','arousal_bin'#,'emo_quad']
weights_loss = 'non_weighted' #Options: 'weighted_loss', 'non_weighted'
arch = 'GCN_basic_proj' #Options: GCN_basic, GAT_basic, MLP, SAGE_basic
eval_ = 'fold_iter'#'overall'
n_splits = 47

test_acc_total = np.zeros((1,n_splits))
test_f1_total = np.zeros((1,n_splits))
test_f1low_total = np.zeros((1,n_splits))
test_f1high_total = np.zeros((1,n_splits))


if eval_ != 'overall':
    #PATH_RESULTS = os.path.join(PATH_FILES,sup_mode,mode,'eval')
    for i in range(n_splits):
        iter_fold_path = os.path.join(PATH_RESULTS, 'fold_'+str(i+1))
        file_path = os.path.join(iter_fold_path, arch +'_' + label +  '_' + 'fold' + str(i+1) + '_loss_' + weights_loss+ '_test_results.pkl')
        testR = load(file_path)
        
        test_acc_total[0,i] = testR['acc']; test_f1_total[0,i] = testR['f1_macro']
        test_f1low_total[0,i] = testR['f1_0']; test_f1high_total[0,i] = testR['f1_1']

    "Calculate stats"
    mean_acc = np.zeros(n_splits,)
    std_acc = np.zeros(n_splits,)
    mean_acc_low = np.zeros(n_splits,)
    mean_acc_high = np.zeros(n_splits,)
    mean_f1 = np.zeros(n_splits,)
    std_f1 = np.zeros(n_splits,)
    mean_f1_low = np.zeros(n_splits,)
    mean_f1_high = np.zeros(n_splits,)
    mean_f1low = np.zeros(n_splits,)
    std_f1low = np.zeros(n_splits,)
    mean_f1low_low = np.zeros(n_splits,)
    mean_f1low_high = np.zeros(n_splits,)
    mean_f1high = np.zeros(n_splits,)
    std_f1high = np.zeros(n_splits,)
    mean_f1high_low = np.zeros(n_splits,)
    mean_f1high_high = np.zeros(n_splits,)
    for k in range(n_splits):
        mean_acc[k],mean_acc_low[k],mean_acc_high[k] = mean_confidence_interval(
                    test_acc_total[:,k], confidence=0.95)
        std_acc[k] = np.std(test_acc_total[:,k])
        mean_f1[k],mean_f1_low[k],mean_f1_high[k] = mean_confidence_interval(
                    test_f1_total[:,k], confidence=0.95)
        std_f1[k] = np.std(test_f1_total[:,k])
        mean_f1low[k],mean_f1low_low[k],mean_f1low_high[k] = mean_confidence_interval(
                    test_f1low_total[:,k], confidence=0.95)
        std_f1low[k] = np.std(test_f1low_total[:,k])
        mean_f1high[k],mean_f1high_low[k],mean_f1high_high[k] = mean_confidence_interval(
                test_f1high_total[:,k], confidence=0.95)
        std_f1high[k] = np.std(test_f1high_total[:,k])

    print('Average Fold Test Accuracy: ',mean_acc)
    print('STD Fold Test Accuracy: ',std_acc)
    print('Average Fold Test F1 macro: ',mean_f1)
    print('STD Fold Test F1 macro: ',std_f1)
    print('Average Fold Test F1 Low: ',mean_f1low)
    print('STD Fold Test F1 Low: ',std_f1low)
    print('Average Fold Test F1 High: ',mean_f1high)
    print('STD Fold Test F1 High: ',std_f1high)
    mm_acc = np.mean(mean_acc)
    std_acc_overall = np.std(test_acc_total,axis = None)
    mm_f1 = np.mean(mean_f1)
    std_f1_overall = np.std(test_f1_total,axis = None)
    mm_f1low = np.mean(mean_f1low)
    std_f1low_overall = np.std(test_f1low_total,axis = None)
    mm_f1high = np.mean(mean_f1high)
    std_f1low_overall = np.std(test_f1low_total,axis = None)
    print(f'Average Overall Test Accuracy: {mm_acc:.4f}')
    print(f'STD Overall Test Accuracy: {std_acc_overall:.4f}')
    print(f'Average Overall Test F1 macro: {mm_f1:.4f}')
    print(f'STD Overall Test F1 macro: {std_f1_overall:.4f}')
    print(f'Average Overall Test F1 Low: {mm_f1low:.4f}')
    print(f'STD Overall Test F1 Low: {std_f1low_overall:.4f}')
    print(f'Average Overall Test F1 High: {mm_f1high:.4f}')
    print(f'STD Overall Test F1 High: {std_f1low_overall:.4f}')

else:
    "Averages over all splits and random iterations" 
    PATH_RESULTS = os.path.join(PATH_FILES,sup_mode,mode,'eval',eval_)
    testR = load(os.path.join(PATH_RESULTS, use_data + '_' + subset + '_' + arch +'_' + label +  '_' + eval_ + '_test_results.pkl'))
    test_acc = testR['acc']; test_f1 = testR['f1_macro']
    test_f1low = testR['f1_0']; test_f1high = testR['f1_1']
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1 macro: {test_f1:.4f}')
    print(f'Test F1 Low: {test_f1low:.4f}')
    print(f'Test F1 High: {test_f1high:.4f}')
# %%
