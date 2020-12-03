from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import numpy as np
import math

def aupr(targets, predictions):
    aupr_array = []
        
    for i in range(targets.shape[1]):        
        precision, recall, _ = precision_recall_curve(targets[:,i], predictions[:,i], 
                                                      pos_label=1)
        
        auPR = auc(recall, precision)
        if not math.isnan(auPR):
            aupr_array.append(np.nan_to_num(auPR))
       
    
    aupr_array = np.array(aupr_array)
    mean = np.mean(aupr_array)
    median = np.median(aupr_array)
    var = np.var(aupr_array)
    
    return (mean, median, var), aupr_array

def auroc(targets, predictions):
    auroc_array = []
    for i in range(targets.shape[1]):        
        auroc = roc_auc_score(targets[:,i], predictions[:,i])
        auroc_array.append(auroc)
    
    auroc_array = np.array(auroc_array)
    mean = np.mean(auroc_array)
    median = np.median(auroc_array)
    var = np.var(auroc_array)
    
    return (mean, median, var), auroc_array


def evaluate_model(model, X_test, Y_test):
    _, aupr_array = aupr(Y_test, model.call(X_test))
    _, auroc_array = auroc(Y_test, model.call(X_test))
    _, test_accuracy = model.evaluate(X_test, Y_test)
    return {
        'accuracy': test_accuracy,
        'aupr': aupr_array[0],
        'auroc': auroc_array[0]
    } 