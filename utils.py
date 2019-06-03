import sys,os
import numpy as np
def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        #test = base[i*n/n_folds:(i+1)*n/n_folds]
        test = base[int(i * n/n_folds):int((i + 1) * n/n_folds)]  #modified by xfuwu on May 30, 2018
        train = list(set(base)-set(test))
        folds.append([train,test])
        # i=0, test=[0,...,599], train=[600,...,6000]
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:  #d[0],d[1],d[2],d[3] with d[2] denoting cosine-distance, d[3] denoting the ground-truth label
        same = 1 if float(d[2]) > threshold else 0  #cos-correlation
        #same = 1 if float(d[2]) < threshold else 0  #euclidian-distance
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn
