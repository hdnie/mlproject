import numpy as np
def print_confusion_matrix(matrix,mode,classnum):
    acc = np.sum(np.diagonal(matrix)) / np.sum(matrix)
    real = np.sum(matrix, axis=0)
    recall = np.array(np.diagonal(matrix) / real)
    meanrecall = np.sum(recall) / classnum
    pred = np.sum(matrix, axis=1)
    precision = np.array(np.diagonal(matrix) / pred)
    meanprecision = np.sum(precision) / classnum
    print("accuracy in "+mode+":", acc)
    print("meanrecall in "+mode+":", meanrecall)
    print("meanprecision in "+mode+":", meanprecision)
    print("recall:",recall)
    print("precision:", precision)
    print(matrix)
def get_acc(matrix):
    return np.sum(np.diagonal(matrix)) / np.sum(matrix)
def get_pre(matrix,classnum):
    pred = np.sum(matrix, axis=1)
    precision = np.array(np.diagonal(matrix) / pred)
    meanprecision = np.sum(precision) / classnum
    return meanprecision
def get_rec(matrix,classnum):
    real = np.sum(matrix, axis=0)
    recall = np.array(np.diagonal(matrix) / real)
    meanrecall = np.sum(recall) / classnum
    return meanrecall
