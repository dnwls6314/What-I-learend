# -*- coding: utf-8 -*-

# DO NOT CHANGE
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import time
import matplotlib.pyplot as plt

#%%
def wkNN(Xtr,ytr,Xts,k,random_state=None):
    # Implement weighted kNN
    # Xtr: training input data set
    # ytr: ouput labels of training set
    # Xts: test data set
    # random_state: random seed
    # return 1-D array containing output labels of test set
    start = time.time()
    
    y_pred = []
    for point in Xts :
        distance = pairwise_distances([point], Xtr)[0]
        index_within_k = np.argsort(distance)[0:k]
        
        if len(np.unique(ytr)) > 2 :
            weight = {0:0, 1:0, 2:0}
        else :
            weight = {0:0, 1:0}

        for index in index_within_k:
            if index == index_within_k[0] :
                weight[ytr[index]] = 1
                
            else :
                weight[ytr[index]] = (distance[index_within_k[-1]]-distance[index])/(distance[index_within_k[-1]]-distance[index_within_k[0]]) + weight[ytr[index]] 
        
        y_pred.append(max(weight, key=weight.get))    
        
    print("execution time", time.time()-start)
    return y_pred

#%%
def PNN(Xtr,ytr,Xts,k,random_state=None):
    # Implement PNN
    # Xtr: training input data set
    # ytr: ouput labels of training set
    # Xts: test data set
    # random_state: random seed
    # return 1-D array containing output labels of test set
    start = time.time()
    
    y_pred = []
    point_per_class = {}
    
    if len(np.unique(ytr)) > 2 :
        class_num = 3
    else :
        class_num = 2
        
    for class_ in range(0,class_num) :
        point_per_class[class_] = Xtr[ytr==class_]
        
    for point in Xts :
        weight = {}
        for class_ in point_per_class :
            weight[class_] = np.sort(pairwise_distances(point_per_class[class_], [point]).reshape(1,-1)[0])[0:k]
            for i in range(1, k+1) :
                weight[class_][i-1] = weight[class_][i-1]/i
        
        sum_of_weight_per_class = {}
        for class_ in point_per_class:
            sum_of_weight_per_class[class_] = sum(weight[class_])
            
        y_pred.append(min(sum_of_weight_per_class, key=sum_of_weight_per_class.get))

    print("execution time ", time.time()-start)
    return y_pred


#%%

def accuracy(pred, test) :
    total_length = len(pred)
    number_of_same_element = sum(pred==test)
    
    return round(number_of_same_element/total_length, 4)

#%%
X1,y1=datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=13)
Xtr1,Xts1, ytr1, yts1=train_test_split(X1,y1,test_size=0.2, random_state=22)

#TODO: Cacluate accuracy with varying k for wkNN and PNN
#TODO: Calculate computation time
#TODO: Draw scatter plot

y_pred_wknn_1 = wkNN(Xtr1, ytr1, Xts1, 5)
y_pred_pnn_1 = PNN(Xtr1, ytr1, Xts1, 5)
accuracy(y_pred_wknn_1, yts1)
accuracy(y_pred_pnn_1, yts1)

#%%
X2,y2=datasets.make_classification(n_samples=1000, n_features=6, n_informative=2, n_redundant=3, n_classes=2, n_clusters_per_class=2, flip_y=0.2,random_state=75)
Xtr2,Xts2, ytr2, yts2=train_test_split(X2,y2,test_size=0.2, random_state=78)

#TODO: Cacluate accuracy with varying k for wkNN and PNN
#TODO: Calculate computation time
#TODO: Draw scatter plot
y_pred_wknn_2 = wkNN(Xtr2, ytr2, Xts2, 5)
y_pred_pnn_2 = PNN(Xtr2, ytr2, Xts2, 5)
accuracy(y_pred_wknn_2, yts2)
accuracy(y_pred_pnn_2, yts2)

