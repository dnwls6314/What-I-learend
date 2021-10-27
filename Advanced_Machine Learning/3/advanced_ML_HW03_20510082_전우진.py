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
    y_pred = []
    
    for a in Xts:
        d = pairwise_distances([a],Xtr)[0]
        ind = np.argsort(d)[0:k]
        
        if len(np.unique(ytr)) > 2 :
            weight = {0:0, 1:0, 2:0}
        else :
            weight = {0:0, 1:0}
            
        for index in ind:
            if index == ind[0]:
                weight[ytr[index]] = 1
            else:
                weight[ytr[index]] = (d[ind[-1]]-d[index])/(d[ind[-1]]-d[ind[0]]) + weight[ytr[index]]
            
        y_pred.append(max(weight, key=weight.get))
    
    return y_pred
#%%
def PNN(Xtr,ytr,Xts,k,random_state=None):
    # Implement PNN
    # Xtr: training input data set
    # ytr: ouput labels of training set
    # Xts: test data set
    # random_state: random seed
    # return 1-D array containing output labels of test set
    y_pred = []
    point_per_class = {}
    
    if len(np.unique(ytr)) > 2:
        class_num = 3
    else:
        class_num = 2
        
    for cl in range(0,class_num):
        point_per_class[cl] = Xtr[ytr==cl]
        
    for point in Xts:
        weight = {}
        for cl2 in point_per_class:
            weight[cl2] = np.sort(pairwise_distances(point_per_class[cl2], [point]).reshape(1,-1)[0])[0:k]
            for i in range(1, k+1):
                weight[cl2][i-1] = weight[cl2][i-1]/i
        
        sum_of_weight_per_class = {}
        for cl3 in point_per_class:
            sum_of_weight_per_class[cl3] = sum(weight[cl3])
            
        y_pred.append(min(sum_of_weight_per_class, key=sum_of_weight_per_class.get))

    return y_pred
#%%    
# 정확도 측정
def accuracy(pred, test) :
    total_pred = len(pred)
    accurate = sum(pred==test)
    
    return round(accurate/total_pred, 4)
#%%
X1,y1=datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=13)
Xtr1,Xts1, ytr1, yts1=train_test_split(X1,y1,test_size=0.2, random_state=22)

#TODO: Cacluate accuracy with varying k for wkNN and PNN
#TODO: Calculate computation time
#TODO: Draw scatter plot
k=[3,5,7,9,11]
index_k1=[]
WKNN1=[]
Pnn1=[]

start = time.time()
for ks in k:
    wknn_1 = wkNN(Xtr1, ytr1, Xts1, ks)
    pnn_1 = PNN(Xtr1, ytr1, Xts1, ks)
    wknn_acrcy = accuracy(wknn_1, yts1)
    pnn_acrcy = accuracy(pnn_1, yts1)
    
    index_k1.append(ks)
    WKNN1.append(wknn_acrcy)
    Pnn1.append(pnn_acrcy)
        
data1_soyo_time = time.time()-start
    
print("Elapsed time: ", data1_soyo_time)
print('-----------------------------------------')
print('  k          wkNN          PNN')
print('-----------------------------------------')
print('  ',index_k1[0],'          ',WKNN1[0],'          ',Pnn1[0])
print('  ',index_k1[1],'          ',WKNN1[1],'          ',Pnn1[1])
print('  ',index_k1[2],'          ',WKNN1[2],'          ',Pnn1[2])
print('  ',index_k1[3],'          ',WKNN1[3],'          ',Pnn1[3])
print('  ',index_k1[4],'         ',WKNN1[4],'          ',Pnn1[4])
print('-----------------------------------------')
#%%
# data1 그래프: atrribute 2개, class 3개
k=7
wkn_TF1=(wknn_1==yts1)
pn_TF1=(pnn_1==yts1)
plt.xlim(-5,5)
plt.ylim(-5,5)

sample_Xtr1_x=[]
sample_Xtr1_y=[]
sample_Xts1_x=[]
sample_Xts1_y=[]
wkn_TF_x1=[]
wkn_TF_y1=[]
pn_TF_x1=[]
pn_TF_y1=[]
point_per_train_class = {}
point_per_test_class = {}

point_per_train_class = {}
point_per_test_class = {}

for cl in range(0,3) :
    point_per_train_class[cl] = Xtr1[ytr1==cl]
    point_per_test_class[cl] = Xts1[yts1==cl]
    
for sample1, sample2 in Xtr1:
    sample_Xtr1_x.append(sample1)
    sample_Xtr1_y.append(sample2)

for sample1, sample2 in Xts1:
    sample_Xts1_x.append(sample1)
    sample_Xts1_y.append(sample2)
    
for ind in range(len(wkn_TF1)):
    if wkn_TF1[ind] == True:
        continue
    else:
        wkn_TF_x1.append(sample_Xtr1_x[ind])
        wkn_TF_y1.append(sample_Xtr1_y[ind])

for ind in range(len(pn_TF1)):
    if pn_TF1[ind] == True:
        continue
    else:
        pn_TF_x1.append(sample_Xtr1_x[ind])
        pn_TF_y1.append(sample_Xtr1_y[ind])


plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(sample_Xtr1_x, sample_Xtr1_y, c='teal', marker='o',s=18)
plt.scatter(sample_Xts1_x, sample_Xts1_y, c='teal', marker='x',s=18)
plt.scatter(wkn_TF_x1, wkn_TF_y1, c='none', marker='s', edgecolors='red', s=40)
plt.scatter(pn_TF_x1, pn_TF_y1, c='none', marker='d', edgecolors='blue', s=40)
plt.legend(['Train','Test','Misclassifed by wkNN','Miscalssified by PNN'], loc='lower right')
plt.scatter(point_per_train_class[0][: ,0], point_per_train_class[0][:,1], color='purple', marker='o',s=18)
plt.scatter(point_per_train_class[1][: ,0], point_per_train_class[1][:,1], color='teal', marker='o',s=18)
plt.scatter(point_per_train_class[2][: ,0], point_per_train_class[2][:,1], color='gold', marker='o',s=18)
            
plt.scatter(point_per_test_class[0][: ,0], point_per_test_class[0][:,1], color='purple', marker='x',s=18)
plt.scatter(point_per_test_class[1][: ,0], point_per_test_class[1][:,1], color='teal', marker='x',s=18)
plt.scatter(point_per_test_class[2][: ,0], point_per_test_class[2][:,1], color='gold', marker='x',s=18)

#%%

X2,y2=datasets.make_classification(n_samples=1000, n_features=6, n_informative=2, n_redundant=3, n_classes=2, n_clusters_per_class=2, flip_y=0.2,random_state=75)
Xtr2,Xts2, ytr2, yts2=train_test_split(X2,y2,test_size=0.2, random_state=78)

#TODO: Cacluate accuracy with varying k for wkNN and PNN
#TODO: Calculate computation time
#TODO: Draw scatter plot
k=[3,5,7,9,11]
index_k2=[]
WKNN2=[]
Pnn2=[]

start = time.time()
for ks in k:
    wknn_2 = wkNN(Xtr2, ytr2, Xts2, ks)
    pnn_2 = PNN(Xtr2, ytr2, Xts2, ks)
    wknn_acrcy = accuracy(wknn_2, yts2)
    pnn_acrcy = accuracy(pnn_2, yts2)
    
    index_k2.append(ks)
    WKNN2.append(wknn_acrcy)
    Pnn2.append(pnn_acrcy)
        
data2_soyo_time = time.time()-start

print("Elapsed time: ", data2_soyo_time)
print('-----------------------------------------')
print('  k          wkNN          PNN')
print('-----------------------------------------')
print('  ',index_k2[0],'          ',WKNN2[0],'          ',Pnn2[0])
print('  ',index_k2[1],'          ',WKNN2[1],'          ',Pnn2[1])
print('  ',index_k2[2],'          ',WKNN2[2],'          ',Pnn2[2])
print('  ',index_k2[3],'          ',WKNN2[3],'          ',Pnn2[3])
print('  ',index_k2[4],'         ',WKNN2[4],'          ',Pnn2[4])
print('-----------------------------------------')
#%%
# data1 그래프: atrribute 6개, class 2개
k=7
wkn_TF2=(wknn_2==yts2)
pn_TF2=(pnn_2==yts2)
plt.xlim(-3,3)
plt.ylim(-5,5)

sample_Xtr2_x=[]
sample_Xtr2_y=[]
sample_Xts2_x=[]
sample_Xts2_y=[]
wkn_TF_x2=[]
wkn_TF_y2=[]
pn_TF_x2=[]
pn_TF_y2=[]
point_per_train_class = {}
point_per_test_class = {}

point_per_train_class = {}
point_per_test_class = {}

for cl in range(0,3) :
    point_per_train_class[cl] = Xtr2[ytr2==cl]
    point_per_test_class[cl] = Xts2[yts2==cl]
    
for sample1, sample2, sample3, sample4, sample5, sample6 in Xtr2:
    sample_Xtr2_x.append(sample1)
    sample_Xtr2_y.append(sample2)

for sample1, sample2, sample3, sample4, sample5, sample6 in Xts2:
    sample_Xts2_x.append(sample1)
    sample_Xts2_y.append(sample2)
    
for ind in range(len(wkn_TF2)):
    if wkn_TF2[ind] == True:
        continue
    else:
        wkn_TF_x2.append(sample_Xtr2_x[ind])
        wkn_TF_y2.append(sample_Xtr2_y[ind])

for ind in range(len(pn_TF2)):
    if pn_TF2[ind] == True:
        continue
    else:
        pn_TF_x2.append(sample_Xtr2_x[ind])
        pn_TF_y2.append(sample_Xtr2_y[ind])


plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(sample_Xtr2_x, sample_Xtr2_y, c='teal', marker='o',s=18)
plt.scatter(sample_Xts2_x, sample_Xts2_y, c='teal', marker='x',s=18)
plt.scatter(wkn_TF_x2, wkn_TF_y2, c='none', marker='s', edgecolors='red', s=40)
plt.scatter(pn_TF_x2, pn_TF_y2, c='none', marker='d', edgecolors='blue', s=40)
plt.legend(['Train','Test','Misclassifed by wkNN','Miscalssified by PNN'], loc='lower right')
plt.scatter(point_per_train_class[0][: ,0], point_per_train_class[0][:,1], color='purple', marker='o',s=18)
plt.scatter(point_per_train_class[1][: ,0], point_per_train_class[1][:,1], color='gold', marker='o',s=18)
            
plt.scatter(point_per_test_class[0][: ,0], point_per_test_class[0][:,1], color='purple', marker='x',s=18)
plt.scatter(point_per_test_class[1][: ,0], point_per_test_class[1][:,1], color='gold', marker='x',s=18)