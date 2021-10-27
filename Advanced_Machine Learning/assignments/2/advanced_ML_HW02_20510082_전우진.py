# -*- coding: utf-8 -*-

# DO NOT CHANGE
import pandas as pd
import numpy as np

def get_order(structure):
    # structure: dictionary of structure
    #            key is variable and value is parents of a variable
    # return list of learning order of variables 
    # ex) ['A', 'R', 'E', 'S', 'T', 'O']
    ordered_list = list()
    for k, v in structure.items():
        if not v:
            ordered_list.append(k)
            
    for k, v in structure.items():
        for p in v:
            if p in ordered_list and k not in ordered_list:
                ordered_list.append(k)
                
    for k, v in structure.items():
        for p2 in v:
            if p2 not in ordered_list:
                ordered_list.append(p2)
                
    for k,v in structure.items():
        if k not in ordered_list:
            ordered_list.append(k)
    
    return ordered_list

def learn_parms(data,structure,var_order):
    # data: training data
    # structure: dictionary of structure
    # var_order: list of learning order of variables
    # return dictionary of trained parameters (key=variable, value=learned parameters)
    dict = {}
    for var in var_order :
        matrix = []
        key_list = []
        
        if len(structure[var]) == 0 :
            df1 = pd.DataFrame(list(data[var].value_counts(normalize=True))).T
            df1.columns = np.unique(data[var])
            dict[var] = df1
            continue ;
        
        group_probs = data.groupby(structure[var])
        row_number = 0
        for key, group in group_probs :
            matrix.append(list(group[var].value_counts(normalize=True)))
            df2 = pd.DataFrame(matrix)
            df2.columns = np.unique(data[var])
            key_list.append(key)
        
        df2.index = key_list
        dict[var] = df2
        
    return dict
                
def print_parms(var_order,parms):
    # var_order: list of learning order of variables
    # parms: dictionary of trained parameters (key=variable, value=learned parameters)
    # print the trained parameters for each variable
    for var in var_order:
        print('-------------------------')
        print('Variable Name=%s'%(var))
        #TODO: print the trained paramters
        print(parms[var])
    
data=pd.read_csv('https://drive.google.com/uc?export=download&id=1taoE9WlUUN4IbzDzHv7mxk_xSj07f-Zt', sep=' ')

str1={'A':[],'S':[],'E':['A','S'],'O':['E'],'R':['E'],'T':['O','R']}
order1=get_order(str1)
parms1=learn_parms(data,str1,get_order(str1))
print('-----First Structure------')
print_parms(order1,parms1)
print('')

str2={'A':['E'],'S':['A','E'],'E':['O','R'],'O':['R','T'],'R':['T'],'T':[]}
order2=get_order(str2)
parms2=learn_parms(data,str2,get_order(str2))
print('-----Second Structure-----')
print_parms(order2,parms2)
print('')