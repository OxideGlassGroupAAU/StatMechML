#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 08:56:53 2021

@author: mikkel
"""

# from deepmind-research import glassy_dynamics as gd

import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from StatMechGlass import stat_mech_glass as smg


# print(current_dir)
# os.chdir("/home/mikkel/OneDrive/Uni/PhD/Python/SMML/Data2")



def binary_comp(name):
    
    current_dir = os.getcwd()
    split_name = re.findall('[A-Z][^A-Z]*', name)
    # print(split_name)
    data = pd.read_csv(current_dir+"/Data2/"+name+".csv")
    
    # os.chdir(current_dir)
    
    glass_comp = {}
    smg_res = {}
    
    for i in range(len(data.columns)):
        next_ind = data.columns[i]
        smg_res[next_ind] = 0
    
    x = 0
    for i in range(len(data[split_name[0]])):
        
    
        glass_comp[split_name[0]] = data[split_name[0]][i]
        glass_comp[split_name[1]] = 100 - data[split_name[0]][i]
        
        # print(glass_comp)
        res = smg.smg_structure(glass_comp, 700)
        
        
        for i2 in res:
            if x == 0:
                smg_res[i2] = [res[i2]]
            else:
                smg_res[i2].append(res[i2])
        if x == 0:
            smg_res[split_name[0]] = [data[split_name[0]][i]]
        else:
            smg_res[split_name[0]].append(data[split_name[0]][i])
        
        x = 1
    
    smg_res_df = pd.DataFrame.from_dict(smg_res)
    print(smg_res_df)
    # print(data)
    data_columns = list(data.columns)
    # print(data_columns)
    
    for i in split_name:
        try:
            ind = data_columns.index(i)
            data_columns.pop(ind)
        except:
            pass
    
    # print(data_columns)
    
    # print(data)
    # print(smg_res_df)
    
    x_plot = []
    y_plot = []
    
    for i in data_columns:
        for i2 in data[i]:
            x_plot.append(i2)
    
    for i in data_columns:
        for i2 in smg_res_df[i]:
            y_plot.append(i2)
    
    return x_plot, y_plot



if __name__ == "__main__":

    data_name = "NaP"
    # data_name2 = "KSi"
    
    x_plot, y_plot = binary_comp(data_name)
    # x_plot2, y_plot2 = binary_comp(data_name2)
    
    
    t = list(range(100))
    
    plt.plot(t, t, 'k--')
    plt.plot(x_plot, y_plot, "rd")
    plt.legend("Quality of model")
    plt.xlabel("Data")
    plt.ylabel("Model")
    plt.axis([0, 100, 0, 100])
    plt.show()

# for i in data[split_name[0]]:
#     print(i)
