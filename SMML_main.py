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
import scipy.optimize
from StatMechGlass import stat_mech_glass as smg
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import timeit
import numpy as np

import warnings
warnings.filterwarnings("ignore")


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
    smg_res_df.to_csv("Data_SM/{}_SM.csv".format(name), index=False)
    # print(smg_res_df)
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

def ternary_comp(name):
    
    current_dir = os.getcwd()
    split_name = re.findall('[A-Z][^A-Z]*', name)
    # print(split_name)
    data = pd.read_csv(current_dir+"/Data2/"+name+".csv")
    # print(data)
    # os.chdir(current_dir)
    
    glass_comp = {}
    smg_res = {}
    
    for i in range(len(data.columns)):
        next_ind = data.columns[i]
        smg_res[next_ind] = 0
    # print(smg_res)
    x = 0
    for i in range(len(data[split_name[0]])):
        for i2 in range(len(split_name)):
            glass_comp[split_name[i2]] = data[split_name[i2]][i]
        # print(glass_comp)
        # print(glass_comp)
        tg = data["Tg"][i]
        # print(tg)
        res = smg.smg_structure(glass_comp, tg)
        # print(res)
        
        for i2 in res:
            if x == 0:
                smg_res[i2] = [res[i2]]
            else:
                smg_res[i2].append(res[i2])
        if x == 0:
            for i3 in range(len(split_name)):
                smg_res[split_name[i3]] = [data[split_name[i3]][i]]
        else:
            for i3 in range(len(split_name)):
                smg_res[split_name[i3]].append(data[split_name[i3]][i])
        
        x = 1
    # print(smg_res)
    smg_res_df = pd.DataFrame.from_dict(smg_res)
    smg_res_df.to_csv("Data_SM/{}_SM.csv".format(name), index=False)
    # print(smg_res_df)
    # print(data)
    data_columns = list(data.columns)
    # print(data_columns)
    
    for i in split_name:
        try:
            ind = data_columns.index(i)
            data_columns.pop(ind)
        except:
            pass
        
    try:
        ind = data_columns.index("Tg")
        data_columns.pop(ind)
    except:
        pass
    
    # print(data_columns)
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
    
    return x_plot, y_plot, data, smg_res_df



def stat_mech_comp(train_names, tst_names, s_name=None):
    
    x_plot1 = []
    y_plot1 = []
    
    for i in tst_names:
        split_name = re.findall('[A-Z][^A-Z]*', i)
        if len(split_name) == 2:
            x_res, y_res = binary_comp(i)
        else:
            x_res, y_res = ternary_comp(i)[0], ternary_comp(i)[1]
        for i2 in x_res:
            x_plot1.append(i2)
        for i2 in y_res:
            y_plot1.append(i2)
    
    
    x_plot2 = []
    y_plot2 = []
    
    for i in train_names:
        x_res, y_res = binary_comp(i)
        for i2 in x_res:
            x_plot2.append(i2)
        for i2 in y_res:
            y_plot2.append(i2)
            
    
    t = list(range(100))
    
    plt.plot(t, t, 'k--')
    # plt.plot(x_plot1, y_plot1, "kd")
    plt.plot(x_plot2, y_plot2, "kd", x_plot1, y_plot1, "rd")
    plt.xlabel("Data")
    plt.ylabel("Model")
    plt.axis([0, 100, 0, 100])
    # plt.savefig(('{}_plot.png'.format("stat_mech_ALL")))
    plt.show()
    
    if s_name:
        m_data = np.column_stack([x_plot2, y_plot2])
        m_data2 = np.column_stack([x_plot1, y_plot1])
        np.savetxt(os.path.join('data_out', "{}_train.csv".format(s_name)), m_data)
        np.savetxt(os.path.join('data_out', "{}_val.csv".format(s_name)), m_data2)


def ml_comp_struc(train_name, tst_name, sm = False, s_name=None):
    
    current_dir = os.getcwd()
    ml_train_data = pd.read_csv(current_dir+"/Data_ML/"+train_name)
    if sm:
        ml_train_data_in = ml_train_data[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al", "Si6SM", "Si5SM", "Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM", "P3SM", "P2SM", "P1SM", "P0SM", "B4SM", "Al4SM", "Al6SM"]]
        # ml_train_data_in = ml_train_data[["Si6SM", "Si6SM", "Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM", "P3SM", "P2SM", "P1SM", "P0SM", "B4SM", "Al4SM", "Al6SM"]]
    else:
        ml_train_data_in = ml_train_data[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al"]]
    ml_train_data_out = ml_train_data[["Si6", "Si5", "Si4", "Si3", "Si2", "Si1", "Si0", "P3", "P2", "P1", "P0", "B4", "Al4", "Al6"]]
    
    clf = MLPRegressor(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
                       beta_2=0.999, early_stopping=False, epsilon=1e-08,
                       hidden_layer_sizes=(13,16), learning_rate='adaptive',
                       learning_rate_init=0.001, max_iter=50000000, momentum=0.9,
                       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=True,
                       warm_start=False)
    
    clf.fit(ml_train_data_in, ml_train_data_out)
    
    ml_predict_in = clf.predict(ml_train_data_in)
    
    ml_tst_data = pd.read_csv(current_dir+"/Data_ML"+tst_name)
    if sm:
        ml_tst_data_in = ml_tst_data[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al", "Si6SM", "Si5SM", "Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM", "P3SM", "P2SM", "P1SM", "P0SM", "B4SM", "Al4SM", "Al6SM"]]
        # ml_tst_data_in = ml_tst_data[["Si6SM", "Si6SM", "Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM", "P3SM", "P2SM", "P1SM", "P0SM", "B4SM", "Al4SM", "Al6SM"]]
    else:
        ml_tst_data_in = ml_tst_data[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al"]]
    ml_tst_data_out = ml_tst_data[["Si6", "Si5", "Si4", "Si3", "Si2", "Si1", "Si0", "P3", "P2", "P1", "P0", "B4", "Al4", "Al6"]]
    # print(ml_tst_data_in)
    
    ml_predict_tst = clf.predict(ml_tst_data_in)
    # print(ml_predict_tst)
    # print(ml_tst_data_out)
    
    error = ((ml_predict_tst-ml_tst_data_out)**2)
    SE = error.sum()
    SSE = SE.sum()
    # print(error)
    # print(SE)
    # print("SSE: {}".format(SSE))
    t = list(range(100))

    
    plt.plot(t, t, 'k--')
    plt.plot(ml_train_data_out, ml_predict_in, "kd", ml_tst_data_out, ml_predict_tst, "rd")
    plt.xlabel("Data")
    plt.ylabel("Model")
    plt.axis([0, 100, 0, 100])
    # plt.savefig(('{}_plot.png'.format("SM_ML_5_10")))
    plt.show()

    if sm:
        ml_tst_NaSi = pd.read_csv(current_dir+"/Data_ML/ml_data_sm_tst_NaSi.csv")
        ml_tst_NaSi = ml_tst_NaSi[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al", "Si6SM", "Si5SM", "Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM", "P3SM", "P2SM", "P1SM", "P0SM", "B4SM", "Al4SM", "Al6SM"]]
        ml_predict_NaSi = clf.predict(ml_tst_NaSi)
        
        NaSi_sm = ml_tst_NaSi[["Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM"]]
        NaSi_prediction = pd.DataFrame(ml_predict_NaSi)
        NaSi_ml = NaSi_prediction[[2, 3, 4, 5, 6]]
        
        Na_sm = ml_tst_NaSi[["Na"]]
        
        current_dir = os.getcwd()
        NaSi_data = pd.read_csv(current_dir+"/Data2/NaSi.csv")
        
        plt.plot(Na_sm, NaSi_sm["Si4SM"], "k-", Na_sm, NaSi_sm["Si3SM"], "r-", Na_sm, NaSi_sm["Si2SM"], "g-", Na_sm, NaSi_sm["Si1SM"], "y-", Na_sm, NaSi_sm["Si0SM"], "b-")
        plt.plot(Na_sm, NaSi_ml[2], "k--", Na_sm, NaSi_ml[3], "r--", Na_sm, NaSi_ml[4], "g--", Na_sm, NaSi_ml[5], "y--", Na_sm, NaSi_ml[6], "b--")
        plt.plot(NaSi_data["Na"], NaSi_data["Si4"], "kd", NaSi_data["Na"], NaSi_data["Si3"], "rd", NaSi_data["Na"], NaSi_data["Si2"], "gd", NaSi_data["Na"], NaSi_data["Si1"], "yd", NaSi_data["Na"], NaSi_data["Si0"], "bd")
        # plt.legend(["SM","ML","Data"])
        plt.xlabel("Na")
        plt.ylabel("Qn")
        plt.axis([0, 66, 0, 100])
        # plt.savefig(('{}_plot.png'.format("SM_ML_5_10_NaSi_ext")))
        plt.show()
        
        # np.savetxt(os.path.join('data_out', "NaSi_SM_train.csv"), NaSi_ml)
    else:
        ml_tst_NaSi = pd.read_csv(current_dir+"/Data_ML/ml_data_sm_tst_NaSi.csv")
        ml_predict_NaSi = clf.predict(ml_tst_NaSi[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al"]])
        
        NaSi_sm = ml_tst_NaSi[["Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM"]]
        NaSi_prediction = pd.DataFrame(ml_predict_NaSi)
        NaSi_ml = NaSi_prediction[[2, 3, 4, 5, 6]]
        
        Na_sm = ml_tst_NaSi[["Na"]]
        
        current_dir = os.getcwd()
        NaSi_data = pd.read_csv(current_dir+"/Data2/NaSi.csv")
        
        plt.plot(Na_sm, NaSi_sm["Si4SM"], "k-", Na_sm, NaSi_sm["Si3SM"], "r-", Na_sm, NaSi_sm["Si2SM"], "g-", Na_sm, NaSi_sm["Si1SM"], "y-", Na_sm, NaSi_sm["Si0SM"], "b-")
        plt.plot(Na_sm, NaSi_ml[2], "k--", Na_sm, NaSi_ml[3], "r--", Na_sm, NaSi_ml[4], "g--", Na_sm, NaSi_ml[5], "y--", Na_sm, NaSi_ml[6], "b--")
        plt.plot(NaSi_data["Na"], NaSi_data["Si4"], "kd", NaSi_data["Na"], NaSi_data["Si3"], "rd", NaSi_data["Na"], NaSi_data["Si2"], "gd", NaSi_data["Na"], NaSi_data["Si1"], "yd", NaSi_data["Na"], NaSi_data["Si0"], "bd")
        # plt.legend(["SM","ML","Data"])
        plt.xlabel("Na")
        plt.ylabel("Qn")
        plt.axis([0, 66, 0, 100])
        # plt.savefig(('{}_plot.png'.format("ML_5_10_NaSi_ext")))
        plt.show()
        
        # np.savetxt(os.path.join('data_out', "NaSi_ML_train.csv"), NaSi_ml)

    ml_tst_NaPSi = pd.read_csv(current_dir+"/Data_ML/NaPSi_tst.csv")
    NaPSi_in = ml_tst_NaPSi[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al", "Si6SM", "Si5SM", "Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM", "P3SM", "P2SM", "P1SM", "P0SM", "B4SM", "Al4SM", "Al6SM"]]
    NaPSi_in2 = ml_tst_NaPSi[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al", "Si6SM", "Si5SM", "Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM", "P3SM", "P2SM", "P1SM", "P0SM", "B4SM", "Al4SM", "Al6SM"]]
    # NaPSi_in2 = ml_tst_NaPSi[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al"]]
    NaPSi_out = ml_tst_NaPSi[["Si6", "Si5", "Si4", "Si3", "Si2", "Si1", "Si0", "P3", "P2", "P1", "P0", "B4", "Al4", "Al6"]]
    
    ml_predict_NaPSi = pd.DataFrame(clf.predict(NaPSi_in2))
    ml_predict_NaPSi_plot = ml_predict_NaPSi[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    
    sm_predict_NaPSi_plot = NaPSi_in[["Si6SM", "Si5SM", "Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM", "P3SM", "P2SM", "P1SM", "P0SM"]]
    
    NaPSi_data_plot = NaPSi_out[["Si6", "Si5", "Si4", "Si3", "Si2", "Si1", "Si0", "P3", "P2", "P1", "P0"]]
    
    t = list(range(100))

    plt.plot(t, t, 'k--')
    plt.plot(NaPSi_data_plot, sm_predict_NaPSi_plot, "bd", NaPSi_data_plot, ml_predict_NaPSi_plot, "gd")
    plt.xlabel("Data")
    plt.ylabel("Model")
    plt.axis([0, 100, 0, 100])
    plt.legend(["1:1", "SM", "ML"])
    # plt.savefig(('NaPSi.png'))
    plt.show()
    
    n_data = np.column_stack([NaPSi_data_plot, ml_predict_NaPSi_plot])
    np.savetxt(os.path.join('data_out', "NaPSi_SMML.csv"), n_data)
    
    if s_name:
        m_data = np.column_stack([ml_train_data_out, ml_predict_in])
        m_data2 = np.column_stack([ml_tst_data_out, ml_predict_tst])
        np.savetxt(os.path.join('data_out', "{}_train.csv".format(s_name)), m_data)
        np.savetxt(os.path.join('data_out', "{}_val.csv".format(s_name)), m_data2)

    return ml_predict_tst


def ml_comp_struc_err(train_name, tst_name):
    
    current_dir = os.getcwd()
    ml_train_data = pd.read_csv(current_dir+"/Data_ML/"+train_name)


    ml_train_data_in = ml_train_data[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al"]]
    ml_train_data_out = ml_train_data[["Si6err", "Si5err", "Si4err", "Si3err", "Si2err", "Si1err", "Si0err", "P3err", 
                                       "P2err", "P1err", "P0err", "B4err", "Al4err", "Al6err"]]
    
    clf = MLPRegressor(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
                       beta_2=0.999, early_stopping=False, epsilon=1e-08,
                       hidden_layer_sizes=(10,15), learning_rate='adaptive',
                       learning_rate_init=0.001, max_iter=50000000, momentum=0.9,
                       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=True,
                       warm_start=False)
    
    clf.fit(ml_train_data_in, ml_train_data_out)
    
    ml_predict_in = clf.predict(ml_train_data_in)
    
    ml_tst_data = pd.read_csv(current_dir+"/Data_ML/"+tst_name)
    

    ml_tst_data_in = ml_tst_data[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al"]]
    ml_tst_data_out = ml_tst_data[["Si6err", "Si5err", "Si4err", "Si3err", "Si2err", "Si1err", "Si0err", "P3err", 
                                   "P2err", "P1err", "P0err", "B4err", "Al4err", "Al6err"]]
    # print(ml_tst_data_in)
    
    ml_predict_tst = clf.predict(ml_tst_data_in)
    # print(ml_predict_tst)
    # print(ml_tst_data_out)
    
    error = ((ml_predict_tst-ml_tst_data_out)**2)
    SE = error.sum()
    SSE = SE.sum()
    # print(error)
    # print(SE)
    # print("SSE: {}".format(SSE))
    t = list(range(100))

    
    plt.plot(t, t, 'k--')
    plt.plot(ml_train_data_out, ml_predict_in, "kd", ml_tst_data_out, ml_predict_tst, "rd")
    plt.xlabel("Data")
    plt.ylabel("Model")
    plt.axis([0, 20, 0, 20])
    # plt.savefig(('{}_plot.png'.format("SM_ML_err_20")))
    plt.show()

    # if sm:
    #     ml_tst_NaSi = pd.read_csv("ml_data_sm_tst_NaSi.csv")
    #     ml_tst_NaSi = ml_tst_NaSi[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al", "Si6SM", "Si5SM", "Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM", "P3SM", "P2SM", "P1SM", "P0SM", "B4SM", "Al4SM", "Al6SM"]]
    #     ml_predict_NaSi = clf.predict(ml_tst_NaSi)
        
    #     NaSi_sm = ml_tst_NaSi[["Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM"]]
    #     NaSi_prediction = pd.DataFrame(ml_predict_NaSi)
    #     NaSi_ml = NaSi_prediction[[2, 3, 4, 5, 6]]
        
    #     Na_sm = ml_tst_NaSi[["Na"]]
        
    #     current_dir = os.getcwd()
    #     NaSi_data = pd.read_csv(current_dir+"/Data2/NaSi.csv")
        
    #     plt.plot(Na_sm, NaSi_sm["Si4SM"], "k-", Na_sm, NaSi_sm["Si3SM"], "r-", Na_sm, NaSi_sm["Si2SM"], "g-", Na_sm, NaSi_sm["Si1SM"], "y-", Na_sm, NaSi_sm["Si0SM"], "b-")
    #     plt.plot(Na_sm, NaSi_ml[2], "k--", Na_sm, NaSi_ml[3], "r--", Na_sm, NaSi_ml[4], "g--", Na_sm, NaSi_ml[5], "y--", Na_sm, NaSi_ml[6], "b--")
    #     plt.plot(NaSi_data["Na"], NaSi_data["Si4"], "kd", NaSi_data["Na"], NaSi_data["Si3"], "rd", NaSi_data["Na"], NaSi_data["Si2"], "gd", NaSi_data["Na"], NaSi_data["Si1"], "yd", NaSi_data["Na"], NaSi_data["Si0"], "bd")
    #     # plt.legend(["SM","ML","Data"])
    #     plt.xlabel("Na")
    #     plt.ylabel("Qn")
    #     plt.axis([0, 66, 0, 100])
    #     # plt.savefig(('{}_plot.png'.format("SM_ML_10_15_NaSi_ext")))
    #     plt.show()
    # else:
    #     ml_tst_NaSi = pd.read_csv("ml_data_sm_tst_NaSi.csv")
    #     ml_predict_NaSi = clf.predict(ml_tst_NaSi[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al"]])
        
    #     NaSi_sm = ml_tst_NaSi[["Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM"]]
    #     NaSi_prediction = pd.DataFrame(ml_predict_NaSi)
    #     NaSi_ml = NaSi_prediction[[2, 3, 4, 5, 6]]
        
    #     Na_sm = ml_tst_NaSi[["Na"]]
        
    #     current_dir = os.getcwd()
    #     NaSi_data = pd.read_csv(current_dir+"/Data2/NaSi.csv")
        
    #     plt.plot(Na_sm, NaSi_sm["Si4SM"], "k-", Na_sm, NaSi_sm["Si3SM"], "r-", Na_sm, NaSi_sm["Si2SM"], "g-", Na_sm, NaSi_sm["Si1SM"], "y-", Na_sm, NaSi_sm["Si0SM"], "b-")
    #     plt.plot(Na_sm, NaSi_ml[2], "k--", Na_sm, NaSi_ml[3], "r--", Na_sm, NaSi_ml[4], "g--", Na_sm, NaSi_ml[5], "y--", Na_sm, NaSi_ml[6], "b--")
    #     plt.plot(NaSi_data["Na"], NaSi_data["Si4"], "kd", NaSi_data["Na"], NaSi_data["Si3"], "rd", NaSi_data["Na"], NaSi_data["Si2"], "gd", NaSi_data["Na"], NaSi_data["Si1"], "yd", NaSi_data["Na"], NaSi_data["Si0"], "bd")
    #     # plt.legend(["SM","ML","Data"])
    #     plt.xlabel("Na")
    #     plt.ylabel("Qn")
    #     plt.axis([0, 66, 0, 100])
    #     # plt.savefig(('{}_plot.png'.format("ML_10_15_NaSi_ext")))
    #     plt.show()

    # ml_tst_NaPSi = pd.read_csv("NaPSi_tst.csv")
    # NaPSi_in = ml_tst_NaPSi[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al", "Si6SM", "Si5SM", "Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM", "P3SM", "P2SM", "P1SM", "P0SM", "B4SM", "Al4SM", "Al6SM"]]
    # NaPSi_out = ml_tst_NaPSi[["Si6", "Si5", "Si4", "Si3", "Si2", "Si1", "Si0", "P3", "P2", "P1", "P0", "B4", "Al4", "Al6"]]
    
    # ml_predict_NaPSi = pd.DataFrame(clf.predict(NaPSi_in))
    # ml_predict_NaPSi_plot = ml_predict_NaPSi[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    
    # sm_predict_NaPSi_plot = NaPSi_in[["Si6SM", "Si5SM", "Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM", "P3SM", "P2SM", "P1SM", "P0SM"]]
    
    # NaPSi_data_plot = NaPSi_out[["Si6", "Si5", "Si4", "Si3", "Si2", "Si1", "Si0", "P3", "P2", "P1", "P0"]]
    
    # t = list(range(100))

    # plt.plot(t, t, 'k--')
    # plt.plot(NaPSi_data_plot, sm_predict_NaPSi_plot, "bd", NaPSi_data_plot, ml_predict_NaPSi_plot, "gd")
    # plt.xlabel("Data")
    # plt.ylabel("Model")
    # plt.axis([0, 100, 0, 100])
    # plt.legend(["1:1", "SM", "ML"])
    # plt.savefig(('NaPSi.png'))
    # plt.show()

    return ml_predict_tst


def cross_val(arc):
    
    current_dir = os.getcwd()
    arc=[int(arc[0]), int(arc[1])]
    
    ml_train_data = pd.read_csv(current_dir+"/Data_ML/ml_data_sm_ext.csv")
    ml_tst_data = pd.read_csv(current_dir+"/Data_ML/ml_data_sm_tst_ext.csv")
    
    ml_tot_data = pd.concat([ml_train_data, ml_tst_data], ignore_index=True)
    
    ml_tot_input = ml_tot_data[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al"]]
    # ml_tot_input = ml_tot_data[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al", "Si6SM", "Si5SM", "Si4SM", "Si3SM", "Si2SM", "Si1SM", "Si0SM", "P3SM", "P2SM", "P1SM", "P0SM", "B4SM", "Al4SM", "Al6SM"]]
    ml_tot_output = ml_tot_data[["Si6", "Si5", "Si4", "Si3", "Si2", "Si1", "Si0", "P3", "P2", "P1", "P0", "B4", "Al4", "Al6"]]

    cross_val_NN = MLPRegressor(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
                       beta_2=0.999, early_stopping=False, epsilon=1e-08,
                       hidden_layer_sizes=arc, learning_rate='adaptive',
                       learning_rate_init=0.001, max_iter=50000000, momentum=0.9,
                       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=True,
                       warm_start=False)

    score = cross_val_score(cross_val_NN, ml_tot_input, ml_tot_output, cv=10, scoring='neg_root_mean_squared_error')
    
    mean_score = abs(score.mean())
    
    return mean_score

def cross_val_plot(s_name=None):
    
    arcs = [[3,6], [7,10], [9,12], [11,14], [13,16], [15,18], [15,50], [15,100], [30, 500]]
    
    res = []
    
    for i in range(len(arcs)):
        current_res = cross_val(arcs[i])
        res.append(current_res)
    
    x_plot = range(len(res))
    
    plt.plot(x_plot, res, "kd", x_plot, res, "k--")
    plt.xlabel("NN architecture (AU.)")
    plt.ylabel("RMSE")
    # plt.axis([0, 100, 0, 100])
    # plt.legend(["1:1", "SM", "ML"])
    plt.savefig(('NN_architecture_MLonly.png'))
    plt.show()
    
    if s_name:
        m_data = np.column_stack([x_plot, res])
        np.savetxt(os.path.join('data_out', "{}_CV.csv".format(s_name)), m_data)

    

def cross_val_err(arc):
    
    current_dir = os.getcwd()
    arc=[int(arc[0]), int(arc[1])]
    
    ml_train_data = pd.read_csv(current_dir+"/Data_ML/ml_data_sm_ext.csv")
    ml_tst_data = pd.read_csv(current_dir+"/Data_ML/ml_data_sm_tst_ext.csv")
    
    ml_tot_data = pd.concat([ml_train_data, ml_tst_data], ignore_index=True)
    
    ml_tot_input = ml_tot_data[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al"]]
    ml_tot_output = ml_tot_data[["Si6", "Si5", "Si4", "Si3", "Si2", "Si1", "Si0", "P3", "P2", "P1", "P0", "B4", "Al4", "Al6"]]

    cross_val_NN = MLPRegressor(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
                       beta_2=0.999, early_stopping=False, epsilon=1e-08,
                       hidden_layer_sizes=arc, learning_rate='adaptive',
                       learning_rate_init=0.001, max_iter=50000000, momentum=0.9,
                       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=True,
                       warm_start=False)

    score = cross_val_score(cross_val_NN, ml_tot_input, ml_tot_output, cv=10, scoring='neg_root_mean_squared_error')
    
    mean_score = abs(score.mean())
    
    return mean_score

def CV_engine(it = 100):
    arc0 = [10, 15]

    minimizer_kwargs = {"method": "BFGS"}
    res = scipy.optimize.basinhopping(cross_val, arc0, niter=it, T=4.0, stepsize=5, 
                                       minimizer_kwargs=minimizer_kwargs, take_step=None, 
                                       accept_test=None, callback=None, interval=20, 
                                       disp=True, niter_success=None, seed=None)

    return res.x


def CV_engine_err(it = 100):
    arc0 = [13, 16]

    minimizer_kwargs = {"method": "BFGS"}
    res = scipy.optimize.basinhopping(cross_val_err, arc0, niter=it, T=4.0, stepsize=5, 
                                       minimizer_kwargs=minimizer_kwargs, take_step=None, 
                                       accept_test=None, callback=None, interval=20, 
                                       disp=True, niter_success=None, seed=None)

    return res.x


if __name__ == "__main__":
    
    # former = "Si"
    # modifier = "Rb"
    # smg.smg_binary_par(former, modifier, it=50)

    # binary_names = ["NaSi", "CsB", "RbSi", "MgSi", "KB", "CsSi", "CaP", "CaSi", 
    #                 "CsP", "KSi", "LiB", "LiP", "LiSi", "MgP", "NaB", "NaP"]
    # data_name = ["CaPSi", "NaPSi", "NaAlB", "LiAlB", "CaAlSi", "CsAlB", "NaCaP", "NaCaPSi", "NaCaSi"]
    
    # binary_names = ["NaSi", "NaB", "NaP", "KSi"]
    # data_name = ["CaPSi", "NaPSi"]
    
    # data_name = ["NaCaSi"]
    
    # stat_mech_comp(binary_names, data_name, s_name = "ALL_SM")
    


    
    # start = timeit.default_timer()
    
    # ml_prediction = ml_comp_struc("ml_data_sm_ext.csv", "ml_data_sm_tst_ext.csv", sm=True)
    # # ml_prediction = ml_comp_struc("ml_data_sm_ext.csv", "ml_data_sm_tst_ext.csv", sm=True, s_name = "ALL_SMML")
    
    # # ml_prediction = ml_comp_struc("ml_data_sm3.csv", "ml_data_sm_tst3.csv", sm=True)
    # # ml_prediction = ml_comp_struc("ml_data_sm3.csv", "ml_data_sm_tst3.csv")
    
    # # cross_score = CV_engine()
    # # print("Cross score: {}".format(cross_score))
    
    # stop = timeit.default_timer()
    # print('Time: ', stop - start, 's')



    start = timeit.default_timer()
    
    # # ml_prediction = ml_comp_struc_err("ml_data_sm_ext_err.csv", "ml_data_sm_tst_ext_err.csv")
    
    cross_val_plot(s_name = "ALL_MLonly")
    # cross_score = cross_val_err([13, 16])
    # print("Cross score: {}".format(cross_score))
    
    stop = timeit.default_timer()
    print('Time: ', stop - start, 's')



    # ml_train_data = pd.read_csv("ml_data_sm_ext.csv")
    # ml_tst_data = pd.read_csv("ml_data_sm_tst_ext.csv")
    
    # tst_err = ml_tst_data[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al"]]
    
    # tst_err["Si6err"] = ml_tst_data["Si6SM"]-ml_tst_data["Si6"]
    # tst_err["Si5err"] = ml_tst_data["Si5SM"]-ml_tst_data["Si5"]
    # tst_err["Si4err"] = ml_tst_data["Si4SM"]-ml_tst_data["Si4"]
    # tst_err["Si3err"] = ml_tst_data["Si3SM"]-ml_tst_data["Si3"]
    # tst_err["Si2err"] = ml_tst_data["Si2SM"]-ml_tst_data["Si2"]
    # tst_err["Si1err"] = ml_tst_data["Si1SM"]-ml_tst_data["Si1"]
    # tst_err["Si0err"] = ml_tst_data["Si0SM"]-ml_tst_data["Si0"]
    
    # tst_err["P3err"] = ml_tst_data["P3SM"]-ml_tst_data["P3"]
    # tst_err["P2err"] = ml_tst_data["P2SM"]-ml_tst_data["P2"]
    # tst_err["P1err"] = ml_tst_data["P1SM"]-ml_tst_data["P1"]
    # tst_err["P0err"] = ml_tst_data["P0SM"]-ml_tst_data["P0"]
    
    # tst_err["B4err"] = ml_tst_data["B4SM"]-ml_tst_data["B4"]
    
    # tst_err["Al4err"] = ml_tst_data["Al4SM"]-ml_tst_data["Al4"]
    # tst_err["Al6err"] = ml_tst_data["Al6SM"]-ml_tst_data["Al6"]
    
    # tst_err.to_csv("ml_data_sm_tst_ext_err.csv", index=False)
    
    
    # trn_err = ml_train_data[["Na", "Li", "K", "Cs", "Ca", "Mg", "Si", "P", "B", "Al"]]
    
    # trn_err["Si6err"] = ml_train_data["Si6SM"]-ml_train_data["Si6"]
    # trn_err["Si5err"] = ml_train_data["Si5SM"]-ml_train_data["Si5"]
    # trn_err["Si4err"] = ml_train_data["Si4SM"]-ml_train_data["Si4"]
    # trn_err["Si3err"] = ml_train_data["Si3SM"]-ml_train_data["Si3"]
    # trn_err["Si2err"] = ml_train_data["Si2SM"]-ml_train_data["Si2"]
    # trn_err["Si1err"] = ml_train_data["Si1SM"]-ml_train_data["Si1"]
    # trn_err["Si0err"] = ml_train_data["Si0SM"]-ml_train_data["Si0"]
    
    # trn_err["P3err"] = ml_train_data["P3SM"]-ml_train_data["P3"]
    # trn_err["P2err"] = ml_train_data["P2SM"]-ml_train_data["P2"]
    # trn_err["P1err"] = ml_train_data["P1SM"]-ml_train_data["P1"]
    # trn_err["P0err"] = ml_train_data["P0SM"]-ml_train_data["P0"]
    
    # trn_err["B4err"] = ml_train_data["B4SM"]-ml_train_data["B4"]
    
    # trn_err["Al4err"] = ml_train_data["Al4SM"]-ml_train_data["Al4"]
    # trn_err["Al6err"] = ml_train_data["Al6SM"]-ml_train_data["Al6"]
    
    # trn_err.to_csv("ml_data_sm_ext_err.csv", index=False)
    
    
# for i in data[split_name[0]]:
#     print(i)


# start = timeit.default_timer()

# buy_tst, sell_tst, short_tst, buyback_tst, profit_percent, avg_profit, bought_stock, shorted_stock = hist_test(symbols)

# stop = timeit.default_timer()
# print('Time: ', stop - start, 's')
    # name_a = ["CaSi"]
    # name_b = ["CaSi"]
    # stat_mech_comp(name_a, name_b)
    
    # former = "P"
    # modifier = "Ca"
    # smg.smg_binary_par(former, modifier, it=1000)
    
    # x.to_csv("temp/{}_data.csv".format(name), index=False)
    # y.to_csv("temp/{}_SM.csv".format(name), index=False)
