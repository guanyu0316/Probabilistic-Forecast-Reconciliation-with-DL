import argparse
import logging
import os
import CRPS.CRPS as pscore
import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
import copy
import model.net as net
from dataloader import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def prepare_data(freq1,freq2):
    data = pd.read_csv(f'data/{data_name}.csv',index_col=0)  # 输入数据格式 sku-group time value

    data['sku']=data[group_names].apply(lambda x: 'total'+'_'+'_'.join(x),axis=1)

    hier_dict={}
    hier_dict['total']=[]
    for group_idx in range(len(group_names)-1):
        group = group_names[group_idx]
        for k in data[group].unique():
            temp_list = list(data.loc[data[group]==k,group_names[group_idx+1]].unique())
            if group_idx==0:
                hier_dict['total_'+k] = ['total_'+k+"_"+t for t in temp_list]
                hier_dict['total'].append('total_'+k)
            else:
                for k2 in sum(list(hier_dict.values())[-1::-1], []):
                    if (k2.find(k)!=-1)&(len(k2.split('_'))==group_idx+2):
                        hier_dict[k2] = [k2+"_"+t for t in temp_list]
                    

    data = pd.pivot_table(data,values=target_name,index=[time_name],columns=['sku'])
    data.index=pd.to_datetime(data.index)
    data = data.asfreq(freq1)
    data.index = pd.PeriodIndex(data.index, freq=freq2)
    data = data.sort_index()


    k_list = list(hier_dict.keys())[-1::-1]
    for k in k_list:
        temp_names = copy.deepcopy(hier_dict[k])
        for tn in temp_names:
            if tn not in data.columns:
                hier_dict[k].remove(tn)
                if tn in hier_dict.keys():
                    del hier_dict[tn]
        if k in hier_dict.keys():
            data[k] = data[hier_dict[k]].sum(axis=1)


    final_hier_dict2 = {}
    for k in hier_dict.keys():
        final_hier_dict2[list(data.columns).index(k)] = [list(data.columns).index(i) for i in hier_dict[k]]


    return data,hier_dict,final_hier_dict2

if __name__=='__main__':

    data_name = 'infant'

    group_names = ['state','gender']
    time_name = 'year'
    target_name = 'deaths'
    data,hier_dict,final_hier_dict2 = prepare_data('Y','Y')

    np.save(f'data/{data_name}/hier_dict_name.npy', hier_dict)
    series_names = list(data.columns)
    pd.DataFrame(series_names).to_csv(f'data/{data_name}/series_names.csv')
    np.save(f'data/{data_name}/hier_dict.npy', final_hier_dict2)

    # train test split
    train_end = '1999-12-31'
    test_start = '2000-12-31'
    train_data = data[:train_end]
    test_data = data[test_start:]

    all_method = ['deepar','deepar-hier','arima_stack_mint_struct','ets_stack_mint_struct']
    all_method2 = ['DeepAR','DeepAR-Hier','Arima_Stack_Mint_Struct','Ets_Stack_Mint_Struct']
    color_l = ['orange','blue','pink','green']

    # evaluate and plot

    f = plt.figure(figsize=(200, 100))
    ncols = 5
    nrows = int(len(series_names)/ncols)+1
    ax = f.subplots(nrows, ncols)

    m=-1
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            m=m+1
            if m<len(series_names):
                ytrue = data[series_names[m]][-8:]
                x_ = list(range(8))
                ytrue.index=x_
                ax[i,j].plot(x_, ytrue.values, color='r',label='truth')
                all_crps = []
                for me, method in enumerate(all_method):
                    mu_pred = pd.read_csv(f'result/{method}/mu_pred.csv')[series_names[m]]
                    mu_pred.index=list(range(4,8))
                    std_pred = pd.read_csv(f'result/{method}/std_pred.csv')[series_names[m]]
                    std_pred.index=list(range(4,8))
                    ax[i,j].plot(mu_pred.index, mu_pred.values, color=color_l[me],label=all_method2[me])
                
                    crps_l=[]
                    for t in mu_pred.index:
                        pred_sample = np.random.normal(loc=mu_pred[t], scale=std_pred[t], size=500)
                        pred_sample = pred_sample[pred_sample>0]
                        if len(pred_sample) ==0:
                            pred_sample=np.array([0,0])
                        crps = pscore(pred_sample,ytrue[t]).compute()[0]
                        crps_l.append(crps)
                    avg_crps = np.mean(crps_l)
                    all_crps.append(avg_crps)
                print(all_crps)
                percentage = -(all_crps[1]-all_crps[-1])/all_crps[-1]
                ax[i,j].legend(loc='best')
                ax[i,j].set_xlabel(f'{series_names[m]}')
                ax[i,j].set_title('percent: {:.2%}'.format(percentage))
                
                print(f'Finish{m}')
    plt.tight_layout()
    plt.savefig('plot_res.jpg',dpi=200)
    plt.close()