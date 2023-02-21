from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import random
from tqdm import trange

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from scipy import stats
from collections import Counter
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tourism', help='Name of the dataset')

def gen_covariates(times, num_covariates):
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 1] = input_time.quarter
        covariates[i, 2] = input_time.year
    for i in range(1,num_covariates):
        covariates[:,i] = stats.zscore(covariates[:,i])
    return covariates[:, :num_covariates]


def prep_data(data, covariates, data_start, train = True):
    #print("train: ", train)
    time_len = data.shape[0]
    #print("time_len: ", time_len)
    input_size = window_size-stride_size
    windows_per_series = np.full((num_series), (time_len-input_size) // stride_size)
    #print("windows pre: ", windows_per_series.shape)
    if train: windows_per_series -= (data_start+stride_size-1) // stride_size 
    #print("data_start: ", data_start.shape)
    #print(data_start)
    #print("windows: ", windows_per_series.shape)
    #print(windows_per_series)
    total_windows = np.sum(windows_per_series)
    x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    v_input = np.zeros((total_windows, 2), dtype='float32')
    #cov = 3: ground truth + age + day_of_week + hour_of_day + num_series
    #cov = 4: ground truth + age + day_of_week + hour_of_day + month_of_year + num_series
    count = 0
    if not train:
        covariates = covariates[-time_len:]
    for series in trange(num_series):
        cov_age = stats.zscore(np.arange(total_time-data_start[series]))
        if train:
            covariates[data_start[series]:time_len, 0] = cov_age[:time_len-data_start[series]]
        else:
            covariates[:, 0] = cov_age[-time_len:]
        for i in range(windows_per_series[series]):
            if train:
                window_start = stride_size*i+data_start[series]
            else:
                window_start = stride_size*i
            window_end = window_start+window_size
            '''
            print("x: ", x_input[count, 1:, 0].shape)
            print("window start: ", window_start)
            print("window end: ", window_end)
            print("data: ", data.shape)
            print("d: ", data[window_start:window_end-1, series].shape)
            '''
            x_input[count, 1:, 0] = data[window_start:window_end-1, series] # 这里少一个是因为z_t-1，但为什么不对齐，为什么不是x_input[count, :-1, 0]
            x_input[count, :, 1:1+num_covariates] = covariates[window_start:window_end, :]
            x_input[count, :, -1] = series # idx
            label[count, :] = data[window_start:window_end, series] # label是这一段序列的值
            nonzero_sum = (x_input[count, 1:input_size, 0]!=0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0 # 这个v好像是论文里面 scale handling部分
            else:
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(),nonzero_sum)+1
                x_input[count, :, 0] = x_input[count, :, 0]/v_input[count, 0]
                if train:
                    label[count, :] = label[count, :]/v_input[count, 0]
            count += 1
    # reformat the data, make every batch has all time series
    res=[]
    for i in range(x_input.shape[0]):
        res.append(x_input[i][:,-1][0])
    n1 = Counter(res)[0]

    new_x_input = []
    for i in range(n1):
        for j in range(num_series):
            # print(i+j*n1)
            new_x_input.append(x_input[i+j*n1])
    new_x_input = np.array(new_x_input)

    new_v_input = []
    for i in range(n1):
        for j in range(num_series):
            # print(i+j*n1)
            new_v_input.append(v_input[i+j*n1])
    new_v_input = np.array(new_v_input)

    new_label = []
    for i in range(n1):
        for j in range(num_series):
            # print(i+j*n1)
            new_label.append(label[i+j*n1])
    new_label = np.array(new_label)
    return new_x_input, new_v_input, new_label



if __name__=='__main__':

    args = parser.parse_args()
    data_name = args.dataset

    window_size = 8
    stride_size = 2

    train_start = '1998-01-01'
    train_end = '2014-10-01'
    test_start = '2014-01-01' #need additional 7 days as given info 
    test_end = '2016-10-01'

    data = pd.read_csv(f'data/{data_name}.csv',index_col=0)  # 输入数据格式 sku-group time value
    group_names = ['State','Region','Purpose']
    time_name = 'Quarter'
    target_name = 'Trips'

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
    data = data.sort_index()


    for k in list(hier_dict.keys())[-1::-1]:
        data[k] = data[hier_dict[k]].sum(axis=1)


    num_covariates = 3
    covariates = gen_covariates(data[train_start:test_end].index, num_covariates)
    


    ## you can import covariates from csv or generating covariates by yourself
    # covariates = pd.read_csv(f'data/{data_name}_cov.csv',index_col=0)
    # num_covariates = len(covariates.columns)-1   #????

    train_data = data[train_start:train_end].values
    test_data = data[test_start:test_end].values
    # data_start = (train_data!=0).argmax(axis=0) #find first nonzero value in each time series
    total_time = data.shape[0] #76
    num_series = data.shape[1] #85
    data_start = np.zeros(num_series,dtype='int64')



    train_x_input, train_v_input, train_label = prep_data(train_data, covariates, data_start)
    test_x_input, test_v_input, test_label = prep_data(test_data, covariates, data_start, train=False)

    np.save(f'data/{data_name}/train_data_tourism', train_x_input)
    np.save(f'data/{data_name}/train_v_tourism', train_v_input)
    np.save(f'data/{data_name}/train_label_tourism',train_label)

    np.save(f'data/{data_name}/test_data_tourism', test_x_input)
    np.save(f'data/{data_name}/test_v_tourism', test_v_input)
    np.save(f'data/{data_name}/test_label_tourism',test_label)

    np.save(f'data/{data_name}/hier_dict_name.npy', hier_dict)


    final_hier_dict2 = {}
    for k in hier_dict.keys():
        final_hier_dict2[list(data.columns).index(k)] = [list(data.columns).index(i) for i in hier_dict[k]]
    
    pd.DataFrame(list(data.columns)).to_csv(f'data/{data_name}/series_names.csv')
    np.save(f'data/{data_name}/hier_dict.npy', final_hier_dict2)

    with open('experiments/base_model/params.json','r+') as f:
        params=json.load(f)
        params['batch_size']=num_series
        params['num_class']=num_series
        params['cov_dim']=num_covariates
        params['predict_batch']=num_series

    params_data = json.dumps(params, indent=1)
    with open('experiments/param_search/params.json', 'w', newline='\n') as f:
        f.write(params_data)

    
    with open("experiments/base_model/params.json", 'w', newline='\n') as f:
        f.write(params_data)

