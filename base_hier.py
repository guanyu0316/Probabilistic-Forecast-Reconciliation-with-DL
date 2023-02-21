
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sktime.forecasting.arima import AutoARIMA 
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
import CRPS.CRPS as pscore
import argparse
import datetime
import hts
import collections
import multiprocessing
import copy
from functools import partial
from sktime.transformations.series.outlier_detection import HampelFilter
import logging
logger = logging.getLogger('arima_hier')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tourism', help='Name of the dataset')
parser.add_argument('--fcst_model', default='arima', help='Base forecast model')
parser.add_argument('--permute_method', default='stack', help='The permutation of sample')
parser.add_argument('--rcc_method', default='mint', help='Reconciliation method')
parser.add_argument('--rcc_covariance', default=None, help='The covariance form of mint')


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

def arima_prob(y_,fh,sp,max_p,max_q,d=None):
    transformer = HampelFilter(window_length=10)
    y = transformer.fit_transform(y_)
    y = y.fillna(method='ffill')
    y = y.fillna(method='bfill')
    forecaster = AutoARIMA(sp=sp, d=d, max_p=max_p, max_q=max_q, suppress_warnings=True) 
    # step 4: fitting the forecaster
    forecaster.fit(y, fh=fh)
    # step 5: querying predictions
    var_pred = forecaster.predict_var()
    mu_pred = forecaster.predict()

    return mu_pred,var_pred[0]

def ets_prob(y_,fh,sp):
  transformer = HampelFilter(window_length=10)
  y = transformer.fit_transform(y_)
  y = y.fillna(method='ffill')
  y = y.fillna(method='bfill')
  if sp!=1:
    model = ExponentialSmoothing(endog=y, trend=True,seasonal=sp)
  else:
    model = ExponentialSmoothing(endog=y, trend=True)
  es_fit_result = model.fit()
  res_df = es_fit_result.get_forecast(len(fh)).summary_frame()
  mu_pred = res_df['mean']
  var_pred = res_df['mean_se']**2

  return mu_pred, var_pred

def permute(pred_dict,perm_method):
    sample_size = pred_dict['total'].shape[1]
    tp_names = list(pred_dict.keys())
    res_df_l = []
    if perm_method=='stack':
        for i in range(sample_size):
            df = pd.concat([pred_dict[n].iloc[:,i] for n in tp_names],axis=1)
            df.columns = tp_names
            res_df_l.append(df)
    elif perm_method=='rank':
        for i in range(sample_size):
            df = pd.concat([pred_dict[n].transform(np.sort,axis=1).iloc[:,i] for n in tp_names],axis=1)
            df.columns = tp_names
            res_df_l.append(df)
    elif perm_method=='random':
        for i in range(sample_size):
            tp_df_l = []
            for n in tp_names:
              df0 = pred_dict[n].sample(frac=1,axis=1,random_state=1)
              df0.columns = list(range(sample_size))
              tp_df_l.append(df0.iloc[:,i])
            df = pd.concat(tp_df_l,axis=1)
            df.columns = tp_names
            res_df_l.append(df)
    else:
        logger.info(f'Wrong permutation method')
    
    return res_df_l


def _lamb_estimate(x: np.ndarray) -> float:
    """Estimate :math`\\lambda` used in :ref:`shrinkage` estimator of mint method.
    :param x: in-sample 1-step-ahead forecast error.
    :return: :math`\\lambda`.
    """
    T = x.shape[0]
    covm = x.T.dot(x)/T
    xs = x / np.sqrt(np.diag(covm))
    corm = xs.T.dot(xs)/T
    np.fill_diagonal(corm, 0)
    d = np.sum(np.square(corm))
    xs2 = np.square(xs)
    v = 1/(T*(T-1))*(xs2.T.dot(xs2) - 1/T*np.square(xs.T.dot(xs)))
    np.fill_diagonal(v, 0)
    lamb = np.max(np.min([np.sum(v)/d, 1]), 0)
    return lamb

def reconcile(final_df,hier_dict,method:str,cov_method=None,error_dict=None,hist_p=None):
    final_df=final_df.copy()
    tree = hts.hierarchy.HierarchyTree.from_nodes(hier_dict, final_df)
    sum_mat, sum_mat_labels = hts.functions.to_sum_mat(tree)

    if method=='mint':
        # if cov_method=='ols':
        #     res_df_l=[]
        #     for i in range(final_df.shape[0]):
        #         forecasts = final_df.iloc[i,:].copy()
        #         pred_dict = collections.OrderedDict()
        #         # Add predictions to dictionary is same order as summing matrix
        #         for label in sum_mat_labels:
        #             pred_dict[label] = pd.DataFrame(data=[forecasts[label]], columns=['yhat'])
        #         revised = hts.functions.optimal_combination(pred_dict, sum_mat, method='OLS', mse={})
        #         # set negative forecast 0
        #         revised[revised<0]=0
        #         # Put reconciled forecasts in nice DataFrame form
        #         revised_forecasts = pd.DataFrame(data=revised[0:,0:],index=[final_df.index[i]],
        #                                             columns=sum_mat_labels)
        #         res_df_l.append(revised_forecasts)
        #     return pd.concat(res_df_l)
        
        if error_dict is not None:
            error = np.array([error_dict[l] for l in sum_mat_labels])
            T = error.shape[1]
            W = error.dot(error.T) / T

        if cov_method=='ols':
            weight_matrix = np.identity(len(sum_mat_labels))
        elif cov_method=='var':
            weight_matrix = np.diag(np.diagonal(W))
        elif cov_method=='shrink':
            lamb = _lamb_estimate(error.T)
            weight_matrix = lamb * np.diag(np.diag(W)) + (1 - lamb) * W
        elif cov_method=='struct':
            weight_matrix = np.diag(sum_mat.dot(np.array([1]*sum_mat.shape[1])))
        else:
            raise ValueError("This wls weighting method is not supported for now.")
        
        res_df_l=[]
        for i in range(final_df.shape[0]):
            forecasts = final_df.iloc[i,:].copy()
            yhat = forecasts[sum_mat_labels].copy().values
            p= np.dot(np.dot(np.linalg.inv(sum_mat.T@np.linalg.inv(weight_matrix)@sum_mat), sum_mat.T), np.linalg.inv(weight_matrix))
            pred_dict = collections.OrderedDict()
            # Add predictions to dictionary is same order as summing matrix
            
            revised=sum_mat@p@yhat
            # revised[revised<0]=0
            # Put reconciled forecasts in nice DataFrame form
            revised_forecasts =pd.DataFrame(revised.reshape(-1, len(revised)),index=[final_df.index[i]],columns=sum_mat_labels)
            res_df_l.append(revised_forecasts)
            
        return pd.concat(res_df_l)
    
    if method=='bu':
        res_df_l=[]
        p = np.c_[np.zeros([sum_mat.shape[1],sum_mat.shape[0]-sum_mat.shape[1]]),np.identity(sum_mat.shape[1])]
        for i in range(final_df.shape[0]):
            forecasts = final_df.iloc[i,:].copy()
            yhat = forecasts[sum_mat_labels].copy().values
            
            revised=sum_mat@p@yhat
            # revised[revised<0]=0
            # Put reconciled forecasts in nice DataFrame form
            revised_forecasts =pd.DataFrame(revised.reshape(-1, len(revised)),index=[final_df.index[i]],columns=sum_mat_labels)
            res_df_l.append(revised_forecasts)
        return pd.concat(res_df_l)

    if method=='ub':
    
        bottom_labels = tree.get_level_order_labels()[-1]
        res_df_l=[]
        p=np.c_[hist_p[bottom_labels].values,np.zeros([sum_mat.shape[1],sum_mat.shape[0]-1])]
        for i in range(final_df.shape[0]):
            forecasts = final_df.iloc[i,:].copy()
            yhat = forecasts[sum_mat_labels].copy().values
            
            revised=sum_mat@p@yhat
            # revised[revised<0]=0
            # Put reconciled forecasts in nice DataFrame form
            revised_forecasts =pd.DataFrame(revised.reshape(-1, len(revised)),index=[final_df.index[i]],columns=sum_mat_labels)
            res_df_l.append(revised_forecasts)
        return pd.concat(res_df_l)


def plot_and_eval(ytrue:pd.Series,pred_mu:pd.Series,pred_std:pd.Series,ax,series_name,origin_mu_pred,origin_std_pred,perm_mu_pred,perm_std_pred):
    '''
    ytrue contains the history
    '''
    # compute crps average value for 4 timestamps
    crps_l=[]
    for t in pred_mu.index:
        pred_sample = np.random.normal(loc=pred_mu[t], scale=pred_std[t], size=500)
        pred_sample = pred_sample[pred_sample>0]
        if len(pred_sample) ==0:
            pred_sample=np.array([0,0])
        crps = pscore(pred_sample,ytrue[t]).compute()[0]
        crps_l.append(crps)
    avg_crps = np.mean(crps_l)
    x = pred_mu.index.strftime("%Y-%m-%d")
    # ax.fill_between(x, pred_mu.values - 2 * pred_var.values,
    #                 pred_mu.values + 2 * pred_var.values, color='blue',
    #                 alpha=0.2,label='95% interval predict')
    x_ = ytrue.index.strftime("%Y-%m-%d")
    ax.plot(x_, ytrue.values, color='r',label='truth')
    ax.axvline(pred_mu.index[0].strftime("%Y-%m-%d"), color='g', linestyle='dashed')
    ax.set_title(f'CRPS:{avg_crps}',fontsize= 15)
    ax.set_xlabel(f'{series_name}',fontsize= 15)
    # ax.set_xticks(range(len(x_)),x_)
    ax.set_xticklabels(x_,rotation=70)
    ax.plot(x, pred_mu.values, color='b',label='predict')
    ax.plot(x, origin_mu_pred.values, color='orange',label='origin_pred')
    ax.plot(x, perm_mu_pred.values, color='pink',label='perm_pred')
    ax.legend(loc='best',fontsize= 15)
    return avg_crps


if __name__=='__main__':

    # np.random.seed(2022)
    args = parser.parse_args()
    data_name = args.dataset
    fcst_model = args.fcst_model
    permute_method = args.permute_method
    rcc_method = args.rcc_method
    rcc_cov = args.rcc_covariance

    res_dir = fcst_model+'_'+permute_method+'_'+rcc_method+'_'+f"{rcc_cov if rcc_cov else 'none'}"

    if not os.path.exists(f'result2/{res_dir}'): 
        os.makedirs(f'result2/{res_dir}')
    group_names = ['State','Region','Purpose']
    time_name = 'Quarter'
    target_name = 'Trips'

    if not os.path.exists(f'result/{fcst_model}_hier'): 
        os.makedirs(f'result/{fcst_model}_hier')

    data,hier_dict,final_hier_dict2 = prepare_data('QS','Q')

    np.save(f'data/{data_name}/hier_dict_name.npy', hier_dict)
    series_names = list(data.columns)
    pd.DataFrame(series_names).to_csv(f'data/{data_name}/series_names.csv')
    np.save(f'data/{data_name}/hier_dict.npy', final_hier_dict2)

    # train test split
    train_end = '2015-10-01'
    test_start = '2016-01-01' 
    train_data = data[:train_end]
    test_data = data[test_start:]

    # set hyper-parameter
    sp = 4 # if quarterly data
    max_p = 3
    max_q = 3
    fh = list(range(1,test_data.shape[0]+1))
    d = None

    # fit and predict
    pred_dict = {}

    for name in series_names:
        y = train_data[name]
        if fcst_model=='arima':
            mu_pred,var_pred = arima_prob(y,fh,sp,max_p,max_q,d=None)
        elif fcst_model=='ets':
            mu_pred,var_pred = ets_prob(y,fh,sp)
        else:
            logger.info('This base model is not supported.')
        df_l1 = []
        for t in mu_pred.index:
            pred_sample = np.random.normal(loc=mu_pred[t], scale=var_pred[t]**0.5, size=200)
            df = pd.DataFrame(pred_sample).T
            df.index = [t]
            df_l1.append(df)
        pred_dict[name] = pd.concat(df_l1)
    
    # permute
    perm_df_l = permute(pred_dict,permute_method) # in this list, every df is a sample

    # reconcile every sample(df)
    rcc_p = partial(reconcile,hier_dict=hier_dict,method=rcc_method,cov_method=rcc_cov)

    with multiprocessing.Pool(processes=10) as pool:
        rcc_df_l = pool.map(rcc_p,perm_df_l)

    # evaluate and plot

    f = plt.figure(figsize=(50, 200))
    ncols = 5
    nrows = int(len(series_names)/ncols)+1
    ax = f.subplots(nrows, ncols)
    
    all_metrics={}
    crps_l=[]
    m=-1
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            m=m+1
            if m<len(series_names):
                res_df = pd.concat([rcc_df_l[kk][series_names[m]] for kk in range(len(rcc_df_l))],axis=1)
                mu_pred = res_df.mean(axis=1)
                std_pred = res_df.std(axis=1)
                origin_mu_pred = pred_dict[series_names[m]].mean(axis=1)
                origin_std_pred = pred_dict[series_names[m]].std(axis=1)
                res_df2 =pd.concat([perm_df_l[pp][series_names[m]] for pp in range(len(perm_df_l))],axis=1)
                perm_mu_pred = res_df2.mean(axis=1)
                perm_std_pred = res_df2.std(axis=1)
                crps = plot_and_eval(data[series_names[m]][-30:],mu_pred,std_pred,ax[i,j],series_names[m],origin_mu_pred,origin_std_pred,perm_mu_pred,perm_std_pred)
                crps_l.append(crps)
                print(f'Finish{m}')
    plt.tight_layout()
    all_metrics['crps']=crps_l
    f.savefig(f'result/{res_dir}/res.png')
    np.save(f'result/{res_dir}/metrics.npy', all_metrics)
    plt.close()

