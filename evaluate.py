import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
import model.net as net
from dataloader import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('DeepAR.Eval')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='infant', help='Name of the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--restore-file', default='best',
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'


def evaluate(model, loss_fn, test_loader, params, plot_num, dataset,sample=True):
    '''Evaluate the model on the test set.
    Args:
        model: (torch.nn.Module) the Deep AR model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
        plot_num: (-1): evaluation from evaluate.py; else (epoch): evaluation on epoch
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    '''
    model.eval()
    with torch.no_grad():
    #   plot_batch = np.random.randint(len(test_loader)-1)

      summary_metric = {}
      raw_metrics = utils.init_metrics(sample=sample)

      # Test_loader: 
      # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
      # id_batch ([batch_size]): one integer denoting the time series id;
      # v ([batch_size, 2]): scaling factor for each window;
      # labels ([batch_size, train_window]): z_{1:T}.
      for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(test_loader)):
          test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
          id_batch = id_batch.unsqueeze(0).to(params.device)
          v_batch = v.to(torch.float32).to(params.device)
          labels = labels.to(torch.float32).to(params.device)
          batch_size = test_batch.shape[1]
          input_mu = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          input_sigma = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          hidden = model.init_hidden(batch_size)
          cell = model.init_cell(batch_size)

          for t in range(params.test_predict_start):
              # if z_t is missing, replace it by output mu from the last time step
              zero_index = (test_batch[t,:,0] == 0)
              if t > 0 and torch.sum(zero_index) > 0:
                  test_batch[t,zero_index,0] = mu[zero_index]

              mu, sigma, hidden, cell = model(test_batch[t].unsqueeze(0), id_batch, hidden, cell)
              input_mu[:,t] = v_batch[:, 0] * mu + v_batch[:, 1]
              input_sigma[:,t] = v_batch[:, 0] * sigma

          if sample:
              samples, sample_mu, sample_sigma = model.test2(test_batch, v_batch, id_batch, hidden, cell, sampling=True)
              print(sample_mu.shape)
              print(sample_sigma.shape)
              raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, params.lam,params.dataset, samples,relative = params.relative_metrics)
          else:
              sample_mu, sample_sigma = model.test2(test_batch, v_batch, id_batch, hidden, cell)
              raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, params.lam,params.dataset, relative = params.relative_metrics)

          if i == 2: ### plot
              if sample:
                  sample_metrics = utils.get_metrics(sample_mu, sample_sigma, labels, params.test_predict_start, samples, relative = params.relative_metrics)
              else:
                  sample_metrics = utils.get_metrics(sample_mu, sample_sigma, labels, params.test_predict_start, relative = params.relative_metrics)                
              # select 10 from samples with highest error and 10 from the rest
            #   top_10_nd_sample = (-sample_metrics['ND']).argsort()[:batch_size // 10]  # hard coded to be 10
            #   chosen = set(top_10_nd_sample.tolist())
            #   all_samples = set(range(batch_size))
            #   not_chosen = np.asarray(list(all_samples - chosen))
            #   if batch_size < 100: # make sure there are enough unique samples to choose top 10 from
            #       random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=True)
            #   else:
            #       random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=False)
            #   if batch_size < 12: # make sure there are enough unique samples to choose bottom 90 from
            #       random_sample_90 = np.random.choice(not_chosen, size=10, replace=True)
            #   else:
            #       random_sample_90 = np.random.choice(not_chosen, size=10, replace=False)
            #   combined_sample = np.concatenate((random_sample_10, random_sample_90))
              combined_sample =np.asarray(list(range(batch_size)))
              label_plot = labels[combined_sample].data.cpu().numpy()
              predict_mu = sample_mu[combined_sample].data.cpu().numpy()
              predict_sigma = sample_sigma[combined_sample].data.cpu().numpy()
              plot_mu = np.concatenate((input_mu[combined_sample].data.cpu().numpy(), predict_mu), axis=1)
              plot_sigma = np.concatenate((input_sigma[combined_sample].data.cpu().numpy(), predict_sigma), axis=1)
              plot_metrics = {_k: _v[combined_sample] for _k, _v in sample_metrics.items()}
              plot_eight_windows(params.plot_dir, plot_mu, plot_sigma, label_plot, params.test_window, params.test_predict_start, plot_num, plot_metrics, dataset,sample)

      summary_metric = utils.final_metrics(raw_metrics, sampling=sample)
      metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
      logger.info('- Full test metrics: ' + metrics_string)
      if plot_num==-1:
        np.save(f'result/{model_type}/metrics.npy', plot_metrics)
    return summary_metric,plot_metrics


def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_sigma,
                       labels,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       dataset,
                       sampling=False):

    series_names = list(pd.read_csv(f'data/{dataset}/series_names.csv',index_col=0)['0'])
    x = np.arange(window_size)
    f = plt.figure(figsize=(500, 100))
    ncols = 5
    nrows = int(predict_values.shape[0]/ncols)+1
    ax = f.subplots(nrows, ncols)

    m=-1
    mu_pred_l = []
    std_pred_l = []
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            m=m+1
            if m<predict_values.shape[0]:
                mu_pred_l.append(pd.DataFrame(predict_values[m, predict_start:]))
                std_pred_l.append(pd.DataFrame(predict_sigma[m, predict_start:]))
    mu_pred_df = pd.concat(mu_pred_l,axis=1)
    std_pred_df = pd.concat(std_pred_l,axis=1)
    mu_pred_df.columns = series_names
    std_pred_df.columns = series_names
    mu_pred_df.to_csv(f'result/{model_type}/mu_pred.csv')
    std_pred_df.to_csv(f'result/{model_type}/std_pred.csv')

    # ls' mean CRPS: {np.mean(plot_metrics['CRPS']): .3f}")

if __name__ == '__main__':

    ############evaluate deepar-hier#############
    model_type='deepar_hier'
    if not os.path.exists(f'result/{model_type}'): 
        os.makedirs(f'result/{model_type}')
    # Load the parameters
    args = parser.parse_args()
    model_dir ='experiments/param_search/lam_0.12-lstm_dropout_0.00-lstm_hidden_dim_35.00'
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    utils.set_logger(os.path.join(model_dir, 'eval.log'))

    params.relative_metrics = args.relative_metrics
    params.sampling = args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    params.dataset = args.dataset
    
    cuda_exist = torch.cuda.is_available()  # use GPU is available

    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params)

    # Create the input data pipeline
    logger.info('Loading the datasets...')

    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, shuffle=False, num_workers=4)
    logger.info('- done.')

    print('model: ', model)
    loss_fn = net.loss_fn

    logger.info('Starting evaluate deepar-hier')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    test_metrics,plot_metrics = evaluate(model, loss_fn, test_loader, params, -1, args.dataset,params.sampling)
    save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)

    ############evaluate deepar#############
    model_type='deepar'
    if not os.path.exists(f'result/{model_type}'): 
        os.makedirs(f'result/{model_type}')
    # Load the parameters
    args = parser.parse_args()
    model_dir ='experiments/param_search/lam_0.00-lstm_dropout_0.00-lstm_hidden_dim_5.00'
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    utils.set_logger(os.path.join(model_dir, 'eval.log'))

    params.relative_metrics = args.relative_metrics
    params.sampling = args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    params.dataset = args.dataset
    
    cuda_exist = torch.cuda.is_available()  # use GPU is available

    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params)

    # Create the input data pipeline
    logger.info('Loading the datasets...')

    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, shuffle=False, num_workers=4)
    logger.info('- done.')

    print('model: ', model)
    loss_fn = net.loss_fn

    logger.info('Starting evaluate deepar')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    test_metrics,plot_metrics = evaluate(model, loss_fn, test_loader, params, -1, args.dataset,params.sampling)
    save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)

