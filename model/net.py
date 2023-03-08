import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import hts


logger = logging.getLogger('DeepAR.Net')

class Net(nn.Module):
    def __init__(self, params):
        '''
        We define a recurrent network that predicts the future values of a time-dependent variable based on
        past inputs and covariates.
        '''
        super(Net, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(params.num_class, params.embedding_dim)

        self.lstm = nn.LSTM(input_size=1+params.cov_dim+params.embedding_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)
        '''self.lstm = nn.LSTM(input_size=1 + params.cov_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)'''
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_presigma = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_sigma = nn.Softplus()

    def forward(self, x, idx, hidden, cell):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, 1+cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            idx ([1, batch_size]): one integer denoting the time series id
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t-1
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        '''
        onehot_embed = self.embedding(idx) #TODO: is it possible to do this only once per window instead of per step?
        lstm_input = torch.cat((x, onehot_embed), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # use h from all three layers to calculate mu and sigma
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma)  # softplus to make sure standard deviation is positive

        mu = torch.squeeze(mu)
        sigma = torch.squeeze(sigma)

        return mu, sigma, hidden, cell

    def init_hidden(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def init_cell(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def test(self, x, v_batch, id_batch, hidden, cell, sampling=False):
        batch_size = x.shape[1]
        if sampling:
            samples = torch.zeros(self.params.sample_times, batch_size, self.params.predict_steps,
                                       device=self.params.device)
            for j in range(self.params.sample_times):
                decoder_hidden = hidden
                decoder_cell = cell
                for t in range(self.params.predict_steps):
                    mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                         id_batch, decoder_hidden, decoder_cell)
                    gaussian = torch.distributions.normal.Normal(mu_de, sigma_de)
                    pred = gaussian.sample()  # not scaled
                    samples[j, :, t] = pred * v_batch[:, 0] + v_batch[:, 1]
                    if t < (self.params.predict_steps - 1):
                        x[self.params.predict_start + t + 1, :, 0] = pred

            hier_dict = np.load(f'data/{self.params.dataset}/hier_dict.npy',allow_pickle=True).item()
            hier_dict_name = np.load(f'data/{self.params.dataset}/hier_dict_name.npy',allow_pickle=True).item()
            series_names = list(pd.read_csv(f'data/{self.params.dataset}/series_names.csv',index_col=0)['0'])
            level_num_list = [len(n.split('_')) for n in series_names]
            bottom_idx = [i for i,x in enumerate(level_num_list) if x==max(level_num_list)]

            for t in range(4):
                for i in list(set(range(len(series_names)))-set(bottom_idx)):
                    samples[:, i, t] = samples[:, hier_dict[i], t].sum(axis=1)
            sample_mu = torch.median(samples, dim=0)[0]
            sample_sigma = samples.std(dim=0)
            return samples, sample_mu, sample_sigma

        else:
            decoder_hidden = hidden
            decoder_cell = cell
            sample_mu = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            sample_sigma = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            for t in range(self.params.predict_steps):
                mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                     id_batch, decoder_hidden, decoder_cell)
                sample_mu[:, t] = mu_de * v_batch[:, 0] + v_batch[:, 1]
                sample_sigma[:, t] = sigma_de * v_batch[:, 0]
                if t < (self.params.predict_steps - 1):
                    x[self.params.predict_start + t + 1, :, 0] = mu_de
            return sample_mu, sample_sigma


    def test2(self, x, v_batch, id_batch, hidden, cell, sampling=False):
        batch_size = x.shape[1]
        if sampling:
            samples = torch.zeros(self.params.sample_times, batch_size, self.params.predict_steps,
                                    device=self.params.device)
            for j in range(self.params.sample_times):
                decoder_hidden = hidden
                decoder_cell = cell
                for t in range(self.params.predict_steps):
                    mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                        id_batch, decoder_hidden, decoder_cell)
                    gaussian = torch.distributions.normal.Normal(mu_de, sigma_de)
                    pred = gaussian.sample()  # not scaled
                    samples[j, :, t] = pred * v_batch[:, 0] + v_batch[:, 1]
                    if t < (self.params.predict_steps - 1):
                        x[self.params.predict_start + t + 1, :, 0] = pred

            sample_mu = torch.median(samples, dim=0)[0]
            sample_sigma = samples.std(dim=0)
            return samples, sample_mu, sample_sigma

        else:
            decoder_hidden = hidden
            decoder_cell = cell
            sample_mu = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            sample_sigma = torch.zeros(batch_size, self.params.predict_steps, device=self.params.device)
            for t in range(self.params.predict_steps):
                mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                    id_batch, decoder_hidden, decoder_cell)
                sample_mu[:, t] = mu_de * v_batch[:, 0] + v_batch[:, 1]
                sample_sigma[:, t] = sigma_de * v_batch[:, 0]
                if t < (self.params.predict_steps - 1):
                    x[self.params.predict_start + t + 1, :, 0] = mu_de
            return sample_mu, sample_sigma



def loss_fn(mu: Variable, sigma: Variable, labels: Variable, lam, dataset,is_target_b=False):
    '''
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''
    hier_dict = np.load(f'data/{dataset}/hier_dict.npy',allow_pickle=True).item()
    series_names = list(pd.read_csv(f'data/{dataset}/series_names.csv',index_col=0)['0'])
    level_num_list = [len(n.split('_')) for n in series_names]
    bottom_level = max(level_num_list)
    bottom_idx = [i for i,x in enumerate(level_num_list) if x==bottom_level]

    
    if is_target_b:
        zero_index = (labels != 0)
        distribution = torch.distributions.normal.Normal(mu[zero_index][bottom_idx], sigma[zero_index][bottom_idx])
        likelihood = distribution.log_prob(labels[zero_index][bottom_idx])
    else:
        zero_index = (labels != 0)
        distribution = torch.distributions.normal.Normal(mu[zero_index], sigma[zero_index])
        likelihood = distribution.log_prob(labels[zero_index])

    # KL reg
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(230)
    fixed_sample = torch.randn(200).to(device) # sample num = 200
    
    final_loss = -torch.mean(likelihood)
    for k in hier_dict.keys():
        # KL_reg1
        obs_sample = fixed_sample*sigma[k]+mu[k]
        obs_dist = torch.distributions.normal.Normal(mu[k], sigma[k])
        obs_log_prob = obs_dist.log_prob(obs_sample)

        pred_mu = mu[hier_dict[k]].sum()
        pred_sigma = (sigma[hier_dict[k]]**2).sum()**0.5
        pred_dist = torch.distributions.normal.Normal(pred_mu, pred_sigma)
        pred_log_prob = pred_dist.log_prob(obs_sample)

        KL_loss1 = nn.KLDivLoss(reduction="mean")
        KL_reg1 = KL_loss1(pred_log_prob.exp().softmax(dim=-1).log(), obs_log_prob.exp().softmax(dim=-1))

        # KL_reg2
        obs_sample = fixed_sample*(sigma[hier_dict[k]]**2).sum()**0.5+mu[hier_dict[k]].sum()
        obs_dist = torch.distributions.normal.Normal(mu[hier_dict[k]].sum(), (sigma[hier_dict[k]]**2).sum()**0.5)
        obs_log_prob = obs_dist.log_prob(obs_sample)

        pred_mu = mu[k]
        pred_sigma = sigma[k]
        pred_dist = torch.distributions.normal.Normal(pred_mu, pred_sigma)
        pred_log_prob = pred_dist.log_prob(obs_sample)

        KL_loss2 = nn.KLDivLoss(reduction="mean")
        KL_reg2 = KL_loss2(pred_log_prob.exp().softmax(dim=-1).log(), obs_log_prob.exp().softmax(dim=-1))
        print(f'KL_reg1:{KL_reg1}')
        print(f'KL_reg2:{KL_reg2}')
        

        final_loss = final_loss+lam*(KL_reg1+KL_reg2)/2
    print(f'final_loss:{final_loss}')
        
    return final_loss


def loss_fn2(mu_: Variable, sigma_: Variable, labels_: Variable, lam, dataset, is_target_b=True):
    '''
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''

    hier_dict = np.load(f'data/{dataset}/hier_dict.npy',allow_pickle=True).item()
    series_names = list(pd.read_csv(f'data/{dataset}/series_names.csv',index_col=0)['0'])
    level_num_list = [len(n.split('_')) for n in series_names]
    bottom_level = max(level_num_list)
    bottom_idx = [i for i,x in enumerate(level_num_list) if x==bottom_level]

    for t in range(mu_.shape[1]):
        mu = mu_[:,t]
        sigma = sigma_[:,t]
        labels = labels_[:,t]
        zero_index = (labels != 0)
        if is_target_b:
            zero_index = (labels != 0)
            distribution = torch.distributions.normal.Normal(mu[zero_index][bottom_idx], sigma[zero_index][bottom_idx])
            likelihood = distribution.log_prob(labels[zero_index][bottom_idx])
        else:
            zero_index = (labels != 0)
            distribution = torch.distributions.normal.Normal(mu[zero_index], sigma[zero_index])
            likelihood = distribution.log_prob(labels[zero_index])

        # KL reg
        torch.manual_seed(230)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        fixed_sample = torch.randn(200).to(device) # sample num = 200
        
        final_loss = -torch.mean(likelihood)
        for k in hier_dict.keys():
            # KL_reg1
            obs_sample = fixed_sample*sigma[k]+mu[k]
            obs_dist = torch.distributions.normal.Normal(mu[k], sigma[k])
            obs_log_prob = obs_dist.log_prob(obs_sample)

            pred_mu = mu[hier_dict[k]].sum()
            pred_sigma = (sigma[hier_dict[k]]**2).sum()**0.5
            pred_dist = torch.distributions.normal.Normal(pred_mu, pred_sigma)
            pred_log_prob = pred_dist.log_prob(obs_sample)

            KL_loss1 = nn.KLDivLoss(reduction="mean")
            KL_reg1 = KL_loss1(pred_log_prob.exp().softmax(dim=-1).log(), obs_log_prob.exp().softmax(dim=-1))

            # KL_reg2
            obs_sample = fixed_sample*(sigma[hier_dict[k]]**2).sum()**0.5+mu[hier_dict[k]].sum()
            obs_dist = torch.distributions.normal.Normal(mu[hier_dict[k]].sum(), (sigma[hier_dict[k]]**2).sum()**0.5)
            obs_log_prob = obs_dist.log_prob(obs_sample)

            pred_mu = mu[k]
            pred_sigma = sigma[k]
            pred_dist = torch.distributions.normal.Normal(pred_mu, pred_sigma)
            pred_log_prob = pred_dist.log_prob(obs_sample)

            KL_loss2 = nn.KLDivLoss(reduction="mean")
            KL_reg2 = KL_loss2(pred_log_prob.exp().softmax(dim=-1).log(), obs_log_prob.exp().softmax(dim=-1))

            final_loss = final_loss+lam*(KL_reg1+KL_reg2)/2

            # TO-DO: lamda
    
    return final_loss

# if relative is set to True, metrics are not normalized by the scale of labels
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return [diff, summation]


def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    diff = torch.sum(torch.mul((mu[zero_index] - labels[zero_index]), (mu[zero_index] - labels[zero_index]))).item()
    if relative:
        return [diff, torch.sum(zero_index).item(), torch.sum(zero_index).item()]
    else:
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        if summation == 0:
            logger.error('summation denominator error! ')
        return [diff, summation, torch.sum(zero_index).item()]


def accuracy_ROU(rou: float, samples: torch.Tensor, labels: torch.Tensor, relative = False):
    numerator = 0
    denominator = 0
    pred_samples = samples.shape[0]
    for t in range(labels.shape[1]):
        zero_index = (labels[:, t] != 0)
        if zero_index.numel() > 0:
            rou_th = math.ceil(pred_samples * (1 - rou))
            rou_pred = torch.topk(samples[:, zero_index, t], dim=0, k=rou_th)[0][-1, :]
            abs_diff = labels[:, t][zero_index] - rou_pred
            numerator += 2 * (torch.sum(rou * abs_diff[labels[:, t][zero_index] > rou_pred]) - torch.sum(
                (1 - rou) * abs_diff[labels[:, t][zero_index] <= rou_pred])).item()
            denominator += torch.sum(labels[:, t][zero_index]).item()
    if relative:
        return [numerator, torch.sum(labels != 0).item()]
    else:
        return [numerator, denominator]


def accuracy_ND_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mu[labels == 0] = 0.

    diff = np.sum(np.abs(mu - labels), axis=1)
    if relative:
        summation = np.sum((labels != 0), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result
    else:
        summation = np.sum(np.abs(labels), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result


def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    mu[mask] = 0.

    diff = np.sum((mu - labels) ** 2, axis=1)
    summation = np.sum(np.abs(labels), axis=1)
    mask2 = (summation == 0)
    if relative:
        div = np.sum(~mask, axis=1)
        div[mask2] = 1
        result = np.sqrt(diff / div)
        result[mask2] = -1
        return result
    else:
        summation[mask2] = 1
        result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))
        result[mask2] = -1
        return result


def accuracy_ROU_(rou: float, samples: torch.Tensor, labels: torch.Tensor, relative = False):
    samples = samples.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    samples[:, mask] = 0.

    pred_samples = samples.shape[0]
    rou_th = math.floor(pred_samples * rou)

    samples = np.sort(samples, axis=0)
    rou_pred = samples[rou_th]

    abs_diff = np.abs(labels - rou_pred)
    abs_diff_1 = abs_diff.copy()
    abs_diff_1[labels < rou_pred] = 0.
    abs_diff_2 = abs_diff.copy()
    abs_diff_2[labels >= rou_pred] = 0.

    numerator = 2 * (rou * np.sum(abs_diff_1, axis=1) + (1 - rou) * np.sum(abs_diff_2, axis=1))
    denominator = np.sum(labels, axis=1)

    mask2 = (denominator == 0)
    denominator[mask2] = 1
    result = numerator / denominator
    result[mask2] = -1
    return result
