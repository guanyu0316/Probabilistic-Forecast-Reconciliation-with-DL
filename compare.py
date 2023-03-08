import pandas as pd
import numpy as np
import itertools

deepar_hier_crps = np.load('result/deepar_hier/metrics.npy',allow_pickle=True).item()['CRPS']
pure_deepar_crps = np.load('result/deepar/metrics.npy',allow_pickle=True).item()['CRPS']
pure_deepar_df = pd.DataFrame({'DeepAR':pure_deepar_crps})
deepar_hier_df = pd.DataFrame({'DeepAR-Hier':deepar_hier_crps})

hier_dict_name = np.load('data/infant/hier_dict_name.npy',allow_pickle=True).item()
hier_dict = np.load('data/infant/hier_dict.npy',allow_pickle=True).item()
series_names = list(pd.read_csv('data/infant/series_names.csv',index_col=0)['0'])
df_l = []
crps_l = []
for fcst_model,permute_method,rcc_method,rcc_covariance in itertools.product(['arima','ets'],['stack','rank','random'],['mint','bu'],['struct','none']):
  try:
    my_metrics = np.load(f'result/{fcst_model}_{permute_method}_{rcc_method}_{rcc_covariance}/metrics.npy',allow_pickle=True).item()['crps']
    df = pd.DataFrame({f'{fcst_model.capitalize()}_{permute_method.capitalize()}_{rcc_method.capitalize()}_{rcc_covariance.capitalize()}':my_metrics})
    df_l.append(df)
  except:
    continue

df_l.append(pure_deepar_df)
df_l.append(deepar_hier_df)

all_df= pd.concat(df_l,axis=1)

print('Average CRPS')
print(all_df.mean())

rank_res = all_df.rank(axis=1)
k=14
N=25
r = 4.743
d = k*(k+1)/(12*N)

import matplotlib.pyplot as plt

plt.style.use('_mpl-gallery')

x = rank_res.mean().sort_values()
y = list(range(0,14,1))
xerr = d**0.5*r

# plot:
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor('white')
ax.errorbar(x, y, xerr = xerr, fmt='o', linewidth=2, capsize=6)
ax.set_yticks(list(range(0,14,1)),list(x.index))
ax.set_title('Dataset:Infant')
# plt.show()
plt.tight_layout()
plt.savefig('infant_res.pdf',dpi=200)

def get_level_agg_metric(metrics,series_names):
    level_agg_crps = {}
    level_num_list = [len(n.split('_')) for n in series_names]
    for level_num in list(set(level_num_list)):
        idx_list = [i for i,x in enumerate(level_num_list) if x==level_num]
        try:
          level_agg_crps[level_num] = np.array(metrics['CRPS'])[idx_list].mean()
        except:
          level_agg_crps[level_num] = np.array(metrics['crps'])[idx_list].mean()
    try:
      avg_crps = metrics['CRPS'].mean()
    except:
      avg_crps = metrics['crps'].mean()
    return level_agg_crps, avg_crps

my_metrics = np.load('result/deepar_hier/metrics.npy',allow_pickle=True).item()
series_names = list(pd.read_csv('data/infant/series_names.csv',index_col=0)['0'])
level_agg_crps,avg_crps = get_level_agg_metric(my_metrics,series_names)
print('Average CRPS at each level \n')
print(level_agg_crps)