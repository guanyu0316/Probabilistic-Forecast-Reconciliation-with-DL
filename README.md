# Probabilistic-Forecast-Reconciliation-with-DL
Code for paper Probabilistic Forecast Reconciliation with Deep Learning 

In this paper, we conducted experiments on three datasets, this code is for Infant dataset.

You can run run.sh to reproduct the results in this paper.

Specifically,
- `preprocess.py`: transfer the data to training instances and generate hierachical structure.
- `train.py`: train the deepar-hier model with specified hyperparameters.
- `search_params.py`: use grid search to tune hyperparameters, especially lambda.
- `evaluate.py`: evaluate deepar-hier and deepar on test data
- `base_hier.py`: run other existing probabilistic forecast reconciliation methods.
- `compare.py`: compare deepar-hier and other methods, output CRPS and MCB results.



