# Probabilistic-Forecast-Reconciliation-with-DL

Code for paper **"Probabilistic Forecast Reconciliation with Kullback-Leibler Divergence Regularization"** accepted in ICDM 2023 Workshop "AI for Time Series Analysis".

In this paper, we conducted experiments on three datasets. This code is for the Infant dataset.

You can run `run.sh` to reproduce the results in this paper.

Specifically,
- `preprocess.py`: transfer the data to training instances and generate a hierarchical structure.
- `train.py`: train the deepar-hier model with specified hyperparameters.
- `search_params.py`: use grid search to tune hyperparameters, especially lambda.
- `evaluate.py`: evaluate "deepar-hier" and "deeper" on test data
- `base_hier.py`: run other existing probabilistic forecast reconciliation methods.
- `compare.py`: compare "deepar-hier" and other methods, output CRPS and MCB results.

DeepAR related code refers to <https://github.com/husnejahan/DeepAR-pytorch>






