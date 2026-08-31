# Decoding Neuromodulators from FSCV data

This repository is the output of the work done to study the use of [Dynamix](https://github.com/DurstewitzLab/DynaMix-python) Foundational Time Series models capabilities in the downstream task of decoding neuromodulators. To this end, decoding_core has the following:

- model.py: Classic MLP(with hidden layers 312, 256, 128) implementation and helper functions to train in different regimes.
- preprocessing.py: Preprocessing methods to output obtain temporal embeddings for arbitrary time series data.
- shape_metrics: Certain distance metric implementations for comparing the expert activations of different Dynamix models during forecasting.
- time_wise_kfold.py: KFold regression over time using linear regression.
- utils.py: Utility functions for loading/manipulating FSCV data.


## Install 

To install the decoding core in your current Python environment:
```bash
    pip install -e .
```
