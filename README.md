# Decoding Neuromodulators from FSCV data

This repository is the output of the work done to study the use of [Dynamix](https://github.com/DurstewitzLab/DynaMix-python) Foundational Time Series models capabilities in the downstream task of decoding neuromodulators. To this end, decoding_core has the following:

```bash
decodingcore/
├── models.py #Implementation of a MLP(with hidden dims 312,256,128) and its training utilities.
├── preprocessing.py #Temporal embedding(positional, delay) helper functions for spit_complaint_data at utils.py.
├── shape_metrics.py #Implementation of 2-Wasserstein and Pruscrustes distance to compare gating weight activations.
├── time_wise_kfold.py #KFold regression per time point.
└── utils.py #Utility functions for loading/manipulating voltammogram/weight data.
```

## Install 
To install the decoding core in your current Python environment:
```bash
    pip install -e .
```
