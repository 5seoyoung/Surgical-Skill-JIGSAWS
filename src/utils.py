import numpy as np, pandas as pd

def zscore_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-8
    return mu, sd

def zscore_transform(X, mu, sd):
    return (X - mu) / sd

def sliding_window(arr, win, stride):
    # arr: [T, C]
    X = []
    for s in range(0, max(1, len(arr)-win+1), stride):
        chunk = arr[s:s+win]
        if len(chunk) < win:
            pad = np.zeros((win-len(chunk), arr.shape[1]))
            chunk = np.vstack([chunk, pad])
        X.append(chunk)
    return np.stack(X) if X else np.zeros((0,win,arr.shape[1]))
