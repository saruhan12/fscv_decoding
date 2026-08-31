"""
Batched embedding utilities for (T, B, N) arrays.

This is the (T, B, N) counterpart of the `Embedding` class in the utils module:
same tau estimators, same eq.-6 positional encoding, same delay embedding, but

  * it operates on a whole batch of sequences at once (numpy, no per-sweep loop),
  * it *returns the embedding parameters*, so the identical embedding (same tau,
    same phases, same trim) can be re-applied to the context / train / test
    splits instead of being re-estimated independently on each.

Shape convention
----------------
    (T,)        -> treated as (T, 1, 1)
    (T, N)      -> treated as (T, 1, N)   [same semantics as Embedding]
    (T, B, N)   -> used as-is

Everything returns float arrays with the dtype of the input.
"""

import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks

__all__ = [
    "to_TBN",
    "estimate_pos_tau",
    "estimate_tdm_tau",
    "snap_tau_to_period",
    "make_phis",
    "make_sines",
    "positional_embedding",
    "zero_embedding",
    "delay_embedding",
    "delay_embedding_random",
    "apply_embedding",
]


# --------------------------------------------------------------------------- #
# shape helpers
# --------------------------------------------------------------------------- #
def to_TBN(X):
    """Coerce input to (T, B, N)."""
    X = np.asarray(X)
    if X.ndim == 1:
        return X[:, None, None]
    if X.ndim == 2:                     # (T, N) -> single batch element
        return X[:, None, :]
    if X.ndim == 3:
        return X
    raise ValueError(f"expected 1D/2D/3D array, got shape {X.shape}")


def _series_matrix(X, channel=0, n_series=64, seed=0):
    """
    Flatten (T, B, N) into a (T, S) matrix of scalar series for ACF estimation,
    optionally subsampling S columns (ACF over 10k sweeps is pointless and slow).

    channel=None uses all channels; channel=int uses one (default: the observed
    data channel 0, which is what you want before anything has been appended).
    """
    X = to_TBN(X)
    T, B, N = X.shape
    A = X if channel is None else X[:, :, channel][:, :, None]
    A = A.reshape(T, -1)
    S = A.shape[1]
    if n_series is not None and S > n_series:
        rng = np.random.default_rng(seed)
        A = A[:, rng.choice(S, size=n_series, replace=False)]
    return np.ascontiguousarray(A, dtype=np.float64)


def _acf_curve(x, nlags):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if not np.any(np.abs(x) > 0):
        return np.zeros(nlags + 1)
    return acf(x, nlags=nlags, fft=True)


def _dominant_peak(curve, min_lag):
    """
    Largest autocorrelation peak at lag > min_lag; falls back to the argmax of
    the tail if no local maximum qualifies. (Identical logic to
    Embedding.estimate_pos_tau, factored out.)
    """
    peaks, _ = find_peaks(curve)
    valid = [i for i in peaks if min_lag < i < len(curve)]
    if valid:
        return int(valid[int(np.argmax(curve[valid]))])
    start = min(min_lag + 1, len(curve) - 1)
    return int(start + int(np.argmax(curve[start:])))


# --------------------------------------------------------------------------- #
# tau estimators
# --------------------------------------------------------------------------- #
def estimate_pos_tau(X, min_lag=None, max_lag=None, channel=0,
                     n_series=64, seed=0, reduce="mean_acf"):
    """
    Autocorrelation time for the positional embedding, over a batch.

    reduce:
        "mean_acf" -- average the ACF curves first, then pick the peak.
                      Much more robust than per-sweep peak picking when the
                      per-sweep SNR is low (the usual FSCV case).
        "median"   -- per-series tau, then median.
        "max"      -- per-series tau, then max (matches the original single
                      series behaviour, but is dominated by outlier sweeps).
    """
    A = _series_matrix(X, channel=channel, n_series=n_series, seed=seed)
    T, S = A.shape
    if max_lag is None:
        max_lag = T - 1
    max_lag = int(min(max_lag, T - 1))
    if min_lag is None:
        min_lag = T // 10
    min_lag = int(max(1, min(min_lag, max_lag - 1)))

    curves = np.stack([_acf_curve(A[:, s], max_lag) for s in range(S)], axis=0)

    if reduce == "mean_acf":
        return _dominant_peak(curves.mean(axis=0), min_lag)

    taus = np.array([_dominant_peak(c, min_lag) for c in curves])
    if reduce == "median":
        return int(np.median(taus))
    if reduce == "max":
        return int(taus.max())
    raise ValueError(f"unknown reduce: {reduce}")


def estimate_tdm_tau(X, acorr_threshold=1.0 / np.e, channel=0,
                     n_series=64, seed=0, reduce="median"):
    """
    Delay-embedding tau: first lag whose autocorrelation drops below
    `acorr_threshold`, aggregated over the batch.
    """
    A = _series_matrix(X, channel=channel, n_series=n_series, seed=seed)
    T, S = A.shape
    nlags = max(1, T // 2)

    taus = np.ones(S, dtype=int)
    for s in range(S):
        c = _acf_curve(A[:, s], nlags)
        below = np.where(c[1:] < acorr_threshold)[0]
        taus[s] = int(below[0] + 1) if len(below) else 1

    if reduce == "median":
        return int(max(1, np.median(taus)))
    if reduce == "max":
        return int(taus.max())
    if reduce == "min":
        return int(taus.min())
    raise ValueError(f"unknown reduce: {reduce}")


def _divisors(n):
    n = int(n)
    ds, i = [], 1
    while i * i <= n:
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
        i += 1
    return sorted(ds)


def snap_tau_to_period(tau, period, min_tau=2):
    """
    Snap tau to the nearest divisor of `period`.

    Why this matters: when a sweep of length `period` is tiled `rep` times along
    the time axis, the data channel is exactly periodic with `period`. If the
    sine channel has an incommensurate period, its phase drifts relative to the
    data across the context, so the model sees a different phase for the same
    voltammogram at every repetition. Snapping keeps the embedding coherent
    with the tiling.
    """
    cands = [d for d in _divisors(period) if d >= min_tau]
    if not cands:
        return int(tau)
    return int(min(cands, key=lambda d: abs(d - tau)))


# --------------------------------------------------------------------------- #
# positional embedding (eq. 6)
# --------------------------------------------------------------------------- #
def make_phis(n_extra, mode="linspace", rng=None):
    """Phase offsets for the sine channels."""
    if n_extra <= 0:
        return np.zeros(0)
    if n_extra == 1:
        return np.zeros(1)
    if mode == "linspace":
        return np.linspace(0.0, np.pi / 2, n_extra)
    if mode == "random":
        rng = np.random.default_rng() if rng is None else rng
        return np.sort(rng.uniform(0.0, np.pi / 2, size=n_extra))
    raise ValueError(f"unknown phi mode: {mode}")


def make_sines(T, B, tau, phis, dtype=np.float32, t0=0):
    """
    sin(2*pi*t/tau + phi_i) broadcast over the batch -> (T, B, len(phis)).

    t runs over t0+1 ... t0+T, so passing t0 lets you continue the phase of a
    previous block instead of restarting it.
    """
    phis = np.atleast_1d(np.asarray(phis, dtype=np.float64))
    if phis.size == 0:
        return np.zeros((T, B, 0), dtype=dtype)
    t = np.arange(t0 + 1, t0 + T + 1, dtype=np.float64)[:, None]      # (T, 1)
    P = np.sin(2.0 * np.pi / float(tau) * t + phis[None, :])          # (T, P)
    P = np.broadcast_to(P[:, None, :], (T, B, phis.size))
    return np.ascontiguousarray(P, dtype=dtype)


def positional_embedding(X, model_dim, tau=None, phis=None, phi_mode="linspace",
                         rng=None, t0=0, **tau_kwargs):
    """
    Append sine channels until X has `model_dim` channels.

    Returns (X_emb, params) with params = {"tau", "phis", "t0"}.
    """
    X = to_TBN(X)
    T, B, N = X.shape
    n_extra = model_dim - N
    if n_extra <= 0:
        return X, {"tau": tau, "phis": np.zeros(0), "t0": t0}

    if tau is None:
        tau = estimate_pos_tau(X, **tau_kwargs)
    if phis is None:
        phis = make_phis(n_extra, mode=phi_mode, rng=rng)
    phis = np.atleast_1d(np.asarray(phis, dtype=np.float64))
    if phis.size != n_extra:
        raise ValueError(f"got {phis.size} phases for {n_extra} missing channels")

    P = make_sines(T, B, tau, phis, dtype=X.dtype, t0=t0)
    return np.concatenate([X, P], axis=2), {"tau": int(tau), "phis": phis, "t0": t0}


# --------------------------------------------------------------------------- #
# zero / delay embeddings
# --------------------------------------------------------------------------- #
def zero_embedding(X, model_dim):
    X = to_TBN(X)
    T, B, N = X.shape
    n_extra = model_dim - N
    if n_extra <= 0:
        return X, {}
    Z = np.zeros((T, B, n_extra), dtype=X.dtype)
    return np.concatenate([X, Z], axis=2), {}


def delay_embedding(X, model_dim, tau=None, source_channel=-1, **tau_kwargs):
    """
    Standard delay embedding, batched.

    NOTE: this shortens the time axis by trim = n_extra * tau. The trim is
    returned in params so the caller can keep the splits aligned (and so any
    per-timestep target can be trimmed the same way).
    """
    X = to_TBN(X)
    T, B, N = X.shape
    n_extra = model_dim - N
    if n_extra <= 0:
        return X, {"tau": tau, "trim": 0, "source_channel": source_channel}

    if tau is None:
        tau = estimate_tdm_tau(X, channel=source_channel, **tau_kwargs)
    tau = int(max(1, tau))

    trim = n_extra * tau
    if trim >= T:                                   # same guard as the original
        tau = max(1, T // (n_extra + 1))
        trim = n_extra * tau

    ts = X[:, :, source_channel]                    # (T, B)
    parts = [X[trim:]]
    for i in range(1, n_extra + 1):
        parts.append(ts[trim - i * tau: T - i * tau][:, :, None])
    out = np.concatenate(parts, axis=2)
    return out, {"tau": int(tau), "trim": int(trim), "source_channel": source_channel}


def delay_embedding_random(X, model_dim, upper_tau=10, lower_tau=3, seed=None,
                           source_channel=0, taus=None):
    X = to_TBN(X)
    T, B, N = X.shape
    n_extra = model_dim - N
    if n_extra <= 0:
        return X, {"taus": [], "trim": 0}

    if taus is None:
        rng = np.random.default_rng(seed)
        taus = rng.integers(lower_tau, upper_tau + 1, size=n_extra).tolist()
    taus = [int(t) for t in taus]
    trim = max(taus)

    ts = X[:, :, source_channel]
    parts = [X[trim:]]
    for tau_i in taus:
        parts.append(ts[trim - tau_i: T - tau_i][:, :, None])
    return np.concatenate(parts, axis=2), {"taus": taus, "trim": int(trim),
                                           "source_channel": source_channel}


# --------------------------------------------------------------------------- #
# dispatcher
# --------------------------------------------------------------------------- #
def apply_embedding(X, model_dim, method="pos_embedding", params=None, **kwargs):
    """
    Embed X to `model_dim` channels.

    Returns (X_emb, params). Feed the returned `params` back in for the other
    splits so context / train / test share one tau and one set of phases:

        Xd, p = apply_embedding(X_data,    n_dim, "pos_embedding")
        Xc, _ = apply_embedding(X_context, n_dim, "pos_embedding", params=p)
        Xt, _ = apply_embedding(X_test,    n_dim, "pos_embedding", params=p)
    """
    params = dict(params or {})
    params.pop("method", None)

    if method == "pos_embedding":
        kwargs.update({k: params[k] for k in ("tau", "phis", "t0") if k in params})
        out, p = positional_embedding(X, model_dim, **kwargs)
    elif method == "zero_embedding":
        out, p = zero_embedding(X, model_dim)
    elif method == "delay_embedding":
        kwargs.update({k: params[k] for k in ("tau", "source_channel") if k in params})
        out, p = delay_embedding(X, model_dim, **kwargs)
    elif method == "delay_embedding_random":
        kwargs.update({k: params[k] for k in ("taus", "source_channel") if k in params})
        out, p = delay_embedding_random(X, model_dim, **kwargs)
    else:
        raise ValueError(f"Unsupported embedding method: {method}")

    p["method"] = method
    return out, p