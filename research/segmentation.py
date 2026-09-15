"""Research-only reverse DTW patching. All arrays are observation-prefix inputs."""
import numpy as np
from numba import njit


@njit(cache=True)
def zscore(x):
    sd = np.std(x)
    return (x - np.mean(x)) / max(sd, 1e-8)


@njit(cache=True)
def dtw_cost(x, y):
    """L1 minimum-total-cost path and its length. Deterministic diagonal-first ties."""
    n, m = len(x), len(y)
    d = np.full((n + 1, m + 1), np.inf)
    count = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a, b = i - 1, j - 1
            if d[i - 1, j] < d[a, b]:
                a, b = i - 1, j
            if d[i, j - 1] < d[a, b]:
                a, b = i, j - 1
            d[i, j] = abs(x[i - 1] - y[j - 1]) + d[a, b]
            count[i, j] = count[a, b] + 1
    return d[n, m], count[n, m]


@njit(cache=True)
def pairwise_dtw(samples):
    n = len(samples)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            cost, length = dtw_cost(samples[i], samples[j])
            d[i, j] = d[j, i] = cost / length
    return d


def learn_prototypes(pool, cfg):
    """Fixed-length historical shapes; seeded DTW k-medoids approximation."""
    rng = np.random.default_rng(cfg["prototype_seed"])
    length = cfg["prototype_length"]
    samples = []
    for close in pool:
        for end in range(length, len(close) + 1, 5):
            segment = close[end-length:end]
            if np.isfinite(segment).all():
                samples.append(zscore(np.log(segment)))
    samples = np.asarray(samples)
    if len(samples) > cfg["prototype_sample_limit"]:
        samples = samples[np.sort(rng.choice(len(samples), cfg["prototype_sample_limit"], replace=False))]
    d = pairwise_dtw(samples)
    ids = [int(rng.integers(len(samples)))]
    for _ in range(1, cfg["prototype_count"]):
        nearest = d[:, ids].min(axis=1)
        nearest[ids] = -1
        ids.append(int(nearest.argmax()))
    for _ in range(3):
        labels = d[:, ids].argmin(axis=1)
        for k in range(len(ids)):
            members = np.flatnonzero(labels == k)
            if len(members):
                candidates = members if len(members) <= 48 else np.sort(rng.choice(members, 48, replace=False))
                ids[k] = int(candidates[d[np.ix_(candidates, members)].mean(axis=1).argmin()])
    return samples[ids], dict(candidate_count=len(samples), medoid_ids=ids)


@njit(cache=True)
def ending_costs(close, prototypes, min_len=5, max_len=25):
    """Memoize distances for each observed endpoint; each cell reads only its prefix."""
    n = len(close)
    raw = np.full((n + 1, max_len + 1), np.inf)
    normalized = np.full((n + 1, max_len + 1), np.inf)
    logc = np.log(close)
    for end in range(min_len, n + 1):
        for length in range(min_len, min(max_len, end) + 1):
            x = zscore(logc[end - length:end])
            for p in prototypes:
                cost, path_len = dtw_cost(x, p)
                raw[end, length] = min(raw[end, length], cost)
                normalized[end, length] = min(normalized[end, length], cost / path_len)
    return raw, normalized


@njit(cache=True)
def reverse_lengths(costs, end, lookback=125, min_len=5, max_len=25):
    remaining = lookback
    result = np.zeros(lookback // min_len + 1, dtype=np.int64)
    count = 0
    while remaining:
        if remaining < min_len:
            length = remaining
        else:
            hi = min(max_len, remaining)
            length = min_len
            for candidate in range(min_len + 1, hi + 1):
                if costs[end, candidate] < costs[end, length]:
                    length = candidate
        result[count] = length
        count += 1
        remaining -= length
        end -= length
    return result[:count][::-1].copy()


def fixed_lengths(lookback, length):
    rem = lookback % length
    return np.array(([rem] if rem else []) + [length] * (lookback // length), dtype=np.int64)


def transformed_bars(frame, channels):
    a = frame[channels].to_numpy(dtype=float, copy=True)
    a[:, :4] = np.log(a[:, :4])
    a[:, 4:] = np.log1p(a[:, 4:])
    return a


@njit(cache=True)
def encode_patches(window, lengths, max_patches=25, points=5):
    channels = window.shape[1]
    out = np.zeros((max_patches, points * channels + 3))
    assert len(lengths) <= max_patches
    offset = max_patches - len(lengths)
    start = 0
    for j, length in enumerate(lengths):
        for k in range(points):
            at = k * (length - 1) / (points - 1)
            left = int(at)
            right = min(left + 1, length - 1)
            w = at - left
            for c in range(channels):
                out[offset + j, k * channels + c] = (1-w) * window[start+left, c] + w * window[start+right, c]
        out[offset+j, -3] = length / len(window)
        out[offset+j, -2] = (start+length) / len(window)
        out[offset+j, -1] = 1.0
        start += length
    assert start == len(window)
    return out.reshape(-1)


def make_dataset(frame, prototypes, cfg, methods):
    close = frame.close.to_numpy(dtype=float)
    raw, norm = ending_costs(close, prototypes, cfg["minimum_length"], cfg["maximum_length"])
    bars = transformed_bars(frame, cfg["channels"])
    lb, h = cfg["lookback"], cfg["horizon"]
    anchors = np.arange(lb - 1, len(frame) - h)
    if "valid_ohlc" in frame:
        good = frame.valid_ohlc.to_numpy(dtype=bool)
        anchors = np.array([t for t in anchors if good[t-lb+1:t+h+1].all()])
    x = {m: [] for m in methods}
    y, mu, sigma, diagnostics = [], [], [], []
    for t in anchors:
        window = bars[t-lb+1:t+1]
        mean, sd = window.mean(axis=0), np.maximum(window.std(axis=0), 1e-6)
        normalized = (window - mean) / sd
        lengths_norm = reverse_lengths(norm, t+1, lb, cfg["minimum_length"], cfg["maximum_length"])
        lengths_raw = reverse_lengths(raw, t+1, lb, cfg["minimum_length"], cfg["maximum_length"])
        for method in methods:
            if method.startswith("fixed_"):
                lengths = fixed_lengths(lb, int(method.split("_")[1]))
            elif method == "adaptive_normalized":
                lengths = lengths_norm
            elif method == "adaptive_raw":
                lengths = lengths_raw
            elif method == "random_partition":
                lengths = np.random.default_rng(cfg["prototype_seed"] + int(t)).permutation(lengths_norm)
            else:
                raise ValueError(method)
            x[method].append(encode_patches(normalized, lengths))
        y.append(((bars[t+1:t+h+1] - mean) / sd).reshape(-1))
        mu.append(mean)
        sigma.append(sd)
        diagnostics.append(dict(anchor=int(t), raw=lengths_raw.tolist(), normalized=lengths_norm.tolist()))
    return ({k: np.array(v) for k, v in x.items()}, np.array(y), np.array(mu),
            np.array(sigma), anchors, diagnostics)
