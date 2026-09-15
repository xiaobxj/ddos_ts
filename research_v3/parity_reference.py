"""Run with the existing Python 3.13 research runtime to export v2 references."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'research_v2'))
from core import economic_window, multi_features, ridge_path

frame = pd.read_csv(ROOT.parent / 'research/data/1_000300.csv')
anchors = np.array([124, 1200, 2000, 3000, 3900])
features = []
for t in anchors:
    channels, base = economic_window(frame.iloc[t-124:t+1])
    features.append(multi_features(channels, base))
rng = np.random.default_rng(89234)
x = rng.normal(size=(130, 12)); y = rng.normal(size=130); xt = rng.normal(size=(9, 12))
prediction = ridge_path(x, y, xt, [0.1, 10.0])
np.savez_compressed(ROOT / 'sources/v2_parity_reference.npz', anchors=anchors, features=features,
                    x=x, y=y, xt=xt, ridge01=prediction['0.1'], ridge10=prediction['10.0'])
print('Independent round-2 feature and ridge references saved.')
