"""treeid: recovering occluded apple-tree branch structure from RGB-D images.

MMA3001 Numerical Methods and Machine Learning project (Monash University, S2 2026).

Modules
-------
io          dataset discovery, filename parsing, loading
preprocess  cropping/resampling, vegetation index, depth normalisation, network input
align       FFT cross-correlation registration of ground-truth masks
metrics     overlap and tolerance-based centreline metrics
baseline    classical ridge + depth detector (the comparison baseline)
model       compact U-Net and FLOP / arithmetic-intensity estimator
train       training loop with early stopping
skeleton    branch skeleton extraction and 3-D back-projection
"""

__version__ = "0.1.0"
