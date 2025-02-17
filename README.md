# Myo_Gait : Myoelectric driven gait phase recognition development platform

This repository contains code for gait phase recognition experiment using lower limb EMG signals benchmarked on SIAT-LLMD publicly available dataset. The project leverages the `LibEMG` library to facilitate preprocessing, feature extraction, model employment, postprocessing, and performance evaluation.
## Overview

The goal of this project is to provide an accessible platform for developing myoelectric driven gait phases recognition schemes. The code is designed to be modular, allowing users to easily test standard myoelectric pattern recognition pipeline as well as allow them to embed their proposed contribution in terms of any block of the general pipeline ( mainly in the feature extraction and models employment ) to pave the way for further boosting the state-of-art performances, potentially pushing forward the lower limb active prosthetic market. This platform was developed during our recent work in which we proposed that spatial feature extraction techniques e.g. Phasors feature extraction scheme provide significant improvements for gait phase recognition motor task  (Tigrini et al., 2024) [1].

## Dataset

The dataset used in this project is SIAT-LLMD publicly available dataset (Wei et al., 2023) [2] and be accessed [here](https://pubmed.ncbi.nlm.nih.gov/37280249/). It contains EMG signals recorded from lower limb muscles during walking, along with corresponding gait phase labels.

## Dependencies

To run the code, you will need the following Python libraries:

- `LibEMG` (for EMG signal processing and feature extraction) (Eddy et al., 2023) [3]
- `NumPy` (for numerical computations)
- `Pandas` (for data manipulation)
- `Scikit-learn` (for machine learning models and evaluation)
- `Matplotlib` (for visualization)

You can install the required libraries using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib

[1] Tigrini, Andrea, et al. "Phasor-based myoelectric synergy features: a fast hand-crafted feature extraction scheme for boosting performance in gait phase recognition." Sensors 24.17 (2024): 5828.
[2] Wei, Wenhao, et al. "Surface electromyogram, kinematic, and kinetic dataset of lower limb walking for movement intent recognition." Scientific Data 10.1 (2023): 358.
[3] Eddy, Ethan, et al. "Libemg: an open source library to facilitate the exploration of myoelectric control." IEEE Access (2023).


Please cite as as
-
-
