# Kinetic Model 2TC Curve Fit for PET data

## Project description
This repository contains a Python script for analysing PET (positron emission tomography) data using a two-compartment model (2TC). The main task of the script is to calculate and store kinetic parameters for different regions of the lung.

## Prerequisites
Before the script can be executed, some dependencies must be installed.

### Required Python libraries:
- `numpy`
- `pandas`
- `torch`
- SimpleITK
- `matplotlib`
- `scipy`
- `natsort`

These can be installed with the following command:
```bash
pip install numpy pandas torch SimpleITK matplotlib scipy natsort
```

## Directory structure
```
/
├── utils/
│ ├── utils_torch.py # Contains functions for torch interpolation and convolution
│ ├── set_root_paths.py # Contains variables for data paths
├── main.py # Main script for PET data processing
├── README.md # This file
```

## Function description

### `reduce_to_600(values)`
Reduces a given time series to 600 values by extracting the maximum value at defined intervals.

### `KineticModel_2TC_curve_fit`
A class for modelling kinetic parameters in PET data. Contains methods:
- `read_idif(sample_time, t)`: Reads and interpolates IDIF values from CSV files.
- `PET_2TC_KM(t, k1, k2, k3, Vb, alpha, beta)`: Calculates the PET data using a two-compartment model.
- `PET_normal(t, k1, k2, k3, Vb)`: Calculates the PET data using a two-compartment model.

Translated with DeepL.com (free version)
