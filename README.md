# pygidFIT: Gaussian fitting for grazing incidence diffraction (GID) data

A Python package for fitting Gaussian functions to GID (Grazing-Incidence Wide-Angle X-ray and Neutron Scattering) data. 
pygidFIT is part of the comprehensive machine learning pipeline for automated analysis of GID data. The focus is on multiparallel execution for real-time sequential processing at the synchrotron and neutron facilities.

<p align="center">
  <img src="docs/images/mlgid_logo_pygidfit.png" width="400" alt="pygidFIT">
</p>

## Installation

So far, only source installation is supported:

```bash
git clone git@github.com:mlgid-project/pygidFIT.git
cd pygidFIT
pip install -e .
```

## Usage

```python
import pygidfit
import numpy as np
filename = r'..\tests\example.h5'
pygidfit.run_scans(filename,
    crit_angle = 1,                     # critical angle to shift the sample horizon (in degrees)
    ratio_threshold=50,                 # h/w ratio for boxes classification  
    clustering_distance_rings=10,       # distance for ring clustering (in pixels)
    clustering_distance_peaks = 10,     # distance for peak clustering (in pixels)
    clustering_extend= 2,               # number of pixels to extend the cluster size
    use_pool = False,                   # whether to use pool of peaks
    debug = True,                       # whether to plot fitting result and parameters
    multiprocessing = False,            # not recommended
    polar_shape = np.array([512,1024])  # shape of the converted polar image 
)
```

## Overview

pygidFIT is part of the machine learning pipeline for automated analysis of GID data. It is designed to analyze scattering data by fitting Gaussian profiles to peaks in both 1D and 2D data. It refines the peak positions revealed by the deep learning-based peak detection by automated conventional fitting during the postprocessing stage. 

## Key Features

- **Efficient Clustering:** Groups nearby peaks for better fitting
- **Parameter Reuse:** Maintains a cache of previous fit parameters to speed up processing of time series
- **Parallel Processing:** Uses multiprocessing for faster fitting of large datasets
- **HDF5 Integration:** Works with HDF5 file format common in synchrotron data

## Authors 

The package is developed by Ekaterina Kneschaurek @ [the Schreiber Lab](https://github.com/schreiber-lab) with the help of [Vladimir Starostin](https://github.com/StarostinV) ([mlcolab](https://github.com/mlcolab)), [Constantin Völter](https://github.com/cvoelt) and [Ainur Abukaev](https://github.com/ainurabukaev99).


## License

MIT 