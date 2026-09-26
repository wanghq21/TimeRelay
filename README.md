# TimeRelay: Relation-Aware Recurrent Variable Modeling for Multivariate Time Series Forecasting


## Overview
TimeRelay is a relation-aware recurrent framework for multivariate time series forecasting. Instead of explicitly materializing dense variable-to-variable interactions, it models cross-variable dependencies from the perspective of predictive information propagation.

Its core module, Relation-Aware State Propagation (RASP), constructs a compact multivariate context for each variable and uses the variable-context relation to adaptively control recurrent state updates. The affine recurrent form further supports work-efficient parallel evaluation with linear arithmetic work in the number of variables and logarithmic dependency depth.


## Highlights

- Relation-aware recurrent propagation for cross-variable dependency modeling

- Compact leave-one-out multivariate context with learnable residual refinement

- Relation-aware gating based on the current variable and multivariate context

- Work-efficient parallel prefix scan for recurrent evaluation

- Strong forecasting performance across ten multivariate benchmarks

- Favorable runtime, memory usage, and scalability


## Architecture

The overall architecture of TimeRelay is shown below:

<p align="center">
  <img src="./figure/architecture.png" width="100%" alt="TimeRelay architecture" />
</p>


## Repository Structure

```text
TimeRelay/
├── data_provider/        # Dataset loading and preprocessing
├── exp/                  # Experiment pipeline
├── models/               # Model definitions
├── utils/                # Utility functions
├── dataset/              # Benchmark datasets
├── figure/               # Figures used in README
├── TimeRelay.sh          # Script for reproducing experiments
├── requirements.txt      # Python dependencies
└── README.md
```

## Installation

1. Create a Python environment with **Python 3.8**.

2. Install dependencies:

```bash
pip install -r requirements.txt
```



## Data Preparation

All benchmark datasets can be downloaded from [Google Drive](https://drive.google.com/file/d/1MKugRwUKN2u9tIBgES-n3-QOT5l6unLl/view?usp=drive_link).

After downloading, place the datasets under the `./dataset` directory. For example:

```text
./dataset/electricity/electricity.csv
```

Please ensure that the dataset directory structure matches the expected paths used in the code.




## Quick Start

To reproduce the main experimental results reported in the paper, run:

```bash
sh TimeRelay.sh
```

This script reproduces the benchmark experiments for TimeRelay across multiple datasets and forecasting settings.


## Experimental Results

### Overall performance on different datasets
<p align="center">
  <img src="./figure/result.png" width="100%" alt="Overall performance on benchmark datasets" />
</p>


### Comparison with different across variable modeling methods
<p align="center">
  <img src="./figure/across_variable.png" width="100%" alt="Comparison with different across variable modeling methods" />
</p>



### Scalability
<p align="center">
  <img src="./figure/scalability.png" width="100%" alt="Scalability of TimeRelay." />
</p>


### Efficiency
<p align="center">
  <img src="./figure/efficiency1.png" width="100%" alt="Computational efficiency of different cross-variable modeling mechanisms." />
</p>

<p align="center">
  <img src="./figure/efficiency1.png" width="100%" alt="Efficiency of different evaluation strategies for the same relation-aware affine recurrence." />
</p>



