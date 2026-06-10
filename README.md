# Adaptive Frequency Neurons

This repository contains the official implementation to reproduce the results presented in the paper: **"Adaptive-Frequency Resonate-and-Fire Neurons for Spectral Estimation of Streaming Radar Signals"**.

Our work introduces a dynamical system of adaptive-frequency neurons—specifically utilizing Adaptive Resonate-and-Fire (ARF) mechanics—designed for efficient, streaming FMCW radar signal estimation.

---

## Repository Structure

```text
├── notebooks/             # Jupyter notebooks for result replication
│   ├── recorded_data/     # Experiments utilizing real-world recorded radar data
│   └── spiking_tests/     # Experiments analyzing spike mechanisms and dynamics
├── src/                   # Core source code
│   ├── py_network/        # Implementation of the Adaptive Resonate-and-Fire network
│   ├── fmcw_simulation/   # Synthetic FMCW radar data generation pipelines
│   └── batch_experiments/ # Framework for running parallel Monte Carlo simulations
├── scripts/               # Automation scripts for batch and Monte Carlo experiments
└── setup.sh               # Installation and compilation script
```

## Instructions

### 1. Prerequisites

The core simulation backend is implemented in C++ for optimal performance and requires the Eigen linear algebra library.

On Ubuntu/Debian-based systems, install it via:
`sudo apt update && sudo apt install libeigen3-dev`

### 2. Run setup script

We provide an automated setup script that handles the Python virtual environment creation, dependency installation, and C++ binding compilation.

Execute the following command from the root directory: `chmod +x setup.sh && ./setup.sh`

### 3. Run Montecarlo experiments

The batch execution script scans the configuration directory for experiment files and processes them sequentially.

To run all simulations, execute:
Bash

`./scripts/run_experiments.sh`

Note: The script processes every configuration file found in scripts/batch_experiments/config_files/. Depending on the number of experiments and trials, this process may take a significant amount of time.

### 4. Visualize Results

Once the simulations are complete, you can analyze and visualize the performance metrics using the provided Jupyter notebook: `notebooks/performance_measurements.ipynb`.

### 5. Download recordings

Download the data from the followinf [link](https://doi.org/10.5281/zenodo.20491183) and uncompress it in the `data` folder.

Run the notebook in `notebook/recorded_data` to visualize the results.

<!-- ## Citation

If you use this code or find our research helpful in your work, please cite our paper:

```
@article{todo2026adaptive,
  title={TODO: Insert Paper Title},
  author={Chiavazza, Stefano and Yuan, Sen ... [Expand fully as needed]},
  journal={TODO: Insert Journal/Conference},
  year={2026}
}
``` -->
