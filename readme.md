# AutoBeocat Optuna Branch

This repository provides an automated machine learning (AutoML) solution that integrates with the Beocat HPC environment. It uses [Optuna](https://optuna.org/) for hyperparameter optimization, allowing you to quickly run multiple trials in parallel on Beocat via SLURM job arrays. The application front-end is built using [Streamlit](https://streamlit.io/), making it easy to specify hyperparameters, resource allocations, and model choices without manually editing code or scripts.

## Key Features

- **Optuna Integration**: Leverages Optuna to sample hyperparameters efficiently rather than relying on exhaustive grid search.
- **HPC-Ready**: Submits jobs to Beocat using SLURM job arrays, with each trial as a separate array task.
- **Model Support**: Supports both:
  - **Standard Neural Network (NN)** for regression/classification tasks.
  - **Conditional Variational Autoencoder (CVAE)** for generative tasks.
- **Resource Specifications**: Set CPU, memory, time limits, and GPU options directly from the Streamlit UI.
- **Result Monitoring**: Periodically checks progress and displays a leaderboard plus visualization of hyperparameter vs. performance relationships when all trials complete.

## Repository Structure

- `automl.py`: The main Streamlit-based application.
  - Upload your dataset (CSV).
  - Choose your model type (Standard NN or CVAE).
  - Select hyperparameters and number of Optuna trials.
  - Specify HPC resources and launch training jobs on Beocat.
  - Monitors job completion and displays final results.

- `training.py`: The training script executed by each SLURM array task.
  - Handles loading data, using Optuna to sample hyperparameters for a single trial.
  - Trains the model and writes metrics to `results.csv`.

- `utils/`:
  - `data_utils.py`: Functions to prepare datasets for NN and CVAE training.
  - `slurm_utils.py`: Utilities to generate the SLURM script and parse job IDs.

- `models/`:
  - `cvae_model.py`: Defines the CVAE model architecture and loss functions.

- `requirements.txt`: Suggested Python dependencies (if provided).

## Setup and Installation

1. **Environment Setup on Beocat**:
   ```bash
   module load Python/3.10.4-GCCcore-11.3.0
   # Optional: create and activate a virtual environment
   python -m venv ~/virtualenvs/automl_env
   source ~/virtualenvs/automl_env/bin/activate
   pip install -r requirements.txt



streamlit run automl.py
when open http://localhost:8501 in your browser.

On Beocat, you might need to use port forwarding or run Streamlit on a local machine.

Running on Beocat
Configure in Streamlit:

Upload your dataset.
Choose model type (NN or CVAE).
Set hyperparameter ranges (Optuna will sample from these).
Specify the number of trials (N) and max concurrent jobs.
Adjust HPC resources (time, memory, CPUs, GPUs) all in the UI.
Start Training:

Click "Start Training" in the UI.
The app creates a main job directory (e.g., job_1), saves data, parameters, and generates a SLURM script.
Submits a job array to Beocat with N tasks, each corresponding to one Optuna trial.
The UI displays progress, waiting until all trials are done.
Monitor Progress:

The UI updates periodically.
Once all trials finish, it loads results.csv and shows a leaderboard.
Interpreting Results:

The leaderboard includes metrics like MSE or Accuracy, depending on the task.
It uses Plotly to visualize hyperparameter vs. metric relationships.
Identifies best hyperparameter sets found by Optuna.
Key Considerations
Optuna Trials: Each trial corresponds to one set of hyperparameters sampled by Optuna. The best trial typically has the best metric score.
Scaling Up: Increase the number of trials for broader hyperparameter coverage. Adjust the concurrency to match Beocat's resources.
GPU Usage: If selected, the SLURM script requests GPUs for trials. Ensure your dataset and model can benefit from GPU acceleration.
Dependencies
Core: Python 3.10, Streamlit, Optuna, PyTorch, scikit-learn, pandas.
Plotting: Plotly for interactive charts.
HPC Integration: SLURM workload manager, pre-loaded Python modules on Beocat.
