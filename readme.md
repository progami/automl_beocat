# AutoBeocat: Automated ML on HPC

## Overview

**AutoBeocat** is a Streamlit-based application that simplifies the process of running large-scale hyperparameter experiments on high-performance computing (HPC) systems managed by Slurm (e.g., Beocat). It allows you to:

1. Upload a custom `training.py` script.
2. Upload and select a dataset for training.
3. Configure hyperparameters to form a grid of all possible combinations.
4. Specify HPC resource requirements.
5. Dispatch all experiments as a Slurm job array.
6. Automatically aggregate results into `results.csv` and display a leaderboard upon completion.

This tool is especially useful for searching hyperparameters for conditional variational autoencoder (CVAE) models, but can be adapted to other training scripts as long as `training.py` reads `params.json` and `data.pkl`.

## Features

- **User-Friendly UI**: A Streamlit interface guides you through uploading `training.py`, dataset selection, hyperparameter choices, and HPC configuration.
- **Data Handling**: Upload a large CSV dataset. The tool preprocesses the dataset (via `prepare_cvae_data` in `utils/data_utils.py`) into train/val/test splits and stores them in `data.pkl`.
- **Grid Search Hyperparameters**: Select multiple values for each hyperparameter. The tool forms a Cartesian product, producing all combinations for a full grid search.
- **HPC Integration**: Generate a Slurm job array that runs each hyperparameter combination as a separate job, respecting concurrency and GPU constraints.
- **Local Testing**: Optional "local test" mode runs just one combination with minimal settings (1 epoch, 1 batch) to quickly validate the setup before launching HPC jobs.
- **Results Aggregation**: Each completed job writes a line to `results.csv`. Once all jobs finish, the UI displays a leaderboard of all tested combinations and their metrics.

## Prerequisites

- **Python Environment**:
  - Python 3.7+ recommended.
  - Install dependencies: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `torch`, `tqdm`.
    ```bash
    pip install streamlit pandas numpy scikit-learn torch tqdm
    ```
- **HPC System with Slurm**:
  - Ensure `sbatch`, `squeue`, and `scancel` are available.
  - A `utils` directory containing:
    - `data_utils.py`: Must define `prepare_cvae_data(data, input_columns, condition_columns)`.
    - `slurm_utils.py`: Must define `generate_slurm_script` and `parse_job_id`.

## Getting Started

1. **Launch the UI**:
   ```bash
   streamlit run automl.py
   ```
   Open the displayed URL in your browser.

2. **Upload Training Script**:
   - In the "Upload Training Script" tab, upload `training.py`.
   - Your `training.py` should:
     - Read hyperparameters from `params.json`.
     - Load data from `data.pkl`.
     - After training, append results to `results.csv`.

3. **Dataset Selection**:
   - In the "Dataset Selection" tab, upload your CSV dataset.
   - Wait until "Dataset uploaded successfully." appears.
   - Choose which columns are input features and which are condition features.

4. **Hyperparameter Configuration**:
   - In the "Hyperparameters" tab, select multiple values for each parameter (latent dims, epochs, batch sizes, LR, activation, etc.).
   - Optionally adjust advanced hyperparameters.
   - The tool will create a grid of all parameter combinations.

5. **Resource Specifications**:
   - In the "Resource Specifications" tab, set HPC resource limits: time, memory per CPU, CPUs per task.
   - Optionally enable GPU usage, specify GPU count and type, and set maximum concurrent jobs.
   - (Optional) Enable local test mode to run a quick test (no Slurm).

6. **Run the Experiment**:
   - In the "Run" tab, click "Start Training".
   - If local test mode is on, it runs locally and shows immediate output.
   - If not, it submits a Slurm job array and gives you the job ID.
   - The UI will periodically check `results.csv` for completed jobs and update progress.

7. **Monitoring & Canceling Jobs**:
   - Monitor jobs via `squeue -j <job_id>`.
   - Use the "Cancel Job" button in the UI or `scancel <job_id>` to terminate early.

8. **Results & Leaderboard**:
   - Once all jobs are complete, a leaderboard table is displayed in the UI.
   - `results.csv` contains hyperparameters and metrics for each job.
   - Additional files like `metrics.txt` and `sample_predictions.txt` are also available for analysis.

## Files and Outputs

- **`params.json`**: Created by the UI, read by `training.py`.
- **`data.pkl`**: Contains preprocessed datasets, created after dataset selection.
- **`results.csv`**: Appended to by each job upon completion. Contains metrics and hyperparameters.
- **`metrics.txt`** & **`sample_predictions.txt`**: Additional output files from `training.py` for analysis.

## Tips

- Test locally first (Local Test Run) to ensure `training.py` and data preprocessing work correctly.
- Adjust HPC resources as needed based on job run times.
- Keep track of `job_x` directories for results from previous runs.

## Cleaning Up

- Remove `job_x` directories as they accumulate after multiple runs.
- Cancel running jobs if needed to stop HPC usage.
