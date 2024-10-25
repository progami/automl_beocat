import os
import pandas as pd
import numpy as np
import pickle
import subprocess
import re
import time
import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset
import streamlit as st
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    st.title("AutoML on Beocat")
    print("Starting AutoML on Beocat application...")  # Console output

    # Allow user to upload a dataset
    st.header("Dataset Selection")
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    if uploaded_file is not None:
        # User uploaded a file
        data = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully.")
        print("Dataset uploaded successfully.")  # Console output
    else:
        st.warning("Please upload a dataset to proceed.")
        print("Awaiting dataset upload...")  # Console output
        return

    # Display columns and let user select target column
    st.header("Select Target Column")
    columns = data.columns.tolist()
    if not columns:
        st.error("The uploaded dataset does not contain any columns.")
        print("Error: The uploaded dataset does not contain any columns.")  # Console output
        return
    target_column = st.selectbox("Select the target (label) column:", columns)

    # Select task type
    st.header("Select Task Type")
    task_type = st.radio("Is this a regression or classification task?", ('Regression', 'Classification'))

    # Hyperparameter Settings
    st.header("Hyperparameter Settings")

    # Batch Size Options
    st.markdown("**Batch Size**")
    batch_size_options = [16, 32, 64, 128, 256]
    batch_sizes = st.multiselect("Select Batch Sizes:", batch_size_options, default=[32, 64])

    # Learning Rate Options
    st.markdown("**Learning Rate**")
    learning_rate_options = [0.0001, 0.001, 0.01, 0.1]
    learning_rates = st.multiselect("Select Learning Rates:", learning_rate_options, default=[0.001, 0.01])

    # Epochs Options
    st.markdown("**Epochs**")
    epochs_options = [50, 100, 150, 200]
    epochs_list = st.multiselect("Select Number of Epochs:", epochs_options, default=[100])

    # Hidden Layer Size Options
    st.markdown("**Hidden Layer Size**")
    hidden_size_options = [32, 64, 128, 256]
    hidden_sizes = st.multiselect("Select Hidden Layer Sizes:", hidden_size_options, default=[64, 128])

    # Optimizer Options
    st.markdown("**Optimizers**")
    optimizer_options = ['Adam', 'SGD', 'RMSprop']
    optimizers = st.multiselect("Select Optimizers:", optimizer_options, default=['Adam'])

    # Loss Function Options
    st.markdown("**Loss Functions**")
    if task_type == 'Regression':
        loss_function_options = ['MSELoss', 'L1Loss', 'SmoothL1Loss']
    else:
        loss_function_options = ['CrossEntropyLoss', 'NLLLoss']
    loss_functions = st.multiselect("Select Loss Functions:", loss_function_options, default=[loss_function_options[0]])

    # Hyperparameters dictionary
    hyperparams = {
        'learning_rates': learning_rates,
        'batch_sizes': batch_sizes,
        'epochs_list': epochs_list,
        'hidden_sizes': hidden_sizes,
        'optimizers': optimizers,
        'loss_functions': loss_functions
    }

    # Generate hyperparameter combinations
    combinations = generate_hyperparameter_combinations(hyperparams)
    num_combinations = len(combinations)
    st.write(f"Total hyperparameter combinations: {num_combinations}")

    # Display combinations
    if num_combinations <= 100:
        combination_df = pd.DataFrame(combinations, columns=[
            'Learning Rate',
            'Batch Size',
            'Epochs',
            'Hidden Size',
            'Optimizer',
            'Loss Function'
        ])
        st.dataframe(combination_df)
    else:
        st.warning("Too many combinations to display.")

    # Partition selection
    st.header("SLURM Partition Selection")
    partitions = ['batch.q', 'killable.q', 'interact.q', 'vis.q']
    selected_partition = st.selectbox("Select a partition for your job:", partitions, index=0)
    st.info(f"Selected partition: {selected_partition}")
    print(f"Selected partition: {selected_partition}")  # Console output

    # Resource specifications
    st.header("Resource Specifications")
    time_limit = st.text_input("Enter time limit (e.g., 01:00:00 for 1 hour):", value="01:00:00")
    memory_per_cpu = st.number_input("Enter memory per CPU in GB:", min_value=1, max_value=512, value=4)
    cpus_per_task = st.number_input("Enter number of CPUs per task:", min_value=1, max_value=32, value=1)

    # GPU Resources
    st.header("GPU Resources")
    use_gpu = st.checkbox("Use GPU for Training")
    if use_gpu:
        gpus = st.number_input("Enter number of GPUs to use:", min_value=1, max_value=8, value=1)
        # List of available GPU types
        gpu_types = [
            'geforce_gtx_1080_ti',
            'geforce_rtx_2080_ti',
            'geforce_rtx_3090',
            'l40s',
            'quadro_gp100',
            'rtx_a4000',
            'rtx_a4500',
            'rtx_a6000'
        ]
        gpu_type = st.selectbox("Select GPU Type:", gpu_types)
    else:
        gpus = 0
        gpu_type = None

    # Limit concurrent jobs
    max_concurrent_jobs = st.number_input("Max Concurrent Jobs (to avoid overloading the system):", min_value=1, max_value=100, value=10)

    # Start training button
    if st.button("Start Training"):
        if num_combinations > 500:
            st.error("Too many hyperparameter combinations selected. Please reduce the number to less than 500.")
            return
        st.info("Preparing data and submitting jobs to SLURM...")
        print("Preparing data and submitting jobs to SLURM...")  # Console output

        # Create a unique main job directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        main_job_dir = f"run_{timestamp}"
        os.makedirs(main_job_dir, exist_ok=True)
        print(f"Created main job directory: {main_job_dir}")  # Console output

        # Preprocess the data
        try:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        except KeyError:
            st.error(f"Target column '{target_column}' not found in the dataset.")
            print(f"Error: Target column '{target_column}' not found in the dataset.")  # Console output
            return

        # Encode categorical features
        X = pd.get_dummies(X)

        # Feature scaling
        X = X.values.astype(np.float32)
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8  # Avoid division by zero
        X = (X - X_mean) / X_std

        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)

        # Process target variable
        if task_type == 'Classification':
            # Encode target labels
            if y.dtype == 'object':
                y = pd.Categorical(y).codes
            else:
                y = y.astype(int)
            y = torch.tensor(y.values, dtype=torch.long)
        else:
            y = y.values.reshape(-1, 1).astype(np.float32)
            y = torch.tensor(y, dtype=torch.float32)

        # Split the data
        dataset = TensorDataset(X, y)
        if len(dataset) < 2:
            st.error("The dataset is too small to split into training and testing sets.")
            print("Error: The dataset is too small to split into training and testing sets.")  # Console output
            return
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        X_train, y_train = zip(*train_dataset)
        X_train = torch.stack(X_train)
        y_train = torch.stack(y_train)

        X_test, y_test = zip(*test_dataset)
        X_test = torch.stack(X_test)
        y_test = torch.stack(y_test)

        # Save data and task_type in the main job directory
        with open(os.path.join(main_job_dir, 'data.pkl'), 'wb') as f:
            pickle.dump((X_train, y_train, X_test, y_test), f)
        with open(os.path.join(main_job_dir, 'params.pkl'), 'wb') as f:
            pickle.dump({'task_type': task_type}, f)
        with open(os.path.join(main_job_dir, 'combinations.pkl'), 'wb') as f:
            pickle.dump(combinations, f)
        print("Data, task_type, and hyperparameter combinations saved.")  # Console output

        # Generate SLURM script for the job array
        job_dir = os.path.join(main_job_dir, 'jobs')
        os.makedirs(job_dir, exist_ok=True)
        slurm_script_content = generate_slurm_script(
            job_name=f'automl_array_job_{timestamp}',
            script_path=os.path.abspath('training.py'),
            output_path=os.path.join(job_dir, 'slurm_output_%A_%a.txt'),
            error_path=os.path.join(job_dir, 'slurm_error_%A_%a.txt'),
            partition=selected_partition,
            time=time_limit,
            mem_per_cpu=f"{memory_per_cpu}G",
            cpus_per_task=cpus_per_task,
            job_dir=os.path.abspath(job_dir),
            main_job_dir=os.path.abspath(main_job_dir),
            gpus=gpus,
            gpu_type=gpu_type,
            array=f"0-{num_combinations-1}%{max_concurrent_jobs}",
            num_combinations=num_combinations
        )

        # Save the SLURM script to the job directory
        slurm_script_path = os.path.join(job_dir, 'job.slurm')
        with open(slurm_script_path, 'w') as f:
            f.write(slurm_script_content)
        print(f"SLURM script saved as '{slurm_script_path}'.")

        # Submit the job array
        submit_command = ['sbatch', slurm_script_path]
        result = subprocess.run(submit_command, capture_output=True, text=True)
        st.text(result.stdout)
        print(result.stdout)
        if result.returncode != 0:
            st.error(f"Job submission failed: {result.stderr}")
            print(f"Job submission failed: {result.stderr}")
        else:
            # Extract job ID from output
            job_id = parse_job_id(result.stdout)
            if job_id:
                st.success(f"Job array submitted successfully. Job ID: {job_id}")
                print(f"Job array submitted successfully. Job ID: {job_id}")
                st.info(f"Monitoring job status (Job ID: {job_id})...")
                print(f"Monitoring job status (Job ID: {job_id})...")
                job_complete = monitor_job(job_id)
                if job_complete:
                    st.success("All jobs completed successfully.")
                    print("All jobs completed successfully.")
                    # Load and display results
                    results = collect_results(main_job_dir, task_type)
                    if results:
                        display_leaderboard(results, task_type)
                    else:
                        st.error("No results found.")
                        print("Error: No results found.")
                else:
                    st.error("Jobs did not complete successfully.")
                    print("Error: Jobs did not complete successfully.")
            else:
                st.error("Could not parse job ID from submission output.")
                print("Error: Could not parse job ID from submission output.")

def generate_hyperparameter_combinations(hyperparams):
    combinations = list(product(
        hyperparams['learning_rates'],
        hyperparams['batch_sizes'],
        hyperparams['epochs_list'],
        hyperparams['hidden_sizes'],
        hyperparams['optimizers'],
        hyperparams['loss_functions']
    ))
    return combinations

def generate_slurm_script(job_name, script_path, output_path, error_path, partition='batch.q', time='01:00:00', mem_per_cpu='4G', cpus_per_task=1, job_dir='', main_job_dir='', gpus=0, gpu_type=None, array=None, num_combinations=1):
    # GPU request line
    gpu_request = ''
    if gpus > 0:
        if gpu_type:
            gpu_request = f"#SBATCH --gres=gpu:{gpu_type}:{gpus}"
        else:
            gpu_request = f"#SBATCH --gres=gpu:{gpus}"

    # Job array line
    if array:
        array_line = f"#SBATCH --array={array}"
    else:
        array_line = ""

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_path}
#SBATCH --error={error_path}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
{gpu_request}
{array_line}

module load Python/3.10.4-GCCcore-11.3.0  # Adjust based on available modules
source ~/virtualenvs/automl_env/bin/activate

cd {job_dir}

python {script_path} --task_id $SLURM_ARRAY_TASK_ID --total_tasks {num_combinations} --main_job_dir {main_job_dir}
"""
    return slurm_script

def parse_job_id(submission_output):
    # Example output: "Submitted batch job 123456"
    match = re.search(r'Submitted batch job (\d+)', submission_output)
    if match:
        return match.group(1)
    else:
        return None

def monitor_job(job_id):
    import subprocess
    import time

    job_running = True
    while job_running:
        result = subprocess.run(['squeue', '-j', str(job_id)], capture_output=True, text=True)
        if str(job_id) in result.stdout:
            print(f"Job {job_id} is still running...")  # Console output
            time.sleep(30)  # Wait before checking again
        else:
            job_running = False
    return True

def collect_results(main_job_dir, task_type):
    results = []
    results_dir = os.path.join(main_job_dir, 'results')
    if os.path.exists(results_dir):
        for file_name in os.listdir(results_dir):
            if file_name.endswith('.pkl'):
                with open(os.path.join(results_dir, file_name), 'rb') as f:
                    result = pickle.load(f)
                    results.append(result)
    else:
        # Check for chunked results
        for chunk_dir_name in os.listdir(main_job_dir):
            if chunk_dir_name.startswith('chunk_'):
                chunk_dir = os.path.join(main_job_dir, chunk_dir_name)
                results_dir = os.path.join(chunk_dir, 'results')
                if os.path.exists(results_dir):
                    for file_name in os.listdir(results_dir):
                        if file_name.endswith('.pkl'):
                            with open(os.path.join(results_dir, file_name), 'rb') as f:
                                result = pickle.load(f)
                                results.append(result)
    return results if results else None

def display_leaderboard(results, task_type):
    st.header("Leaderboard")
    print("Generating leaderboard...")
    # Prepare DataFrame
    leaderboard_data = []
    for result in results:
        entry = {
            'Task ID': result['Task ID'],
            'Learning Rate': result['Hyperparameters']['learning_rate'],
            'Batch Size': result['Hyperparameters']['batch_size'],
            'Epochs': result['Hyperparameters']['epochs'],
            'Hidden Size': result['Hyperparameters']['hidden_size'],
            'Optimizer': result['Hyperparameters']['optimizer'],
            'Loss Function': result['Hyperparameters']['loss_function'],
        }
        if task_type == 'Regression':
            entry['MSE'] = result['MSE']
            entry['RMSE'] = result['RMSE']
            entry['R2'] = result['R2']
        else:
            entry['Accuracy'] = result['Accuracy']
        leaderboard_data.append(entry)

    leaderboard_df = pd.DataFrame(leaderboard_data)
    if task_type == 'Regression':
        leaderboard_df = leaderboard_df.sort_values(by='MSE')
    else:
        leaderboard_df = leaderboard_df.sort_values(by='Accuracy', ascending=False)
    st.dataframe(leaderboard_df)
    print("Leaderboard displayed.")

    # Visualization Section
    st.header("Performance Visualization")
    st.write("Select hyperparameters and metrics to visualize their relationships.")

    # List of hyperparameters and metrics
    hyperparameters = ['Learning Rate', 'Batch Size', 'Epochs', 'Hidden Size', 'Optimizer', 'Loss Function']
    if task_type == 'Regression':
        metrics = ['MSE', 'RMSE', 'R2']
    else:
        metrics = ['Accuracy']

    x_axis = st.selectbox("Select X-axis (Hyperparameter):", hyperparameters)
    y_axis = st.selectbox("Select Y-axis (Metric):", metrics)
    hue_option = st.selectbox("Select Hue (Optional):", [None] + hyperparameters)

    # Convert categorical variables to strings
    for col in ['Optimizer', 'Loss Function']:
        leaderboard_df[col] = leaderboard_df[col].astype(str)

    plt.figure(figsize=(10, 6))

    if leaderboard_df[x_axis].dtype == 'object' or leaderboard_df[x_axis].dtype == 'str':
        # Categorical x-axis
        sns.boxplot(x=leaderboard_df[x_axis], y=leaderboard_df[y_axis])
    else:
        # Numerical x-axis
        if hue_option and hue_option != 'None':
            if leaderboard_df[hue_option].dtype == 'object' or leaderboard_df[hue_option].dtype == 'str':
                sns.scatterplot(x=leaderboard_df[x_axis], y=leaderboard_df[y_axis], hue=leaderboard_df[hue_option])
            else:
                sns.scatterplot(x=leaderboard_df[x_axis], y=leaderboard_df[y_axis], hue=leaderboard_df[hue_option].astype(str))
        else:
            sns.scatterplot(x=leaderboard_df[x_axis], y=leaderboard_df[y_axis])
        sns.lineplot(x=leaderboard_df[x_axis], y=leaderboard_df[y_axis], estimator='mean', ci=None)
    plt.title(f"{y_axis} vs {x_axis}")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    st.pyplot(plt)
    print("Visualization displayed.")

if __name__ == '__main__':
    main()

