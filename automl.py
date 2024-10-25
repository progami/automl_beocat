# automl.py

import os
import pandas as pd
import numpy as np
import pickle
import subprocess
import re
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import streamlit as st


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

    # Since we're focusing on neural networks, we set algorithms to 'Neural Network'
    algorithms = ['Neural Network']
    st.info("Algorithm selected: Neural Network")
    print("Algorithm selected: Neural Network")  # Console output

    # Partition selection
    st.header("SLURM Partition Selection")
    partitions = ['batch.q', 'killable.q', 'interact.q', 'vis.q']
    selected_partition = st.selectbox("Select a partition for your job:", partitions, index=0)
    st.info(f"Selected partition: {selected_partition}")
    print(f"Selected partition: {selected_partition}")  # Console output

    # Resource specifications
    st.header("Resource Specifications")
    time_limit = st.text_input("Enter time limit (e.g., 01:00:00 for 1 hour):", value="01:00:00")
    memory_per_cpu = st.text_input("Enter memory per CPU (e.g., 4G for 4 GB):", value="4G")
    cpus_per_task = st.number_input("Enter number of CPUs per task:", min_value=1, max_value=32, value=1)

    # Start training button
    if st.button("Start Training"):
        st.info("Preparing data and submitting job(s) to SLURM...")
        print("Preparing data and submitting job(s) to SLURM...")  # Console output

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
        print("Data and task_type saved.")  # Console output

        # Since we're only using Neural Network, we can proceed directly
        algorithm = 'Neural Network'
        st.info(f"Submitting job for algorithm: {algorithm}")
        print(f"Submitting job for algorithm: {algorithm}")

        # Create a directory for this algorithm's job inside the main job directory
        algorithm_safe = algorithm.replace(' ', '_').lower()
        job_dir = os.path.join(main_job_dir, algorithm_safe)
        os.makedirs(job_dir, exist_ok=True)

        # Generate SLURM script for this algorithm
        slurm_script_content = generate_slurm_script(
            job_name=f'automl_job_{algorithm_safe}_{timestamp}',
            script_path=os.path.abspath('training.py'),  # Use absolute path
            output_path=os.path.join(job_dir, 'slurm_output.txt'),
            error_path=os.path.join(job_dir, 'slurm_error.txt'),
            partition=selected_partition,
            time=time_limit,
            mem_per_cpu=memory_per_cpu,
            cpus_per_task=cpus_per_task,
            job_dir=os.path.abspath(job_dir),
            algorithm=algorithm,
            main_job_dir=os.path.abspath(main_job_dir)
        )

        # Save the SLURM script to the job directory
        slurm_script_path = os.path.join(job_dir, 'job.slurm')
        with open(slurm_script_path, 'w') as f:
            f.write(slurm_script_content)
        print(f"SLURM script for {algorithm} saved as '{slurm_script_path}'.")

        # Submit the job
        submit_command = ['sbatch', slurm_script_path]
        result = subprocess.run(submit_command, capture_output=True, text=True)
        st.text(result.stdout)
        print(result.stdout)
        if result.returncode != 0:
            st.error(f"Job submission failed for {algorithm}: {result.stderr}")
            print(f"Job submission failed for {algorithm}: {result.stderr}")
        else:
            # Extract job ID from output
            job_id = parse_job_id(result.stdout)
            if job_id:
                st.success(f"Job submitted successfully for {algorithm}. Job ID: {job_id}")
                print(f"Job submitted successfully for {algorithm}. Job ID: {job_id}")
                job_info = (job_id, algorithm, job_dir)

                # Monitor the submitted job
                st.info(f"Monitoring job status for {algorithm} (Job ID: {job_id})...")
                print(f"Monitoring job status for {algorithm} (Job ID: {job_id})...")
                job_complete = monitor_job(job_id)
                if job_complete:
                    st.success(f"Job completed successfully for {algorithm}.")
                    print(f"Job completed successfully for {algorithm}.")
                    # Load and display results
                    results_path = os.path.join(job_dir, 'results.pkl')
                    if os.path.exists(results_path):
                        with open(results_path, 'rb') as f:
                            results = pickle.load(f)
                        display_results([results], task_type)
                    else:
                        st.error(f"Results file not found for {algorithm}.")
                        print(f"Error: Results file not found for {algorithm}.")
                else:
                    st.error(f"Job did not complete successfully for {algorithm}.")
                    print(f"Error: Job did not complete successfully for {algorithm}.")
            else:
                st.error(f"Could not parse job ID for {algorithm} from submission output.")
                print(f"Error: Could not parse job ID for {algorithm} from submission output.")


def generate_slurm_script(job_name, script_path, output_path, error_path, partition='batch.q', time='01:00:00', mem_per_cpu='4G', cpus_per_task=1, job_dir='', algorithm='', main_job_dir=''):
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

module load Python/3.10.4-GCCcore-11.3.0  # Adjust based on available modules
source ~/virtualenvs/automl_env/bin/activate

cd {job_dir}

python {script_path} --job_dir {job_dir} --algorithm "{algorithm}" --main_job_dir {main_job_dir}
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
    # Optionally, check slurm_output.txt or slurm_error.txt for details
    return True


def display_results(results, task_type):
    for result in results:
        st.write(f"### Algorithm: {result['Algorithm']}")
        print(f"Algorithm: {result['Algorithm']}")  # Console output
        if task_type == 'Regression':
            st.write(f"**Mean Squared Error (MSE)**: {result['MSE']:.4f}")
            st.write(f"**Root Mean Squared Error (RMSE)**: {result['RMSE']:.4f}")
            st.write(f"**R-squared (R²)**: {result['R2']:.4f}")
            print(f"Mean Squared Error (MSE): {result['MSE']:.4f}")  # Console output
            print(f"Root Mean Squared Error (RMSE): {result['RMSE']:.4f}")  # Console output
            print(f"R-squared (R²): {result['R2']:.4f}")  # Console output
        else:
            st.write(f"**Accuracy**: {result['Accuracy']:.4f}")
            st.write("**Classification Report:**")
            st.dataframe(pd.DataFrame(result['Classification Report']).transpose())
            st.write("**Confusion Matrix:**")
            st.write(result['Confusion Matrix'])
            print(f"Accuracy: {result['Accuracy']:.4f}")  # Console output
            print("Classification Report:")  # Console output
            print(pd.DataFrame(result['Classification Report']).transpose())  # Console output
            print("Confusion Matrix:")  # Console output
            print(result['Confusion Matrix'])  # Console output
        st.write("**Best Hyperparameters:**")
        st.write(result['Best Params'])
        print("Best Hyperparameters:")  # Console output
        print(result['Best Params'])  # Console output


if __name__ == '__main__':
    main()

