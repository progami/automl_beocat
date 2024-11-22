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
import plotly.express as px

def main():
    st.set_page_config(layout="wide")
    st.title("AutoML on Beocat")
    total_resources_placeholder = st.empty()  # Placeholder for total resources
    print("Starting AutoML on Beocat application...")  # Console output

    # Custom CSS for the sticky info box in the upper right corner
    st.markdown(
        """
        <style>
        .total-resources {
            position: fixed;
            top: 70px;
            right: 20px;
            width: 300px;
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            z-index: 100;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Function to display total resources in the custom div
    def display_total_resources(info_text):
        total_resources_placeholder.markdown(
            f"""
            <div class="total-resources">
            {info_text}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Dataset and Task Selection
    st.header("Dataset and Task Selection")

    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    if uploaded_file is not None:
        # User uploaded a file
        data = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully.")
        print("Dataset uploaded successfully.")  # Console output

        # Model Selection
        st.header("Model Selection")
        model_options = ['Standard Neural Network', 'Conditional Variational Autoencoder (CVAE)']
        selected_model = st.selectbox("Select the model to train:", model_options)

        columns = data.columns.tolist()
        if not columns:
            st.error("The uploaded dataset does not contain any columns.")
            print("Error: The uploaded dataset does not contain any columns.")  # Console output
            return

        if selected_model == 'Standard Neural Network':
            col1, col2 = st.columns(2)
            with col1:
                target_column = st.selectbox("Select the target (label) column:", columns)
            with col2:
                task_type = st.radio("Task Type:", ('Regression', 'Classification'))
        elif selected_model == 'Conditional Variational Autoencoder (CVAE)':
            st.write("Select input columns (to be reconstructed):")
            input_columns = st.multiselect("Input Columns:", columns, default=columns[:9])

            st.write("Select condition columns (conditioning variables):")
            condition_columns = st.multiselect("Condition Columns:", columns, default=columns[9:])
    else:
        st.warning("Please upload a dataset to proceed.")
        print("Awaiting dataset upload...")  # Console output
        return

    # Hyperparameter Settings
    if selected_model == 'Standard Neural Network':
        st.header("Hyperparameter Settings")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Batch Size**")
            batch_size_options = [16, 32, 64, 128, 256]
            batch_sizes = st.multiselect("Select Batch Sizes:", batch_size_options, default=[32, 64])

            st.markdown("**Learning Rate**")
            learning_rate_options = [0.0001, 0.001, 0.01, 0.1]
            learning_rates = st.multiselect("Select Learning Rates:", learning_rate_options, default=[0.001, 0.01])

        with col2:
            st.markdown("**Epochs**")
            epochs_options = [50, 100, 150, 200]
            epochs_list = st.multiselect("Select Number of Epochs:", epochs_options, default=[100])

            st.markdown("**Hidden Layer Size**")
            hidden_size_options = [32, 64, 128, 256]
            hidden_sizes = st.multiselect("Select Hidden Layer Sizes:", hidden_size_options, default=[64, 128])

        with col3:
            st.markdown("**Optimizers**")
            optimizer_options = ['Adam', 'SGD', 'RMSprop']
            optimizers = st.multiselect("Select Optimizers:", optimizer_options, default=['Adam'])

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

    elif selected_model == 'Conditional Variational Autoencoder (CVAE)':
        st.header("Hyperparameter Settings for CVAE")

        col1, col2, col3 = st.columns(3)

        with col1:
            latent_dims = st.multiselect("Select Latent Dimensions:", [512, 1024, 2048], default=[512, 1024])

            epochs_list = st.multiselect("Select Number of Epochs:", [100, 200, 300], default=[100])

            batch_sizes = st.multiselect("Select Batch Sizes:", [256, 512, 1024], default=[256, 512])

            learning_rates = st.multiselect("Select Learning Rates:", [0.01, 0.001, 0.0001, 0.00001], default=[0.001, 0.0001])

        with col2:
            activation_functions = st.multiselect("Select Activation Functions:", ['Sigmoid', 'ReLU', 'Tanh', 'ELU'], default=['ReLU'])

            position_norm_methods = st.multiselect("Select Position Normalization Methods:", ['StandardScaler', 'None'], default=['StandardScaler'])

            momenta_norm_methods = st.multiselect("Select Momenta Normalization Methods:", ['StandardScaler', 'None'], default=['StandardScaler'])

        with col3:
            use_l1 = st.checkbox("Use L1 Regularization")
            if use_l1:
                l1_lambdas = st.multiselect("Select L1 Lambda Values:", [0.001, 0.01, 0.1, 0.5], default=[0.001])
            else:
                l1_lambdas = [0.0]

            use_l2 = st.checkbox("Use L2 Regularization")
            if use_l2:
                l2_lambdas = st.multiselect("Select L2 Lambda Values:", [0.001, 0.01, 0.1, 0.5], default=[0.001])
            else:
                l2_lambdas = [0.0]

            num_hidden_layers = st.multiselect("Select Number of Hidden Layers:", [1, 2, 3, 4], default=[2])

            hidden_layer_sizes = st.multiselect("Select Hidden Layer Sizes:", [64, 128, 256], default=[128])

        # Fixed hyperparameters
        patience = 100
        min_delta = 1e-5

        # Hyperparameters dictionary for CVAE
        hyperparams = {
            'LATENT_DIM': latent_dims,
            'EPOCHS': epochs_list,
            'BATCH_SIZE': batch_sizes,
            'LEARNING_RATE': learning_rates,
            'PATIENCE': [patience],
            'MIN_DELTA': [min_delta],
            'activation_name': activation_functions,
            'position_norm_method': position_norm_methods,
            'momenta_norm_method': momenta_norm_methods,
            'use_l1': [use_l1],
            'L1_LAMBDA': l1_lambdas,
            'use_l2': [use_l2],
            'L2_LAMBDA': l2_lambdas,
            'num_hidden_layers': num_hidden_layers,
            'hidden_layer_size': hidden_layer_sizes
        }

        # Generate hyperparameter combinations for CVAE
        combinations = generate_hyperparameter_combinations_cvae(hyperparams)
        num_combinations = len(combinations)
        st.write(f"Total hyperparameter combinations: {num_combinations}")

    # Display combinations if not too many
    if num_combinations <= 100:
        combination_df = pd.DataFrame(combinations)
        st.dataframe(combination_df)
    else:
        st.warning("Too many combinations to display.")

    # Resource Specifications
    st.header("Resource Specifications")

    col1, col2, col3 = st.columns(3)

    with col1:
        time_limit_hours = st.number_input(
            "Time Limit (Hours):",
            min_value=1,
            max_value=168,  # Maximum 7 days
            value=1
        )

    with col2:
        memory_per_cpu = st.number_input(
            "Memory per CPU (GB):",
            min_value=1,
            max_value=512,
            value=4
        )

    with col3:
        cpus_per_task = st.number_input(
            "CPUs per Task:",
            min_value=1,
            max_value=32,
            value=1
        )

    col4, col5 = st.columns(2)

    with col4:
        max_concurrent_jobs = st.number_input(
            "Max Concurrent Jobs:",
            min_value=1,
            max_value=100,
            value=10
        )

    with col5:
        use_gpu = st.checkbox("Use GPU for Training")
        if use_gpu:
            gpus = st.number_input("Number of GPUs:", min_value=1, max_value=8, value=1)
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
            gpu_type = st.selectbox("GPU Type:", gpu_types)
        else:
            gpus = 0
            gpu_type = None

    # Calculate total resources
    # First, ensure all necessary inputs are available
    if time_limit_hours and memory_per_cpu and cpus_per_task and num_combinations:
        # Calculate total CPUs
        total_cpus = num_combinations * cpus_per_task

        # Total time in hours
        total_time_hours = time_limit_hours

        # Calculate total CPU hours
        total_cpu_hours = total_cpus * total_time_hours

        # Calculate total memory
        total_memory_gb = num_combinations * cpus_per_task * memory_per_cpu

        # Calculate total GPUs
        if use_gpu:
            total_gpus = num_combinations * gpus
        else:
            total_gpus = 0

        # Update the placeholder at the top
        info_text = f"""
        **Total Resources Requested:**

        - Total Jobs: {num_combinations}
        - Total CPUs: {total_cpus}
        - Total CPU Hours: {total_cpu_hours:.2f} hours
        - Total Memory: {total_memory_gb} GB
        - Total GPUs: {total_gpus}
        """
        display_total_resources(info_text)
    else:
        display_total_resources("Please complete all inputs to calculate total resources.")

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
        if selected_model == 'Standard Neural Network':
            # Existing code for data preprocessing
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
                params = {'task_type': task_type, 'selected_model': selected_model}
                pickle.dump(params, f)
            with open(os.path.join(main_job_dir, 'combinations.pkl'), 'wb') as f:
                pickle.dump(combinations, f)
            print("Data, task_type, and hyperparameter combinations saved.")  # Console output

        elif selected_model == 'Conditional Variational Autoencoder (CVAE)':
            # For CVAE, we need to split the data into input and condition
            try:
                input_data = data[input_columns]
                condition_data = data[condition_columns]
            except KeyError as e:
                st.error(f"Column not found in the dataset: {e}")
                print(f"Error: Column not found in the dataset: {e}")  # Console output
                return

            # Convert to numpy arrays
            input_data = input_data.values
            condition_data = condition_data.values

            # Split data into train, validation, test sets
            from sklearn.model_selection import train_test_split

            train_input, temp_input, train_condition, temp_condition = train_test_split(
                input_data, condition_data, test_size=0.3, random_state=85, shuffle=True
            )
            val_input, test_input, val_condition, test_condition = train_test_split(
                temp_input, temp_condition, test_size=0.5, random_state=42, shuffle=True
            )

            # Save data and selected model in the main job directory
            with open(os.path.join(main_job_dir, 'data.pkl'), 'wb') as f:
                pickle.dump((train_input, val_input, test_input, train_condition, val_condition, test_condition), f)
            with open(os.path.join(main_job_dir, 'params.pkl'), 'wb') as f:
                params = {'selected_model': selected_model}
                pickle.dump(params, f)
            with open(os.path.join(main_job_dir, 'combinations.pkl'), 'wb') as f:
                pickle.dump(combinations, f)
            print("Data, selected model, and hyperparameter combinations saved.")

        # Generate SLURM script for the job array
        job_dir = os.path.join(main_job_dir, 'jobs')
        os.makedirs(job_dir, exist_ok=True)
        slurm_script_content = generate_slurm_script(
            job_name=f'automl_array_job_{timestamp}',
            script_path=os.path.abspath('training.py'),
            output_path=os.path.join(job_dir, 'slurm_output_%A_%a.txt'),
            error_path=os.path.join(job_dir, 'slurm_error_%A_%a.txt'),
            time_hours=time_limit_hours,
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
                    # Load and store results
                    results = collect_results(main_job_dir, selected_model)
                    if results:
                        # Store results in session state
                        st.session_state['results'] = results
                        st.session_state['selected_model'] = selected_model
                        # Do not call display_leaderboard here
                        # It will be called at the end of main()
                    else:
                        st.error("No results found.")
                        print("Error: No results found.")
                else:
                    st.error("Jobs did not complete successfully.")
                    print("Error: Jobs did not complete successfully.")
            else:
                st.error("Could not parse job ID from submission output.")
                print("Error: Could not parse job ID from submission output.")

    # Check if results are available in session state for visualization
    if 'results' in st.session_state and 'selected_model' in st.session_state:
        display_leaderboard(st.session_state['results'], st.session_state['selected_model'])

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

def generate_hyperparameter_combinations_cvae(hyperparams):
    keys, values = zip(*hyperparams.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    return combinations

def generate_slurm_script(job_name, script_path, output_path, error_path, time_hours=1, mem_per_cpu='4G', cpus_per_task=1, job_dir='', main_job_dir='', gpus=0, gpu_type=None, array=None, num_combinations=1, partition='batch.q'):
    # Convert time_hours to SLURM time format (HH:MM:SS)
    time_limit = f"{int(time_hours):02d}:00:00"

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
#SBATCH --time={time_limit}
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
            # Check if job completed or failed
            sacct_result = subprocess.run(['sacct', '-j', str(job_id), '--format=State'], capture_output=True, text=True)
            if 'COMPLETED' in sacct_result.stdout:
                print(f"Job {job_id} completed successfully.")
                job_running = False
                return True
            else:
                print(f"Job {job_id} did not complete successfully.")
                return False
    return True

def collect_results(main_job_dir, selected_model):
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

def display_leaderboard(results, selected_model):
    st.header("Leaderboard")
    print("Generating leaderboard...")
    # Prepare DataFrame
    leaderboard_data = []
    for result in results:
        entry = {
            'Task ID': result['Task ID'],
        }
        entry.update(result['Hyperparameters'])
        if selected_model == 'Standard Neural Network':
            if 'MSE' in result:
                entry['MSE'] = result['MSE']
                entry['RMSE'] = result['RMSE']
                entry['R2'] = result['R2']
            else:
                entry['Accuracy'] = result['Accuracy']
        elif selected_model == 'Conditional Variational Autoencoder (CVAE)':
            metrics = result.get('Metrics', {})
            entry.update(metrics)
        leaderboard_data.append(entry)

    leaderboard_df = pd.DataFrame(leaderboard_data)
    st.dataframe(leaderboard_df)
    print("Leaderboard displayed.")

    # Visualization Section
    st.header("Performance Visualization")
    st.write("Select hyperparameters and metrics to visualize their relationships.")

    if selected_model == 'Standard Neural Network':
        # List of hyperparameters and metrics
        hyperparameters = ['learning_rate', 'batch_size', 'epochs', 'hidden_size', 'optimizer', 'loss_function']
        if 'MSE' in leaderboard_df.columns:
            metrics = ['MSE', 'RMSE', 'R2']
        else:
            metrics = ['Accuracy']
    elif selected_model == 'Conditional Variational Autoencoder (CVAE)':
        hyperparameters = list(result['Hyperparameters'].keys())
        metrics = [key for key in leaderboard_df.columns if key.endswith('_mre') or key.endswith('_mse')]

    x_axis = st.selectbox("Select X-axis (Hyperparameter):", hyperparameters, key='x_axis_selectbox')
    y_axis = st.selectbox("Select Y-axis (Metric):", metrics, key='y_axis_selectbox')
    hue_option = st.selectbox("Select Hue (Optional):", ['None'] + hyperparameters, key='hue_option_selectbox')

    # Convert categorical variables to strings
    for col in hyperparameters:
        if leaderboard_df[col].dtype == 'object':
            leaderboard_df[col] = leaderboard_df[col].astype(str)

    # Prepare data for plotting
    plot_df = leaderboard_df.copy()
    plot_df[x_axis] = plot_df[x_axis].astype(str) if plot_df[x_axis].dtype == 'object' else plot_df[x_axis]
    plot_df[y_axis] = pd.to_numeric(plot_df[y_axis], errors='coerce')
    if hue_option != 'None':
        plot_df[hue_option] = plot_df[hue_option].astype(str)

    # Create interactive plot using Plotly
    if hue_option != 'None':
        fig = px.scatter(
            plot_df,
            x=x_axis,
            y=y_axis,
            color=hue_option,
            hover_data=hyperparameters + metrics
        )
    else:
        fig = px.scatter(
            plot_df,
            x=x_axis,
            y=y_axis,
            hover_data=hyperparameters + metrics
        )

    fig.update_layout(title=f"{y_axis} vs {x_axis}", xaxis_title=x_axis, yaxis_title=y_axis)
    st.plotly_chart(fig)
    print("Visualization displayed.")

if __name__ == '__main__':
    main()

