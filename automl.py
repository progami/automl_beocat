# automl.py

import os
import pandas as pd
import pickle
import subprocess
import time
import json
import streamlit as st
import torch
from sklearn.model_selection import train_test_split
from utils.data_utils import prepare_standard_nn_data, prepare_cvae_data
from utils.slurm_utils import generate_slurm_script, parse_job_id
from datetime import datetime

def print_with_time(message: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")

def format_minutes(m):
    return f"{m:.2f} min."

if "job_running" not in st.session_state:
    st.session_state["job_running"] = False

def main():
    st.set_page_config(layout="wide")
    st.title("AutoBeocat: Automated ML on Beocat")

    print_with_time("Starting AutoBeocat application...")

    st.header("Dataset and Task Selection")

    disabled = st.session_state["job_running"]

    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'], disabled=disabled)
    if uploaded_file is None:
        st.warning("Please upload a dataset to proceed.")
        return

    @st.cache_data
    def load_data(uploaded_file):
        data = pd.read_csv(uploaded_file)
        print_with_time("Dataset loaded successfully.")
        return data

    data = load_data(uploaded_file)
    st.success("Dataset uploaded successfully.")

    st.header("Model Selection")
    model_options = ['Standard Neural Network', 'Conditional Variational Autoencoder (CVAE)']
    selected_model = st.selectbox("Select the model to train:", model_options, disabled=disabled)

    columns = data.columns.tolist()
    if not columns:
        st.error("No columns in dataset.")
        print_with_time("No columns in dataset.")
        return

    inputs_valid = True
    if selected_model == 'Standard Neural Network':
        st.subheader("Select target and features")
        possible_targets = ['Select a column...'] + columns
        target_column = st.selectbox("Target (label) column:", possible_targets, disabled=disabled)
        if target_column == 'Select a column...':
            st.error("Please select a valid target column.")
            inputs_valid = False

        feature_columns = st.multiselect("Feature columns:", columns, disabled=disabled)
        if target_column != 'Select a column...' and not feature_columns:
            st.error("Please select at least one feature column.")
            inputs_valid = False

        task_type = st.radio("Task Type:", ('Regression', 'Classification'), disabled=disabled)

        st.subheader("Standard NN Hyperparameters")
        nn_lr_options = [0.0001, 0.001, 0.01, 0.1]
        lr_set = st.multiselect("Learning rates:", nn_lr_options, default=[0.001], disabled=disabled)
        if not lr_set:
            st.error("At least one learning rate required.")
            inputs_valid = False

        batch_options = st.multiselect("Batch sizes:", [16,32,64,128,256], default=[16,32], disabled=disabled)
        if not batch_options:
            st.error("At least one batch size required.")
            inputs_valid = False

        epoch_options = st.multiselect("Epoch counts:", [50,100,150,200], default=[50,100], disabled=disabled)
        if not epoch_options:
            st.error("At least one epoch count required.")
            inputs_valid = False

        hidden_size_options = st.multiselect("Hidden sizes:", [32,64,128,256], default=[32,64], disabled=disabled)
        if not hidden_size_options:
            st.error("At least one hidden size required.")
            inputs_valid = False

        optimizer_options = st.multiselect("Optimizers:", ["Adam","SGD","RMSprop"], default=["Adam","SGD"], disabled=disabled)
        if not optimizer_options:
            st.error("At least one optimizer required.")
            inputs_valid = False

        if task_type == 'Regression':
            loss_options = ["MSELoss","L1Loss","SmoothL1Loss"]
        else:
            loss_options = ["CrossEntropyLoss","NLLLoss"]
        loss_selection = st.multiselect("Loss functions:", loss_options, default=[loss_options[0]], disabled=disabled)
        if not loss_selection:
            st.error("At least one loss function required.")
            inputs_valid = False

        nn_hparams = {
            'lr_set': lr_set,
            'batch_sizes': batch_options,
            'epochs': epoch_options,
            'hidden_sizes': hidden_size_options,
            'optimizers': optimizer_options,
            'loss_functions': loss_selection
        }

    elif selected_model == 'Conditional Variational Autoencoder (CVAE)':
        st.subheader("Select input and condition columns for CVAE")
        input_columns = st.multiselect("Input Columns:", columns, disabled=disabled)
        if not input_columns:
            st.error("Please select at least one input column.")
            inputs_valid = False

        condition_columns = st.multiselect("Condition Columns:", columns, disabled=disabled)
        if not condition_columns:
            st.error("Please select at least one condition column.")
            inputs_valid = False

        st.subheader("CVAE Hyperparameters")
        latent_dims_set = st.multiselect("Latent dims:", [8,16,32,64,128], default=[8,32], disabled=disabled)
        if not latent_dims_set:
            st.error("At least one latent dim required.")
            inputs_valid = False

        epoch_choices = st.multiselect("Epochs:", [50,100,200,300], default=[50,100], disabled=disabled)
        if not epoch_choices:
            st.error("At least one epoch count required for CVAE.")
            inputs_valid = False

        batch_sizes_cvae = st.multiselect("Batch sizes (CVAE):", [16,32,64,128], default=[16,32], disabled=disabled)
        if not batch_sizes_cvae:
            st.error("At least one batch size required for CVAE.")
            inputs_valid = False

        cvae_lr_options = [0.0001,0.001,0.005,0.01,0.02]
        cvae_lr_set = st.multiselect("CVAE learning rates:", cvae_lr_options, default=[0.001,0.005], disabled=disabled)
        if not cvae_lr_set:
            st.error("At least one CVAE learning rate required.")
            inputs_valid = False

        activations = st.multiselect("Activations (CVAE):", ["Sigmoid","ReLU","Tanh","ELU"], default=["ReLU","Tanh"], disabled=disabled)
        if not activations:
            st.error("At least one activation required.")
            inputs_valid = False

        nhl = st.multiselect("Num hidden layers (CVAE):", [1,2,3,4], default=[1,2], disabled=disabled)
        if not nhl:
            st.error("At least one num hidden layers required.")
            inputs_valid = False

        hsize = st.multiselect("Hidden layer sizes (CVAE):", [64,128,256], default=[64,128], disabled=disabled)
        if not hsize:
            st.error("At least one hidden layer size required.")
            inputs_valid = False

        l1_options = [0.0,0.0001,0.001,0.01]
        l1_vals = st.multiselect("L1 values (0.0 or none means off):", l1_options, default=[], disabled=disabled)

        l2_options = [0.0,0.0001,0.001,0.01]
        l2_vals = st.multiselect("L2 values (0.0 or none means off):", l2_options, default=[], disabled=disabled)

        cvae_hparams = {
            'latent_dims': latent_dims_set,
            'epochs': epoch_choices,
            'batch_sizes': batch_sizes_cvae,
            'lr_set': cvae_lr_set,
            'activations': activations,
            'num_hidden_layers': nhl,
            'hidden_layer_sizes': hsize,
            'l1_set': l1_vals,
            'l2_set': l2_vals
        }

    if not inputs_valid:
        return

    st.header("Resource Specifications")
    col1, col2, col3 = st.columns(3)
    with col1:
        time_limit_hours = st.number_input("Time Limit (Hours):", min_value=1, max_value=168, value=1, disabled=st.session_state["job_running"])
    with col2:
        memory_per_cpu = st.number_input("Memory per CPU (GB):", min_value=1, max_value=512, value=4, disabled=st.session_state["job_running"])
    with col3:
        cpus_per_task = st.number_input("CPUs per Task:", min_value=1, max_value=32, value=1, disabled=st.session_state["job_running"])

    col4, col5 = st.columns(2)
    with col4:
        total_trials = st.number_input("Number of Trials (Optuna):", min_value=1, max_value=1000, value=10, disabled=st.session_state["job_running"])
    with col5:
        use_gpu = st.checkbox("Use GPU for Training", value=False, disabled=st.session_state["job_running"])
        if use_gpu:
            gpus = st.number_input("Number of GPUs:", min_value=1, max_value=8, value=1, disabled=st.session_state["job_running"])
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
            gpu_type = st.selectbox("GPU Type:", gpu_types, disabled=st.session_state["job_running"])
        else:
            gpus = 0
            gpu_type = None

    max_concurrent_jobs = st.number_input("Max Concurrent Jobs:", min_value=1, max_value=100, value=10, disabled=st.session_state["job_running"])

    num_combinations = total_trials
    max_cpus = max_concurrent_jobs * cpus_per_task
    max_memory_gb = max_concurrent_jobs * cpus_per_task * memory_per_cpu
    if use_gpu:
        max_gpus = max_concurrent_jobs * gpus
    else:
        max_gpus = 0

    total_cpu_hours = num_combinations * cpus_per_task * time_limit_hours
    total_gpu_hours = num_combinations * gpus * time_limit_hours if use_gpu else 0.0

    with st.sidebar.expander("Resource Summary", expanded=True):
        st.write(f"**Total Trials (Optuna):** {num_combinations}")
        st.write(f"**Max Concurrent Jobs:** {max_concurrent_jobs}")
        st.write("**Resources per Job:**")
        st.write(f"- CPUs per Task: {cpus_per_task}")
        st.write(f"- Memory per CPU: {memory_per_cpu} GB")
        st.write(f"- GPUs per Task: {gpus if use_gpu else 0}")
        st.write(f"- Time per Job: {time_limit_hours} hours")

        st.write("**Maximum Concurrent Resources:**")
        st.write(f"- Total CPUs: {max_cpus}")
        st.write(f"- Total Memory: {max_memory_gb} GB")
        st.write(f"- Total GPUs: {max_gpus}")

        st.write("**Total Resources over Entire Job:**")
        st.write(f"- Total CPU Hours: {total_cpu_hours:.2f} hours")
        st.write(f"- Total GPU Hours: {total_gpu_hours:.2f} hours")

    start_button = st.button("Start Training", disabled=st.session_state["job_running"])
    if start_button:
        st.session_state["job_running"] = True  # lock the UI

        main_job_dir = create_main_job_dir()
        if selected_model == 'Standard Neural Network':
            if target_column == 'Select a column...':
                st.error("No valid target column selected.")
                return
            X_train, y_train, X_test, y_test = prepare_standard_nn_data(data, feature_columns, target_column, task_type)
            if X_train is None:
                st.error("Data preprocessing failed.")
                return
            with open(os.path.join(main_job_dir, 'data.pkl'), 'wb') as f:
                pickle.dump((X_train, y_train, X_test, y_test), f)
            params = {
                'task_type': task_type,
                'selected_model': selected_model,
                'input_columns': feature_columns,
                'target_column': target_column,
                'nn_hparams': nn_hparams
            }

        else:  # CVAE
            train_input, val_input, test_input, train_condition, val_condition, test_condition = prepare_cvae_data(data, input_columns, condition_columns)
            if train_input is None:
                st.error("Data preprocessing failed.")
                return
            with open(os.path.join(main_job_dir, 'data.pkl'), 'wb') as f:
                pickle.dump((train_input, val_input, test_input, train_condition, val_condition, test_condition), f)
            params = {
                'selected_model': selected_model,
                'input_columns': input_columns,
                'condition_columns': condition_columns,
                'cvae_hparams': cvae_hparams
            }

        with open(os.path.join(main_job_dir, 'params.json'), 'w') as f:
            json.dump(params, f)

        study_name = "autobeocat_study"
        study_info = {
            'study_name': study_name,
            'total_trials': int(num_combinations)
        }
        with open(os.path.join(main_job_dir, 'study_info.json'), 'w') as f:
            json.dump(study_info, f)

        job_dir = os.path.join(main_job_dir, 'jobs')
        os.makedirs(job_dir, exist_ok=True)
        slurm_script_content = generate_slurm_script(
            job_name=f'automl_job_array_{main_job_dir}',
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

        slurm_script_path = os.path.join(job_dir, 'job.slurm')
        with open(slurm_script_path, 'w') as f:
            f.write(slurm_script_content)
        print_with_time(f"SLURM script saved as '{slurm_script_path}'.")

        submit_command = ['sbatch', slurm_script_path]
        result = subprocess.run(submit_command, capture_output=True, text=True)
        st.text(result.stdout)
        print_with_time(result.stdout)
        if result.returncode != 0:
            st.error(f"Job submission failed: {result.stderr}")
            print_with_time(f"Job submission failed: {result.stderr}")
        else:
            job_id = parse_job_id(result.stdout)
            if job_id:
                st.success(f"Job array submitted successfully. Job ID: {job_id}")
                print_with_time(f"Job array submitted successfully. Job ID: {job_id}")

                with open(os.path.join(main_job_dir, 'study_info.json'), 'r') as f:
                    si = json.load(f)
                si['job_id'] = job_id
                with open(os.path.join(main_job_dir, 'study_info.json'), 'w') as f:
                    json.dump(si, f)

                st.info(f"Jobs submitted with Job ID: {job_id}.")
                st.write("You can monitor with:")
                st.code(f"squeue -j {job_id}")

                st.info("Waiting for jobs to complete...")

                prev_msg = ""
                # Wait indefinitely for results
                while True:
                    if os.path.exists(os.path.join(main_job_dir, 'results.csv')):
                        results = pd.read_csv(os.path.join(main_job_dir, 'results.csv'))
                        completed_jobs = len(results)
                        if completed_jobs == num_combinations:
                            st.success("All jobs have completed.")
                            display_leaderboard(results, selected_model=None)
                            break
                        else:
                            remaining = num_combinations - completed_jobs
                            msg = f"{completed_jobs}/{num_combinations} done."
                            if msg != prev_msg:
                                st.info(msg)
                                prev_msg = msg
                            print_with_time(msg)
                    else:
                        msg = "Results file not found yet, waiting..."
                        if msg != prev_msg:
                            st.info(msg)
                            prev_msg = msg
                        print_with_time(msg)
                    time.sleep(30)

def create_main_job_dir():
    base_dir = "job_"
    existing_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith(base_dir)]
    numbers = [int(d.replace(base_dir, '')) for d in existing_dirs if d.replace(base_dir, '').isdigit()]
    if numbers:
        next_number = max(numbers) + 1
    else:
        next_number = 1
    main_job_dir = f"{base_dir}{next_number}"
    os.makedirs(main_job_dir, exist_ok=False)
    return main_job_dir

def display_leaderboard(results, selected_model=None):
    st.header("Leaderboard")
    print_with_time("Generating leaderboard...")
    st.dataframe(results)
    print_with_time("Leaderboard displayed.")

if __name__ == '__main__':
    main()

