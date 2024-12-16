import streamlit as st
import os
import pandas as pd
import pickle
import subprocess
import time
import json
import torch
from sklearn.model_selection import train_test_split
from utils.data_utils import prepare_cvae_data
from utils.slurm_utils import generate_slurm_script, parse_job_id
from datetime import datetime
from itertools import product

# Ensure keys are always defined
if "job_running" not in st.session_state:
    st.session_state["job_running"] = False
if "current_job_id" not in st.session_state:
    st.session_state["current_job_id"] = None

def print_with_time(message: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")

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

def display_leaderboard(results):
    st.header("Leaderboard")
    print_with_time("Generating leaderboard...")
    st.dataframe(results)
    print_with_time("Leaderboard displayed.")

def main():
    st.set_page_config(page_title="AutoBeocat: Automated ML on HPC", layout="wide")
    st.title("AutoBeocat: Automated ML on Beocat")

    disabled = st.session_state.get("job_running", False)

    tab_train_script, tab_dataset, tab_hparams, tab_resources, tab_run = st.tabs([
        "Upload Training Script", "Dataset Selection", "Hyperparameters", "Resource Specifications", "Run"
    ])

    with tab_train_script:
        st.header("Upload Training Script")
        training_file = st.file_uploader("Upload training.py", type=["py"], disabled=disabled)
        if training_file is None:
            st.warning("Please upload your training.py file.")
            return

    with tab_dataset:
        st.header("Dataset Selection")
        uploaded_file = st.file_uploader("Upload a CSV dataset", type=['csv'], disabled=disabled)
        if uploaded_file is None:
            st.warning("Please upload a dataset.")
            return

        def load_data(uploaded_file):
            data = pd.read_csv(uploaded_file)
            print_with_time("Dataset loaded successfully.")
            return data

        data = load_data(uploaded_file)
        st.success("Dataset uploaded successfully.")

        st.write("Select columns for input and condition features:")
        columns = data.columns.tolist()
        if not columns:
            st.error("No columns in dataset.")
            return

        input_columns = st.multiselect("Input Columns:", columns, disabled=disabled)
        if not input_columns:
            st.error("At least one input column required.")
            return

        condition_columns = st.multiselect("Condition Columns:", columns, disabled=disabled)
        if not condition_columns:
            st.error("At least one condition column required.")
            return

    with tab_hparams:
        st.subheader("Hyperparameters for Grid Search")
        latent_dims_set = st.multiselect("Latent dimensions:", [8,16,32,64,128], default=[32], disabled=disabled)
        if not latent_dims_set:
            st.error("At least one latent dimension required.")
            return

        epoch_choices = st.multiselect("Epoch counts:", [10,20,50,100], default=[50], disabled=disabled)
        if not epoch_choices:
            st.error("At least one epoch count required.")
            return

        batch_sizes_cvae = st.multiselect("Batch sizes:", [16,32,64,128], default=[32], disabled=disabled)
        if not batch_sizes_cvae:
            st.error("At least one batch size required.")
            return

        cvae_lr_set = st.multiselect("Learning rates:", [0.0001,0.001,0.01], default=[0.001], disabled=disabled)
        if not cvae_lr_set:
            st.error("At least one learning rate required.")
            return

        activations = st.multiselect("Activations:", ["elu","relu","tanh","sigmoid"], default=["elu"], disabled=disabled)
        if not activations:
            st.error("At least one activation required.")
            return

        nhl = st.multiselect("Number of hidden layers:", [1,2,3,4], default=[4], disabled=disabled)
        if not nhl:
            st.error("At least one number of hidden layers required.")
            return

        hsize = st.multiselect("Hidden layer sizes:", [16,32,64,128], default=[16,32,64,128], disabled=disabled)
        if not hsize:
            st.error("At least one hidden layer size required.")
            return

        with st.expander("Advanced Hyperparameters"):
            MSE_WEIGHT = st.number_input("MSE Weight:", min_value=0.0, value=0.11767, step=0.0000001, format="%.8f", disabled=disabled)
            KLD_WEIGHT = st.number_input("KLD Weight:", min_value=0.0, value=10.22048, step=0.0000001, format="%.8f", disabled=disabled)
            MRE2_WEIGHT = st.number_input("MRE2 Weight:", min_value=0.0, value=0.0002569, step=0.0000001, format="%.8f", disabled=disabled)
            ENERGY_DIFF_WEIGHT = st.number_input("Energy Diff Weight:", min_value=0.0, value=0.0005698, step=0.0000001, format="%.8f", disabled=disabled)

        st.write("Early stopping with patience=5 and min_delta=0.01 is used internally.")

    with tab_resources:
        st.header("Resource Specifications")
        col1, col2, col3 = st.columns(3)
        with col1:
            time_limit_hours = st.number_input("Time Limit (Hours):", min_value=1, max_value=168, value=1, disabled=disabled)
        with col2:
            memory_per_cpu = st.number_input("Memory per CPU (GB):", min_value=1, max_value=512, value=4, disabled=disabled)
        with col3:
            cpus_per_task = st.number_input("CPUs per Task:", min_value=1, max_value=32, value=1, disabled=disabled)

        col4, col5 = st.columns(2)
        with col4:
            st.write("All chosen combinations will be tested on HPC.")

        with col5:
            use_gpu = st.checkbox("Use GPU for Training", value=False, disabled=disabled)
            if use_gpu:
                gpus = st.number_input("Number of GPUs:", min_value=1, max_value=8, value=1, disabled=disabled)
                gpu_types = [
                    'Any GPU',
                    'geforce_gtx_1080_ti',
                    'geforce_rtx_2080_ti',
                    'geforce_rtx_3090',
                    'l40s',
                    'quadro_gp100',
                    'rtx_a4000',
                    'rtx_a4500',
                    'rtx_a6000'
                ]
                gpu_type = st.selectbox("GPU Type:", gpu_types, disabled=disabled)
            else:
                gpus = 0
                gpu_type = None

        max_concurrent_jobs = st.number_input("Max Concurrent Jobs:", min_value=1, max_value=10, value=2, disabled=disabled)

        local_test = st.checkbox("Local Test Run (no Slurm, 1 combo, 1 epoch, 1 batch)", disabled=disabled)
        if local_test:
            st.info("Local test run will run only the first combination for 1 epoch and break after 1 batch.")

    combos = []
    for ld, ep, bs, lr_val, act, layers in product(
        latent_dims_set, epoch_choices, batch_sizes_cvae, cvae_lr_set, activations, nhl
    ):
        combos.append({
            'latent_dim': ld,
            'epochs': ep,
            'batch_size': bs,
            'lr': lr_val,
            'activation': act,
            'num_hidden_layers': layers,
            'hidden_layer_sizes': hsize,
            'MSE_WEIGHT': MSE_WEIGHT,
            'KLD_WEIGHT': KLD_WEIGHT,
            'MRE2_WEIGHT': MRE2_WEIGHT,
            'ENERGY_DIFF_WEIGHT': ENERGY_DIFF_WEIGHT
        })

    num_combinations = len(combos)
    st.sidebar.write(f"**Number of Combinations:** {num_combinations}")
    if num_combinations > 0:
        combos_df = pd.DataFrame(combos)
        st.sidebar.write("**Hyperparameter Combinations:**")
        st.sidebar.dataframe(combos_df)

    with tab_run:
        start_button = st.button("Start Training", disabled=st.session_state.get("job_running", False))
        if start_button:
            if num_combinations == 0:
                st.error("No combinations chosen!")
                return

            st.session_state["job_running"] = True
            st.session_state["current_job_id"] = None

            main_job_dir = create_main_job_dir()

            # Save training.py
            with open(os.path.join(main_job_dir, "training.py"), "wb") as f:
                f.write(training_file.getvalue())

            # Prepare data
            train_input, val_input, test_input, train_condition, val_condition, test_condition = prepare_cvae_data(data, input_columns, condition_columns)
            if train_input is None:
                st.error("Data preprocessing failed.")
                return
            with open(os.path.join(main_job_dir, 'data.pkl'), 'wb') as f:
                pickle.dump((train_input, val_input, test_input, train_condition, val_condition, test_condition), f)

            combos_df.to_csv(os.path.join(main_job_dir, 'combos.csv'), index=False)

            params_dict = {
                'selected_model': 'CVAE',
                'input_columns': input_columns,
                'condition_columns': condition_columns,
                'PATIENCE': 5,
                'MIN_DELTA': 1e-2,
                'local_test': local_test,
                'main_job_dir': os.path.abspath(main_job_dir)
            }
            with open(os.path.join(main_job_dir, 'params.json'), 'w') as f:
                json.dump(params_dict, f)

            if local_test:
                # Local test run: just run training.py once locally
                os.chdir(main_job_dir)
                st.info("Running a single local test run (1 combo, 1 epoch, 1 batch)...")
                result = subprocess.run(['python', 'training.py'], capture_output=True, text=True)
                st.text(result.stdout)
                if result.returncode != 0:
                    st.error(f"Local run failed: {result.stderr}")
                else:
                    st.success("Local test run completed.")
                st.session_state["job_running"] = False
            else:
                # Normal Slurm submission:
                job_dir = os.path.join(main_job_dir, 'jobs')
                os.makedirs(job_dir, exist_ok=True)
                training_script_path = os.path.join(main_job_dir, "training.py")

                array_str = f"0-{num_combinations-1}%{max_concurrent_jobs}"
                chosen_gpu_type = gpu_type if use_gpu else None
                slurm_script_content = generate_slurm_script(
                    job_name=f'autobeocat_job_array_{main_job_dir}',
                    script_path=training_script_path,
                    output_path=os.path.join(job_dir, 'slurm_output_%A_%a.txt'),
                    error_path=os.path.join(job_dir, 'slurm_error_%A_%a.txt'),
                    time_hours=time_limit_hours,
                    mem_per_cpu=f"{memory_per_cpu}G",
                    cpus_per_task=cpus_per_task,
                    job_dir=os.path.abspath(job_dir),
                    main_job_dir=os.path.abspath(main_job_dir),
                    gpus=gpus,
                    gpu_type=chosen_gpu_type if chosen_gpu_type is not None else "Any GPU",
                    array=array_str,
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
                        st.session_state["current_job_id"] = job_id

                        st.info(f"Jobs submitted with Job ID: {job_id}.")
                        st.write("Monitor with:")
                        st.code(f"squeue -j {job_id}")

                        st.info("Waiting for jobs to complete...")

                        my_bar = st.progress(0)
                        prev_msg = ""

                        cancel_button = st.button("Cancel Job", disabled=(not st.session_state.get("job_running", False) or st.session_state.get("current_job_id") is None))
                        while st.session_state.get("job_running", False):
                            # Check if user canceled the job
                            if cancel_button and st.session_state.get("current_job_id") is not None:
                                cancel_cmd = ['scancel', st.session_state["current_job_id"]]
                                cancel_result = subprocess.run(cancel_cmd, capture_output=True, text=True)
                                if cancel_result.returncode == 0:
                                    st.warning("Job cancelled successfully.")
                                    print_with_time(f"Job {st.session_state['current_job_id']} cancelled by user.")
                                    st.session_state["job_running"] = False
                                    st.session_state["current_job_id"] = None
                                    break
                                else:
                                    st.error(f"Failed to cancel job: {cancel_result.stderr}")

                            results_path = os.path.join(main_job_dir, 'results.csv')
                            if os.path.exists(results_path):
                                results_df = pd.read_csv(results_path)
                                completed_jobs = len(results_df)
                                if completed_jobs == num_combinations:
                                    st.success("All jobs have completed.")
                                    display_leaderboard(results_df)
                                    my_bar.progress(100)
                                    st.session_state["job_running"] = False
                                    st.session_state["current_job_id"] = None
                                    break
                                else:
                                    completion_ratio = int((completed_jobs / num_combinations) * 100)
                                    my_bar.progress(completion_ratio)
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
                            time.sleep(10)

if __name__ == '__main__':
    main()

