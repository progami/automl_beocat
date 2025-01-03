# utils/slurm_utils.py

import os
import subprocess
import time
import re
import pandas as pd

def generate_slurm_script(
    job_name,
    script_path,
    output_path,
    error_path,
    time_hours,
    mem_per_cpu,
    cpus_per_task,
    job_dir,
    main_job_dir,
    gpus,
    gpu_type,
    array,
    num_combinations
):
    gpu_line = ''
    if gpus > 0:
        gpu_line = f'#SBATCH --gres=gpu:{gpu_type}:{gpus}'

    # Ensure script_path is an absolute path
    script_path = os.path.abspath(script_path)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)

    # Adjust the module load command and virtual environment path
    module_load_command = 'module load Python/3.10.4-GCCcore-11.3.0-bare'
    venv_activate_command = 'source /homes/jarrar/virtualenvs/automl_env/bin/activate'

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_path}
#SBATCH --error={error_path}
#SBATCH --time={int(time_hours)}:00:00
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --array={array}
{gpu_line}

# Load the required module
{module_load_command}

# Activate the virtual environment
{venv_activate_command}

# Change to the directory containing training.py
cd {script_dir}

echo "Starting task ${{SLURM_ARRAY_TASK_ID}}"

TOTAL_TASKS={num_combinations}

srun python {script_name} --task_id ${{SLURM_ARRAY_TASK_ID}} --total_tasks $TOTAL_TASKS --main_job_dir {main_job_dir}
"""

    return slurm_script

def parse_job_id(submit_output):
    match = re.search(r'Submitted batch job (\d+)', submit_output)
    if match:
        return match.group(1)
    else:
        return None

def collect_results(main_job_dir, selected_model):
    results_csv_path = os.path.join(main_job_dir, 'results.csv')
    if not os.path.exists(results_csv_path):
        print("Results file not found.")
        return None

    # Read the aggregated results
    leaderboard_df = pd.read_csv(results_csv_path)
    return leaderboard_df

def get_job_status(job_id):
    # Function to get the status of a SLURM job
    cmd = f"sacct -j {job_id} --format=JobID,State"
    try:
        output = subprocess.check_output(cmd, shell=True).decode('utf-8')
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error getting job status: {e}")
        return None

def cancel_job(job_id):
    # Function to cancel a SLURM job
    cmd = f"scancel {job_id}"
    try:
        subprocess.check_call(cmd, shell=True)
        print(f"Job {job_id} cancelled.")
    except subprocess.CalledProcessError as e:
        print(f"Error cancelling job {job_id}: {e}")

def monitor_job(job_id):
    # Function to monitor the status of a SLURM job
    try:
        cmd = f"squeue -j {job_id}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if stdout:
            output = stdout.decode('utf-8')
            if job_id in output:
                return True  # Job is still running
            else:
                return False  # Job has completed
        else:
            return False
    except Exception as e:
        print(f"Error monitoring job {job_id}: {e}")
        return False

