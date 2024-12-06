# utils/slurm_utils.py

import os
import re

def generate_slurm_script(job_name, script_path, output_path, error_path, time_hours=1, mem_per_cpu='4G', cpus_per_task=1, job_dir='', main_job_dir='', gpus=0, gpu_type=None, array=None, num_combinations=1, partition='batch.q'):
    time_limit = f"{int(time_hours):02d}:00:00"

    gpu_request = ''
    if gpus > 0:
        if gpu_type:
            gpu_request = f"#SBATCH --gres=gpu:{gpu_type}:{gpus}"
        else:
            gpu_request = f"#SBATCH --gres=gpu:{gpus}"

    if array:
        array_line = f"#SBATCH --array={array}"
    else:
        array_line = ""

    module_load_command = 'module load Python/3.10.4-GCCcore-11.3.0-bare'
    venv_activate_command = 'source /homes/jarrar/virtualenvs/automl_env/bin/activate'

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

{module_load_command}
{venv_activate_command}

cd {os.path.dirname(script_path)}

echo "Starting task ${{SLURM_ARRAY_TASK_ID}} on `hostname` at `date`"
TOTAL_TASKS={num_combinations}

srun python {os.path.basename(script_path)} --task_id ${{SLURM_ARRAY_TASK_ID}} --total_tasks $TOTAL_TASKS --main_job_dir {main_job_dir}
echo "Task ${{SLURM_ARRAY_TASK_ID}} completed at `date`"
"""
    return slurm_script

def parse_job_id(submission_output):
    match = re.search(r'Submitted batch job (\d+)', submission_output)
    if match:
        return match.group(1)
    else:
        return None

