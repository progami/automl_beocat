# wrapper.py
import sys
import subprocess

if len(sys.argv) < 2:
    print("Usage: python wrapper.py path/to/params.json")
    sys.exit(1)

params_path = sys.argv[1]

# Run training.py locally
command = ["python", "training.py"]
print("Running:", " ".join(command))
result = subprocess.run(command, capture_output=True, text=True)

if result.returncode == 0:
    print("training.py completed successfully.")
    print("STDOUT:", result.stdout)
else:
    print("training.py failed.")
    print("STDERR:", result.stderr)
    sys.exit(1)

