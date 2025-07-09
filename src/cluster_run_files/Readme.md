# Cluster Run Files

This folder contains `.run` files intended for running code from the `src` directory on a High Performance Cluster (HPC). Each `.run` file is designed to execute a corresponding `.py` file with the same name.

## Usage Instructions

1. **Move the desired `.run` script** from this folder to the `src` directory (the parent folder).
2. **Load required modules** on the HPC:
    ```bash
    module load gcc python cuda
    ```
3. **Create the virtual environment** (only needed once):
    ```bash
    python -m venv --system-site-packages venvs/main_env
    ```
4. **Activate the virtual environment**:
    ```bash
    source venvs/main_env/bin/activate
    ```
5. **Install required Python libraries**:
    ```bash
    pip install --no-cache-dir -r ../requirements.txt
    ```

After completing these steps once, you can run any script by moving its `.run` file to the `src` directory and executing it on the cluster.
