# workspace.py
import os
import numpy as np
import torch
import random
from pathlib import Path

def get_project_root():
    """
    Returns the absolute path to the project root directory.
    This is useful for constructing paths to data, models, and reports in a way that is independent of the current working directory.
    """
    return Path(__file__).parent.parent.parent.resolve()

# Set random seeds for reproducibility
def set_seed(seed_value=42):
    print(f"Setting global random seed {seed_value} for reproducibility.")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_dirs(config):
    """
    Creates all necessary directories based on the provided configuration dictionary.
    """
    print("Ensuring directory structure exists...")
    for key, value in config.items():
        if isinstance(value, (str, Path)) and str(config['root']) in str(value):
            path=Path(value)
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: CONFIGURATION
# ============================================================================
# Define all hyperparameters and paths in one place.
# This makes it easy to modify and experiment.
def get_config(task, seed):
    """
    Returns a tight configuration dictionary isolating the raw data path, 
    the processed multiscale tensors, and specific KAN 2.0 / VQC output targets.
    """
    root = get_project_root()
    seed_dir = f"seed_{seed}"
    
    # Core directories
    data_out_dir = os.path.join(root, "data", "processed", task, seed_dir)
    outputs_dir = os.path.join(root, "outputs", task, seed_dir)
    CONFIG = {
        # Base Engine Paths
        "root": root,
        "task": task,
        "seed": seed,

        # Origin and Destination of Data
        "raw_data_dir": os.path.join(root, "data", "raw"),
        "processed_data_dir": data_out_dir,
        "scaler_path": os.path.join(data_out_dir, "global_scaler.pkl"),
        "cache_file": os.path.join(data_out_dir, "preprocessed_data.pt"),
        
        # Output targets for models and reports
        "models_dir": os.path.join(outputs_dir, "models"),
        "plots_dir": os.path.join(outputs_dir, "plots"),
        "results_dir": os.path.join(outputs_dir, "results")
    }
    return CONFIG
