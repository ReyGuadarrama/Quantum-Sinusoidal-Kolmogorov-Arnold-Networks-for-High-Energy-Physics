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
def get_config(seed):
    """
    Returns a tight configuration dictionary isolating the raw data path, 
    the processed multiscale tensors, and specific KAN 2.0 / VQC output targets.
    """
    root = get_project_root()
    seed_dir = f"seed_{seed}"
    
    # Core directories
    data_out_dir = os.path.join(root, "data", seed_dir)
    models_out_dir = os.path.join(root, "production_models", seed_dir)
    reports_out_dir = os.path.join(root, "reports", seed_dir)

    CONFIG = {
        # Base Engine Paths
        "root": root,
        "raw_data_train": os.path.join(root, "data", "train.h5"),
        "raw_data_top_val": os.path.join(root, "data", "val.h5"),
        "raw_data_top_test": os.path.join(root, "data", "test.h5"),
        "processed_data_dir": data_out_dir,
        "scaler_path": os.path.join(data_out_dir, "scaler_top.pkl"),
        
        # Matrix Tensors Outputs
        "train_extended_matrix": os.path.join(data_out_dir, "tt_processed_inputs.npy"), # [N, 32]
        
        # Model Production Targets (Saves learned weights/state_dicts)
        "kan_classical_model": os.path.join(models_out_dir, "kan", "multkan_checkpoint.pt"),
        "vqc_quantum_model": os.path.join(models_out_dir, "vqc", "vqc_dru_checkpoint.pt"),
        
        # Benchmarks Comparison Checkpoints (Baselines SMART Phase 1)
        "mlp_baseline": os.path.join(models_out_dir, "baselines", "mlp_model.pt"),
        "rf_baseline": os.path.join(models_out_dir, "baselines", "rf_model.pkl"),
        
        # Performance and EDA Plot Reports (SMART Phase 4 Validation)
        "eda_reports_dir": os.path.join(reports_out_dir, "eda"),         # Subplots de Pareto y NMI
        "model_reports_dir": os.path.join(reports_out_dir, "evaluation") # Curvas ROC/AUC benchmarks
    }
    return CONFIG