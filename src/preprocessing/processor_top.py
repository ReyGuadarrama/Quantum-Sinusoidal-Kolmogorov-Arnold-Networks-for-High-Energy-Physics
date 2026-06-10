# processor_top.py
import os
import h5py
import hdf5plugin
import pickle
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from src.utils.workspace import get_config, set_seed

# ============================================================================
# ============================================================================
# Vectorized Jet Kinematics Engine (Calculations directly on raw NumPy arrays)
# ============================================================================
# ============================================================================

def _compute_physics_features(raw_matrix, is_train, config):
    """
    Core math transformation. Combines macro-physics and Pareto sub-structure 
    into a structured [N, 32] matrix ready for KAN / Random Forest baselines.
    """
    n_events = raw_matrix.shape[0]
    
    # 1. EXTRACT CARTESIAN FLOWS (Stride indexing)
    p_x = raw_matrix[:, 0:800:4] # Shapes: [N_events, 200]
    p_y = raw_matrix[:, 1:800:4]
    p_z = raw_matrix[:, 2:800:4]
    E   = raw_matrix[:, 3:800:4]
    
    # Reconstruct macro-jet from global entries at the tail
    px_jet = raw_matrix[:, 800]
    py_jet = raw_matrix[:, 801]
    pz_jet = raw_matrix[:, 802]
    E_jet  = raw_matrix[:, 803]
    
    # 2. COMPUTE GLOBAL ENVIRONMENT FEATURES
    multiplicity = np.sum(E > 0, axis=1, dtype=np.float32)
    
    mass_sq = E_jet**2 - (px_jet**2 + py_jet**2 + pz_jet**2)
    mass_sq = np.clip(mass_sq, 0.0, None) # Guard against floating-point underflow
    invariant_mass = np.sqrt(mass_sq)
    
    # 3. COMPUTE LOCAL SUB-STRUCTURE FEATURES (N=15 Pareto block)
    pt_all = np.sqrt(p_x**2 + p_y**2)
    pt_block = pt_all[:, :15]
    
    # Dynamic closed-loop sum to eliminate energy leakage bias across splits
    sum_pt_block = np.sum(pt_block, axis=1)
    sum_pt_safe = np.where(sum_pt_block == 0, 1.0, sum_pt_block)
    z_effective = pt_block / sum_pt_safe[:, None] # Dynamic fraction
    
    # Shift axes to relative coordinate cylinder centred on jet
    pt_jet = np.sqrt(px_jet**2 + py_jet**2)
    eta_jet = -np.log(np.tan(np.arctan2(pt_jet, pz_jet) / 2.0) + 1e-12)
    phi_jet = np.arctan2(py_jet, px_jet)
    
    theta_i = np.arctan2(pt_all[:, :15], p_z[:, :15])
    eta_i = -np.log(np.tan(theta_i / 2.0) + 1e-12)
    phi_i = np.arctan2(p_y[:, :15], p_x[:, :15])
    
    d_eta = eta_i - eta_jet[:, None]
    d_phi = np.arctan2(np.sin(phi_i - phi_jet[:, None]), np.cos(phi_i - phi_jet[:, None]))
    d_R = np.sqrt(d_eta**2 + d_phi**2)
    
    # 4. QUANTUM ASYMPTOTIC COMPRESSION
    m_scaled = np.tanh(np.log(invariant_mass + 1.0))
    
    if is_train:
        scaler = StandardScaler()
        M_normalized = scaler.fit_transform(multiplicity.reshape(-1, 1)).flatten()
        with open(config["scaler_path"], "wb") as f:
            pickle.dump(scaler, f)
    else:
        with open(config["scaler_path"], "rb") as f:
            scaler = pickle.load(f)
        M_normalized = scaler.transform(multiplicity.reshape(-1, 1)).flatten()
        
    M_scaled = np.tanh(M_normalized)
    
    # 5. INTERLEAVE AND PACK COLUMNS [N, 32]
    processed_matrix = np.zeros((n_events, 32), dtype=np.float32)
    processed_matrix[:, 0] = m_scaled
    processed_matrix[:, 1] = M_scaled
    
    for idx in range(15):
        processed_matrix[:, 2 + 2*idx] = d_R[:, idx]      # Geometric parity columns
        processed_matrix[:, 3 + 2*idx] = z_effective[:, idx] # Energetic fraction columns
        
    return processed_matrix, scaler

# ============================================================================
# ============================================================================
# Load and preprocess data
# ============================================================================
# ============================================================================

def load_and_preprocess_data(data_dir, processed_dir, task, seed=42, force_process=False):
    """ 
    Processes separate train.h5, val.h5, and test.h5 files sequentially.
    """
    set_seed(seed)
    config = get_config(task, seed)
    
    DATA_DIR = Path(data_dir)
    PROCESSED_DIR = Path(processed_dir)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    cache_file = PROCESSED_DIR / "preprocessed_data.pt"
    scaler_file = config["scaler_path"]

    # --- STEP 1: CACHE SYSTEM DETECTOR ---
    if cache_file.exists() and not force_process:
        print(f"\n[CACHE DETECTED] Loading preprocessed matrices from: '{cache_file}'")
        try:
            cached_data = torch.load(cache_file)
            if scaler_file.exists():
                with open(scaler_file, "rb") as f:
                    scaler = pickle.load(f)
            print(">> Multi-scale arrays successfully loaded from cache environment.")
            return (
                cached_data['X_train_tensor'], cached_data['y_train_tensor'],
                cached_data['X_val_tensor'], cached_data['y_val_tensor'],
                cached_data['X_test_tensor'], cached_data['y_test_tensor'],
                cached_data['X_train_sample'], scaler
            )
        except Exception as e:
            print(f"CRITICAL cache error: {e}. Falling back to execution loops.")

    # --- STEP 2: SEQUENTIAL PROCESS (Train, Val, Test) ---
    raw_files = {
        "train": DATA_DIR / "train.h5",
        "val": DATA_DIR / "val.h5",
        "test": DATA_DIR / "test.h5"
    }
    
    processed_tensors = {}
    scaler = None

    for split, file_path in raw_files.items():
        print("\n*---------------------------------------------------*")
        print(f"Loading and processing raw split: [{split.upper()}] from HDF5")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Missing mandatory TopTagging partition file: {file_path.name}")
            
        with h5py.File(file_path, "r") as f:
            # Reconstruct structured event tables
            f = f["table"]["table"] # Navigate to the nested group containing the data
            raw_matrix = f["values_block_0"][:]
            # SOTA label: index 1 holds the categorical value (1: Top Signal, 0: QCD Background)
            raw_labels = f["values_block_1"][:, 1] 
            
        print(f"Data chunk successfully mounted in RAM. Extracted shape: {raw_matrix.shape}")
        
        # Transform vector components
        is_train = (split == "train")
        X_norm, split_scaler = _compute_physics_features(raw_matrix, is_train, config)
        
        if is_train:
            scaler = split_scaler

        # Convert straight to standalone float Torch tensors
        processed_tensors[f"X_{split}"] = torch.from_numpy(X_norm).float()
        
        y_tensor = torch.from_numpy(raw_labels).float()
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1) # Match required [N, 1] output dimension
        processed_tensors[f"y_{split}"] = y_tensor

    print("\n--- Final balanced datasets built (Vectorized Slices Framework) ---")
    print(f"X_train shape: {processed_tensors['X_train'].shape} | y_train shape: {processed_tensors['y_train'].shape}")
    print(f"X_val shape:   {processed_tensors['X_val'].shape} | y_val shape:   {processed_tensors['y_val'].shape}")
    print(f"X_test shape:  {processed_tensors['X_test'].shape} | y_test shape:  {processed_tensors['y_test'].shape}")

    # --- STEP 3: SYMBOLIC INTERPOLATION SAMPLE (10k Sub-sample) ---
    print(f"\nIsolating clean sub-sample for high-speed symbolic KAN regressions...")
    sample_size = 10000
    X_train_all = processed_tensors["X_train"]
    
    if len(X_train_all) > sample_size:
        # Uniform sampling permutation over the GPU/CPU data graph boundary
        random_indices = torch.randperm(len(X_train_all))[:sample_size]
        X_train_sample = X_train_all[random_indices]
    else:
        X_train_sample = X_train_all
    print(f"Warm-up tensor isolated. Size: {len(X_train_sample)} physics target nodes.")

    # --- STEP 4: MEMORY MAP CHECKPOINT SERIALIZATION ---
    processed_data = {
        'X_train_tensor': processed_tensors["X_train"],
        'y_train_tensor': processed_tensors["y_train"],
        'X_val_tensor':   processed_tensors["X_val"],
        'y_val_tensor':   processed_tensors["y_val"],
        'X_test_tensor':  processed_tensors["X_test"],
        'y_test_tensor':  processed_tensors["y_test"],
        'X_train_sample': X_train_sample
    }
    
    torch.save(processed_data, cache_file)
    print(f"\n[CACHE WRITTEN] Saving preprocessed database block into: '{cache_file}'")
    print(f"Inference metrics scaler object written into workspace folder structures.")

    return (
        processed_data['X_train_tensor'], processed_data['y_train_tensor'],
        processed_data['X_val_tensor'], processed_data['y_val_tensor'],
        processed_data['X_test_tensor'], processed_data['y_test_tensor'],
        processed_data['X_train_sample'], scaler
    )
