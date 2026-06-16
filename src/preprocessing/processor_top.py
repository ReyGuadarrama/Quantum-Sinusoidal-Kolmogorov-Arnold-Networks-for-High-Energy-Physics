# processor_top.py
import os
import h5py
import hdf5plugin
import pickle
import numpy as np
import torch
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from src.utils.workspace import get_config, set_seed

# ============================================================================
# ============================================================================
# Vectorized Jet Kinematics Engine (Calculations directly on raw NumPy arrays)
# ============================================================================
# ============================================================================

def _compute_physics_features(raw_matrix, config, scaler=None):
    """
    Core math transformation. Combines macro-physics and Pareto sub-structure 
    into a structured [N, 32] matrix ready for KAN / Random Forest baselines.
    """
    # -----------------------
    # 1. GLOBAL JET MASKING
    # -----------------------
    # Extract and sanitize global jet properties
    px_jet = raw_matrix[:, 800]
    py_jet = raw_matrix[:, 801]
    pz_jet = raw_matrix[:, 802]
    E_jet = raw_matrix[:, 803]

    # Jet angles
    pt_jet = np.sqrt(px_jet**2 + py_jet**2)
    eta_jet = -np.log(np.tan(np.arctan2(pt_jet, pz_jet + 1e-12) / 2.0) + 1e-12)
    phi_jet = np.arctan2(py_jet, px_jet + 1e-12)

    n_events = raw_matrix.shape[0]

    mass_sq = np.clip(E_jet**2 - (px_jet**2 + py_jet**2 + pz_jet**2), 0.0, None)
    invariant_mass = np.sqrt(mass_sq)

    # Clean up raw jet arrays to free memory
    del px_jet, py_jet, pz_jet, E_jet, pt_jet
    gc.collect()

    # ----------------------------------------
    # 2. Local masking and feature engineering
    # ----------------------------------------
    # EXTRACT CARTESIAN FLOWS
    p_x = raw_matrix[:, 0:60:4].copy()
    p_y = raw_matrix[:, 1:60:4].copy()
    p_z = raw_matrix[:, 2:60:4].copy()
    E   = raw_matrix[:, 3:60:4].copy()
    
    pt_block = np.sqrt(p_x**2 + p_y**2)

    multiplicity = np.sum(raw_matrix[:, 3:800:4] > 1e-5, axis=1)

    is_ghost = (pt_block < 1e-5) # Mask for zero-energy particles
    
    pt_block[is_ghost] = 1.0
    p_z[is_ghost] = 0.0
    # CONSTITUENT MASKING (Layer 1)
    # Identify valid particles based on detector coverage (|eta| < 3.0) and energy
    theta_i = np.arctan2(pt_block, p_z + 1e-12)
    eta_i = -np.log(np.tan(theta_i / 2.0) + 1e-12)
    phi_i = np.arctan2(p_y, p_x + 1e-12)

    pt_block[is_ghost] = 0.0
    eta_i[is_ghost] = 0.0
    phi_i[is_ghost] = 0.0

    # Calculate relative coordinates
    d_eta_i = eta_i - eta_jet[:, None]
    d_phi_i = np.arctan2(
                    np.sin(phi_i - phi_jet[:, None]),
                    np.cos(phi_i - phi_jet[:, None])
                    )
    d_R = np.sqrt(d_eta_i**2 + d_phi_i**2)

    out_of_cone = (d_R > 0.8) | is_ghost

    d_R[out_of_cone] = 0.0
    pt_block[out_of_cone] = 0.0

    # Clean up large intermediate arrays to free memory
    del p_x, p_y, p_z, E, theta_i#, eta_i #phi_i particle_mask
    del d_eta_i, d_phi_i
    gc.collect()

    # DYNAMIC NORMALIZATION
    sum_pt = np.sum(pt_block, axis=1)
    sum_pt[sum_pt <= 0] = 1.0
    z_effective = pt_block / sum_pt[:, None]

    z_effective[is_ghost] = 0.0
    # -------------------------------------------------------------------------
    # 3. PHYSICAL VALIDATION LOGGING
    # -------------------------------------------------------------------------
    if (~is_ghost).any():
        print(
            f"Rango de eta_i (procesado): {eta_i[~is_ghost].min():.2f} a {eta_i[~is_ghost].max():.2f}"
        )
    valid_dR = d_R[d_R > 0]
    if valid_dR.size > 0:
        print(
            f"Rango de dR (partículas reales): {valid_dR.min():.2f} a {valid_dR.max():.2f}"
        )

    # Check energy conservation constraints internally
    sum_z_test = np.sum(z_effective, axis=1)
    active_jets = sum_z_test > 0
    if active_jets.any():
        is_normalized = np.allclose(sum_z_test[active_jets], 1.0, atol=1e-3)
        print(f"Test de Normalización Relativa (sum(z_i) == 1.0): {is_normalized}")

    # 4. QUANTUM ASYMPTOTIC COMPRESSION
    m_scaled = np.tanh(np.log(invariant_mass + 1.0))
    
    if scaler is None:
        scaler = StandardScaler()
        M_normalized = scaler.fit_transform(multiplicity.reshape(-1, 1)).flatten()
    else:
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
    scaler_file = Path(config["scaler_path"])

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
            raw_matrix = f["values_block_0"][:20000]
            # Index 1 holds the categorical value (1: Top Signal, 0: QCD Background)
            raw_labels = f["values_block_1"][:20000, 1] 


        print(f"Data chunk successfully mounted in RAM. Extracted shape: {raw_matrix.shape}")
        
        # DIAGNÓSTICO ANTES DEL PROCESAMIENTO
        print(f"--> Distribución CRUDA en disco de etiquetas para [{split.upper()}]:")
        clases, conteos = np.unique(raw_labels, return_counts=True)
        for c, n in zip(clases, conteos):
            print(f"    Clase {c}: {n} eventos")

        # Transform vector components
        X_norm, split_scaler = _compute_physics_features(raw_matrix, config, scaler=scaler)
        
        if split == "train":
            with open(scaler_file, "wb") as f:
                pickle.dump(split_scaler, f)
            print(f"Global scaler object saved to: '{scaler_file}'")

        # Convert straight to standalone float Torch tensors
        processed_tensors[f"X_{split}"] = torch.from_numpy(X_norm).float()
        #raw_labels = raw_labels[split_mask]
        y_tensor = torch.from_numpy(raw_labels).float()
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1) # Match required [N, 1] output dimension
        processed_tensors[f"y_{split}"] = y_tensor

        del raw_matrix, raw_labels, X_norm, y_tensor
        gc.collect()

    print("\n--- Final balanced datasets built (Vectorized Slices Framework) ---")
    print(f"X_train shape: {processed_tensors['X_train'].shape} | y_train shape: {processed_tensors['y_train'].shape}")
    print(f"X_val shape:   {processed_tensors['X_val'].shape} | y_val shape:   {processed_tensors['y_val'].shape}")
    print(f"X_test shape:  {processed_tensors['X_test'].shape} | y_test shape:  {processed_tensors['y_test'].shape}")

    # --- STEP 3: SYMBOLIC INTERPOLATION SAMPLE (10k Sub-sample) ---
    print(f"\nIsolating clean sub-sample for high-speed symbolic KAN regressions...")
    sample_size = int(0.05*len(processed_tensors["X_train"]))
    X_train_all = processed_tensors["X_train"]
    
    if len(X_train_all) > sample_size:
        # Uniform sampling permutation over the GPU/CPU data graph boundary
        random_indices = torch.randperm(len(X_train_all))[:sample_size]
        X_train_sample = X_train_all[random_indices]
    else:
        X_train_sample = X_train_all
    print(f"Warm-up tensor isolated. Size: {len(X_train_sample)} physics target nodes.")

    # logging con clases únicas y conteo de ellas
    print(f"Clases únicas en Train: {torch.unique(processed_tensors['y_train'])} con conteos: {torch.bincount(processed_tensors['y_train'].long().squeeze())}")
    print(f"Clases únicas en Val: {torch.unique(processed_tensors['y_val'])} con conteos: {torch.bincount(processed_tensors['y_val'].long().squeeze())}")

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
        processed_data['X_train_sample'], split_scaler
    )
