import sys
from pathlib import Path
# Añadir la raíz al path para importar src y workspace
sys.path.append(str(Path(__file__).parent.parent.resolve()))
import src.utils.workspace as workspace
import src.preprocessing.processor_top as processor
import src.architectures.classic_kan as classic
import src.utils.metrics as viz
from src.architectures.classic_kan import clean_memory

import torch
import time
import os
import json
import numpy as np
import argparse
import gc
from pathlib import Path

def main(args):
    """Main fuction to run the full training pipeline."""
    torch.set_num_threads(4)  # Limit PyTorch to use 4 CPU threads for data loading and processing
    workspace.set_seed(args.seed)
    
    CONFIG = workspace.get_config(task="top", seed=args.seed)
    workspace.make_dirs(CONFIG)
    
    start_time = time.time()
    print(f"Starting the automated training pipeline. Seed: {args.seed}.")


    # ============================================================================
    # STEP 2: DATA LOADING AND PREPROCESSING
    # ============================================================================
    print("\n--- Step 1: Loading and Preprocessing Data ---")
    top_path = os.path.join(CONFIG["raw_data_dir"], "top")
    X_train, y_train, \
    X_val, y_val, \
    X_test, y_test, \
    X_sample, scaler = processor.load_and_preprocess_data(
                                        data_dir=top_path,
                                        processed_dir=CONFIG["processed_data_dir"],
                                        task=CONFIG["task"],
                                        force_process=False
                                    )
    del top_path
    gc.collect()
    
    # Saving scaler to inference

    # ============================================================================
    # STEP 3: BASE TRAINING
    # ============================================================================
    print("\n--- Step 2: Starting Base Training ---")
    base_model_prefix = os.path.join(CONFIG["models_dir"], "01_base")
    base_model_state_path = f"{base_model_prefix}_state"

    if os.path.exists(base_model_state_path) and not args.force:
        print(f"Base model found at {base_model_prefix}")
    else:
        history_base = classic.train_kan_model(
            width=CONFIG["width"],
            grid=CONFIG["grid"],
            k=CONFIG["k"],
            learning_rate=CONFIG["base_lr"],
            num_epochs=CONFIG["base_epochs"],
            batch_size=CONFIG["base_batch_size"], 
            early_stop_patience=CONFIG["base_patience"],
            early_stop_min_delta=CONFIG["base_early_stop_delta"],
            lamb=CONFIG["base_lamb"],
            lamb_l1=CONFIG["base_lamb_l1"],
            lamb_entropy=CONFIG["base_lamb_entropy"],
            lamb_coef=CONFIG["base_lamb_coef"],
            lamb_coefdiff=CONFIG["base_lamb_coefdiff"],
            update_grid_freq=CONFIG["base_update_grid_freq"],
            model_save_path=base_model_prefix,
            X_train_tensor=X_train[:],
            y_train_tensor=y_train[:],
            X_val_tensor=X_val[:10000], 
            y_val_tensor=y_val[:10000],
            num_workers=CONFIG["num_workers"]
        )
        
        with open(CONFIG["base_train_history_data"], 'w') as f:
            json.dump(history_base, f, indent=4)

        viz.plot_loss_history(history_base, save_path=CONFIG["base_train_loss_plot"])
        viz.plot_auc_history(history_base, save_path=CONFIG["base_train_auc_plot"])

        print("\n--- Evaluation of Base Model ---")
        model_base, eval_data_base, metrics_base = classic.evaluate_kan_model(
            model_save_path=base_model_prefix,
            X_test_tensor=X_test[:10000],
            y_test_tensor=y_test[:10000],
            conf_matrix_save_path=CONFIG["base_eval_cm"],
            save_path_roc_curve=CONFIG["base_eval_roc"],
            save_path_pr_curve=CONFIG["base_eval_pr"]
        )

        np.save(CONFIG["base_eval_data_true"], eval_data_base[0])
        np.save(CONFIG["base_eval_data_probs"], eval_data_base[1])
        np.save(CONFIG["base_eval_data_binary"], eval_data_base[2])

        with open(CONFIG["base_eval_metrics"], 'w') as f:
            json.dump(metrics_base, f, indent=4)

        print("\n--- Plotting Base Model Splines ---")
        model_base.plot(
            folder=CONFIG["base_model_plot_folder"],
            save_path=CONFIG["base_model_plot_save_path"],
            beta=12.0,
            metric="backward",
            in_vars=CONFIG["features"],
            scale=1.0,
            varscale=0.5
        )

        print("Cleaning up base model from memory...")
        clean_memory(model_base, eval_data_base, metrics_base, history_base)

# ============================================================================
# STEP 4: PRUNING
# ============================================================================
    print("\n--- Step 3: Starting Pruning ---")
    pruned_model_prefix = os.path.join(CONFIG["pruned_model_path"], "02_pruned")
    pruned_model_state_path = f"{pruned_model_prefix}_state"

    if os.path.exists(pruned_model_state_path) and not args.force:
        print(f"Pruned model found. Skiping pruning.")
    else:
        pruned_model = classic.prune_and_save_kan(
            original_model_path=base_model_prefix,
            pruned_model_path=pruned_model_prefix,
            activation_data=X_sample, # Use sample for fast pruning
            node_th=CONFIG["prune_node_th"],
            edge_th=CONFIG["prune_edge_th"]
        )
        # del pruned_model
        gc.collect()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated KAN Training Pipeline")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument("--force", action='store_true', help='Overwrite existing models')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
