# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# classic_kan.py

# Import necessary packages
import time #, math
import warnings

#from scipy.stats import ks_2samp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, auc, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    precision_score, recall_score, roc_curve
)
from kan import KAN
import gc
import json
#import traceback
from kan.utils import SYMBOLIC_LIB
#import sympy
from sympy.utilities.lambdify import lambdify

import src.utils.metrics as viz



# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# ============================================================================
# Define the neural network model
# ============================================================================
# ============================================================================

def train_kan_model(width, grid, k, learning_rate, num_epochs, batch_size,
                           early_stop_patience, early_stop_min_delta, 
                           model_save_path,
                           X_train_tensor,
                           y_train_tensor,
                           X_val_tensor,
                           y_val_tensor,
                           lamb=0.01, 
                           lamb_l1=1.0, 
                           lamb_entropy=2.0,
                           lamb_coef=0.0,
                           lamb_coefdiff=0.0,
                           reg_metric='edge_forward_spline_n',
                           update_grid_freq=5,
                           num_workers=4):
    """
    Trains a KAN model using a custom loop that includes key pykan features:
    regularization and adaptive grid updates.

    Args:
        width (list): The architecture of the KAN model.
        grid (int): The number of grid points for the splines.
        k (int): The order of the splines.
        learning_rate (float): The learning rate for the Adam optimizer.
        num_epochs (int): The maximum number of epochs for training.
        batch_size (int): The size of the batches for training.
        early_stop_patience (int): Number of epochs with no improvement to wait before stopping.
        early_stop_min_delta (float): Minimum change in validation loss to be considered an improvement.
        model_save_path (str): Path to save the best model checkpoint.
        X_train_tensor, y_train_tensor: Training data and labels as PyTorch tensors.
        X_val_tensor, y_val_tensor: Validation data and labels as PyTorch tensors.
        lamb (float): Overall regularization strength.
        lamb_l1 (float): L1 regularization strength for sparsity.
        lamb_entropy (float): Entropy regularization strength.
        update_grid_freq (int): The frequency (in epochs) to update the spline grid.
        num_workers (int): Number of worker processes for data loading.
    Returns:
        dict: A dictionary containing the training history (losses and AUCs).
    """
    print(f"\n--- Starting Training for KAN Model ---")
    print(f"Parameters: width={width}, grid={grid}, k={k}, lr={learning_rate}")

    # 1. Define the model, criterion, and optimizer
    model = KAN(width, grid, k, symbolic_enabled=True, auto_save=False)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"KAN model created with {num_params} parameters.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 2. Set up optimized DataLoaders
      # Use a reasonable number of workers based on CPU cores
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, persistent_workers=False)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, persistent_workers=False)
    
    # 3. Training and Validation Loop
    best_val_auc = 0.0
    epochs_no_improve = 0
    best_val_loss = float('inf')
    start_time = time.time()

    history = {'train_loss': [], 
               'val_loss': [],
               'train_auc': [],
                'val_auc': []
               }

    print(f"\nStarting training using {num_workers} workers...")
    for epoch in range(num_epochs):
        # --- Update the grid adaptively ---
        if epoch > 0 and epoch % update_grid_freq == 0:
            
            # New grid size
            new_grid_size = model.grid + 3
            
            # Creating a new model with the new grid size
            new_model = KAN(width, new_grid_size, k, symbolic_enabled=True, auto_save=False)
            
            # Transfer weights from the old model to the new model
            new_model.initialize_from_another_model(model, X_train_tensor)
            model = new_model
            
            # Adjust learning rate and early stopping delta for the grid update
            #learning_rate *= 0.8
            early_stop_min_delta = max(early_stop_min_delta * 0.6, 1e-6) # Reduce delta but not below a minimum threshold
            
            # Reset early stopping counters after grid update
            epochs_no_improve = 0
            best_val_loss = float('inf')

            print(f"\nUpdating grid at epoch {epoch+1}...")
            print(f"  -> Current grid: {model.grid}")
            print(f"learning_rate for grid update: {learning_rate:.6f}")
            print(f"early_stop_min_delta for grid update: {early_stop_min_delta:.6f}")
            
            print(f"Model grid updated to {model.grid} points. Continuing training with the new grid...")
            
            # Use the training set to inform the grid update
            with torch.no_grad():
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                #model.update_grid(X_train_tensor) 

        model.train()
        train_loss = 0.0

        all_train_probs_list = []
        all_train_true_list = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Calculate the classification loss
            bce_loss = criterion(outputs, batch_y)
            
            # --- KAN Regularization Loss ---
            reg_value = model.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
            
            # Combine both losses
            total_loss = bce_loss + lamb*reg_value

            total_loss.backward()
            optimizer.step()
            # We only track the classification loss for logging purposes
            train_loss += bce_loss.item() * batch_X.size(0)

            all_train_probs_list.append(torch.sigmoid(outputs).cpu().detach())
            all_train_true_list.append(batch_y.cpu().detach())
            
        train_loss /= len(train_dataset)

        # Validation process
        model.eval()
        val_loss = 0.0

        all_val_probs_list = []
        all_val_true_list = []
        
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                outputs_val = model(batch_X_val)
                loss_val = criterion(outputs_val, batch_y_val)
                val_loss += loss_val.item() * batch_X_val.size(0)

                all_val_probs_list.append(torch.sigmoid(outputs_val).cpu().detach())
                all_val_true_list.append(batch_y_val.cpu().detach())
        val_loss /= len(val_dataset)

        # Concatenate all predictions and true labels
        all_val_probs = torch.cat(all_val_probs_list, dim=0).numpy()
        all_val_true = torch.cat(all_val_true_list, dim=0).numpy()

        all_train_probs = torch.cat(all_train_probs_list, dim=0).numpy()
        all_train_true = torch.cat(all_train_true_list, dim=0).numpy()

        # AUC calculation for all validation data
        val_auc = roc_auc_score(all_val_true, all_val_probs)
        train_auc = roc_auc_score(all_train_true, all_train_probs)

        # logging for every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Training Loss: {train_loss:.5f}, "
                f"Validation Loss: {val_loss:.5f}, "
                f"Training AUC: {train_auc:.5f}, "
                f"Validation AUC: {val_auc:.5f}")

        # Save losses and AUC
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        # Check for improvement in validation AUC and val loss for early stopping and model checkpointing
        if ((val_auc > best_val_auc + early_stop_min_delta) \
        or (val_loss < best_val_loss - early_stop_min_delta)):
            best_val_auc = val_auc
            #epochs_no_improve = 0
            model.saveckpt(model_save_path)

            training_time_seconds = time.time() - start_time
            metadata = {
                'num_params': num_params,
                'training_time_seconds': training_time_seconds,
                'final_val_auc': best_val_auc,
                'hyperparameters': {
                    'width': width,
                    'grid': grid,
                    'k': k,
                    'learning_rate': learning_rate
                }
            }
            try:
                with open(f"{model_save_path}_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=4)
            except Exception as e:
                print(f"Warning: Could not save metadata to JSON file. Error: {e}")
            print(f"Epoch [{epoch+1}/{num_epochs}] - Model checkpoint saved (val_auc: {best_val_auc:.5f} - val_loss: {val_loss:.5f}).")    
        #else:
            #epochs_no_improve += 1

        # Early stop based on loss improvement
        if val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print("Early Stopping triggered!")
            break
            
    training_time_seconds = time.time() - start_time
    print(f"\nTraining finished in {training_time_seconds:.2f} seconds!")
    
    # --- Save a dictionary with model state and metadata ---
    print(f"Saving final model and metadata to '{model_save_path}'...")

    return history

def clean_memory(*args):
    """Utility function to clean up memory by deleting variables and clearing GPU cache."""
    for obj in args:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def evaluate_kan_model(model_save_path, 
                       X_test_tensor, y_test_tensor, 
                       save_path_roc_curve=None, 
                       conf_matrix_save_path=None,
                       save_path_pr_curve=None):
    """
    Loads a KAN model and its metadata from a checkpoint file and evaluates 
    its performance on the test set.
    args:
        model_save_path (str): Path to the saved model checkpoint (.pth file).
        X_test_tensor (torch.Tensor): Test features as a PyTorch tensor.
        y_test_tensor (torch.Tensor): Test labels as a PyTorch tensor.
        save_path_roc_curve (str): Path to save the ROC curve plot.
        conf_matrix_save_path (str): Path to save the confusion matrix plot.
        save_path_pr_curve (str): Path to save the Precision-Recall curve plot.
    return:
        tuple: A tuple containing the loaded model, 
               a tuple of (true labels, predicted probabilities, predicted classes),
               and a dictionary of evaluation metrics.
    """
    print("\n--- Starting Evaluation of the Best Model ---")

    model = KAN.loadckpt(model_save_path)
    
    # 3. Load the trained weights into the model
    model.eval()
    print(f"Model loaded from '{model_save_path}'.")

    # 5. Evaluate on the test set (The rest of the function is the same)
    criterion = nn.BCEWithLogitsLoss()
    test_true = y_test_tensor.cpu().numpy()

    with torch.no_grad():
        outputs_test = model(X_test_tensor)
        loss_test = criterion(outputs_test, y_test_tensor)
        
        probs_test = torch.sigmoid(outputs_test)
        predicted_classes_test = (probs_test > 0.5).float()
        
        test_preds_probs = probs_test.cpu().numpy()
        test_preds_binary = predicted_classes_test.cpu().numpy()

    # 3. Calculate and display final metrics
    test_loss = loss_test.item()
    test_accuracy = accuracy_score(test_true, test_preds_binary)
    test_f1 = f1_score(test_true, test_preds_binary)
    test_auc = roc_auc_score(test_true, test_preds_probs)
    test_precision = precision_score(test_true, test_preds_binary)
    test_recall = recall_score(test_true, test_preds_binary)
    conf_matrix = confusion_matrix(test_true, test_preds_binary)
        
    # --- Print Final Metrics ---
    # Now using the metadata loaded from the file
    print("\n--- Final Metrics on the Test Set ---")
    print(f"Test Loss: {test_loss:.5f}")
    print(f"Test Accuracy: {test_accuracy:.5f}")
    print(f"Test F1 Score: {test_f1:.5f}")
    print(f"Test AUC: {test_auc:.5f}")
    print(f"Test Precision: {test_precision:.5f}")
    print(f"Test Recall: {test_recall:.5f}")

    # Print Confusion Matrix
    print("\nMatriz de Confusión:")
    print(conf_matrix)

    # plot ROC curve
    viz.plot_roc_curve(test_true, test_preds_probs, save_path=save_path_roc_curve)

    # plot Precision-Recall curve
    viz.plot_precision_recall_curve(test_true, test_preds_probs, save_path=save_path_pr_curve)

    # Plot Confusion Matrix
    viz.plot_confusion_matrix(conf_matrix, save_path=conf_matrix_save_path)

    metrics = {
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy,
        "Test F1 Score": test_f1,
        "Test AUC": test_auc,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Confusion Matrix": conf_matrix.tolist() # .tolist() to make it JSON serializable
    }

    return model, (test_true, test_preds_probs, test_preds_binary), metrics

# ============================================================================
# Pruning and Retraining Functions
# ============================================================================

def prune_and_save_kan(original_model_path,
                        pruned_model_path,
                        activation_data,
                        node_th=1e-2,
                        edge_th=3e-2
                       ):
    """
    Load a trained KAN model, prune it, and save a new checkpoint
    compatible with the evaluation function.

    Args:
        original_model_path (str): Path to the .pth file of the trained model.
        pruned_model_path (str): Path where the new checkpoint of the pruned model will be saved.
        prune_threshold (float): Threshold for pruning nodes and edges.

    Returns:
        KAN: The pruned model object, ready for retraining.
    """
    print(f"\n--- Starting Pruning Process ---")

    # 1. Load the original complete checkpoint
    print(f"Loading original model from (prefix): '{original_model_path}'")
    original_model = KAN.loadckpt(original_model_path)

        # 2. Prune the model
    print(f"Pruning the model with a threshold of {node_th} and {edge_th}...")
    # The model needs data to compute activations before pruning
    # We assume training data is available (you can pass X_train_tensor if not)
    original_model.get_act(activation_data) 
    pruned_model = original_model.prune(node_th=node_th, edge_th=edge_th)
    print(f"New architecture after pruning: {pruned_model.width}")

    # 3. Create a new compatible checkpoint for the pruned model
    print(f"Creating and saving the new checkpoint at: '{pruned_model_path}'")

    pruned_model.saveckpt(pruned_model_path)
    
    print("Checkpoint of the pruned model saved successfully.")
    
    return pruned_model
