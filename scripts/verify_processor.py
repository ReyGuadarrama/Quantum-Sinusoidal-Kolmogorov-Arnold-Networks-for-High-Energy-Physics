from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.utils.workspace import get_config

def verify_physics(config):
    # 1. Cargar el bloque de entrenamiento procesado
    data = torch.load(config["cache_file"])
    X = data['X_train_tensor'] # Debe ser [N, 32]
    
    print(f"--- Verificación Física del Pipeline ---")
    print(f"Tensor shape: {X.shape}")
    
    # 2. Validación de Normalización (El test del "sum=1.0")
    # Recordamos que z_i debe sumar 1.0 por evento en el bloque truncado
    z_indices = [3 + 2*i for i in range(15)]
    z_block = X[:, z_indices]
    sum_z = torch.sum(z_block, axis=1)
    
    # Tolerancia a errores de punto flotante
    is_normalized = torch.allclose(sum_z, torch.ones_like(sum_z), atol=1e-5)
    print(f"Test de Normalización Relativa (sum(z_i) == 1.0): {is_normalized}")
    
    if not is_normalized:
        print("¡ALERTA! El normalizado dinámico falló. Los jets aún tienen sesgo de energía.")

    # 3. Test de Invariancia Rotacional (Validación física de dR)
    # dR no debe tener una media sesgada hacia el norte o sur (d_eta, d_phi)
    dR_indices = [2 + 2*i for i in range(15)]
    mean_dR = torch.mean(X[:, dR_indices], axis=0)
    print(f"Media de dR por partícula líder: {mean_dR[:5].numpy()}")
    
    # 4. Inspección Visual de "Sanity Check"
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(X[:, 0].numpy(), bins=50, alpha=0.7, label='Log-Mass Invariante')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(X[:, 2].numpy(), bins=50, alpha=0.7, color='orange', label='dR_1 (Líder)')
    plt.legend()
    # 4. Guardar, no mostrar
    plt.tight_layout()
    plt.savefig("verificacion_fisica.png")
    print("Gráfica guardada como 'verificacion_fisica.png'. Ábrela para inspeccionar.")

if __name__ == "__main__":
    config = get_config(task="top", seed=42)
    verify_physics(config)
