import os
import time
import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(epochs, train_losses, val_losses, train_accuracies, val_accuracies, title_prefix=''):
    """Plot training and validation loss and accuracy curves."""
    plt.figure(figsize=(12, 5))
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix} Loss Curve')
    plt.legend()
    plt.grid(True)
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title_prefix} Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_results_to_excel(results_df, excel_path):
    """Save the results DataFrame to Excel."""
    if os.path.exists(excel_path):
        try:
            existing_df = pd.read_excel(excel_path, sheet_name="Results")
        except Exception:
            existing_df = pd.DataFrame(columns=results_df.columns)
        final_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        final_df = results_df

    with pd.ExcelWriter(excel_path, mode="w", engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="Results", index=False)
    print(f"Results stored successfully in Excel at {excel_path}.")

def compute_mean_vector(seq, embedding_matrix, vector_size):
    """
    Compute mean vector for a sequence using the given embedding matrix.
    If no valid tokens, returns a zero vector.
    """
    valid_vectors = [embedding_matrix[token] for token in seq if token != 0 and token < len(embedding_matrix)]
    return (sum(valid_vectors) / len(valid_vectors)) if valid_vectors else [0.0] * vector_size
