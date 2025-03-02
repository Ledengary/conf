#!/usr/bin/env python
"""
Main Script for Contrastive Learning on Last Hidden States

This script implements a contrastive learning approach to analyze model confidence,
with the following components:
1. Contrastive learning on last hidden states
2. Projection of embeddings using the trained contrastive model
3. Training a classifier on the projected embeddings
4. Training a combined model using both original and projected embeddings

The implementation is designed to be modular and flexible, with proper handling of
directory structures and class balancing.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import datetime
from collections import defaultdict
import random
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Import from our custom modules
from contrastivelearning import (
    train_contrastive_model, 
    apply_contrastive_projection,
    ContrastiveProjectionHead,
    save_model,
    load_contrastive_model,
    ContrastiveClassifierHead
)

from probes import (
    CombinedProbeModel,
    ClassifierHead,
    create_data_loaders,
    create_original_embeddings_dataloaders
)

# Add utils directory and import utilities
def setup_paths():
    """Add utility path to system path."""
    original_sys_path = sys.path.copy()
    utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils"))
    if utils_path not in sys.path:
        sys.path.append(utils_path)
    return original_sys_path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run contrastive learning experiment on last hidden states")
    
    # Dataset paths
    parser.add_argument("--wconf_dir_path", type=str, required=True, 
                        help="Path to the WConf dataset directory")
    parser.add_argument("--postfix", type=str, required=True, 
                        help="Postfix used in input file names, e.g. '_wconf', '_wconf_wid', etc.")
    parser.add_argument("--shots_dir_path", type=str, required=True, 
                        help="Path to the few-shot examples directory")
    parser.add_argument("--nshots", type=int, default=10, 
                        help="Number of shots in few-shot prompting")
    parser.add_argument("--dataset_name", type=str, required=True, 
                        help="Name of the dataset")
    parser.add_argument("--final_hidden_states_path", type=str, required=True, 
                        help="Path to the final hidden states directory")
    
    # Model settings
    parser.add_argument("--llm_id", type=str, required=True, 
                        help="ID of the language model")
    parser.add_argument("--max_seq_length", type=int, default=128, 
                        help="Maximum sequence length")
    parser.add_argument("--output_dim", type=int, default=4096,
                        help="Output dimension for contrastive model")
    
    # Experiment settings
    parser.add_argument("--goal", type=str, default="true", choices=["true", "conf"], 
                        help="Goal of the model: predicting correctness ('true') or confidence ('conf')")
    parser.add_argument("--contrastive_epochs", type=int, default=10, 
                        help="Number of epochs for contrastive learning")
    parser.add_argument("--classifier_epochs", type=int, default=10, 
                        help="Number of epochs for classifier training")
    
    # Output and environment settings
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=23, 
                        help="Random seed")
    parser.add_argument("--visible_cudas", type=str, default="0", 
                        help="CUDA devices to use")
    
    # Added arguments for additional functionality
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--run_combined_model_only", action="store_true",
                        help="Only run the combined model experiment, not the contrastive part")
    parser.add_argument("--contrastive_model_path", type=str, default=None,
                        help="Path to a pretrained contrastive model to use instead of training a new one")
    parser.add_argument("--freeze_embedded_contrastive", action="store_true",
                        help="Freeze the contrastive model when training the combined model (default: contrastive model will be fine-tuned)")
    parser.add_argument("--freeze_classifier_contrastive", action="store_true",
                        help="Freeze the contrastive model when training the classifier (default: contrastive model will be fine-tuned)")
    parser.add_argument("--use_model_for_classifier", action="store_true",
                        help="Use the contrastive model directly in classifier instead of pre-computed projections")
    parser.add_argument("--criter", type=str, default='ntxent',
                        help="Contrastive loss criterion to use (default: ntxent, other options: infonce, triplet, multi)")
    return parser.parse_args()


def evaluate_model(loader, model, device, goal="true"):
    """
    Evaluate a model on the given dataloader.
    
    Args:
        loader: DataLoader for evaluation
        model: Model to evaluate
        device: Device to run evaluation on
        goal: 'true' for correctness prediction, 'conf' for confidence prediction
        
    Returns:
        avg_loss: Average loss over the dataset
        metrics: Dictionary of evaluation metrics
        targets: Ground truth labels
        preds: Model predictions
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.BCELoss()
    count = 0
    
    with torch.no_grad():
        for embeddings, targets in tqdm(loader, desc="Evaluating", leave=False):
            embeddings = embeddings.to(device).float()
            targets = torch.as_tensor(targets, device=device, dtype=torch.float)
            preds = model(embeddings)
            loss = criterion(preds, targets)
            total_loss += loss.item() * targets.size(0)
            count += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = total_loss / count
    
    # Import compute_metrics at runtime to avoid circular imports
    from general import compute_metrics
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds), goal=goal)
    return avg_loss, metrics, np.array(all_targets), np.array(all_preds)


def train_classifier(train_loader, val_loader, test_loader, model, device, 
                    goal='true', epochs=10, lr=1e-4, save_dir=None):
    """
    Train a classifier model.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        model: Model to train
        device: Device to train on
        goal: 'true' for correctness prediction, 'conf' for confidence prediction
        epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save model checkpoints
        
    Returns:
        model: Trained model
        metrics_history: Dictionary of metrics history during training
        final_test_metrics: Final metrics on the test set
        test_preds_final: Final predictions on the test set
        test_targets_final: Ground truth labels for the test set
    """
    model.to(device)
    
    # Create loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Setup for model checkpoints
    best_model_path = os.path.join(save_dir, "classifier_best.pt") if save_dir else None
    best_val_loss = float('inf')
    
    # Initialize metrics history
    metrics_history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "BCE": {"val": [], "test": []},
        "Brier": {"val": [], "test": []},
        "ECE": {"val": [], "test": []},
        "ECE Manual": {"val": [], "test": []},
    }
    
    if goal == 'true':
        metrics_history.update({
            "accuracy": {"val": [], "test": []},
            "precision": {"val": [], "test": []},
            "recall": {"val": [], "test": []},
            "f1": {"val": [], "test": []},
            "aucroc": {"val": [], "test": []},
            "aucpr": {"val": [], "test": []}
        })
    
    print(f"Training classifier for {epochs} epochs")
    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1}/{epochs} =====")
        
        # Training
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for embeddings, targets in tqdm(train_loader, desc="Training", leave=False):
            embeddings = embeddings.to(device).float()
            targets = targets.to(device).float()
            
            outputs = model(embeddings)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * embeddings.size(0)
            train_count += embeddings.size(0)
        
        train_loss /= train_count
        print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")
        
        # Evaluate on validation set
        val_loss, val_metrics, val_targets, val_preds = evaluate_model(val_loader, model, device, goal)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
        print("Validation Metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Evaluate on test set
        test_loss, test_metrics, test_targets_epoch, test_preds_epoch = evaluate_model(test_loader, model, device, goal)
        print(f"Epoch {epoch+1} Test Loss: {test_loss:.4f}")
        
        # Log losses
        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["test_loss"].append(test_loss)
        
        # Log additional metrics (BCE loss is taken as avg_loss here)
        metrics_history["BCE"]["val"].append(val_loss)
        metrics_history["BCE"]["test"].append(test_loss)
        metrics_history["Brier"]["val"].append(val_metrics["Brier Score"])
        metrics_history["Brier"]["test"].append(test_metrics["Brier Score"])
        metrics_history["ECE"]["val"].append(val_metrics["ECE"])
        metrics_history["ECE"]["test"].append(test_metrics["ECE"])
        metrics_history["ECE Manual"]["val"].append(val_metrics["ECE Manual"])
        metrics_history["ECE Manual"]["test"].append(test_metrics["ECE Manual"])
        
        # For goal "true", log additional classification metrics
        if goal == 'true':
            metrics_history["accuracy"]["val"].append(val_metrics["accuracy"])
            metrics_history["accuracy"]["test"].append(test_metrics["accuracy"])
            metrics_history["precision"]["val"].append(val_metrics["precision"])
            metrics_history["precision"]["test"].append(test_metrics["precision"])
            metrics_history["recall"]["val"].append(val_metrics["recall"])
            metrics_history["recall"]["test"].append(test_metrics["recall"])
            metrics_history["f1"]["val"].append(val_metrics["f1"])
            metrics_history["f1"]["test"].append(test_metrics["f1"])
            metrics_history["aucroc"]["val"].append(val_metrics["aucroc"])
            metrics_history["aucroc"]["test"].append(test_metrics["aucroc"])
            metrics_history["aucpr"]["val"].append(val_metrics["aucpr"])
            metrics_history["aucpr"]["test"].append(test_metrics["aucpr"])
        
        # Update best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_model_path:
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model updated at epoch {epoch+1}")
    
    # Load best model for final evaluation
    print("Loading best model for test evaluation...")
    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation
    final_test_loss, final_test_metrics, test_targets_final, test_preds_final = evaluate_model(test_loader, model, device, goal)
    print(f"Final Test Loss: {final_test_loss:.4f}")
    print("Final Test Metrics:")
    for key, value in final_test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save predictions and targets
    if save_dir:
        preds_dir = os.path.join(save_dir, "predictions")
        os.makedirs(preds_dir, exist_ok=True)
        np.save(os.path.join(preds_dir, "test_targets.npy"), test_targets_final)
        np.save(os.path.join(preds_dir, "test_preds.npy"), test_preds_final)
    
    # Plot metrics
    if save_dir:
        # Import at runtime to avoid circular imports
        from general import plot_all_metrics
        plot_all_metrics(metrics_history, save_dir, epochs, goal=goal)
    
    return model, metrics_history, final_test_metrics, test_preds_final, test_targets_final


def visualize_embeddings(embeddings, labels, output_path, title="t-SNE Visualization of Embeddings", max_samples=5000):
    """
    Create a t-SNE visualization of embeddings colored by labels.
    
    Args:
        embeddings: Dictionary of embeddings or list of embedding tensors
        labels: Dictionary of labels or list of label values
        output_path: Path to save the visualization
        title: Title for the plot
        max_samples: Maximum number of samples to use for t-SNE to avoid memory issues
    """
    # Convert dictionaries to lists if necessary
    if isinstance(embeddings, dict):
        emb_list = []
        label_list = []
        for key in embeddings:
            emb_list.append(embeddings[key].numpy() if isinstance(embeddings[key], torch.Tensor) else embeddings[key])
            label_list.append(labels[key])
        embeddings = emb_list
        labels = label_list
    
    # Convert torch tensors to numpy if needed
    embeddings = [e.numpy() if isinstance(e, torch.Tensor) else e for e in embeddings]
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Check if we need to subsample
    total_samples = len(embeddings)
    if total_samples > max_samples:
        print(f"Subsampling from {total_samples} to {max_samples} for t-SNE visualization")
        # Create stratified sample if possible
        try:
            from sklearn.model_selection import train_test_split
            # For stratified sampling we need to convert float labels to categories
            # Group similar values together for stratification
            if labels.dtype == np.float64 or labels.dtype == np.float32:
                # Convert to categorical for stratification
                bins = 10  # Adjust based on your data
                categorical_labels = np.digitize(labels, np.linspace(min(labels), max(labels), bins))
            else:
                categorical_labels = labels
                
            # Take stratified sample
            indices = np.arange(total_samples)
            _, _, _, sample_indices = train_test_split(
                embeddings, indices, 
                test_size=max_samples/total_samples,
                stratify=categorical_labels,
                random_state=42
            )
            
            # Use the sampled indices
            embeddings = embeddings[sample_indices]
            labels = labels[sample_indices]
        except:
            # Fall back to random sampling if stratified fails
            indices = np.random.choice(total_samples, max_samples, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]
    
    # Apply t-SNE with parameters suitable for the dataset size
    print(f"Computing t-SNE on {len(embeddings)} samples...")
    from sklearn.manifold import TSNE
    
    # Adjust perplexity based on sample size (rule of thumb: sqrt(n)/3)
    perplexity = min(30, int(np.sqrt(len(embeddings))/3))
    perplexity = max(5, perplexity)  # Ensure perplexity is at least 5
    
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        learning_rate='auto',  # Use 'auto' instead of a fixed value
        n_iter=1000,
        n_jobs=-1  # Use all available processors
    )
    
    try:
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Label')
        plt.title(f"{title}\n({len(embeddings)} samples)")
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {output_path}")
        
        # Also create a version with discrete bins for clearer visualization
        if labels.dtype == np.float64 or labels.dtype == np.float32:
            plt.figure(figsize=(10, 8))
            # Create 5 bins
            bins = 5
            binned_labels = np.digitize(labels, np.linspace(min(labels), max(labels), bins))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=binned_labels, cmap='viridis', alpha=0.7)
            cbar = plt.colorbar(scatter, ticks=range(1, bins+1))
            bin_edges = np.linspace(min(labels), max(labels), bins)
            bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(bins-1)]
            cbar.set_ticklabels(bin_labels)
            plt.title(f"{title} (Binned values)\n({len(embeddings)} samples)")
            plt.xlabel('t-SNE dimension 1')
            plt.ylabel('t-SNE dimension 2')
            binned_path = output_path.replace('.png', '_binned.png')
            plt.savefig(binned_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Binned visualization saved to {binned_path}")
            
    except Exception as e:
        print(f"Error computing t-SNE: {e}")
        print("Saving raw embedding PCA visualization instead")
        
        # Try PCA as a fallback
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Label')
            plt.title(f"{title} (PCA)\n({len(embeddings)} samples)")
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            pca_path = output_path.replace('.png', '_pca.png')
            plt.savefig(pca_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"PCA visualization saved to {pca_path}")
        except Exception as e2:
            print(f"Error computing PCA: {e2}")
            print("Unable to generate visualization")


def run_combined_model_experiment(train_records, val_records, test_records, 
                                 contrastive_model_path, embeddings_dir, 
                                 device, output_dir, goal='true', seed=23,
                                 classifier_epochs=10, batch_size=64, lr=1e-4,
                                 freeze_contrastive=False, output_dim=4096):
    """
    Run the experiment with the combined model using both original embeddings
    and contrastive projections.
    
    Args:
        train_records: List of training records
        val_records: List of validation records
        test_records: List of testing records
        contrastive_model_path: Path to the saved contrastive model
        embeddings_dir: Directory containing embedding files
        device: Device to run on
        output_dir: Output directory for saving results
        goal: 'true' for correctness prediction, 'conf' for confidence prediction
        seed: Random seed
        classifier_epochs: Number of epochs for training the classifier
        batch_size: Batch size for training
        lr: Learning rate for training
        freeze_contrastive: Whether to freeze the contrastive model weights
        
    Returns:
        model: Trained combined model
        metrics: Final metrics on the test set
    """
    # Set random seed
    from general import seed_everything
    seed_everything(seed)
    
    # Setup directories
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    experiment_dir = os.path.join(output_dir, f"combined_model_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Load the trained contrastive model
    # First need to get the input dimension
    first_id = train_records[0]["run_hash_id"]
    first_emb_path = os.path.join(embeddings_dir, "train", f"{first_id}.pt")
    first_emb = torch.load(first_emb_path, map_location="cpu")
    input_dim = first_emb.shape[0]
    
    # Determine output dimension from the contrastive model
    contrastive_model = ContrastiveProjectionHead(input_dim, output_dim)
    contrastive_model.load_state_dict(torch.load(contrastive_model_path))
    contrastive_model.to(device)
    print('contrastive_model architecture:', contrastive_model)
    
    # Create the combined model with the specified freeze setting
    combined_model = CombinedProbeModel(contrastive_model, input_dim, freeze_contrastive=freeze_contrastive)
    combined_model.to(device)
    print('combined_model architecture:', combined_model)
    
    # Create data loaders using original embeddings
    train_loader, val_loader, test_loader = create_original_embeddings_dataloaders(
        train_records, val_records, test_records,
        embeddings_dir, goal, batch_size
    )
    
    # Train the combined model
    model, metrics_history, final_metrics, test_preds, test_targets = train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=combined_model,
        device=device,
        goal=goal,
        epochs=classifier_epochs,
        lr=lr,
        save_dir=experiment_dir
    )
    
    # Save results
    results_dir = os.path.join(experiment_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "final_metrics.json"), "w") as f:
        # Convert numpy values to Python native types for JSON serialization
        serialized_metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in final_metrics.items()}
        json.dump(serialized_metrics, f, indent=2)
    
    with open(os.path.join(results_dir, "config.txt"), "w") as f:
        config = {
            "goal": goal,
            "classifier_epochs": classifier_epochs,
            "seed": seed,
            "model_type": "combined",
            "freeze_contrastive": freeze_contrastive,
            "train_records": len(train_records),
            "val_records": len(val_records),
            "test_records": len(test_records),
            "learning_rate": lr,
            "output_dim": output_dim
        }
        f.write(str(config))
    
    return model, final_metrics


def run_contrastive_experiment(train_records, val_records, test_records, 
                              final_hidden_states_path, tokenizer, device, 
                              output_dir, goal='true', seed=23,
                              contrastive_epochs=10, classifier_epochs=10,
                              batch_size=64, lr=1e-4, contrastive_model_path=None,
                              freeze_contrastive=False, output_dim=4096,
                              freeze_classifier_contrastive=True,
                              use_model_for_classifier=False, criter='ntxent'):
    """
    Run the full contrastive learning experiment.
    
    1. Train contrastive model on train records (or load pre-trained model)
    2. Project all embeddings using the trained model or use model directly
    3. Train classifier on projected embeddings or with contrastive model
    4. Train combined model using original and projected embeddings
    5. Evaluate and save results
    
    Args:
        train_records: List of training records
        val_records: List of validation records
        test_records: List of testing records
        final_hidden_states_path: Path to the final hidden states directory
        tokenizer: Tokenizer instance with embedding dimension
        device: Device to run on
        output_dir: Output directory for saving results
        goal: 'true' for correctness prediction, 'conf' for confidence prediction
        seed: Random seed
        contrastive_epochs: Number of epochs for contrastive learning
        classifier_epochs: Number of epochs for classifier training
        batch_size: Batch size for training
        lr: Learning rate for training
        contrastive_model_path: Optional path to a pre-trained contrastive model
        freeze_contrastive: Whether to freeze the contrastive model in the combined model
        freeze_classifier_contrastive: Whether to freeze the contrastive model in the classifier
        use_model_for_classifier: Whether to use the contrastive model directly in classifier
        criter: Contrastive loss criterion to use
        
    Returns:
        contrastive_model: Trained contrastive model
        classifier_model: Trained classifier model
        final_metrics: Final metrics on the test set
        experiment_dir: Directory where results are saved
    """
    # Set random seed
    from general import seed_everything
    seed_everything(seed)
    
    # Get the tokenizer embedding dimension
    print(f"Tokenizer embedding dimension: {output_dim}")
    
    # Setup directories for saving artifacts
    model_name = getattr(tokenizer, 'name_or_path', 'model').replace("/", "-")
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    experiment_dir = os.path.join(output_dir, model_name, goal, f"contrastive_exp_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    contrastive_dir = os.path.join(experiment_dir, "contrastive")
    os.makedirs(contrastive_dir, exist_ok=True)
    classifier_dir = os.path.join(experiment_dir, "classifier")
    os.makedirs(classifier_dir, exist_ok=True)
    
    # Save experiment configuration
    with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
        config = {
            "goal": goal,
            "contrastive_epochs": contrastive_epochs,
            "classifier_epochs": classifier_epochs,
            "output_dim": output_dim,
            "seed": seed,
            "train_records": len(train_records),
            "val_records": len(val_records),
            "test_records": len(test_records),
            "batch_size": batch_size,
            "learning_rate": lr,
            "pre_trained_model": contrastive_model_path is not None,
            "freeze_contrastive": freeze_contrastive,
            "freeze_classifier_contrastive": freeze_classifier_contrastive,
            "use_model_for_classifier": use_model_for_classifier
        }
        f.write(str(config))
    
    # 1. Train or load contrastive model
    if contrastive_model_path:
        print(f"Loading pre-trained contrastive model from {contrastive_model_path}")
        # Get input dimension from first embedding
        first_id = train_records[0]["run_hash_id"]
        first_emb_path = os.path.join(final_hidden_states_path, "train", f"{first_id}.pt")
        first_emb = torch.load(first_emb_path, map_location="cpu")
        input_dim = first_emb.shape[0]
        
        contrastive_model = load_contrastive_model(input_dim, output_dim, contrastive_model_path, device)
    else:
        print("Training new contrastive model")
        contrastive_model = train_contrastive_model(
            train_records=train_records,
            val_records=val_records,
            embeddings_dir=final_hidden_states_path,
            output_dim=output_dim,
            device=device,
            goal=goal,
            oversample=True,  
            epochs=contrastive_epochs,
            batch_size=batch_size,
            lr=lr,
            save_dir=contrastive_dir,
            criter=criter
        )
        
        # Save final contrastive model using the utility function
        contrastive_model_path_final = os.path.join(contrastive_dir, "contrastive_model_final.pt")
        save_model(contrastive_model, contrastive_model_path_final)
    
    # load the best contrastive model
    contrastive_model_path_best = os.path.join(contrastive_dir, "contrastive_model_best.pt")
    contrastive_model.load_state_dict(torch.load(contrastive_model_path_best))

    # 2. If using pre-computed projections, project and save all embeddings
    projected_embeddings = None
    if not use_model_for_classifier:
        print("Projecting embeddings for classifier...")
        all_records = train_records + val_records + test_records
        
        # Create a function to get the appropriate embeddings directory for each record
        def get_embeddings_dir(record):
            if record in train_records:
                return os.path.join(final_hidden_states_path, "train")
            elif record in val_records:
                return os.path.join(final_hidden_states_path, "val")
            else:
                return os.path.join(final_hidden_states_path, "test")
        
        projected_embeddings = apply_contrastive_projection(
            records=all_records,
            embeddings_dir=get_embeddings_dir,
            model=contrastive_model,
            device=device
        )
        
        # Save projected embeddings
        print("Saving projected embeddings...")
        projected_dir = os.path.join(experiment_dir, "projected_embeddings")
        os.makedirs(projected_dir, exist_ok=True)
        for run_id, embedding in tqdm(projected_embeddings.items()):
            torch.save(embedding, os.path.join(projected_dir, f"{run_id}.pt"))
    
    # Create label dictionary for visualization
    all_records = train_records + val_records + test_records
    label_dict = {}
    for record in all_records:
        run_id = record["run_hash_id"]
        if goal == 'conf':
            label_dict[run_id] = float(record["empirical_confidence"])
        else:
            label_dict[run_id] = float(record["correct_or_not"])
    
    # Visualize embeddings only if projections are computed
    if projected_embeddings:
        print("Creating t-SNE visualization of projected embeddings...")
        vis_dir = os.path.join(experiment_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        visualize_embeddings(
            projected_embeddings, 
            label_dict, 
            os.path.join(vis_dir, "projected_embeddings_tsne.png"),
            f"t-SNE Visualization of Projected Embeddings (goal: {goal})"
        )
    
    # 3. Create dataloaders and train classifier
    print("Training classifier...")
    if use_model_for_classifier:
        print(f"Using contrastive model directly in classifier (freeze_contrastive={freeze_classifier_contrastive})")
        # Create ContrastiveClassifierHead with contrastive model
        classifier_model = ContrastiveClassifierHead(
            contrastive_model=contrastive_model,
            output_dim=output_dim,
            freeze_contrastive=freeze_classifier_contrastive
        )
        print('classifier_model architecture:', classifier_model)
        
        # Create data loaders using original embeddings
        train_loader, val_loader, test_loader = create_original_embeddings_dataloaders(
            train_records, val_records, test_records,
            final_hidden_states_path, goal, batch_size
        )
    else:
        print("Using pre-computed projections for classifier")
        # Create a simple classifier head
        classifier_model = ClassifierHead(output_dim)
        print('classifier_model architecture:', classifier_model)
        
        # Create dataloaders for projected embeddings
        train_loader, val_loader, test_loader = create_data_loaders(
            train_records, val_records, test_records,
            projected_embeddings, goal, batch_size
        )
    
    # Train the classifier
    classifier_model, metrics_history, final_metrics, test_preds, test_targets = train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=classifier_model,
        device=device,
        goal=goal,
        epochs=classifier_epochs,
        lr=lr,
        save_dir=classifier_dir
    )
    
    # 4. Save results
    results_dir = os.path.join(classifier_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics history
    with open(os.path.join(results_dir, "metrics_history.json"), "w") as f:
        # Handle nested dictionaries and convert numpy values to Python native types
        serialized_history = {}
        for k, v in metrics_history.items():
            if isinstance(v, dict):
                serialized_history[k] = {k2: [float(i) for i in v2] for k2, v2 in v.items()}
            else:
                serialized_history[k] = [float(i) for i in v]
        json.dump(serialized_history, f, indent=2)
    
    # Save final metrics
    with open(os.path.join(results_dir, "final_metrics.json"), "w") as f:
        # Convert numpy values to Python native types
        serialized_metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in final_metrics.items()}
        json.dump(serialized_metrics, f, indent=2)
    
    # 5. Train the combined model
    print("Training combined model...")
    combined_model_path = os.path.join(experiment_dir, "combined_model")
    os.makedirs(combined_model_path, exist_ok=True)
    
    combined_model, combined_metrics = run_combined_model_experiment(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        contrastive_model_path=contrastive_model_path_best,
        embeddings_dir=final_hidden_states_path,
        device=device,
        output_dir=combined_model_path,
        goal=goal,
        seed=seed,
        classifier_epochs=classifier_epochs,
        batch_size=batch_size,
        lr=lr,
        freeze_contrastive=freeze_contrastive,
        output_dim=output_dim
    )
    
    # Save comparison of contrastive-only vs combined model metrics
    with open(os.path.join(results_dir, "model_comparison.json"), "w") as f:
        comparison = {
            "contrastive_only": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in final_metrics.items()},
            "combined_model": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in combined_metrics.items()}
        }
        json.dump(comparison, f, indent=2)
    
    print("Experiment completed successfully!")
    print(f"Contrastive classifier metrics: {final_metrics}")
    print(f"Combined model metrics: {combined_metrics}")
    print(f"Results saved in {experiment_dir}")
    
    return contrastive_model, classifier_model, final_metrics, experiment_dir


def main():
    """Main function to run the contrastive learning experiment."""
    # Parse arguments
    args = parse_arguments()
    
    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cudas
    
    # Set up paths and import utilities
    original_sys_path = setup_paths()
    
    # Import utilities now that path is set up
    from general import (
        load_wconf_dataset, 
        load_few_shot_examples, 
        load_system_prompt,
        flatten_dataset, 
        assert_flatten_dataset, 
        initialize_tokenizer, 
        seed_everything
    )
    
    # Set random seed
    seed_everything(args.seed)
    print(f"Random seed set to {args.seed}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset and prepare data
    print("Loading dataset...")
    dataset = load_wconf_dataset(args.wconf_dir_path, args.postfix)
    
    # Initialize tokenizer
    print(f"Initializing tokenizer: {args.llm_id}")
    tokenizer = initialize_tokenizer(args.llm_id, args.max_seq_length)
    
    # Load few-shot examples and system prompt
    few_shot_prompt = load_few_shot_examples(args.shots_dir_path, args.nshots)
    system_prompt = load_system_prompt(args.dataset_name)
    
    # Flatten dataset
    print("Preparing dataset...")
    dataset_flat = flatten_dataset(dataset, few_shot_prompt, system_prompt, tokenizer, goal=args.goal)
    assert_flatten_dataset(dataset_flat, tokenizer)
    
    # Extract records
    train_records = dataset_flat["train"]
    val_records = dataset_flat["val"]
    test_records = dataset_flat["test"]
    
    print(f"Dataset loaded: {len(train_records)} train, {len(val_records)} val, {len(test_records)} test")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine experiment flow based on arguments
    if args.run_combined_model_only and args.contrastive_model_path:
        print(f"Running combined model only with pre-trained contrastive model: {args.contrastive_model_path}")
        print(f"Contrastive model will be {'frozen' if args.freeze_embedded_contrastive else 'fine-tuned'} during training")
        
        combined_model, combined_metrics = run_combined_model_experiment(
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            contrastive_model_path=args.contrastive_model_path,
            embeddings_dir=args.final_hidden_states_path,
            device=device,
            output_dir=args.output_dir,
            goal=args.goal,
            seed=args.seed,
            classifier_epochs=args.classifier_epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            freeze_contrastive=args.freeze_embedded_contrastive,
            output_dim=args.output_dim
        )
        print("Combined model experiment completed.")
        print(f"Final metrics: {combined_metrics}")
    else:
        # Run the full contrastive experiment
        print("Running complete contrastive learning experiment...")
        print(f"In combined model, contrastive model will be {'frozen' if args.freeze_embedded_contrastive else 'fine-tuned'} during training")
        print(f"In classifier, contrastive model will be {'frozen' if args.freeze_classifier_contrastive else 'fine-tuned'} during training")
        print(f"Classifier will {'use contrastive model directly' if args.use_model_for_classifier else 'use pre-computed projections'}")
        
        contrastive_model, classifier_model, final_metrics, experiment_dir = run_contrastive_experiment(
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            final_hidden_states_path=args.final_hidden_states_path,
            tokenizer=tokenizer,
            device=device,
            output_dir=args.output_dir,
            goal=args.goal,
            seed=args.seed,
            contrastive_epochs=args.contrastive_epochs,
            classifier_epochs=args.classifier_epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            contrastive_model_path=args.contrastive_model_path,
            freeze_contrastive=args.freeze_embedded_contrastive,
            freeze_classifier_contrastive=args.freeze_classifier_contrastive,
            use_model_for_classifier=args.use_model_for_classifier,
            output_dim=args.output_dim,
            criter=args.criter
        )
        
        print(f"Experiment completed. Results saved to {experiment_dir}")
        print("Final metrics:", final_metrics)
    
    # Restore original system path
    sys.path = original_sys_path


if __name__ == "__main__":
    main()