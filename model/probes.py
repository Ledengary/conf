"""
Probe Model Module

This module implements the classifier models for probing embeddings,
including the combined model that uses both original embeddings and 
contrastive projections.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ProjectedEmbeddingDataset(Dataset):
    """Dataset for classifier using projected embeddings."""
    def __init__(self, records, projected_embeddings, goal):
        self.records = records
        self.projected_embeddings = projected_embeddings
        self.goal = goal
        self.targets = []
        
        for rec in self.records:
            if self.goal == 'conf':
                self.targets.append(float(rec["empirical_confidence"]))
            else:
                self.targets.append(float(rec["correct_or_not"]))
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        run_id = self.records[idx]["run_hash_id"]
        return self.projected_embeddings[run_id], self.targets[idx]


class ClassifierHead(nn.Module):
    """Simple classifier head with a sigmoid output."""
    def __init__(self, input_dim, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
            
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x).squeeze(-1)


class CombinedProbeModel(nn.Module):
    """
    Combined model that uses both original embeddings and contrastive projections.
    This model takes original embeddings, applies contrastive projection, and then
    combines both for final prediction.
    """
    def __init__(self, contrastive_model, input_dim, freeze_contrastive=False, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        self.contrastive_model = contrastive_model
        
        # Freeze the contrastive model parameters if requested
        if freeze_contrastive:
            for param in self.contrastive_model.parameters():
                param.requires_grad = False
            print("Contrastive model frozen - weights will not be updated during training")
        else:
            print("Contrastive model unfrozen - weights will be fine-tuned during training")
            
        # Get contrastive output dimension from the model
        contrastive_output_dim = self._get_contrastive_output_dim()
        
        # Combined dimension is the original input plus the contrastive projection
        combined_dim = input_dim + contrastive_output_dim
        
        if hidden_dim is None:
            hidden_dim = combined_dim // 2
        
        # Classifier on top of the combined features
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def _get_contrastive_output_dim(self):
        """Helper method to determine contrastive model output dimension"""
        # Find the last layer of the projection head
        last_layer = list(self.contrastive_model.projection.children())[-1]
        if isinstance(last_layer, nn.Linear):
            return last_layer.out_features
        else:
            # If not found, use a default
            return 768  # Common embedding size
    
    def forward(self, x):
        """
        Forward pass through the combined model.
        
        Args:
            x: Original input embedding
            
        Returns:
            Probability prediction
        """
        # Get contrastive projection - use no_grad only if contrastive model is frozen
        if next(self.contrastive_model.parameters()).requires_grad:
            # Model is being trained, use regular forward pass
            contrastive_proj = self.contrastive_model(x)
        else:
            # Model is frozen, use no_grad to save memory
            with torch.no_grad():
                contrastive_proj = self.contrastive_model(x)
        
        # Concatenate original embedding and contrastive projection
        combined = torch.cat([x, contrastive_proj], dim=1)
        
        # Apply classifier
        return self.classifier(combined).squeeze(-1)


class OriginalEmbeddingDataset(Dataset):
    """Dataset for classifier using original embeddings."""
    def __init__(self, records, embeddings_dir, goal):
        self.records = records
        self.embeddings_dir = embeddings_dir
        self.goal = goal
        self.run_ids = [rec["run_hash_id"] for rec in records]
        self.targets = []
        
        for rec in self.records:
            if self.goal == 'conf':
                self.targets.append(float(rec["empirical_confidence"]))
            else:
                self.targets.append(float(rec["correct_or_not"]))
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        run_id = self.run_ids[idx]
        emb_path = os.path.join(self.embeddings_dir, f"{run_id}.pt")
        embedding = torch.load(emb_path, map_location="cpu", weights_only=True)
        return embedding, self.targets[idx]


def get_embeddings_dir(record, embeddings_dir, train_records, val_records, test_records):
    """
    Determine the appropriate embeddings directory for a record based on its split.
    
    Args:
        record: Record to get directory for
        embeddings_dir: Base embeddings directory
        train_records: List of training records
        val_records: List of validation records
        test_records: List of testing records
        
    Returns:
        Path to the appropriate embeddings directory
    """
    if record in train_records:
        return os.path.join(embeddings_dir, "train")
    elif record in val_records:
        return os.path.join(embeddings_dir, "val")
    else:
        return os.path.join(embeddings_dir, "test")


class OriginalEmbeddingDatasetWithDirFunc(Dataset):
    """Dataset for original embeddings with a function to determine embeddings directory."""
    def __init__(self, records, get_dir_func, goal):
        self.records = records
        self.get_dir_func = get_dir_func
        self.goal = goal
        
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        run_id = rec["run_hash_id"]
        
        # Get embedding directory for this record
        emb_dir = self.get_dir_func(rec)
        emb_path = os.path.join(emb_dir, f"{run_id}.pt")
        embedding = torch.load(emb_path, map_location="cpu", weights_only=True)
        
        # Get target
        if self.goal == 'conf':
            target = float(rec["empirical_confidence"])
        else:
            target = float(rec["correct_or_not"])
            
        return embedding, target


def create_function_embeddings_dir(embeddings_dir, train_records, val_records, test_records):
    """
    Create a function that determines the appropriate embeddings directory for a record.
    
    Args:
        embeddings_dir: Base embeddings directory
        train_records: List of training records
        val_records: List of validation records
        test_records: List of testing records
        
    Returns:
        Function that takes a record and returns the appropriate embeddings directory
    """
    def get_dir(record):
        return get_embeddings_dir(record, embeddings_dir, train_records, val_records, test_records)
    
    return get_dir


def create_original_embeddings_dataloaders(train_records, val_records, test_records,
                                         embeddings_dir, goal='true', batch_size=64, num_workers=4):
    """
    Create DataLoaders for original embeddings, handling the directory structure correctly.
    
    Args:
        train_records: List of training records
        val_records: List of validation records
        test_records: List of testing records
        embeddings_dir: Base directory for embeddings
        goal: 'true' for correctness prediction, 'conf' for confidence prediction
        batch_size: Batch size for the DataLoaders
        num_workers: Number of worker processes for the DataLoaders
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create function to determine embeddings directory for each record
    get_dir = create_function_embeddings_dir(embeddings_dir, train_records, val_records, test_records)
    
    # Create datasets
    train_dataset = OriginalEmbeddingDatasetWithDirFunc(train_records, get_dir, goal)
    val_dataset = OriginalEmbeddingDatasetWithDirFunc(val_records, get_dir, goal)
    test_dataset = OriginalEmbeddingDatasetWithDirFunc(test_records, get_dir, goal)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def create_data_loaders(train_records, val_records, test_records, data_source, 
                       goal='true', batch_size=64, num_workers=4):
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_records: List of training records
        val_records: List of validation records
        test_records: List of testing records
        data_source: Either projected_embeddings dict or embeddings_dir string
        goal: 'true' for correctness prediction, 'conf' for confidence prediction
        batch_size: Batch size for the DataLoaders
        num_workers: Number of worker processes for the DataLoaders
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Check if we're using projected embeddings or original embeddings
    if isinstance(data_source, dict):
        # Using projected embeddings
        train_dataset = ProjectedEmbeddingDataset(train_records, data_source, goal)
        val_dataset = ProjectedEmbeddingDataset(val_records, data_source, goal)
        test_dataset = ProjectedEmbeddingDataset(test_records, data_source, goal)
    else:
        # Using original embeddings directory - handle directory structure
        return create_original_embeddings_dataloaders(
            train_records, val_records, test_records,
            data_source, goal, batch_size, num_workers
        )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader