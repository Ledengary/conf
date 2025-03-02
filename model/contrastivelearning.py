"""
Contrastive Learning Module

This module provides functionality for training a contrastive learning model,
applying the contrastive projections, and managing the datasets required for training.

The module separates the contrastive learning logic from the main workflow, allowing
for better code organization and reusability.

Enhanced with:
1. Hard Negative Mining
2. Deeper Projection Head with Layer Normalization
3. Question-Aware Contrastive Learning
4. Supervised Contrastive Learning
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from collections import defaultdict


class ContrastiveProjectionHead(nn.Module):
    """
    Neural network model for projecting embeddings into a contrastive learning space.
    Enhanced with deeper architecture and layer normalization.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None, deeper=True):
        """
        Initialize the contrastive projection head.
        
        Args:
            input_dim: Dimension of the input embeddings
            output_dim: Dimension of the output projections
            hidden_dim: Dimension of the hidden layer (defaults to input_dim // 2)
            deeper: Whether to use the deeper architecture with layer norm
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        if deeper:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim),
            )
        else:
            # Original simpler architecture
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        
    def forward(self, x):
        """
        Forward pass through the projection head.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Projected tensor of shape [batch_size, output_dim]
        """
        return self.projection(x)


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (Information Noise-Contrastive Estimation Loss)
    A more numerically stable and efficient alternative to NT-Xent loss.
    
    This implementation applies temperature scaling and supports both
    standard and hard negative mining approaches.
    """
    def __init__(self, temperature=0.07, class_weights=None, use_hard_negatives=True, n_hard_negatives=10):
        """
        Initialize the InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for scaling similarities (default: 0.07)
            class_weights: Optional weights for different classes
            use_hard_negatives: Whether to use hard negative mining
            n_hard_negatives: Number of hard negatives to select per sample
        """
        super().__init__()
        self.temperature = temperature
        self.class_weights = class_weights
        self.use_hard_negatives = use_hard_negatives
        self.n_hard_negatives = n_hard_negatives
        self.eps = 1e-8  # Small epsilon to prevent numerical instability
            
    def forward(self, z_i, z_j, labels=None):
        """
        Compute the InfoNCE loss.
        
        Args:
            z_i: First set of normalized embeddings [batch_size, embed_dim]
            z_j: Second set of normalized embeddings [batch_size, embed_dim]
            labels: Optional class labels (not used in standard InfoNCE)
            
        Returns:
            InfoNCE loss value
        """
        batch_size = z_i.size(0)
        device = z_i.device
        
        # Apply L2 normalization
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Combine all embeddings
        z_all = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, embed_dim]
        
        # Compute similarity matrix (cosine similarity)
        sim_matrix = torch.matmul(z_all, z_all.T) / self.temperature  # [2*batch_size, 2*batch_size]
        
        # Create mask for positive pairs
        pos_mask = torch.zeros((2*batch_size, 2*batch_size), dtype=torch.bool, device=device)
        
        # Mark positive pairs: (i,j+batch_size) and (j+batch_size,i)
        for i in range(batch_size):
            pos_mask[i, i+batch_size] = True
            pos_mask[i+batch_size, i] = True
        
        # Create identity mask to exclude self-comparisons
        identity_mask = torch.eye(2*batch_size, dtype=torch.bool, device=device)
        
        # Mask for all valid negatives (exclude self-comparisons and positive pairs)
        neg_mask = ~(pos_mask | identity_mask)
        
        # If using hard negative mining, select top-k most similar negatives
        if self.use_hard_negatives:
            # Extract positive similarities first - shape is [2*batch_size]
            pos_sim = []
            for i in range(2*batch_size):
                # Get the positive similarity for this sample
                if i < batch_size:
                    # For first half of batch, positive is at i+batch_size
                    pos_idx = i + batch_size
                else:
                    # For second half of batch, positive is at i-batch_size
                    pos_idx = i - batch_size
                pos_sim.append(sim_matrix[i, pos_idx])
            
            # Convert to tensor
            pos_sim = torch.stack(pos_sim)
            
            # Hard negative selection
            hard_neg_sim = []
            
            for i in range(2*batch_size):
                # Get negative similarities for this sample
                sample_neg_sim = sim_matrix[i][neg_mask[i]]
                
                # Find the hardest negatives (most similar)
                k = min(self.n_hard_negatives, sample_neg_sim.size(0))
                if k > 0:
                    # Get top-k hardest negatives
                    _, hard_indices = torch.topk(sample_neg_sim, k)
                    hard_neg_sim.append(sample_neg_sim[hard_indices])
                else:
                    # No valid negatives (unlikely but handle this edge case)
                    hard_neg_sim.append(torch.tensor([], device=device))
            
            # Compute loss for each sample
            losses = []
            
            for i in range(2*batch_size):
                if hard_neg_sim[i].size(0) > 0:
                    # Calculate log softmax: log(exp(pos) / (exp(pos) + sum(exp(neg))))
                    # Make sure both tensors have same dimension before concatenating
                    sample_logits = torch.cat([pos_sim[i].unsqueeze(0), hard_neg_sim[i]])
                    loss_i = -F.log_softmax(sample_logits, dim=0)[0]
                    losses.append(loss_i)
            
            if len(losses) > 0:
                loss = torch.stack(losses).mean()
            else:
                # Fallback if no valid pairs (should not happen in practice)
                loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # Original InfoNCE approach (without hard negative mining)
            # Extract positive similarities with proper reshaping
            pos_sim_list = []
            for i in range(2*batch_size):
                pos_indices = torch.where(pos_mask[i])[0]
                if len(pos_indices) > 0:
                    pos_sim_list.append(sim_matrix[i, pos_indices[0]].unsqueeze(0))
                else:
                    # This shouldn't happen with your pairing structure, but handle just in case
                    pos_sim_list.append(torch.tensor([float('-inf')], device=device))
            
            pos_sim = torch.cat(pos_sim_list).unsqueeze(1)  # [2*batch_size, 1]
            
            # Get all negative similarities
            neg_sim = []
            for i in range(2*batch_size):
                neg_indices = torch.where(neg_mask[i])[0]
                if len(neg_indices) > 0:
                    neg_sim.append(sim_matrix[i, neg_indices])
                else:
                    # Handle edge case with no negatives
                    neg_sim.append(torch.tensor([float('-inf')], device=device))
            
            # Pad negatives to same length
            max_neg_len = max(neg.size(0) for neg in neg_sim)
            padded_neg_sim = []
            for neg in neg_sim:
                if neg.size(0) < max_neg_len:
                    padding = torch.full((max_neg_len - neg.size(0),), float('-inf'), device=device)
                    padded_neg_sim.append(torch.cat([neg, padding]))
                else:
                    padded_neg_sim.append(neg)
            
            # Stack to get [2*batch_size, max_neg_len]
            neg_sim = torch.stack(padded_neg_sim)
            
            # Compute log softmax across all samples (positives and negatives)
            logits = torch.cat([pos_sim, neg_sim], dim=1)  # [2*batch_size, 1+max_neg_len]
            labels = torch.zeros(2*batch_size, dtype=torch.long, device=device)  # Positive is at index 0
            
            # Calculate cross-entropy loss
            loss = F.cross_entropy(logits, labels, weight=self.class_weights, reduction='mean')
        
        return loss


class MultiPositiveInfoNCELoss(nn.Module):
    """
    Extension of InfoNCE that handles multiple positives per anchor.
    This is useful for supervised contrastive learning where all samples
    from the same class are considered positives.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8
        
    def forward(self, features, labels):
        """
        Compute loss with multiple positives.
        
        Args:
            features: Feature embeddings [batch_size, feature_dim]
            labels: Class labels [batch_size]
            
        Returns:
            Loss value
        """
        batch_size = features.size(0)
        device = features.device
        
        # L2 normalization
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positives (same class)
        pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        # Exclude self-similarity
        self_mask = torch.eye(batch_size, device=device)
        pos_mask = pos_mask - self_mask
        
        # Calculate number of positives per sample
        num_positives = pos_mask.sum(1)
        
        # Handle case where a sample has no positives of the same class
        # by adding a small epsilon to avoid division by zero
        num_positives = torch.max(num_positives, torch.ones_like(num_positives) * self.eps)
        
        # Compute log-probabilities
        exp_logits = torch.exp(sim_matrix) * (1 - self_mask)
        sum_exp_logits = exp_logits.sum(1, keepdim=True)
        
        # Compute log-prob of positives
        pos_exp_logits = exp_logits * pos_mask
        log_prob = torch.log(pos_exp_logits / sum_exp_logits + self.eps)
        
        # Compute mean of log-likelihood over positive samples
        loss = -torch.sum(log_prob * pos_mask) / torch.sum(num_positives)
        
        return loss


class TripletMarginLoss(nn.Module):
    """
    Triplet Margin Loss - another alternative for contrastive learning.
    Uses triplets of (anchor, positive, negative) samples.
    
    It aims to minimize the distance between anchor and positive,
    while maximizing the distance between anchor and negative.
    """
    def __init__(self, margin=1.0, p=2, reduction='mean'):
        """
        Args:
            margin: Margin value in the loss formula
            p: The norm degree for pairwise distance
            reduction: Reduction method: 'none', 'mean', 'sum'
        """
        super().__init__()
        self.margin = margin
        self.p = p
        self.reduction = reduction
        
    def forward(self, anchors, positives, negatives):
        """
        Compute triplet loss.
        
        Args:
            anchors: Anchor embeddings [batch_size, embed_dim]
            positives: Positive embeddings [batch_size, embed_dim]
            negatives: Negative embeddings [batch_size, embed_dim]
            
        Returns:
            Triplet loss value
        """
        # Normalize embeddings
        anchors = F.normalize(anchors, dim=1)
        positives = F.normalize(positives, dim=1)
        negatives = F.normalize(negatives, dim=1)
        
        # Compute distances
        dist_pos = torch.norm(anchors - positives, p=self.p, dim=1)
        dist_neg = torch.norm(anchors - negatives, p=self.p, dim=1)
        
        # Compute triplet loss
        losses = torch.relu(dist_pos - dist_neg + self.margin)
        
        # Apply reduction
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
        

class NTXentLoss(nn.Module):
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss)
    Used for contrastive learning.
    Enhanced with hard negative mining capability.
    """
    def __init__(self, temperature=0.5, class_weights=None, use_hard_negatives=True, n_hard_negatives=10):
        """
        Initialize the NT-Xent loss.
        
        Args:
            temperature: Temperature parameter for scaling similarities
            class_weights: Optional weights for different classes
            use_hard_negatives: Whether to use hard negative mining
            n_hard_negatives: Number of hard negatives to select per sample
        """
        super().__init__()
        self.temperature = temperature
        self.class_weights = class_weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
        self.use_hard_negatives = use_hard_negatives
        self.n_hard_negatives = n_hard_negatives
        
    def forward(self, z_i, z_j, labels=None):
        """
        Compute the NT-Xent loss with optional hard negative mining.
        
        Args:
            z_i: First set of embeddings from the projection head
            z_j: Second set of embeddings from the projection head
            labels: Optional binary labels (0 or 1) for correct/incorrect predictions
            
        Returns:
            NT-Xent loss value
        """
        batch_size = z_i.size(0)
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Compute similarity matrix
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        # Extract positive pairs
        sim_i_j = torch.diag(similarity_matrix, batch_size)  # z_i to z_j
        sim_j_i = torch.diag(similarity_matrix, -batch_size)  # z_j to z_i
        
        # Combine positives from both directions
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)
        
        # Create mask for all valid negatives (excluding self and positive pairs)
        mask_samples_from_same_repr = torch.eye(batch_size, device=z_i.device).bool()
        mask_samples_from_same_repr = torch.block_diag(
            mask_samples_from_same_repr, 
            mask_samples_from_same_repr
        )
        
        # Create mask for positive pairs
        mask_positives = torch.zeros(2*batch_size, 2*batch_size, device=z_i.device).bool()
        # Set the diagonal blocks to False (z_i's positives are in z_j and vice versa)
        mask_positives[:batch_size, batch_size:] = torch.eye(batch_size, device=z_i.device).bool()
        mask_positives[batch_size:, :batch_size] = torch.eye(batch_size, device=z_i.device).bool()
        
        # Combine masks - we want to exclude both self-comparisons and positive pairs
        mask_negatives = ~(mask_samples_from_same_repr | mask_positives)
        
        # If using hard negative mining, select top-k most similar negatives
        if self.use_hard_negatives:
            # Get negative similarities
            neg_similarities = similarity_matrix.masked_fill(~mask_negatives, float('-inf'))
            
            # For each sample, find the hardest negatives (most similar)
            hard_neg_values = []
            for i in range(2*batch_size):
                # Get negatives for this sample
                sample_negs = neg_similarities[i]
                
                # Find the hardest negatives (most similar)
                hard_indices = torch.topk(sample_negs, min(self.n_hard_negatives, (sample_negs > float('-inf')).sum().item()), dim=0)[1]
                
                # Get the actual similarity values
                hard_values = similarity_matrix[i, hard_indices]
                
                # Add padding if needed
                if hard_values.size(0) < self.n_hard_negatives:
                    pad_size = self.n_hard_negatives - hard_values.size(0)
                    hard_values = torch.cat([hard_values, torch.ones(pad_size, device=hard_values.device) * float('-inf')])
                
                hard_neg_values.append(hard_values)
            
            # Stack to get [2*batch_size, n_hard_negatives] tensor
            negatives = torch.stack(hard_neg_values)
        else:
            # Get all negatives (original approach)
            negatives = similarity_matrix[mask_negatives].reshape(2*batch_size, -1)
        
        # Scale by temperature
        positives = positives / self.temperature
        negatives = negatives / self.temperature
        
        # Concatenate positive and negative similarities
        logits = torch.cat([positives.view(-1, 1), negatives], dim=1)
        
        # Create labels - the positive is always the first element (index 0)
        target_labels = torch.zeros(2*batch_size, dtype=torch.long, device=z_i.device)
        
        # Calculate cross-entropy loss
        loss = self.criterion(logits, target_labels)
        
        # Return average loss
        return loss / (2*batch_size)


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Uses label information to create positive pairs from same class.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        Compute the supervised contrastive loss.
        
        Args:
            features: Feature embeddings [batch_size, feature_dim]
            labels: Class labels [batch_size]
            
        Returns:
            Loss value
        """
        batch_size = features.size(0)
        labels = labels.contiguous().view(-1, 1)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Create mask for positives (same label)
        mask = torch.eq(labels, labels.T).float()
        
        # Exclude self-similarity
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive samples
        # Weighted by number of positive samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss


class QuestionAwareContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning that is question-aware.
    Creates triplets of (anchor, positive, negative) where positive and negative 
    come from the same question but have different correctness.
    """
    def __init__(self, records, embeddings_dir, goal):
        """
        Initialize the question-aware contrastive dataset.
        
        Args:
            records: List of records
            embeddings_dir: Directory containing embedding files
            goal: 'true' for correctness prediction, 'conf' for confidence prediction
        """
        self.records = records
        self.embeddings_dir = embeddings_dir
        self.goal = goal
        
        # Group records by record_hash_id (question)
        self.questions = defaultdict(list)
        for rec in self.records:
            qid = rec["record_hash_id"]
            self.questions[qid].append(rec)
        
        # Load all embeddings and cache them
        self.embedding_cache = {}
        self.label_cache = {}
        
        print("Loading embeddings into memory...")
        for qid, records in tqdm(self.questions.items()):
            for rec in records:
                run_id = rec["run_hash_id"]
                if run_id not in self.embedding_cache:
                    emb_path = os.path.join(self.embeddings_dir, f"{run_id}.pt")
                    embedding = torch.load(emb_path, map_location="cpu", weights_only=True)
                    self.embedding_cache[run_id] = embedding
                    
                    if self.goal == 'conf':
                        self.label_cache[run_id] = float(rec["empirical_confidence"])
                    else:  # 'true'
                        self.label_cache[run_id] = int(float(rec["correct_or_not"]))
        
        # Create triplets (anchor, positive, negative)
        # where positive has same label as anchor, negative has different label
        # and both come from the same question
        self.triplets = []
        
        # For each question
        for qid, question_records in self.questions.items():
            # Split by labels
            correct_records = [r for r in question_records if self.label_cache[r["run_hash_id"]] == 1]
            incorrect_records = [r for r in question_records if self.label_cache[r["run_hash_id"]] == 0]
            
            # If we have both correct and incorrect examples for this question
            if correct_records and incorrect_records:
                # Use each correct record as anchor with a positive from correct and negative from incorrect
                for anchor in correct_records:
                    positives = [r for r in correct_records if r["run_hash_id"] != anchor["run_hash_id"]]
                    if positives:  # If we have at least one positive
                        positive = random.choice(positives)
                        negative = random.choice(incorrect_records)
                        self.triplets.append((
                            anchor["run_hash_id"],
                            positive["run_hash_id"],
                            negative["run_hash_id"]
                        ))
                
                # Use each incorrect record as anchor with a positive from incorrect and negative from correct
                for anchor in incorrect_records:
                    positives = [r for r in incorrect_records if r["run_hash_id"] != anchor["run_hash_id"]]
                    if positives:  # If we have at least one positive
                        positive = random.choice(positives)
                        negative = random.choice(correct_records)
                        self.triplets.append((
                            anchor["run_hash_id"],
                            positive["run_hash_id"],
                            negative["run_hash_id"]
                        ))
        
        print(f"Created {len(self.triplets)} question-aware triplets")

    def __len__(self):
        """Return the number of triplets in the dataset."""
        return len(self.triplets)

    def __getitem__(self, idx):
        """Get a triplet of embeddings and their labels at the given index."""
        anchor_id, positive_id, negative_id = self.triplets[idx]
        
        anchor_emb = self.embedding_cache[anchor_id]
        positive_emb = self.embedding_cache[positive_id]
        negative_emb = self.embedding_cache[negative_id]
        
        anchor_label = self.label_cache[anchor_id]
        positive_label = self.label_cache[positive_id]
        negative_label = self.label_cache[negative_id]
        
        return (
            anchor_emb, positive_emb, negative_emb, 
            torch.tensor(anchor_label), torch.tensor(positive_label), torch.tensor(negative_label)
        )


class ContrastiveEmbeddingDataset(Dataset):
    """
    Dataset for contrastive learning with proper class balancing.
    """
    def __init__(self, records, embeddings_dir, goal, oversample=True, question_aware=False):
        """
        Initialize the contrastive embedding dataset.
        
        Args:
            records: List of records for training
            embeddings_dir: Directory containing embedding files
            goal: 'true' for correctness prediction, 'conf' for confidence prediction
            oversample: Whether to balance classes through oversampling
            question_aware: Whether to use question-aware sampling
        """
        self.records = records
        self.embeddings_dir = embeddings_dir
        self.goal = goal
        self.question_aware = question_aware
        
        # Group records by record_hash_id (question)
        self.questions = defaultdict(list)
        for rec in self.records:
            qid = rec["record_hash_id"]
            self.questions[qid].append(rec)
        
        # Load all embeddings and cache them
        self.embedding_cache = {}
        self.label_cache = {}
        
        print("Loading embeddings into memory...")
        for qid, records in tqdm(self.questions.items()):
            for rec in records:
                run_id = rec["run_hash_id"]
                if run_id not in self.embedding_cache:
                    emb_path = os.path.join(self.embeddings_dir, f"{run_id}.pt")
                    embedding = torch.load(emb_path, map_location="cpu", weights_only=True)
                    self.embedding_cache[run_id] = embedding
                    
                    if self.goal == 'conf':
                        self.label_cache[run_id] = float(rec["empirical_confidence"])
                    else:  # 'true'
                        self.label_cache[run_id] = int(float(rec["correct_or_not"]))
        
        # Create contrastive pairs with proper class balancing
        self.pairs = []
        
        if question_aware:
            # Create question-aware pairs - prioritize pairs from same question
            for qid, question_records in self.questions.items():
                # Split by labels
                correct_records = [r for r in question_records if self.label_cache[r["run_hash_id"]] == 1]
                incorrect_records = [r for r in question_records if self.label_cache[r["run_hash_id"]] == 0]
                
                # Create pairs within each class (correct-correct, incorrect-incorrect)
                # Correct pairs
                if len(correct_records) >= 2:
                    for i, anchor in enumerate(correct_records):
                        for j, positive in enumerate(correct_records):
                            if i != j:  # Don't pair with self
                                self.pairs.append((anchor["run_hash_id"], positive["run_hash_id"]))
                
                # Incorrect pairs
                if len(incorrect_records) >= 2:
                    for i, anchor in enumerate(incorrect_records):
                        for j, positive in enumerate(incorrect_records):
                            if i != j:  # Don't pair with self
                                self.pairs.append((anchor["run_hash_id"], positive["run_hash_id"]))
                                
        elif oversample:
            # Count total samples in each class before oversampling
            correct_count = sum(1 for rec in records if self.label_cache[rec["run_hash_id"]] == 1)
            incorrect_count = sum(1 for rec in records if self.label_cache[rec["run_hash_id"]] == 0)
            
            # The majority class size will be the target for the minority class
            majority_class_size = max(correct_count, incorrect_count)
            
            # For each question, create balanced positive and negative samples
            for qid, question_records in self.questions.items():
                # Split by labels
                correct_records = [r for r in question_records if self.label_cache[r["run_hash_id"]] == 1]
                incorrect_records = [r for r in question_records if self.label_cache[r["run_hash_id"]] == 0]
                
                # Determine oversampling factors based on global class imbalance
                # This ensures we're only oversampling the minority class
                if correct_count < incorrect_count:
                    # Need to oversample correct records
                    correct_oversample_factor = majority_class_size / correct_count if correct_count > 0 else 0
                    incorrect_oversample_factor = 1.0
                else:
                    # Need to oversample incorrect records
                    correct_oversample_factor = 1.0
                    incorrect_oversample_factor = majority_class_size / incorrect_count if incorrect_count > 0 else 0
                
                # Create pairs for correct records (with oversampling if needed)
                if len(correct_records) >= 2:
                    num_pairs = int(len(correct_records) * correct_oversample_factor)
                    for _ in range(num_pairs):
                        # Choose two different correct records
                        anchor, positive = random.sample(correct_records, 2)
                        self.pairs.append((anchor["run_hash_id"], positive["run_hash_id"]))
                
                # Create pairs for incorrect records (with oversampling if needed)
                if len(incorrect_records) >= 2:
                    num_pairs = int(len(incorrect_records) * incorrect_oversample_factor)
                    for _ in range(num_pairs):
                        # Choose two different incorrect records
                        anchor, positive = random.sample(incorrect_records, 2)
                        self.pairs.append((anchor["run_hash_id"], positive["run_hash_id"]))
        else:
            # Simple pair creation without oversampling
            for qid, records in self.questions.items():
                for i, anchor_rec in enumerate(records):
                    anchor_id = anchor_rec["run_hash_id"]
                    anchor_label = self.label_cache[anchor_id]
                    
                    # Find a positive pair (same class, different run)
                    positive_candidates = [r for r in records 
                                         if self.label_cache[r["run_hash_id"]] == anchor_label 
                                         and r["run_hash_id"] != anchor_id]
                    
                    if positive_candidates:
                        pos_rec = random.choice(positive_candidates)
                        pos_id = pos_rec["run_hash_id"]
                        
                        # Add the pair (anchor, positive)
                        self.pairs.append((anchor_id, pos_id))

        print(f"Created {len(self.pairs)} contrastive pairs")

    def __len__(self):
        """Return the number of pairs in the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Get a pair of embeddings and their labels at the given index."""
        anchor_id, positive_id = self.pairs[idx]
        
        anchor_emb = self.embedding_cache[anchor_id]
        positive_emb = self.embedding_cache[positive_id]
        
        anchor_label = self.label_cache[anchor_id]
        positive_label = self.label_cache[positive_id]
        
        return anchor_emb, positive_emb, torch.tensor(anchor_label), torch.tensor(positive_label)


# Create a ContrastiveClassifierHead class that uses the contrastive model
class ContrastiveClassifierHead(nn.Module):
    """
    Classifier model that uses a contrastive model for projection.
    Can be set to freeze or fine-tune the contrastive part.
    """
    def __init__(self, contrastive_model, output_dim, freeze_contrastive=True):
        super().__init__()
        self.contrastive_model = contrastive_model
        
        # Freeze the contrastive model parameters if requested
        if freeze_contrastive:
            for param in self.contrastive_model.parameters():
                param.requires_grad = False
            print("Contrastive model frozen in classifier - weights will not be updated during training")
        else:
            print("Contrastive model unfrozen in classifier - weights will be fine-tuned during training")
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Pass through contrastive model
        if next(self.contrastive_model.parameters()).requires_grad:
            # Model is being trained, use regular forward pass
            proj = self.contrastive_model(x)
        else:
            # Model is frozen, use no_grad to save memory
            with torch.no_grad():
                proj = self.contrastive_model(x)
        
        # Pass through classifier
        return self.classifier(proj).squeeze(-1)


def evaluate_contrastive_model(model, val_data, device, temperature=0.5, supervised=False, criter='ntxent'):
    """
    Evaluate the contrastive model on validation data.
    
    Args:
        model: ContrastiveProjectionHead model
        val_data: ValidationDataLoader
        device: Device to run evaluation on
        temperature: Temperature parameter for loss function
        supervised: Whether to use supervised contrastive loss
        criter: Loss criterion to use ('ntxent', 'infonce', 'triplet', or 'multi')
        
    Returns:
        Average loss over validation set
    """
    model.eval()
    
    # Get the first batch to determine the structure of the data
    for batch in val_data:
        batch_structure = len(batch)
        break
    
    # Choose the appropriate loss function
    if supervised:
        criterion = SupervisedContrastiveLoss(temperature=temperature)
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            # Handle different dataset formats
            if batch_structure == 2:  # SupervisedDataset format
                for embs, labels in tqdm(val_data, desc="Validating"):
                    embs = embs.to(device).float()
                    labels = labels.to(device)
                    features = model(embs)
                    all_features.append(features)
                    all_labels.append(labels)
            elif batch_structure == 6:  # QuestionAwareContrastiveDataset format
                for anchor_emb, pos_emb, neg_emb, anchor_label, pos_label, neg_label in tqdm(val_data, desc="Validating"):
                    # Process all embeddings
                    all_embs = torch.cat([anchor_emb, pos_emb, neg_emb], dim=0).to(device).float()
                    all_labs = torch.cat([anchor_label, pos_label, neg_label], dim=0).to(device)
                    features = model(all_embs)
                    all_features.append(features)
                    all_labels.append(all_labs)
            else:  # ContrastiveEmbeddingDataset format
                for anchor_emb, positive_emb, anchor_label, positive_label in tqdm(val_data, desc="Validating"):
                    # Process all embeddings
                    all_embs = torch.cat([anchor_emb, positive_emb], dim=0).to(device).float()
                    all_labs = torch.cat([anchor_label, positive_label], dim=0).to(device)
                    features = model(all_embs)
                    all_features.append(features)
                    all_labels.append(all_labs)
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        loss = criterion(all_features, all_labels)
        return loss.item()
    else:
        # Initialize the appropriate loss criterion
        if criter == 'ntxent':
            criterion = NTXentLoss(temperature=temperature)
        elif criter == 'infonce':
            criterion = InfoNCELoss(temperature=temperature)
        elif criter == 'triplet':
            criterion = TripletMarginLoss(margin=1.0)
        elif criter == 'multi':
            criterion = MultiPositiveInfoNCELoss(temperature=temperature)
        else:
            raise ValueError(f"Invalid criterion: {criter}, must be one of 'ntxent', 'infonce', 'triplet', 'multi'")
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            if batch_structure == 6:  # QuestionAwareContrastiveDataset format
                for anchor_emb, pos_emb, neg_emb, anchor_label, pos_label, neg_label in tqdm(val_data, desc="Validating"):
                    anchor_emb = anchor_emb.to(device).float()
                    pos_emb = pos_emb.to(device).float()
                    neg_emb = neg_emb.to(device).float()
                    
                    # Project embeddings
                    anchor_proj = model(anchor_emb)
                    pos_proj = model(pos_emb)
                    neg_proj = model(neg_emb)
                    
                    if criter == 'triplet':
                        # For triplet loss
                        loss = criterion(anchor_proj, pos_proj, neg_proj)
                    else:
                        # For other loss functions, use positive pairs
                        if criter == 'multi':
                            # MultiPositiveInfoNCELoss expects (features, labels) format
                            combined_features = torch.cat([anchor_proj, positive_proj], dim=0)
                            combined_labels = torch.cat([anchor_label.to(device), positive_label.to(device)], dim=0)
                            loss = criterion(combined_features, combined_labels)
                        else:
                            # Other loss functions expect (z_i, z_j, labels) format
                            z_i = torch.cat([anchor_proj, positive_proj], dim=0)
                            z_j = torch.cat([positive_proj, anchor_proj], dim=0)
                            labels = torch.cat([anchor_label.to(device), positive_label.to(device)], dim=0)
                            loss = criterion(z_i, z_j, labels)

                    total_loss += loss.item() * anchor_emb.size(0)
                    total_samples += anchor_emb.size(0)
            else:  # ContrastiveEmbeddingDataset format
                for anchor_emb, positive_emb, anchor_label, positive_label in tqdm(val_data, desc="Validating"):
                    anchor_emb = anchor_emb.to(device).float()
                    positive_emb = positive_emb.to(device).float()
                    
                    # Project embeddings
                    anchor_proj = model(anchor_emb)
                    positive_proj = model(positive_emb)
                    
                    if criter == 'triplet':
                        # For triplet loss, we need to generate negatives
                        batch_size = anchor_proj.size(0)
                        if batch_size > 1:
                            # Use other positives as negatives
                            negatives = []
                            for i in range(batch_size):
                                neg_indices = [j for j in range(batch_size) if j != i]
                                neg_idx = random.choice(neg_indices) if neg_indices else 0
                                negatives.append(positive_proj[neg_idx])
                            negative_proj = torch.stack(negatives)
                            
                            loss = criterion(anchor_proj, positive_proj, negative_proj)
                        else:
                            # Skip single-sample batches
                            continue
                    else:
                        # For other loss functions
                        z_i = torch.cat([anchor_proj, positive_proj], dim=0)
                        z_j = torch.cat([positive_proj, anchor_proj], dim=0)
                        labels = torch.cat([
                            anchor_label.to(device), 
                            positive_label.to(device)
                        ], dim=0)
                        
                        loss = criterion(z_i, z_j, labels)
                    
                    total_loss += loss.item() * anchor_emb.size(0)
                    total_samples += anchor_emb.size(0)
        
        if total_samples == 0:
            return float('inf')  # Return a high loss if no valid batches
        
        return total_loss / total_samples


def train_supervised_contrastive_model(train_records, val_records, embeddings_dir, output_dim, device, 
                                     goal='true', batch_size=64, epochs=10, temperature=0.1, 
                                     lr=1e-4, save_dir=None, deeper_model=True):
    """
    Train a contrastive model using supervised contrastive learning.
    
    Args:
        train_records: List of records for training
        val_records: List of records for validation
        embeddings_dir: Directory containing embedding files
        output_dim: Dimension of the output projection (should match tokenizer embedding dim)
        device: Device to train on (cuda or cpu)
        goal: 'true' for correctness prediction, 'conf' for confidence prediction
        batch_size: Batch size for training
        epochs: Number of training epochs
        temperature: Temperature parameter for supervised contrastive loss
        lr: Learning rate
        save_dir: Directory to save model checkpoints
        deeper_model: Whether to use the deeper projection head architecture
    
    Returns:
        Trained ContrastiveProjectionHead model
    """
    # Create simple dataset that just loads embeddings and labels (no pairs)
    class SupervisedDataset(Dataset):
        def __init__(self, records, embeddings_dir, goal):
            self.records = records
            self.embeddings_dir = embeddings_dir
            self.goal = goal
            
            # Load embeddings and labels
            self.embeddings = []
            self.labels = []
            
            print("Loading embeddings into memory...")
            for rec in tqdm(records):
                run_id = rec["run_hash_id"]
                emb_path = os.path.join(embeddings_dir, f"{run_id}.pt")
                embedding = torch.load(emb_path, map_location="cpu", weights_only=True)
                self.embeddings.append(embedding)
                
                if goal == 'conf':
                    self.labels.append(float(rec["empirical_confidence"]))
                else:  # 'true'
                    self.labels.append(int(float(rec["correct_or_not"])))
        
        def __len__(self):
            return len(self.records)
        
        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]
    
    # Create datasets and dataloaders
    train_dataset = SupervisedDataset(train_records, os.path.join(embeddings_dir, "train"), goal)
    val_dataset = SupervisedDataset(val_records, os.path.join(embeddings_dir, "val"), goal)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Get input dimension from the first embedding
    first_emb = train_dataset.embeddings[0]
    input_dim = first_emb.shape[0]
    
    # Create model
    model = ContrastiveProjectionHead(input_dim, output_dim, deeper=deeper_model)
    model.to(device)
    
    # Create loss and optimizer
    criterion = SupervisedContrastiveLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"Training supervised contrastive model for {epochs} epochs")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        all_features = []
        all_labels = []
        
        # Collect all features and labels in a batch (for computing full-batch loss)
        for embeddings, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} collecting"):
            embeddings = embeddings.to(device).float()
            labels = torch.tensor(labels, device=device)
            
            # Project embeddings
            features = model(embeddings)
            
            # Save for loss computation
            all_features.append(features)
            all_labels.append(labels)
        
        # Concatenate all features and labels for the entire epoch
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute loss and optimize
        loss = criterion(all_features, all_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
        
        # Validation phase - note this is expensive as it loads all val data
        model.eval()
        val_all_features = []
        val_all_labels = []
        
        with torch.no_grad():
            for embeddings, labels in tqdm(val_loader, desc="Validating"):
                embeddings = embeddings.to(device).float()
                labels = torch.tensor(labels, device=device)
                features = model(embeddings)
                val_all_features.append(features)
                val_all_labels.append(labels)
        
        val_all_features = torch.cat(val_all_features, dim=0)
        val_all_labels = torch.cat(val_all_labels, dim=0)
        val_loss = criterion(val_all_features, val_all_labels).item()
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                best_model_path = os.path.join(save_dir, "contrastive_model_best.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model at epoch {epoch+1} with val loss: {val_loss:.4f}")
    
    # Load best model if saved
    if save_dir:
        best_model_path = os.path.join(save_dir, "contrastive_model_best.pt")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return model


def train_contrastive_model(train_records, val_records, embeddings_dir, output_dim, device, 
                          goal='true', oversample=True, batch_size=64, 
                          epochs=10, temperature=0.5, lr=1e-4, save_dir=None,
                          deeper_model=True, question_aware=False,
                          use_hard_negatives=False, n_hard_negatives=10,
                          use_supervised=False, criter='ntxent'):
    """
    Train a contrastive model on the final hidden states.
    
    Args:
        train_records: List of records for training
        val_records: List of records for validation
        embeddings_dir: Directory containing embedding files
        output_dim: Dimension of the output projection (should match tokenizer embedding dim)
        device: Device to train on (cuda or cpu)
        goal: 'true' for correctness prediction, 'conf' for confidence prediction
        oversample: Whether to oversample minority class
        batch_size: Batch size for training
        epochs: Number of training epochs
        temperature: Temperature parameter for NT-Xent loss
        lr: Learning rate
        save_dir: Directory to save model checkpoints
        deeper_model: Whether to use deeper projection head architecture
        question_aware: Whether to use question-aware contrastive learning
        use_hard_negatives: Whether to use hard negative mining
        n_hard_negatives: Number of hard negatives to use per sample
        use_supervised: Whether to use supervised contrastive learning
        criter: Criterion to use for contrastive learning
    
    Returns:
        Trained ContrastiveProjectionHead model
    """
    # If using supervised contrastive learning, delegate to specialized function
    if use_supervised:
        return train_supervised_contrastive_model(
            train_records=train_records,
            val_records=val_records,
            embeddings_dir=embeddings_dir,
            output_dim=output_dim,
            device=device,
            goal=goal,
            batch_size=batch_size,
            epochs=epochs,
            temperature=temperature,
            lr=lr,
            save_dir=save_dir,
            deeper_model=deeper_model
        )
        
    # Create dataset and dataloader for training
    if question_aware:
        # Use QuestionAwareContrastiveDataset for triplet-based learning
        train_dataset = QuestionAwareContrastiveDataset(
            train_records, 
            os.path.join(embeddings_dir, "train"), 
            goal
        )
        val_dataset = QuestionAwareContrastiveDataset(
            val_records, 
            os.path.join(embeddings_dir, "val"), 
            goal
        )
    else:
        # Use standard ContrastiveEmbeddingDataset
        train_dataset = ContrastiveEmbeddingDataset(
            train_records, 
            os.path.join(embeddings_dir, "train"), 
            goal, 
            oversample=oversample,
            question_aware=False  # This is different from the question_aware triplet approach
        )
        val_dataset = ContrastiveEmbeddingDataset(
            val_records, 
            os.path.join(embeddings_dir, "val"), 
            goal, 
            oversample=False,  # No oversampling for validation
            question_aware=False
        )
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Get input dimension from the first embedding
    first_id = train_records[0]["run_hash_id"]
    first_emb_path = os.path.join(embeddings_dir, "train", f"{first_id}.pt")
    first_emb = torch.load(first_emb_path, map_location="cpu", weights_only=True)
    input_dim = first_emb.shape[0]
    
    # Create model with the specified architecture
    model = ContrastiveProjectionHead(input_dim, output_dim, deeper=deeper_model)
    model.to(device)
    
    # Create loss and optimizer
    if criter == 'ntxent':
        criterion = NTXentLoss(temperature=temperature)
    elif criter == 'infonce':
        criterion = InfoNCELoss(temperature=temperature)
    elif criter == 'triplet':
        criterion = TripletMarginLoss(margin=1.0)
    elif criter == 'multi':
        criterion = MultiPositiveInfoNCELoss(temperature=temperature)
    else:
        raise ValueError(f"Invalid criterion: {criter}, must be one of 'ntxent', 'infonce', 'triplet', 'multi'")
                         
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"Training contrastive model for {epochs} epochs")
    print(f"  Using deeper model: {deeper_model}")
    print(f"  Using hard negatives: {use_hard_negatives}")
    print(f"  Using question-aware: {question_aware}")
    print(f"  Temperature: {temperature}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        if question_aware:
            # For question-aware triplet learning
            for anchor_emb, pos_emb, neg_emb, anchor_label, pos_label, neg_label in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                anchor_emb = anchor_emb.to(device).float()
                pos_emb = pos_emb.to(device).float()
                neg_emb = neg_emb.to(device).float()
                
                # Project embeddings
                anchor_proj = model(anchor_emb)
                pos_proj = model(pos_emb)
                neg_proj = model(neg_emb)
                
                # Compute triplet loss directly
                anchor_proj = F.normalize(anchor_proj, dim=1)
                pos_proj = F.normalize(pos_proj, dim=1)
                neg_proj = F.normalize(neg_proj, dim=1)
                
                # Positive similarity
                pos_sim = torch.sum(anchor_proj * pos_proj, dim=1)
                # Negative similarity
                neg_sim = torch.sum(anchor_proj * neg_proj, dim=1)
                
                # Loss is -log(exp(pos_sim/t) / (exp(pos_sim/t) + exp(neg_sim/t)))
                loss = -torch.log(torch.exp(pos_sim/temperature) / 
                                 (torch.exp(pos_sim/temperature) + torch.exp(neg_sim/temperature)))
                loss = loss.mean()
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * anchor_emb.size(0)
        else:
            # For standard contrastive learning
            for anchor_emb, positive_emb, anchor_label, positive_label in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                anchor_emb = anchor_emb.to(device).float()
                positive_emb = positive_emb.to(device).float()
                anchor_label = anchor_label.to(device)
                positive_label = positive_label.to(device)
                
                # Project embeddings
                anchor_proj = model(anchor_emb)
                positive_proj = model(positive_emb)
                
                # Compute loss
                # We use both anchors and positives as separate examples
                # Inside the for loop where you're computing the loss
                if criter == 'multi':
                    # MultiPositiveInfoNCELoss expects (features, labels) format
                    combined_features = torch.cat([anchor_proj, positive_proj], dim=0)
                    combined_labels = torch.cat([anchor_label, positive_label], dim=0)
                    loss = criterion(combined_features, combined_labels)
                else:
                    # Other loss functions expect (z_i, z_j, labels) format
                    z_i = torch.cat([anchor_proj, positive_proj], dim=0)
                    z_j = torch.cat([positive_proj, anchor_proj], dim=0)
                    labels = torch.cat([anchor_label, positive_label], dim=0)
                    loss = criterion(z_i, z_j, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * anchor_emb.size(0)
        
        avg_train_loss = train_loss / len(train_dataset)
        
        # Validation phase
        val_loss = evaluate_contrastive_model(model, val_dataloader, device, temperature, criter)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                best_model_path = os.path.join(save_dir, "contrastive_model_best.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model at epoch {epoch+1} with val loss: {val_loss:.4f}")
    
    # Load best model if saved
    if save_dir:
        best_model_path = os.path.join(save_dir, "contrastive_model_best.pt")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return model


def apply_contrastive_projection(records, embeddings_dir, model, device):
    """
    Apply the trained contrastive projection model to all embeddings.
    
    Args:
        records: List of records
        embeddings_dir: Directory containing embedding files or function to get directory
        model: Trained ContrastiveProjectionHead model
        device: Device to run inference on
    
    Returns:
        Dictionary mapping run_hash_id to projected embedding
    """
    model.eval()
    projected_embeddings = {}
    
    with torch.no_grad():
        for rec in tqdm(records, desc="Projecting embeddings"):
            run_id = rec["run_hash_id"]
            if run_id not in projected_embeddings:
                # Handle embeddings_dir as either a string or a function
                if callable(embeddings_dir):
                    dir_path = embeddings_dir(rec)
                else:
                    dir_path = embeddings_dir
                
                emb_path = os.path.join(dir_path, f"{run_id}.pt")
                embedding = torch.load(emb_path, map_location="cpu", weights_only=True).to(device).float()
                
                # Project embedding
                projected = model(embedding.unsqueeze(0)).squeeze(0)
                projected_embeddings[run_id] = projected.cpu()
    
    return projected_embeddings


def save_model(model, path):
    """Save a model to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model_class, path, **kwargs):
    """Load a model from the specified path."""
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model


def load_contrastive_model(input_dim, output_dim, path, device=None):
    """
    Load a pre-trained contrastive model.
    
    Args:
        input_dim: Input dimension of the model
        output_dim: Output dimension of the model
        path: Path to the saved model state
        device: Device to load the model on
        
    Returns:
        Loaded ContrastiveProjectionHead model
    """
    model = ContrastiveProjectionHead(input_dim, output_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    if device:
        model = model.to(device)
    model.eval()
    return model