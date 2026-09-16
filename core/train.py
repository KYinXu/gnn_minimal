"""
Training and evaluation utilities for GNN models.

Includes:
- Training loop with early stopping
- Evaluation metrics (AUC-ROC, AUC-PR, F1, MCC, etc.)
- Cross-validation support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef
)
from torch_geometric.loader import DataLoader
import time


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    logit_penalty: float = 0.0,
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: GNN model
        loader: Training DataLoader
        optimizer: Optimizer
        device: torch device
        class_weights: Optional class weights for imbalanced data
        label_smoothing: Cross-entropy label smoothing (reduces logit saturation)
        logit_penalty: L2 mean on logits added to loss (further softens extreme margins)
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    n_batches = 0
    
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )
    
    for batch in loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        out = model(batch)
        
        # Ensure target shape matches predictions
        target = batch.y.view(-1)
        if target.numel() == 0:
            continue
        
        loss = criterion(out, target)
        if logit_penalty > 0:
            loss = loss + float(logit_penalty) * (out ** 2).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: GNN model
        loader: Evaluation DataLoader
        device: torch device
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0
    n_batches = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        
        # Ensure target shape matches predictions
        target = batch.y.view(-1)
        if target.numel() == 0:
            continue
        
        loss = criterion(out, target)
        total_loss += loss.item()
        n_batches += 1
        
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        labels = target.cpu().numpy()
        
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    metrics = {
        'loss': total_loss / n_batches,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'mcc': matthews_corrcoef(all_labels, all_preds),
    }
    
    # AUC metrics (need both classes present)
    if len(np.unique(all_labels)) > 1:
        metrics['auc_roc'] = roc_auc_score(all_labels, all_probs)
        metrics['auc_pr'] = average_precision_score(all_labels, all_probs)
    else:
        metrics['auc_roc'] = 0.0
        metrics['auc_pr'] = 0.0
    
    return metrics


@torch.no_grad()
def evaluate_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) for validation set (for ROC/PR/calibration)."""
    model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        labels = batch.y.view(-1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels)
    return np.array(all_labels), np.array(all_probs)


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    min_delta: float = 1e-4,
    class_weights: Optional[torch.Tensor] = None,
    verbose: bool = True,
    label_smoothing: float = 0.08,
    logit_penalty: float = 1e-4,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """
    Full training loop with early stopping.
    
    Args:
        model: GNN model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: torch device
        epochs: Maximum epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
        class_weights: Optional class weights
        verbose: Print progress
        label_smoothing: CE smoothing (default 0.08 mitigates saturated softmax / tied logits)
        logit_penalty: Weight on mean(logits^2) added to training loss
        
    Returns:
        history: Dict of training metrics per epoch
        best_metrics: Best validation metrics
    """
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # History tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc_roc': [],
        'val_f1': [],
    }
    
    best_val_auc = 0.0
    best_metrics = {}
    best_state = None
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights,
            label_smoothing=label_smoothing,
            logit_penalty=logit_penalty,
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_metrics['auc_roc'])
        
        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc_roc'].append(val_metrics['auc_roc'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Check for improvement
        if val_metrics['auc_roc'] > best_val_auc + min_delta:
            best_val_auc = val_metrics['auc_roc']
            best_metrics = val_metrics.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s")
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return history, best_metrics


def cross_validate(
    model_fn,
    dataset,
    cv_splits,
    device: torch.device,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    class_weights: Optional[torch.Tensor] = None,
    verbose: bool = True,
    label_smoothing: float = 0.08,
    logit_penalty: float = 1e-4,
) -> Dict[str, List[float]]:
    """
    Perform cross-validation.
    
    Args:
        model_fn: Function that returns a fresh model instance
        dataset: Full dataset (list of Data objects)
        cv_splits: List of (train_idx, val_idx) tuples
        device: torch device
        batch_size: Batch size
        epochs: Max epochs per fold
        lr: Learning rate
        patience: Early stopping patience
        class_weights: Optional class weights for imbalanced data
        verbose: Print progress
        
    Returns:
        Dictionary of metric lists (one value per fold)
    """
    cv_results = {
        'auc_roc': [],
        'auc_pr': [],
        'f1': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'mcc': [],
    }
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fold {fold + 1}/{len(cv_splits)}")
            print(f"{'='*60}")
        
        # Create dataloaders for this fold
        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        # Fresh model
        model = model_fn()
        
        # Train
        _, best_metrics = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            patience=patience,
            class_weights=class_weights,
            verbose=verbose,
            label_smoothing=label_smoothing,
            logit_penalty=logit_penalty,
        )

        # Record results
        for metric in cv_results:
            cv_results[metric].append(best_metrics.get(metric, 0.0))
        
        if verbose:
            print(f"Fold {fold + 1} Results: "
                  f"AUC-ROC={best_metrics['auc_roc']:.4f}, "
                  f"F1={best_metrics['f1']:.4f}, "
                  f"Prec={best_metrics['precision']:.4f}, "
                  f"Rec={best_metrics['recall']:.4f}, "
                  f"MCC={best_metrics['mcc']:.4f}")
    
    return cv_results


def print_cv_summary(cv_results: Dict[str, List[float]]):
    """Print cross-validation summary statistics."""
    print("\n" + "="*60)
    print("Cross-Validation Summary")
    print("="*60)
    
    for metric, values in cv_results.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric:15s}: {mean:.4f} ± {std:.4f}")
