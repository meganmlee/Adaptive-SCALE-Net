"""
Evaluate a trained model checkpoint and print final results

Usage:
    python evaluate_checkpoint.py --checkpoint ./checkpoints/best_ssvep_model.pth
    python evaluate_checkpoint.py --checkpoint ./checkpoints/best_ssvep_model.pth --task SSVEP
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
import pandas as pd

# Import from scale_net_adaptive
from scale_net_adaptive import (
    AdaptiveSCALENet,
    evaluate,
    unwrap_model,
    setup_device,
    collect_weights
)
from dataset import load_dataset, TASK_CONFIGS, create_dataloaders


def load_model_from_checkpoint(checkpoint_path, device):
    """
    Load model from checkpoint file
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        model, config, task
    """
    print(f"\n{'='*70}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config and task from checkpoint
    config = checkpoint.get('config', {})
    task = checkpoint.get('task', None)
    
    if task is None:
        # Try to infer from filename
        basename = os.path.basename(checkpoint_path)
        if 'ssvep' in basename.lower():
            task = 'SSVEP'
        elif 'p300' in basename.lower():
            task = 'P300'
        elif 'mi' in basename.lower():
            task = 'MI'
        else:
            raise ValueError("Could not determine task. Please specify --task")
    
    print(f"Task: {task}")
    print(f"Best Val Acc (from checkpoint): {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    
    # Get task config for defaults
    task_config = TASK_CONFIGS.get(task, {})
    
    # Load data to get dimensions
    datasets = load_dataset(
        task=task,
        data_dir=config.get('data_dir', task_config.get('data_dir')),
        num_seen=config.get('num_seen', task_config.get('num_seen')),
        seed=config.get('seed', 44)
    )
    
    if not datasets:
        raise ValueError(f"Failed to load data for task: {task}")
    
    # STFT config
    stft_config = {
        'fs': config.get('stft_fs', task_config.get('sampling_rate', 250)),
        'nperseg': config.get('stft_nperseg', task_config.get('stft_nperseg', 128)),
        'noverlap': config.get('stft_noverlap', task_config.get('stft_noverlap', 112)),
        'nfft': config.get('stft_nfft', task_config.get('stft_nfft', 512))
    }
    
    # Create a temporary loader to get dimensions
    temp_loaders = create_dataloaders(
        datasets,
        stft_config,
        batch_size=1,
        num_workers=0,
        augment_train=False,
        seed=config.get('seed', 44)
    )
    
    sample_x, _ = next(iter(temp_loaders['train']))
    sample_x_time, sample_x_spec = sample_x
    _, n_channels, T_raw = sample_x_time.shape
    _, _, freq_bins, time_bins = sample_x_spec.shape
    
    print(f"\nModel Dimensions:")
    print(f"  Channels: {n_channels}")
    print(f"  T_raw: {T_raw}")
    print(f"  STFT: ({freq_bins}, {time_bins})")
    
    # Create model
    n_classes = config.get('n_classes', task_config.get('num_classes', 26))
    model = AdaptiveSCALENet(
        freq_bins=freq_bins,
        time_bins=time_bins,
        n_channels=n_channels,
        n_classes=n_classes,
        T_raw=T_raw,
        cnn_filters=config.get('cnn_filters', 16),
        lstm_hidden=config.get('lstm_hidden', 128),
        pos_dim=config.get('pos_dim', 16),
        dropout=config.get('dropout', 0.3),
        cnn_dropout=config.get('cnn_dropout', 0.2),
        use_hidden_layer=config.get('use_hidden_layer', False),
        hidden_dim=config.get('hidden_dim', 64),
        fusion_mode=config.get('fusion_mode', 'global_attention'),
        fusion_temperature=config.get('fusion_temperature', 2.0)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Fusion Mode: {config.get('fusion_mode', 'global_attention')}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config, task, datasets, stft_config


def evaluate_checkpoint(checkpoint_path, task=None, batch_size=16):
    """
    Evaluate a checkpoint and print final results
    
    Args:
        checkpoint_path: Path to checkpoint file
        task: Task name (optional, will try to infer from checkpoint)
        batch_size: Batch size for evaluation
    """
    device, _ = setup_device()
    
    # Load model
    model, config, task, datasets, stft_config = load_model_from_checkpoint(
        checkpoint_path, device
    )
    
    # Create data loaders
    loaders = create_dataloaders(
        datasets,
        stft_config,
        batch_size=batch_size,
        num_workers=4,
        augment_train=False,
        seed=config.get('seed', 44)
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test1_loader = loaders.get('test1')
    test2_loader = loaders.get('test2')
    
    # Determine if binary classification
    n_classes = config.get('n_classes', TASK_CONFIGS.get(task, {}).get('num_classes', 26))
    is_binary = (n_classes == 2)
    
    # Loss function (for loss calculation)
    if is_binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION RESULTS - {task}")
    print(f"{'='*70}")
    
    # Evaluate validation set
    val_loss, val_acc, val_metrics = evaluate(
        model, val_loader, device, criterion, is_binary=is_binary, return_metrics=True
    )
    print(f"\nValidation Set:")
    print(f"  Accuracy:    {val_acc:.2f}%")
    print(f"  Loss:        {val_loss:.4f}")
    if val_metrics.get('f1') is not None:
        print(f"  F1 Score:    {val_metrics['f1']:.2f}%")
        print(f"  Recall:      {val_metrics['recall']:.2f}%")
        if val_metrics.get('auc') is not None:
            print(f"  AUC:         {val_metrics['auc']:.2f}%")
    
    # Evaluate test1 (if available)
    if test1_loader:
        test1_loss, test1_acc, test1_metrics = evaluate(
            model, test1_loader, device, criterion, is_binary=is_binary, return_metrics=True
        )
        print(f"\nTest1 (Seen Subjects):")
        print(f"  Accuracy:    {test1_acc:.2f}%")
        print(f"  Loss:        {test1_loss:.4f}")
        if test1_metrics.get('f1') is not None:
            print(f"  F1 Score:    {test1_metrics['f1']:.2f}%")
            print(f"  Recall:      {test1_metrics['recall']:.2f}%")
            if test1_metrics.get('auc') is not None:
                print(f"  AUC:         {test1_metrics['auc']:.2f}%")
    
    # Evaluate test2 (if available)
    if test2_loader:
        test2_loss, test2_acc, test2_metrics = evaluate(
            model, test2_loader, device, criterion, is_binary=is_binary, return_metrics=True
        )
        print(f"\nTest2 (Unseen Subjects):")
        print(f"  Accuracy:    {test2_acc:.2f}%")
        print(f"  Loss:        {test2_loss:.4f}")
        if test2_metrics.get('f1') is not None:
            print(f"  F1 Score:    {test2_metrics['f1']:.2f}%")
            print(f"  Recall:      {test2_metrics['recall']:.2f}%")
            if test2_metrics.get('auc') is not None:
                print(f"  AUC:         {test2_metrics['auc']:.2f}%")
    
    print(f"\n{'='*70}")
    
    # ====== COLLECT AND DISPLAY ATTENTION WEIGHTS ======
    print(f"\n{'='*70}")
    print(f"ATTENTION WEIGHTS ANALYSIS - {task}")
    print(f"{'='*70}")
    
    # Use Test2 if available, otherwise use Val
    weights_loader = test2_loader if test2_loader else val_loader
    weights_type = 'Test2 (Unseen)' if test2_loader else 'Validation'
    
    if weights_loader:
        all_weights_array = collect_weights(model, weights_loader, device)
        
        if all_weights_array is not None and len(all_weights_array) > 0:
            # Ensure 2D shape (N, 2)
            if all_weights_array.ndim == 3:
                all_weights_array = all_weights_array.mean(axis=1)
            
            # Convert to DataFrame
            weights_df = pd.DataFrame(
                all_weights_array,
                columns=['Spectral_Weight', 'Temporal_Weight']
            )
            
            # Display statistics
            print(f"\n{weights_type} Set Attention Weights:")
            print(f"  Total Trials: {len(weights_df)}")
            print(f"\n  Statistics:")
            print(f"    Spectral Weight:")
            print(f"      Mean:   {weights_df['Spectral_Weight'].mean():.4f}")
            print(f"      Std:    {weights_df['Spectral_Weight'].std():.4f}")
            print(f"      Min:    {weights_df['Spectral_Weight'].min():.4f}")
            print(f"      Max:    {weights_df['Spectral_Weight'].max():.4f}")
            print(f"      Median: {weights_df['Spectral_Weight'].median():.4f}")
            
            print(f"\n    Temporal Weight:")
            print(f"      Mean:   {weights_df['Temporal_Weight'].mean():.4f}")
            print(f"      Std:    {weights_df['Temporal_Weight'].std():.4f}")
            print(f"      Min:    {weights_df['Temporal_Weight'].min():.4f}")
            print(f"      Max:    {weights_df['Temporal_Weight'].max():.4f}")
            print(f"      Median: {weights_df['Temporal_Weight'].median():.4f}")
            
            # Show first few examples
            print(f"\n  First 10 Examples:")
            print(weights_df.head(10).to_string(index=True))
            
            # Save to CSV
            checkpoint_basename = os.path.basename(checkpoint_path).replace('.pth', '')
            csv_filename = f'{checkpoint_basename}_attention_weights.csv'
            weights_df.to_csv(csv_filename, index=False)
            print(f"\n  ✓ Weights saved to: {csv_filename}")
        else:
            print(f"\n⚠ No attention weights available (fusion mode: {config.get('fusion_mode', 'unknown')})")
            print(f"  Note: Some fusion modes (static, glu, multiplicative, bilinear) don't return weights")
    
    print(f"\n{'='*70}")
    
    # Return results dictionary
    results = {
        'val': {
            'acc': val_acc,
            'loss': val_loss,
            'f1': val_metrics.get('f1'),
            'recall': val_metrics.get('recall'),
            'auc': val_metrics.get('auc')
        }
    }
    
    if test1_loader:
        results['test1'] = {
            'acc': test1_acc,
            'loss': test1_loss,
            'f1': test1_metrics.get('f1'),
            'recall': test1_metrics.get('recall'),
            'auc': test1_metrics.get('auc')
        }
    
    if test2_loader:
        results['test2'] = {
            'acc': test2_acc,
            'loss': test2_loss,
            'f1': test2_metrics.get('f1'),
            'recall': test2_metrics.get('recall'),
            'auc': test2_metrics.get('auc')
        }
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate a trained model checkpoint and print final results'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (e.g., ./checkpoints/best_ssvep_model.pth)'
    )
    parser.add_argument(
        '--task',
        type=str,
        default=None,
        choices=['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300', 'BI2014b_P300'],
        help='Task name (optional, will try to infer from checkpoint)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for evaluation (default: 16)'
    )
    
    args = parser.parse_args()
    
    try:
        results = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            task=args.task,
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
