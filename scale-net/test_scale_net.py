"""
Test script for SCALENet + SE Block
Uses best_ssvep_model.pth checkpoint
"""

import torch
import numpy as np
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import confusion_matrix

# Import model and dataset functions from model_cldnn_se.py
from train_scale_net import SCALENet
from dataset import load_dataset, TASK_CONFIGS, create_dataloaders, get_stft_dimensions


# Note: Using model and dataset functions from model_cldnn_se.py and dataset.py


# ==================== Evaluation ====================

def evaluate(model, loader, device, is_binary=False):
    """
    Evaluate model on a dataset
    
    Args:
        model: Model to evaluate
        loader: DataLoader
        device: Device to run on
        is_binary: Whether this is binary classification (n_classes=2)
    
    Returns:
        accuracy, predictions, labels
    """
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Evaluating', ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Prediction: binary uses sigmoid threshold, multi-class uses argmax
            if is_binary:
                pred = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()  # (B, 1) → (B,)
            else:
                _, pred = outputs.max(1)
            
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return 100. * correct / total, np.array(all_preds), np.array(all_labels)


# ==================== Confusion Matrix ====================

def print_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Print confusion matrix as text in terminal
    Shows counts and percentages for each class
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(cm)
    class_labels = [chr(65 + i) for i in range(n_classes)]
    
    # Normalize to percentages
    cm_percent = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) * 100
    
    print(f"\n{title}")
    
    # Calculate column width (need space for 3-digit numbers)
    col_width = 4  # Each cell is 4 characters wide
    
    # Total width: "True\\Pred" (9) + n_classes * col_width + " Total" (6)
    total_width = 9 + n_classes * col_width + 6
    print("=" * total_width)
    
    # Header: Predicted labels
    header = "True\\Pred"
    for label in class_labels:
        header += f"{label:>{col_width}s}"
    header += " Total"
    print(header)
    print("-" * total_width)
    
    # Each row: True label, predicted counts, total
    for i in range(n_classes):
        row = f"   {class_labels[i]:>3s}  "
        total_true = cm[i].sum()
        for j in range(n_classes):
            count = cm[i, j]
            row += f"{count:>{col_width}d}"
        row += f"{total_true:>6d}"
        print(row)
    
    print("-" * total_width)
    
    # Column totals
    col_totals = cm.sum(axis=0)
    row = "  Total  "
    for total in col_totals:
        row += f"{total:>{col_width}d}"
    row += f"{len(y_true):>6d}"
    print(row)
    print("=" * total_width)
    
    # Summary statistics
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = 100. * correct / total
    
    print(f"\nSummary:")
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Overall accuracy: {accuracy:.2f}%")
    
    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for i in range(n_classes):
        if cm[i, i] > 0 or cm[i].sum() > 0:
            cls_total = cm[i].sum()
            cls_correct = cm[i, i]
            cls_acc = 100. * cls_correct / cls_total if cls_total > 0 else 0.0
            print(f"  Class {i:2d} ({class_labels[i]}): {cls_acc:5.1f}% ({cls_correct}/{cls_total})")


# ==================== Main Test ====================

def test_task(task, checkpoint_path, batch_size, checkpoint_dir='./checkpoints'):
    """
    Loads model and evaluates it for a specific task.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_upper = task.upper() # Use uppercase for printing/keys
    
    print(f"\n{'='*70}")
    print(f"Testing Task: {task_upper}")
    print(f"{'='*70}")
    
    # 1. Load Checkpoint First to Get Config
    if not os.path.exists(checkpoint_path):
        print(f"Skipping task {task_upper}: Checkpoint not found at {checkpoint_path}")
        return {'task': task_upper, 'error': 'Checkpoint not found'}
        
    try:
        # Load the checkpoint dictionary
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Skipping task {task_upper}: Failed to load checkpoint. Error: {e}")
        return {'task': task_upper, 'error': f'Checkpoint loading failed: {e}'}

    # Get config from checkpoint or use defaults
    saved_config = checkpoint.get('config', {})
    task_defaults = TASK_CONFIGS.get(task_upper, {})
    
    # Merge configs (Simplified merge for testing)
    config = {
        # General params
        'num_seen': saved_config.get('num_seen', task_defaults.get('num_seen', 33)),
        'seed': saved_config.get('seed', 44),
        'data_dir': saved_config.get('data_dir', task_defaults.get('data_dir', './data')),
        'n_classes': saved_config.get('n_classes', task_defaults.get('num_classes', 26)),
        # Model params
        'cnn_filters': saved_config.get('cnn_filters', 16),
        'lstm_hidden': saved_config.get('lstm_hidden', 128),
        'pos_dim': saved_config.get('pos_dim', 16),
        'dropout': saved_config.get('dropout', 0.3),
        'cnn_dropout': saved_config.get('cnn_dropout', 0.2),
        'use_hidden_layer': saved_config.get('use_hidden_layer', False),
        'hidden_dim': saved_config.get('hidden_dim', 64),
        # STFT params
        'stft_fs': saved_config.get('stft_fs', task_defaults.get('sampling_rate', 250)),
        'stft_nperseg': saved_config.get('stft_nperseg', task_defaults.get('stft_nperseg', 128)),
        'stft_noverlap': saved_config.get('stft_noverlap', task_defaults.get('stft_noverlap', 112)),
        'stft_nfft': saved_config.get('stft_nfft', task_defaults.get('stft_nfft', 512)),
    }
    
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data Dir: {config['data_dir']}")
    print("Model Configuration loaded from checkpoint.")

    # 2. Load Data and Create Data Loaders
    datasets = load_dataset(
        task=task_upper,
        data_dir=config['data_dir'],
        num_seen=config['num_seen'],
        seed=config['seed']
    )
    
    stft_config = {
        'fs': config['stft_fs'],
        'nperseg': config['stft_nperseg'],
        'noverlap': config['stft_noverlap'],
        'nfft': config['stft_nfft']
    }
    
    loaders = create_dataloaders(
        datasets,
        stft_config,
        batch_size=batch_size,
        num_workers=4,
        augment_train=False
    )
    
    val_loader = loaders['val']
    test1_loader = loaders.get('test1')
    test2_loader = loaders.get('test2')
    
    
    # 3. Create Model and Load Checkpoint
    n_classes = config['n_classes']
    is_binary = (n_classes == 2)
    
    model = SCALENet(
        freq_bins=freq_bins,
        time_bins=time_bins,
        n_channels=n_channels,
        n_classes=n_classes,
        cnn_filters=config['cnn_filters'],
        lstm_hidden=config['lstm_hidden'],
        pos_dim=config['pos_dim'],
        dropout=config['dropout'],
        cnn_dropout=config['cnn_dropout'],
        use_hidden_layer=config['use_hidden_layer'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # In case only state_dict was saved
    
    print("✓ Checkpoint weights loaded successfully!")
    
    # 4. Evaluate
    print(f"\n{'='*70}")
    print("Evaluating...")
    print(f"{'='*70}")
    
    # Initialize results
    test1_acc, test1_preds, test1_labels = None, None, None
    test2_acc, test2_preds, test2_labels = None, None, None

    # Always evaluate validation set
    val_acc, val_preds, val_labels = evaluate(model, val_loader, device, is_binary=is_binary)
    
    if test1_loader:
        test1_acc, test1_preds, test1_labels = evaluate(model, test1_loader, device, is_binary=is_binary)
    
    if test2_loader:
        test2_acc, test2_preds, test2_labels = evaluate(model, test2_loader, device, is_binary=is_binary)
    
    # 5. Results and Confusion Matrix
    print(f"\n{'='*70}")
    print(f"RESULTS - {task_upper}")
    print(f"{'='*70}")
    print(f"Validation Acc:  {val_acc:.2f}%")
    if test1_acc is not None:
        print(f"Test1 (Seen):    {test1_acc:.2f}%")
    if test2_acc is not None:
        print(f"Test2 (Unseen):  {test2_acc:.2f}%")
    print(f"{'='*70}")
    
    if test1_preds is not None and test1_labels is not None:
        print_confusion_matrix(
            test1_labels, test1_preds, 
            title=f"Confusion Matrix - {task_upper} Test1 (Seen Subjects)"
        )
    
    results = {
        'task': task_upper,
        'val_acc': val_acc,
        'test1_acc': test1_acc,
        'test2_acc': test2_acc
    }
    
    return results

# ==================== Test All Tasks ====================

def main_test_all(tasks, checkpoint_dir, batch_size):
    """Function to test all specified tasks and summarize results."""
    
    all_results = {}
    
    print("\n" + "=" * 80)
    print("SCALENet + SE Block - Multi-Task Evaluation")
    print("=" * 80)
    
    for task in tasks:
        # Construct the standardized checkpoint path
        task_lower = task.lower().replace('_', '') # e.g., imagined_speech -> imaginedspeech
        checkpoint_path = os.path.join(checkpoint_dir, f'best_{task_lower}_model.pth')
        
        # Call the testing function for the task
        results = test_task(task, checkpoint_path, batch_size)
        all_results[task] = results
        
    # Final Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY OF ALL TASKS")
    print(f"{'='*80}")
    
    for task, results in all_results.items():
        print("-" * 30)
        if 'error' in results:
            print(f"🚨 {task}: FAILED - {results['error']}")
        else:
            print(f"✅ {task}:")
            print(f"  Val Acc:   {results['val_acc']:.2f}%")
            if results['test1_acc'] is not None:
                print(f"  Test1 Acc: {results['test1_acc']:.2f}% (Seen)")
            if results['test2_acc'] is not None:
                print(f"  Test2 Acc: {results['test2_acc']:.2f}% (Unseen)")

    print(f"\n{'='*80}")
    return all_results

# ==================== Argument Parsing and Entry Point ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test SCALENet on a specific EEG task or all tasks."
    )
    
    tasks_available = ['SSVEP', 'P300', 'MI', 'Imagined_speech']
    
    parser.add_argument(
        '--task', 
        type=str, 
        default='SSVEP',
        choices=tasks_available + ['all'],
        help=f"Task to test on. Choices: {tasks_available + ['all']} (default: SSVEP)"
    )
    parser.add_argument(
        '--checkpoint_dir', 
        type=str, 
        default='./checkpoints',
        help='Directory where model checkpoints are saved'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch size for testing (default: 32)'
    )
    
    args = parser.parse_args()
    
    if args.task == 'all':
        # Run all tasks
        main_test_all(
            tasks=tasks_available, 
            checkpoint_dir=args.checkpoint_dir, 
            batch_size=args.batch_size
        )
    else:
        # Run a single task
        # Construct the checkpoint path based on the task name
        task_lower = args.task.lower().replace('_', '')
        checkpoint_name = f'best_{task_lower}_model.pth'
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
        
        test_task(
            task=args.task, 
            checkpoint_path=checkpoint_path, 
            batch_size=args.batch_size
        )
