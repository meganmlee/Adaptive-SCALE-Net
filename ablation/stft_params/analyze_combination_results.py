"""
Analyze and Organize STFT 27-Config Ablation Results

This script collects, analyzes, and organizes results from a fixed 3×3×3 (27-config)
ablation studies into comprehensive CSV and JSON files.

Usage:
    python analyze_combination_results.py --results_dir ./ablation_results --output_dir ./analysis
"""

import os
import sys
import json
import argparse
import glob
import re
from typing import Dict, List, Optional
import pandas as pd
import torch
from datetime import datetime

# Add scale-net directory to path
scale_net_path = os.path.join(os.path.dirname(__file__), '..', '..', 'scale-net')
sys.path.insert(0, scale_net_path)

from dataset import TASK_CONFIGS


def parse_config_name(filename: str) -> Optional[Dict]:
    """
    Parse STFT configuration from model filename
    
    Args:
        filename: Model filename like 'ssvep_stft_nperseg128_overlap87pct_nfft512_model.pth'
        
    Returns:
        Dictionary with parsed parameters or None if parsing fails
    """
    # Pattern: {task}_stft_nperseg{val}_overlap{val}pct_nfft{val}_model.pth
    pattern = r'(\w+)_stft_nperseg(\d+)_overlap(\d+)pct_nfft(\d+)_model\.pth'
    match = re.match(pattern, os.path.basename(filename))
    
    if match:
        task, nperseg, overlap_pct, nfft = match.groups()
        return {
            'task': task,
            'nperseg': int(nperseg),
            'overlap_pct': int(overlap_pct),
            'noverlap': int(int(nperseg) * int(overlap_pct) / 100),
            'nfft': int(nfft),
            'config_name': f'nperseg{nperseg}_overlap{overlap_pct}pct_nfft{nfft}',
            'model_path': filename
        }
    return None


def extract_results_from_checkpoint(checkpoint_path: str, 
                                    evaluate_model: bool = True) -> Optional[Dict]:
    """
    Extract results from model checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        evaluate_model: Whether to evaluate model on test sets to get test1/test2 acc
        
    Returns:
        Dictionary with results or None if loading fails
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract information
        result = {
            'model_path': checkpoint_path,
            'epoch': checkpoint.get('epoch', None),
            'best_val_acc': checkpoint.get('best_val_acc', None),
            'task': checkpoint.get('task', None),
            'config': checkpoint.get('config', {}),
            'n_channels': checkpoint.get('n_channels', None),
            'n_samples': checkpoint.get('n_samples', None),
        }
        
        # Extract STFT parameters from config
        if 'config' in checkpoint:
            config = checkpoint['config']
            result['stft_fs'] = config.get('stft_fs', None)
            result['stft_nperseg'] = config.get('stft_nperseg', None)
            result['stft_noverlap'] = config.get('stft_noverlap', None)
            result['stft_nfft'] = config.get('stft_nfft', None)
        
        # Evaluate model to get test1 and test2 accuracy
        if evaluate_model and result['task']:
            test_results = evaluate_model_from_checkpoint(checkpoint_path, result['task'])
            if test_results:
                result['test1_acc'] = test_results.get('test1_acc')
                result['test2_acc'] = test_results.get('test2_acc')
                result['test1_loss'] = test_results.get('test1_loss')
                result['test2_loss'] = test_results.get('test2_loss')
        
        return result
    except Exception as e:
        print(f"Warning: Failed to load {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_model_from_checkpoint(checkpoint_path: str, task: str) -> Optional[Dict]:
    """
    Evaluate model from checkpoint on test sets
    
    Args:
        checkpoint_path: Path to model checkpoint
        task: Task name
        
    Returns:
        Dictionary with test1_acc, test2_acc, test1_loss, test2_loss or None if fails
    """
    try:
        # Import here to avoid circular imports
        sys.path.insert(0, scale_net_path)
        from train_scale_net import SCALENet, evaluate
        from dataset import load_dataset, create_dataloaders
        import torch.nn as nn
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint.get('config', {})
        task_config = TASK_CONFIGS.get(task, {})
        
        if not config:
            return None
        
        # Get STFT config
        stft_config = {
            'fs': config.get('stft_fs', task_config.get('sampling_rate', 250)),
            'nperseg': config.get('stft_nperseg', task_config.get('stft_nperseg', 128)),
            'noverlap': config.get('stft_noverlap', task_config.get('stft_noverlap', 112)),
            'nfft': config.get('stft_nfft', task_config.get('stft_nfft', 512))
        }
        
        # Load data
        datasets = load_dataset(
            task=task,
            data_dir=config.get('data_dir', task_config.get('data_dir')),
            num_seen=config.get('num_seen', task_config.get('num_seen')),
            seed=config.get('seed', 44)
        )
        
        if not datasets:
            return None
        
        # Create data loaders
        loaders = create_dataloaders(
            datasets,
            stft_config,
            batch_size=config.get('batch_size', 16),
            num_workers=0,  # Use 0 to avoid multiprocessing issues
            augment_train=False,
            seed=config.get('seed', 44)
        )
        
        test1_loader = loaders.get('test1')
        test2_loader = loaders.get('test2')
        
        if not test1_loader and not test2_loader:
            return None
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get model dimensions
        sample_x, _ = next(iter(loaders['val']))
        # For regular scale_net, inputs is a list [x_time, x_spec], use spectral only
        if isinstance(sample_x, list):
            sample_x_spec = sample_x[1]
        else:
            sample_x_spec = sample_x
        _, n_channels, freq_bins, time_bins = sample_x_spec.shape
        
        # Create model
        n_classes = config.get('n_classes', task_config.get('num_classes', 26))
        is_binary = (n_classes == 2)
        
        model = SCALENet(
            freq_bins=freq_bins,
            time_bins=time_bins,
            n_channels=n_channels,
            n_classes=n_classes,
            cnn_filters=config.get('cnn_filters', 16),
            lstm_hidden=config.get('lstm_hidden', 128),
            pos_dim=config.get('pos_dim', 16),
            dropout=config.get('dropout', 0.3),
            cnn_dropout=config.get('cnn_dropout', 0.2),
            use_hidden_layer=config.get('use_hidden_layer', False),
            hidden_dim=config.get('hidden_dim', 64)
        ).to(device)
        
        # Load model weights (handle DataParallel wrapper)
        state_dict = checkpoint['model_state_dict']
        # Remove 'module.' prefix if present (from DataParallel)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        
        # Create criterion
        if is_binary:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        results = {}
        
        # Evaluate on test1
        if test1_loader:
            test1_loss, test1_acc = evaluate(model, test1_loader, device, criterion, is_binary=is_binary)
            results['test1_acc'] = test1_acc
            results['test1_loss'] = test1_loss
        
        # Evaluate on test2
        if test2_loader:
            test2_loss, test2_acc = evaluate(model, test2_loader, device, criterion, is_binary=is_binary)
            results['test2_acc'] = test2_acc
            results['test2_loss'] = test2_loss
        
        return results
        
    except Exception as e:
        print(f"  Warning: Failed to evaluate model from {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def collect_results_from_directory(results_dir: str) -> Dict:
    """
    Collect all results from a directory
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        Dictionary with organized results
    """
    print(f"\n{'='*80}")
    print(f"Collecting results from: {results_dir}")
    print(f"{'='*80}")
    
    all_results = {}
    
    # Find all model checkpoint files
    model_files = glob.glob(os.path.join(results_dir, '*_model.pth'))
    print(f"Found {len(model_files)} model checkpoint files")
    
    # Find CSV files
    csv_files = glob.glob(os.path.join(results_dir, '*_results.csv'))
    print(f"Found {len(csv_files)} CSV result files")
    
    # Find JSON summary files
    json_files = glob.glob(os.path.join(results_dir, '*_summary.json'))
    print(f"Found {len(json_files)} JSON summary files")
    
    # Process CSV files
    csv_results = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            task = os.path.basename(csv_file).split('_')[0]
            key = f"{task}"
            csv_results[key] = {
                'file': csv_file,
                'data': df,
                'task': task,
                'num_configs': len(df)
            }
            print(f"  Loaded CSV: {csv_file} ({len(df)} configurations)")
        except Exception as e:
            print(f"  Warning: Failed to load {csv_file}: {e}")
    
    # Process JSON files
    json_results = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            task = data.get('task', os.path.basename(json_file).split('_')[0])
            mode = data.get('mode', 'unknown')
            
            key = f"{task}_{mode}"
            json_results[key] = {
                'file': json_file,
                'data': data,
                'task': task,
                'mode': mode
            }
            print(f"  Loaded JSON: {json_file}")
        except Exception as e:
            print(f"  Warning: Failed to load {json_file}: {e}")
    
    # Process model checkpoints (if CSV/JSON not available)
    checkpoint_results = {}
    if not csv_files and not json_files:
        print("\nNo CSV/JSON files found. Extracting from model checkpoints...")
        print("Note: This will evaluate models on test sets to get test1/test2 accuracy.")
        print("This may take some time...")
        
        for idx, model_file in enumerate(model_files, 1):
            config = parse_config_name(model_file)
            if config:
                print(f"\nProcessing checkpoint {idx}/{len(model_files)}: {os.path.basename(model_file)}")
                checkpoint_data = extract_results_from_checkpoint(
                    model_file, 
                    evaluate_model=True  # Enable evaluation to get test1/test2 acc
                )
                if checkpoint_data:
                    task = config['task']
                    if task not in checkpoint_results:
                        checkpoint_results[task] = []
                    
                    result = {
                        **config,
                        **checkpoint_data
                    }
                    checkpoint_results[task].append(result)
                    print(f"  ✓ Extracted: Val Acc={result.get('best_val_acc', 'N/A')}, "
                          f"Test1 Acc={result.get('test1_acc', 'N/A')}, "
                          f"Test2 Acc={result.get('test2_acc', 'N/A')}")
    
    all_results = {
        'csv_results': csv_results,
        'json_results': json_results,
        'checkpoint_results': checkpoint_results,
        'total_model_files': len(model_files),
        'total_csv_files': len(csv_files),
        'total_json_files': len(json_files)
    }
    
    return all_results


def create_consolidated_results(collected_results: Dict, output_dir: str):
    """
    Create consolidated CSV and JSON files from collected results
    
    Args:
        collected_results: Results collected from directory
        output_dir: Directory to save consolidated results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Creating Consolidated Results")
    print(f"{'='*80}")
    
    # Consolidate CSV results
    all_csv_data = []
    for key, csv_data in collected_results['csv_results'].items():
        df = csv_data['data'].copy()
        df['source_file'] = os.path.basename(csv_data['file'])
        df['mode'] = csv_data['mode']
        all_csv_data.append(df)
    
    # Process checkpoint results if CSV files are not available
    if not all_csv_data and collected_results['checkpoint_results']:
        print("\nConverting checkpoint results to DataFrame...")
        for task, checkpoint_list in collected_results['checkpoint_results'].items():
            if checkpoint_list:
                # Convert checkpoint results to DataFrame format
                rows = []
                for checkpoint in checkpoint_list:
                    row = {
                        'task': checkpoint.get('task', task),
                        'config_name': checkpoint.get('config_name', 'unknown'),
                        'nperseg': checkpoint.get('nperseg') or checkpoint.get('stft_nperseg'),
                        'noverlap': checkpoint.get('noverlap') or checkpoint.get('stft_noverlap'),
                        'nfft': checkpoint.get('nfft') or checkpoint.get('stft_nfft'),
                        'overlap_ratio': checkpoint.get('overlap_pct', 0) / 100 if 'overlap_pct' in checkpoint else None,
                        'val_acc': checkpoint.get('best_val_acc'),
                        'test1_acc': None,  # Not available in checkpoint
                        'test2_acc': None,  # Not available in checkpoint
                        'model_path': checkpoint.get('model_path', ''),
                        'epoch': checkpoint.get('epoch'),
                    }
                    # Calculate overlap_ratio if not available
                    if row['overlap_ratio'] is None and row['nperseg'] and row['noverlap']:
                        row['overlap_ratio'] = row['noverlap'] / row['nperseg'] if row['nperseg'] > 0 else None
                    rows.append(row)
                
                if rows:
                    df = pd.DataFrame(rows)
                    df['source_file'] = 'checkpoint_extraction'
                    df['mode'] = 'checkpoint'
                    all_csv_data.append(df)
                    print(f"  Converted {len(rows)} configurations from {task} checkpoints")
    
    consolidated_df = None
    if all_csv_data:
        consolidated_df = pd.concat(all_csv_data, ignore_index=True)
        
        # Sort by task, then by validation accuracy
        consolidated_df = consolidated_df.sort_values(
            by=['task', 'val_acc'], 
            ascending=[True, False],
            na_position='last'
        )
        
        # Save consolidated CSV
        consolidated_csv = os.path.join(output_dir, 'consolidated_stft_27_results.csv')
        consolidated_df.to_csv(consolidated_csv, index=False)
        print(f"✓ Consolidated CSV saved: {consolidated_csv}")
        print(f"  Total configurations: {len(consolidated_df)}")
        
        # Create summary by task
        task_summaries = []
        for task in consolidated_df['task'].unique():
            task_df = consolidated_df[consolidated_df['task'] == task]
            successful = task_df[~task_df['val_acc'].isna()]
            
            if len(successful) > 0:
                best = successful.iloc[0]
                task_summaries.append({
                    'task': task,
                    'total_configs': len(task_df),
                    'successful_configs': len(successful),
                    'best_val_acc': best['val_acc'],
                    'best_test1_acc': best.get('test1_acc', None),
                    'best_test2_acc': best.get('test2_acc', None),
                    'best_config_name': best['config_name'],
                    'best_nperseg': best['nperseg'],
                    'best_noverlap': best['noverlap'],
                    'best_nfft': best['nfft'],
                })
        
        if task_summaries:
            summary_df = pd.DataFrame(task_summaries)
            summary_csv = os.path.join(output_dir, 'task_summary.csv')
            summary_df.to_csv(summary_csv, index=False)
            print(f"✓ Task summary CSV saved: {summary_csv}")
    
    # Consolidate JSON results
    all_json_data = {
        'timestamp': datetime.now().isoformat(),
        'tasks': {},
        'overall_stats': {}
    }
    
    for key, json_data in collected_results['json_results'].items():
        data = json_data['data']
        task = json_data['task']
        mode = json_data['mode']
        
        if task not in all_json_data['tasks']:
            all_json_data['tasks'][task] = {}
        
        all_json_data['tasks'][task][mode] = data
    
    # Calculate overall statistics
    if consolidated_df is not None:
        successful = consolidated_df[~consolidated_df['val_acc'].isna()]
        
        all_json_data['overall_stats'] = {
            'total_configurations_tested': len(consolidated_df),
            'successful_configurations': len(successful),
            'failed_configurations': len(consolidated_df) - len(successful),
            'average_val_acc': float(successful['val_acc'].mean()) if len(successful) > 0 else None,
            'best_val_acc': float(successful['val_acc'].max()) if len(successful) > 0 else None,
            'worst_val_acc': float(successful['val_acc'].min()) if len(successful) > 0 else None,
            'tasks_tested': list(consolidated_df['task'].unique()) if len(consolidated_df) > 0 else []
        }
    
    # Save consolidated JSON
    consolidated_json = os.path.join(output_dir, 'consolidated_stft_27_summary.json')
    with open(consolidated_json, 'w') as f:
        json.dump(all_json_data, f, indent=2)
    print(f"✓ Consolidated JSON saved: {consolidated_json}")
    
    # Create detailed analysis
    if consolidated_df is not None and len(consolidated_df) > 0:
        create_detailed_analysis(consolidated_df, output_dir)
    else:
        print("\n⚠ No data available to create detailed analysis")


def create_detailed_analysis(df: pd.DataFrame, output_dir: str):
    """
    Create detailed analysis of results
    
    Args:
        df: Consolidated dataframe
        output_dir: Directory to save analysis
    """
    print(f"\nCreating detailed analysis...")
    
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'by_task': {},
        'by_parameter': {},
        'top_configurations': {}
    }
    
    # Analysis by task
    for task in df['task'].unique():
        task_df = df[df['task'] == task].copy()
        successful = task_df[~task_df['val_acc'].isna()]
        
        if len(successful) > 0:
            analysis['by_task'][task] = {
                'total_configs': len(task_df),
                'successful_configs': len(successful),
                'best_val_acc': successful['val_acc'].max(),
                'worst_val_acc': successful['val_acc'].min(),
                'mean_val_acc': successful['val_acc'].mean(),
                'std_val_acc': successful['val_acc'].std(),
                'top_5_configs': successful.nlargest(5, 'val_acc')[
                    ['config_name', 'nperseg', 'noverlap', 'nfft', 'val_acc', 'test1_acc', 'test2_acc']
                ].to_dict('records')
            }
    
    # Analysis by parameter ranges
    successful_df = df[~df['val_acc'].isna()].copy()
    
    if len(successful_df) > 0:
        # By nperseg
        nperseg_analysis = successful_df.groupby('nperseg')['val_acc'].agg(['mean', 'std', 'count']).to_dict('index')
        analysis['by_parameter']['nperseg'] = nperseg_analysis
        
        # By overlap ratio
        if 'overlap_ratio' in successful_df.columns:
            overlap_analysis = successful_df.groupby('overlap_ratio')['val_acc'].agg(['mean', 'std', 'count']).to_dict('index')
            analysis['by_parameter']['overlap_ratio'] = overlap_analysis
        
        # By nfft
        nfft_analysis = successful_df.groupby('nfft')['val_acc'].agg(['mean', 'std', 'count']).to_dict('index')
        analysis['by_parameter']['nfft'] = nfft_analysis
        
        # Top configurations overall
        analysis['top_configurations']['top_10'] = successful_df.nlargest(
            10, 'val_acc'
        )[['task', 'config_name', 'nperseg', 'noverlap', 'nfft', 'val_acc', 'test1_acc', 'test2_acc']].to_dict('records')
    
    # Save analysis
    analysis_json = os.path.join(output_dir, 'detailed_analysis.json')
    with open(analysis_json, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Detailed analysis saved: {analysis_json}")
    
    # Create parameter effect analysis CSV
    if len(successful_df) > 0:
        param_effects = []
        
        # Analyze nperseg effect
        for nperseg in successful_df['nperseg'].unique():
            subset = successful_df[successful_df['nperseg'] == nperseg]
            param_effects.append({
                'parameter': 'nperseg',
                'value': nperseg,
                'mean_acc': subset['val_acc'].mean(),
                'std_acc': subset['val_acc'].std(),
                'count': len(subset),
                'best_acc': subset['val_acc'].max()
            })
        
        # Analyze nfft effect
        for nfft in successful_df['nfft'].unique():
            subset = successful_df[successful_df['nfft'] == nfft]
            param_effects.append({
                'parameter': 'nfft',
                'value': nfft,
                'mean_acc': subset['val_acc'].mean(),
                'std_acc': subset['val_acc'].std(),
                'count': len(subset),
                'best_acc': subset['val_acc'].max()
            })
        
        # Analyze overlap effect
        if 'overlap_ratio' in successful_df.columns:
            for overlap in successful_df['overlap_ratio'].unique():
                subset = successful_df[successful_df['overlap_ratio'] == overlap]
                param_effects.append({
                    'parameter': 'overlap_ratio',
                    'value': overlap,
                    'mean_acc': subset['val_acc'].mean(),
                    'std_acc': subset['val_acc'].std(),
                    'count': len(subset),
                    'best_acc': subset['val_acc'].max()
                })
        
        param_effects_df = pd.DataFrame(param_effects)
        param_effects_csv = os.path.join(output_dir, 'parameter_effects_analysis.csv')
        param_effects_df.to_csv(param_effects_csv, index=False)
        print(f"✓ Parameter effects analysis saved: {param_effects_csv}")


def main(results_dir: str, output_dir: str):
    """
    Main function to analyze and organize results
    
    Args:
        results_dir: Directory containing ablation results
        output_dir: Directory to save organized results
    """
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Collect results
    collected_results = collect_results_from_directory(results_dir)
    
    # Create consolidated results
    create_consolidated_results(collected_results, output_dir)
    
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - consolidated_stft_27_results.csv")
    print(f"  - task_summary.csv")
    print(f"  - consolidated_stft_27_summary.json")
    print(f"  - detailed_analysis.json")
    print(f"  - parameter_effects_analysis.csv")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze and Organize STFT 27-Config Ablation Results'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./ablation_results',
        help='Directory containing ablation results (default: ./ablation_results)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./analysis',
        help='Directory to save organized results (default: ./analysis)'
    )
    
    args = parser.parse_args()
    
    main(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
