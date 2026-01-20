import torch
import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
import os
from datetime import datetime
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# Import from model_cldnn_se.py
from model_cldnn_se import (
    train_task,
    setup_device,
    TASK_CONFIGS
)

# Baseline thresholds from paper (best performance for each task)
BASELINE_THRESHOLDS = {
    'SSVEP': {
        'seen': 86.79,    # MultiDiffNet + Mixup
        'unseen': 85.25,  # MultiDiffNet + Mixup
    },
    'P300': {
        'seen': 88.79,    # EEGNet
        'unseen': 87.24,  # EEGNet
    },
    'MI': {
        'seen': 67.01,    # EEGNet
        'unseen': 46.18,  # EEGNet
    },
    'Imagined_speech': {
        'seen': 17.57,    # MultiDiffNet + Mixup
        'unseen': 12.12,  # MultiDiffNet + Mixup
    }
}


class OptunaCallback:
    """Callback for early stopping within a trial"""
    def __init__(self, trial: Trial, monitor: str = 'val_acc'):
        self.trial = trial
        self.monitor = monitor
        
    def __call__(self, epoch: int, metrics: Dict):
        # Report intermediate value
        value = metrics.get(self.monitor)
        if value is not None:
            self.trial.report(value, epoch)
            
            # Prune trial if not promising
            if self.trial.should_prune():
                raise optuna.TrialPruned()


def objective(trial: Trial, task: str, base_config: Dict, tune_config: Dict, 
              baseline_thresholds: Optional[Dict] = None) -> float:
    """
    Objective function for Optuna
    
    Args:
        trial: Optuna trial object
        task: EEG task name
        base_config: Base configuration (fixed parameters)
        tune_config: Tuning configuration (search spaces)
        baseline_thresholds: Dict with 'seen' and 'unseen' thresholds
        
    Returns:
        Best validation accuracy
    """
    
    # Build config by suggesting hyperparameters
    config = base_config.copy()
    
    # ====== Architecture Parameters ======
    if tune_config.get('tune_cnn_filters', True):
        config['cnn_filters'] = trial.suggest_categorical('cnn_filters', [8, 16, 32, 64])
    
    if tune_config.get('tune_lstm_hidden', True):
        config['lstm_hidden'] = trial.suggest_categorical('lstm_hidden', [32, 64, 128, 256])
    
    if tune_config.get('tune_pos_dim', True):
        config['pos_dim'] = trial.suggest_categorical('pos_dim', [8, 16, 32, 64])
    
    # ====== Hidden Layer ======
    if tune_config.get('tune_hidden_layer', True):
        config['use_hidden_layer'] = trial.suggest_categorical('use_hidden_layer', [True, False])
        if config['use_hidden_layer']:
            config['hidden_dim'] = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
    
    # ====== Regularization ======
    if tune_config.get('tune_dropout', True):
        config['dropout'] = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        config['cnn_dropout'] = trial.suggest_float('cnn_dropout', 0.1, 0.4, step=0.1)
    
    if tune_config.get('tune_weight_decay', True):
        config['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    
    # ====== Training Parameters ======
    if tune_config.get('tune_lr', True):
        config['lr'] = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    
    if tune_config.get('tune_batch_size', True):
        config['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # ====== STFT Parameters (if enabled) ======
    if tune_config.get('tune_stft', False):
        task_config = TASK_CONFIGS.get(task, {})
        fs = task_config.get('sampling_rate', 250)
        
        # nperseg: window size (ms)
        nperseg_ms = trial.suggest_categorical('stft_nperseg_ms', [100, 250, 500, 1000])
        config['stft_nperseg'] = int(nperseg_ms * fs / 1000)
        
        # overlap percentage
        overlap_pct = trial.suggest_categorical('stft_overlap_pct', [50, 75, 87.5, 93.75])
        config['stft_noverlap'] = int(config['stft_nperseg'] * overlap_pct / 100)
        
        # nfft (power of 2)
        config['stft_nfft'] = trial.suggest_categorical('stft_nfft', [256, 512, 1024, 2048])
    
    # ====== Scheduler ======
    if tune_config.get('tune_scheduler', False):
        config['scheduler'] = trial.suggest_categorical('scheduler', 
                                                        ['ReduceLROnPlateau', 'CosineAnnealingLR'])
    
    # Print trial parameters
    print(f"\n{'='*70}")
    print(f"Trial #{trial.number} Parameters:")
    print(f"{'='*70}")
    for key, value in config.items():
        if key.startswith('stft_') or key in ['cnn_filters', 'lstm_hidden', 'pos_dim', 
                                                'dropout', 'cnn_dropout', 'lr', 'batch_size',
                                                'weight_decay', 'use_hidden_layer', 'hidden_dim',
                                                'scheduler']:
            print(f"  {key}: {value}")
    
    try:
        # Train model
        model_path = f'./optuna_trials/trial_{trial.number}_{task.lower()}_model.pth'
        os.makedirs('./optuna_trials', exist_ok=True)
        
        print(f"\n[Trial {trial.number}] Starting training...")
        model, results = train_task(
            task=task,
            config=config,
            model_path=model_path
        )
        
        print(f"[Trial {trial.number}] Training completed. Results: {results}")
        
        # Get validation and test accuracies
        val_acc = results.get('val', 0.0)
        test1_acc = results.get('test1', 0.0)  # Seen test
        test2_acc = results.get('test2', 0.0)  # Unseen test
        
        print(f"[Trial {trial.number}] Accuracies - Val: {val_acc:.2f}%, Test1: {test1_acc:.2f}%, Test2: {test2_acc:.2f}%")
        
        # Check if meets baseline threshold (either seen OR unseen)
        meets_baseline = True
        if baseline_thresholds is not None:
            seen_threshold = baseline_thresholds.get('seen', 0.0)
            unseen_threshold = baseline_thresholds.get('unseen', 0.0)
            
            # Check if EITHER test1 (seen) or test2 (unseen) exceeds baseline
            seen_pass = test1_acc >= seen_threshold
            unseen_pass = test2_acc >= unseen_threshold
            meets_baseline = seen_pass or unseen_pass
            
            print(f"\nTrial {trial.number} Performance vs Baseline:")
            print(f"  Seen:   {test1_acc:.2f}% {'✓' if seen_pass else '✗'} (threshold: {seen_threshold:.2f}%)")
            print(f"  Unseen: {test2_acc:.2f}% {'✓' if unseen_pass else '✗'} (threshold: {unseen_threshold:.2f}%)")
            
            if meets_baseline:
                print(f"✓ Trial {trial.number} PASSED: Meets baseline on {'both' if (seen_pass and unseen_pass) else 'seen' if seen_pass else 'unseen'}")
            else:
                print(f"✗ Trial {trial.number} FAILED: Does not meet baseline on either test set")
                # Delete model file if below baseline
                if os.path.exists(model_path):
                    os.remove(model_path)
                    print(f"  Model deleted: {model_path}")
        
        # Save trial results
        trial.set_user_attr('val_acc', val_acc)
        trial.set_user_attr('test1_acc', test1_acc)
        trial.set_user_attr('test2_acc', test2_acc)
        trial.set_user_attr('meets_baseline', meets_baseline)
        
        return val_acc
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0  # Return worst score on failure


def auto_tune(
    task: str = 'SSVEP',
    n_trials: int = 50,
    timeout: Optional[int] = None,
    base_config: Optional[Dict] = None,
    tune_config: Optional[Dict] = None,
    study_name: Optional[str] = None,
    save_dir: str = './optuna_results',
    use_baseline: bool = True,
    custom_baseline: Optional[Dict] = None
) -> optuna.Study:
    """
    Automatic hyperparameter tuning using Optuna
    
    Args:
        task: EEG task name ('SSVEP', 'P300', 'MI', 'Imagined_speech')
        n_trials: Number of trials to run
        timeout: Time limit in seconds (None for no limit)
        base_config: Base configuration (fixed parameters)
        tune_config: Tuning configuration (which parameters to tune)
        study_name: Name for the study (auto-generated if None)
        save_dir: Directory to save results
        use_baseline: Whether to use baseline thresholds (default: True)
        custom_baseline: Custom baseline dict {'seen': X, 'unseen': Y} (overrides default)
        
    Returns:
        Optuna study object
    """
    
    # Default base config
    if base_config is None:
        task_config = TASK_CONFIGS.get(task, {})
        base_config = {
            'num_seen': task_config.get('num_seen', 33),
            'seed': 44,
            'n_classes': task_config.get('num_classes', 26),
            'num_epochs': 50,  # Reduced for faster tuning
            'patience': 10,    # Early stopping
            # Task-specific STFT defaults
            'stft_fs': task_config.get('sampling_rate', 250),
            'stft_nperseg': task_config.get('stft_nperseg', 128),
            'stft_noverlap': task_config.get('stft_noverlap', 112),
            'stft_nfft': task_config.get('stft_nfft', 512),
        }
    
    # Default tune config (tune everything)
    if tune_config is None:
        tune_config = {
            'tune_cnn_filters': True,
            'tune_lstm_hidden': True,
            'tune_pos_dim': True,
            'tune_hidden_layer': True,
            'tune_dropout': True,
            'tune_weight_decay': True,
            'tune_lr': True,
            'tune_batch_size': True,
            'tune_stft': False,  # STFT tuning is expensive, disabled by default
            'tune_scheduler': False,
        }
    
    # Create study name
    if study_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        study_name = f"{task}_tune_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"AUTOMATIC HYPERPARAMETER TUNING")
    print(f"{'='*80}")
    print(f"Task: {task}")
    print(f"Study: {study_name}")
    print(f"Trials: {n_trials}")
    print(f"Timeout: {timeout if timeout else 'None'}")
    print(f"Save Dir: {save_dir}")
    print(f"\nBase Config:")
    for key, value in base_config.items():
        print(f"  {key}: {value}")
    print(f"\nTuning:")
    for key, value in tune_config.items():
        if value:
            print(f"  ✓ {key.replace('tune_', '')}")
    print(f"{'='*80}\n")
    
    # Setup baseline thresholds
    baseline_thresholds = None
    if use_baseline:
        if custom_baseline is not None:
            baseline_thresholds = custom_baseline
        else:
            # Default baseline thresholds (can be adjusted per task)
            baseline_thresholds = {
                'seen': 50.0,    # 50% accuracy on seen subjects
                'unseen': 30.0   # 30% accuracy on unseen subjects
            }
        print(f"Baseline Thresholds:")
        print(f"  Seen:   {baseline_thresholds['seen']:.2f}%")
        print(f"  Unseen: {baseline_thresholds['unseen']:.2f}%")
        print()
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # Maximize validation accuracy
        sampler=TPESampler(seed=base_config['seed']),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Create objective wrapper with proper closure
    def optimization_objective(trial):
        return objective(trial, task, base_config, tune_config, baseline_thresholds)
    
    # Run optimization
    study.optimize(
        optimization_objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        catch=(Exception,)
    )
    
    # Check if any trials completed successfully
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"\n{'='*80}")
    print(f"TRIAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed: {len(completed_trials)}")
    print(f"Failed: {len(failed_trials)}")
    
    if len(completed_trials) == 0:
        print(f"\n{'='*80}")
        print(f"ERROR: NO TRIALS COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"All {len(study.trials)} trials failed.")
        print(f"Please check the error messages above.")
        print(f"{'='*80}")
        
        # Print failed trial details
        if failed_trials:
            print(f"\nFailed trial details:")
            for trial in failed_trials[:5]:  # Show first 5 failures
                print(f"  Trial {trial.number}: {trial.system_attrs.get('fail_reason', 'Unknown error')}")
        
        return study
    
    # Check if best_trial exists and is valid
    if study.best_trial is None or study.best_trial.state != optuna.trial.TrialState.COMPLETE:
        print(f"\n{'='*80}")
        print(f"WARNING: BEST TRIAL IS NOT VALID")
        print(f"{'='*80}")
        print(f"Best trial state: {study.best_trial.state if study.best_trial else 'None'}")
        print(f"Selecting best from completed trials...")
        
        # Find best from completed trials
        best_completed = max(completed_trials, key=lambda t: t.value if t.value is not None else -1)
        study.best_trial = best_completed
        study.best_value = best_completed.value if best_completed.value is not None else 0.0
    
    # Print results
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETED")
    print(f"{'='*80}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.2f}%")
    
    # Print test accuracies
    if study.best_trial.user_attrs:
        if 'test1_acc' in study.best_trial.user_attrs:
            test1_acc = study.best_trial.user_attrs['test1_acc']
            print(f"Test1 (Seen) accuracy: {test1_acc:.2f}%")
            if baseline_thresholds:
                improvement = test1_acc - baseline_thresholds.get('seen', 0.0)
                print(f"  Improvement over baseline: {improvement:+.2f}%")
        
        if 'test2_acc' in study.best_trial.user_attrs:
            test2_acc = study.best_trial.user_attrs['test2_acc']
            print(f"Test2 (Unseen) accuracy: {test2_acc:.2f}%")
            if baseline_thresholds:
                improvement = test2_acc - baseline_thresholds.get('unseen', 0.0)
                print(f"  Improvement over baseline: {improvement:+.2f}%")
        
        # Check if best trial meets baseline
        if baseline_thresholds and 'meets_baseline' in study.best_trial.user_attrs:
            if study.best_trial.user_attrs['meets_baseline']:
                print(f"\n✓ Best model PASSES baseline criteria!")
            else:
                print(f"\n✗ Best model does NOT pass baseline criteria")
    else:
        print(f"\n⚠ WARNING: Best trial has no user attributes (user_attrs is empty)")
        print(f"This may indicate the trial did not complete successfully.")
    
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Print test accuracies if available
    if study.best_trial.user_attrs:
        if 'test1_acc' in study.best_trial.user_attrs:
            print(f"\nTest1 accuracy: {study.best_trial.user_attrs['test1_acc']:.2f}%")
        if 'test2_acc' in study.best_trial.user_attrs:
            print(f"Test2 accuracy: {study.best_trial.user_attrs['test2_acc']:.2f}%")
        if 'meets_baseline' in study.best_trial.user_attrs:
            meets = study.best_trial.user_attrs['meets_baseline']
            print(f"Meets baseline: {'✓ Yes' if meets else '✗ No'}")
    
    # Save results
    results_path = os.path.join(save_dir, f'{study_name}_results.json')
    
    # Prepare best_trial info safely
    best_trial_num = study.best_trial.number if study.best_trial else None
    best_user_attrs = study.best_trial.user_attrs if study.best_trial else {}
    best_params = study.best_params if study.best_trial else {}
    
    results = {
        'study_name': study_name,
        'task': task,
        'n_trials': len(study.trials),
        'n_completed': len(completed_trials),
        'n_failed': len(failed_trials),
        'best_trial': best_trial_num,
        'best_value': study.best_value,
        'best_params': best_params,
        'best_user_attrs': best_user_attrs,
        'base_config': base_config,
        'tune_config': tune_config,
        'baseline_thresholds': baseline_thresholds,
    }
    
    # Add warning if best_trial has no user_attrs
    if not best_user_attrs:
        results['warning'] = 'Best trial has no user_attrs - training may not have completed successfully'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Save study for later analysis
    study_path = os.path.join(save_dir, f'{study_name}_study.pkl')
    import joblib
    joblib.dump(study, study_path)
    print(f"Study saved to: {study_path}")
    
    # Generate optimization history plot
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        import matplotlib.pyplot as plt
        
        # Optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_html(os.path.join(save_dir, f'{study_name}_history.html'))
        
        # Parameter importance
        fig2 = plot_param_importances(study)
        fig2.write_html(os.path.join(save_dir, f'{study_name}_importance.html'))
        
        print(f"Plots saved to: {save_dir}")
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print(f"{'='*80}\n")
    
    return study


def tune_all_tasks(
    tasks: Optional[list] = None,
    n_trials: int = 50,
    save_dir: str = './optuna_results',
    **kwargs
) -> Dict[str, optuna.Study]:
    """
    Tune hyperparameters for all tasks
    
    Args:
        tasks: List of task names (default: all tasks)
        n_trials: Number of trials per task
        save_dir: Directory to save results
        **kwargs: Additional arguments for auto_tune
        
    Returns:
        Dictionary of studies for each task
    """
    if tasks is None:
        tasks = ['SSVEP', 'P300', 'MI', 'Imagined_speech']
    
    studies = {}
    
    print("=" * 80)
    print("MULTI-TASK HYPERPARAMETER TUNING")
    print("=" * 80)
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TUNING: {task}")
        print(f"{'='*60}")
        
        try:
            study = auto_tune(
                task=task,
                n_trials=n_trials,
                save_dir=os.path.join(save_dir, task.lower()),
                **kwargs
            )
            studies[task] = study
            
        except Exception as e:
            print(f"Error tuning {task}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("TUNING SUMMARY")
    print(f"{'='*80}")
    
    for task, study in studies.items():
        print(f"\n{task}:")
        print(f"  Best Val Acc: {study.best_value:.2f}%")
        if 'test1_acc' in study.best_trial.user_attrs:
            print(f"  Test1 Acc: {study.best_trial.user_attrs['test1_acc']:.2f}%")
        if 'test2_acc' in study.best_trial.user_attrs:
            print(f"  Test2 Acc: {study.best_trial.user_attrs['test2_acc']:.2f}%")
        print(f"  Best Trial: #{study.best_trial.number}")
    
    print(f"\n{'='*80}")
    print("MULTI-TASK TUNING COMPLETED!")
    print(f"{'='*80}")
    
    return studies


def load_and_apply_best_params(results_path: str) -> Dict:
    """
    Load best parameters from tuning results
    
    Args:
        results_path: Path to results JSON file
        
    Returns:
        Configuration dict with best parameters
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    config = results['base_config'].copy()
    config.update(results['best_params'])
    
    return config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Auto-tune SCALENet/AdaptiveSCALENet with Multi-GPU support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        
            Examples:
            # Basic usage (uses all available GPUs automatically)
            python auto_tune.py --task SSVEP --n_trials 50
            
            # Tune all tasks
            python auto_tune.py --task all --n_trials 30
            
            # Custom baseline thresholds
            python auto_tune.py --task SSVEP --n_trials 50 --custom_seen 90.0 --custom_unseen 85.0
            
            # Disable baseline filtering
            python auto_tune.py --task SSVEP --n_trials 50 --no_baseline
            
            # Tune STFT parameters (more expensive)
            python auto_tune.py --task SSVEP --n_trials 100 --tune_stft
            
            # Set time limit (2 hours)
            python auto_tune.py --task SSVEP --n_trials 100 --timeout 7200

            GPU Usage:
            - All available GPUs are automatically detected and used via DataParallel
            - No additional configuration needed for multi-GPU training
            - Each trial will use all available GPUs for faster training
            
        """
    )
    parser.add_argument('--task', type=str, default='SSVEP',
                        choices=['SSVEP', 'P300', 'MI', 'Imagined_speech', 'all'],
                        help='Task to tune')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of tuning trials')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Time limit in seconds')
    parser.add_argument('--save_dir', type=str, default='./optuna_results',
                        help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max epochs per trial (reduced for faster tuning)')
    parser.add_argument('--tune_stft', action='store_true',
                        help='Also tune STFT parameters (expensive)')
    parser.add_argument('--no_baseline', action='store_true',
                        help='Disable baseline threshold filtering')
    parser.add_argument('--custom_seen', type=float, default=None,
                        help='Custom seen test baseline threshold')
    parser.add_argument('--custom_unseen', type=float, default=None,
                        help='Custom unseen test baseline threshold')
    
    args = parser.parse_args()
    
    # Base config
    base_config = {
        'num_epochs': args.epochs,
        'patience': 10,
        'seed': 44,
    }
    
    # Tune config
    tune_config = {
        'tune_cnn_filters': True,
        'tune_lstm_hidden': True,
        'tune_pos_dim': True,
        'tune_hidden_layer': True,
        'tune_dropout': True,
        'tune_weight_decay': True,
        'tune_lr': True,
        'tune_batch_size': True,
        'tune_stft': args.tune_stft,
        'tune_scheduler': False,
    }
    
    # Custom baseline
    custom_baseline = None
    if args.custom_seen is not None or args.custom_unseen is not None:
        custom_baseline = {
            'seen': args.custom_seen if args.custom_seen is not None else 0.0,
            'unseen': args.custom_unseen if args.custom_unseen is not None else 0.0,
        }
    
    if args.task == 'all':
        studies = tune_all_tasks(
            n_trials=args.n_trials,
            timeout=args.timeout,
            save_dir=args.save_dir,
            base_config=base_config,
            tune_config=tune_config,
            use_baseline=not args.no_baseline,
            custom_baseline=custom_baseline
        )
    else:
        study = auto_tune(
            task=args.task,
            n_trials=args.n_trials,
            timeout=args.timeout,
            save_dir=args.save_dir,
            base_config=base_config,
            tune_config=tune_config,
            use_baseline=not args.no_baseline,
            custom_baseline=custom_baseline
        )