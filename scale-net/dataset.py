"""
Multi-Task EEG Dataset Loaders
Supports: SSVEP, P300, MI (Motor Imagery), Imagined Speech

Based on spectral_cnn_lstm3.ipynb implementation
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal
from collections import defaultdict
from typing import Dict, Tuple, Optional, List
import pandas as pd
import scipy.io

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: mne not installed. MI data loading will not work.")


# ==================== Task Configurations ====================

TASK_CONFIGS = {
    "SSVEP": {
        "num_classes": 26,
        "num_subjects": 35,
        "num_seen": 33,
        "data_dir": "/ocean/projects/cis250213p/shared/ssvep",
        "sampling_rate": 250,
        "stft_nperseg": 128,
        "stft_noverlap": 112,
        "stft_nfft": 512,
    },
    "Lee2019_SSVEP": {
        "num_classes": 4,
        "num_subjects": 54,
        "num_seen": 40,
        "data_dir": "/ocean/projects/cis250213p/shared/lee2019_ssvep_processed",
        "sampling_rate": 250,
        "stft_nperseg": 128,
        "stft_noverlap": 112,
        "stft_nfft": 512,
    },
    "P300": {
        "num_classes": 2,
        "num_subjects": 43,
        "num_seen": 36,
        "data_dir": "/ocean/projects/cis250213p/shared/p300",
        "sampling_rate": 256,
        "stft_nperseg": 256,
        "stft_noverlap": 240,
        "stft_nfft": 512,
    },
    "BNCI2014_P300": {
        "num_classes": 2,
        "num_subjects": 10,
        "num_seen": 7,
        "data_dir": "/ocean/projects/cis250213p/shared/bnci2014_p300",
        "sampling_rate": 512,
        "stft_nperseg": 256,
        "stft_noverlap": 240,
        "stft_nfft": 512,
    },
    "MI": {
        "num_classes": 4,
        "num_subjects": 9,
        "num_seen": 7,
        "data_dir": "/ocean/projects/cis250213p/shared/mi",
        "sampling_rate": 250,
        "stft_nperseg": 128,
        "stft_noverlap": 112,
        "stft_nfft": 512,
    },
    "Lee2019_MI": {
        "num_classes": 2,
        "num_subjects": 54,
        "num_seen": 40,
        "data_dir": "/ocean/projects/cis250213p/shared/lee2019_mi_processed",
        "sampling_rate": 250,
        "stft_nperseg": 128,
        "stft_noverlap": 112,
        "stft_nfft": 512,
    },
    "Imagined_speech": {
        "num_classes": 11,
        "num_subjects": 14,
        "num_seen": 8,
        "data_dir": "/ocean/projects/cis250213p/shared/img_speech",
        "sampling_rate": 1000,
        "stft_nperseg": 256,
        "stft_noverlap": 224,
        "stft_nfft": 512,
    },
}

# ==================== STFT Transformation ====================

def apply_stft_transform(data, fs=250, nperseg=128, noverlap=112, nfft=512):
    """
    Apply STFT transform to EEG data
    
    Args:
        data: (samples, channels, time) or (channels, time)
        fs: sampling frequency
        nperseg: length of each segment
        noverlap: overlap between segments
        nfft: FFT size
        
    Returns:
        (samples, channels, freq_bins, time_bins) or (channels, freq_bins, time_bins)
    """
    single_sample = False
    if len(data.shape) == 2:
        data = data[np.newaxis, :, :]
        single_sample = True
    
    n_samples, n_channels, n_times = data.shape
    stft_data = []
    
    for sample_idx in range(n_samples):
        channels_stft = []
        for ch in range(n_channels):
            f, t, Zxx = signal.stft(
                data[sample_idx, ch, :],
                fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft
            )
            power = np.abs(Zxx) ** 2
            channels_stft.append(power)
        stft_data.append(np.stack(channels_stft, axis=0))
    
    stft_data = np.array(stft_data)
    
    if single_sample:
        return stft_data[0]
    return stft_data

# ==================== MI Lee2019 Data Loading ====================

def load_lee2019_data(data_dir: str, num_seen: int = 40, seed: int = 44) -> Dict:

    all_subjects = list(range(1, 55)) # 54 subjects total
    
    random.seed(seed)
    seen_subjects = random.sample(all_subjects, num_seen)
    unseen_subjects = [s for s in all_subjects if s not in seen_subjects]

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test1, y_test1 = [], [] 
    X_test2, y_test2 = [], [] 

    print(f"Source Directory: {data_dir}")

    # 2. Load Seen Subjects (Split into Train/Val/Test1)
    for sid in seen_subjects:
        file_path = os.path.join(data_dir, f"S{sid}_preprocessed.npz")
        
        if not os.path.exists(file_path):
            print(f"  [WARNING] Preprocessed file for Subject {sid} not found.")
            continue
            
        with np.load(file_path) as data:
            X = data['X'] # Shape: (Total_Trials, 22, 1000)
            y = data['y'] # Shape: (Total_Trials,)
            
            # Since we combined sessions in preprocessing, we split by trial index
            # Standard 70/15/15 split for the seen subject's data
            n_trials = len(X)
            train_split = int(n_trials * 0.7)
            val_split = int(n_trials * 0.85)
            
            X_train.append(X[:train_split])
            y_train.append(y[:train_split])
            
            X_val.append(X[train_split:val_split])
            y_val.append(y[train_split:val_split])
            
            X_test1.append(X[val_split:])
            y_test1.append(y[val_split:])

    # 3. Load Unseen Subjects (Test2)
    for sid in unseen_subjects:
        file_path = os.path.join(data_dir, f"S{sid}_preprocessed.npz")
        if os.path.exists(file_path):
            with np.load(file_path) as data:
                X_test2.append(data['X'])
                y_test2.append(data['y'])

    # 4. Final Concatenation
    result = {}
    splits = [
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test1', X_test1, y_test1),
        ('test2', X_test2, y_test2)
    ]

    for name, X_list, y_list in splits:
        if X_list:
            result[name] = (np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0))
            print(f"  Loaded {name}: {result[name][0].shape}")
        else:
            print(f"  [ERROR] {name} split is empty!")

    return result
# ==================== SSVEP Data Loading ====================

def load_ssvep_data(data_dir: str, num_seen: int = 33, seed: int = 44) -> Dict:
    """
    Load SSVEP data with subject-wise split
    
    Returns:
        Dictionary with 'train', 'val', 'test1', 'test2' splits
    """
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Loading SSVEP data from: {data_dir}")
    
    all_subjects = list(range(1, 36))
    random.shuffle(all_subjects)
    
    seen_subjects = all_subjects[:num_seen]
    unseen_subjects = all_subjects[num_seen:]
    
    print(f"[SSVEP Split] Seen: {len(seen_subjects)} subjects")
    print(f"[SSVEP Split] Unseen: {unseen_subjects}")
    
    train_sessions = [0, 1, 2, 3]
    val_session = 4
    test1_session = 5
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test1, y_test1 = [], []
    X_test2, y_test2 = [], []
    
    for sid in seen_subjects:
        filepath = os.path.join(data_dir, f"S{sid}_chars.npy")
        if not os.path.exists(filepath):
            continue
        data = np.load(filepath)
        
        for char_idx in range(26):
            for sess in train_sessions:
                X_train.append(data[char_idx, sess])
                y_train.append(char_idx)
            X_val.append(data[char_idx, val_session])
            y_val.append(char_idx)
            X_test1.append(data[char_idx, test1_session])
            y_test1.append(char_idx)
    
    for sid in unseen_subjects:
        filepath = os.path.join(data_dir, f"S{sid}_chars.npy")
        if not os.path.exists(filepath):
            continue
        data = np.load(filepath)
        
        for char_idx in range(26):
            for sess in range(6):
                X_test2.append(data[char_idx, sess])
                y_test2.append(char_idx)
    
    datasets = {
        'train': (np.array(X_train), np.array(y_train)),
        'val': (np.array(X_val), np.array(y_val)),
        'test1': (np.array(X_test1), np.array(y_test1)),
        'test2': (np.array(X_test2), np.array(y_test2))
    }
    
    print(f"  Train: {datasets['train'][0].shape}")
    print(f"  Val: {datasets['val'][0].shape}")
    print(f"  Test1 (seen): {datasets['test1'][0].shape}")
    print(f"  Test2 (unseen): {datasets['test2'][0].shape}")
    
    return datasets


# ==================== P300 Data Loading ====================

def _P300_load_subject_data(subject_id: int, root_dir: str) -> Dict:
    """Load P300 data for one subject"""
    folder = os.path.join(root_dir, f"subject_{subject_id:02d}")
    X = np.load(os.path.join(folder, "X.npy"))  # (n_trials, C, T)
    Y = np.load(os.path.join(folder, "y.npy"))  # (n_trials,)
    Y = np.array([1 if label == 'Target' else 0 for label in Y])

    meta = pd.read_csv(os.path.join(folder, "metadata.csv"))

    trials_per_repetition = 12
    reps_per_level = 8

    level_list = []
    repetition_list = []

    for sess in sorted(meta["session"].unique()):
        session_idxs = meta.index[meta["session"] == sess].tolist()
        for i, idx in enumerate(session_idxs):
            rep = i // trials_per_repetition
            level = rep // reps_per_level
            repetition = rep % reps_per_level
            level_list.append(level)
            repetition_list.append(repetition)

    meta["level"] = level_list
    meta["repetition"] = repetition_list

    return {
        "X": X,
        "Y": Y,
        "session": meta["session"].tolist(),
        "level": meta["level"].tolist(),
        "repetition": meta["repetition"].tolist()
    }


def _P300_split_repetitions(subject_data: Dict, min_reps: int = 3) -> Tuple[List, List, List]:
    """Split P300 data by repetitions"""
    repetition_arr = np.array(subject_data["repetition"])
    
    train_idx, val_idx, test_idx = [], [], []
    
    for rep in range(max(repetition_arr) + 1):
        rep_indices = np.where(repetition_arr == rep)[0]
        if len(rep_indices) >= min_reps:
            n_train = int(len(rep_indices) * 0.6)
            n_val = int(len(rep_indices) * 0.2)
            
            train_idx.extend(rep_indices[:n_train])
            val_idx.extend(rep_indices[n_train:n_train + n_val])
            test_idx.extend(rep_indices[n_train + n_val:])
    
    return train_idx, val_idx, test_idx


def load_p300_data(data_dir: str, num_seen: int = 36, seed: int = 44) -> Dict:
    """
    Load P300 dataset with proper split
    
    Returns:
        Dictionary with 'train', 'val', 'test1', 'test2' splits
    """
    random.seed(seed)
    
    # Find available subjects (excluding problematic ones)
    all_subjects = sorted([
        int(f.split('_')[1]) 
        for f in os.listdir(data_dir)
        if f.startswith('subject_') and os.path.isdir(os.path.join(data_dir, f))
    ])
    all_subjects = [s for s in all_subjects if s not in [1, 27]]  # Exclude problematic subjects
    
    if num_seen >= len(all_subjects):
        print(f"Warning: Not enough subjects. Requested {num_seen}, available {len(all_subjects)}")
        num_seen = len(all_subjects) - 2
        
    seen_subjects = random.sample(all_subjects, num_seen)
    unseen_subjects = [s for s in all_subjects if s not in seen_subjects]

    print(f"[P300 Split] Seen: {len(seen_subjects)} subjects")
    print(f"[P300 Split] Unseen: {unseen_subjects}")

    X_train_all, Y_train_all = [], []
    X_val_all, Y_val_all = [], []
    X_test1_all, Y_test1_all = [], []

    for sid in seen_subjects:
        try:
            data = _P300_load_subject_data(sid, data_dir)
            X, Y = torch.tensor(data['X']).float(), torch.tensor(data['Y']).long()
            train_idx, val_idx, test1_idx = _P300_split_repetitions(data)
            
            X_train_all.append(X[train_idx])
            Y_train_all.append(Y[train_idx])
            X_val_all.append(X[val_idx])
            Y_val_all.append(Y[val_idx])
            X_test1_all.append(X[test1_idx])
            Y_test1_all.append(Y[test1_idx])
        except Exception as e:
            print(f"Error loading P300 subject {sid}: {e}")
            continue

    X_test2_all, Y_test2_all = [], []
    for sid in unseen_subjects:
        try:
            data = _P300_load_subject_data(sid, data_dir)
            X, Y = torch.tensor(data['X']).float(), torch.tensor(data['Y']).long()
            X_test2_all.append(X)
            Y_test2_all.append(Y)
        except Exception as e:
            print(f"Error loading P300 subject {sid}: {e}")
            continue

    datasets = {}
    if X_train_all:
        datasets['train'] = (torch.cat(X_train_all, dim=0).numpy(), torch.cat(Y_train_all, dim=0).numpy())
    if X_val_all:
        datasets['val'] = (torch.cat(X_val_all, dim=0).numpy(), torch.cat(Y_val_all, dim=0).numpy())
    if X_test1_all:
        datasets['test1'] = (torch.cat(X_test1_all, dim=0).numpy(), torch.cat(Y_test1_all, dim=0).numpy())
    if X_test2_all:
        datasets['test2'] = (torch.cat(X_test2_all, dim=0).numpy(), torch.cat(Y_test2_all, dim=0).numpy())

    for split in datasets:
        print(f"  {split}: {datasets[split][0].shape}")

    return datasets

def load_bnci_p300(data_dir: str, num_seen: int = 7, seed: int = 44) -> Dict:

    random.seed(seed)
    np.random.seed(seed)
    
    # 1. Subject-level split (Seen vs. Unseen)
    # BNCI2014-009 has 10 subjects: A01S.mat to A10S.mat
    all_subjects = list(range(1, 11)) 
    random.shuffle(all_subjects)
    seen_subjects = all_subjects[:num_seen]
    unseen_subjects = [s for s in all_subjects if s not in seen_subjects]

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test1, y_test1 = [], [] # Test split for seen subjects
    X_test2, y_test2 = [], [] # Test split for unseen subjects

    for sid in all_subjects:
        file_path = os.path.join(data_dir, f"A{sid:02d}S.mat")
        
        if not os.path.exists(file_path):
            print(f"  [Warning] File NOT found: {file_path}")
            continue
        
        mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        sessions = mat['data'] 
        if not isinstance(sessions, (list, np.ndarray)):
            sessions = [sessions]
        
        sub_x, sub_y = [], []
        for sess_idx, session in enumerate(sessions):
            raw_x = session.X            # (Samples, 16 channels)
            # Find flash starts: session.y is continuous. 1=Target, 2=Non-Target, 0=None.
            # We find indices where y > 0 and the previous value was 0.
            stim_raw = session.y
            flash_starts = np.where((stim_raw[:-1] == 0) & (stim_raw[1:] > 0))[0] + 1
            
            extracted_count = 0
            for start in flash_starts:
                # BNCI Label Mapping: 1 (Target) -> 1, 2 (Non-Target) -> 0
                label = 1 if stim_raw[start] == 1 else 0
                
                # Extract 1s window @ 256Hz (256 samples)
                end = start + 256 
                if end <= raw_x.shape[0]:
                    trial = raw_x[start:end, :].T # (16, 256)
                    
                    # Resample to 512 samples to match the 512Hz Task Config
                    # This ensures consistency with your STFT parameters
                    trial = signal.resample(trial, 512, axis=-1)
                    
                    sub_x.append(trial.astype(np.float32))
                    sub_y.append(label)
                    extracted_count += 1

        if not sub_x:
            continue

        # 2. Intra-subject split (Matching your 60/20/20 logic)
        X_sub = np.array(sub_x)
        y_sub = np.array(sub_y)
        
        # Shuffle trials within subject before splitting
        indices = np.arange(len(X_sub))
        np.random.shuffle(indices)
        X_sub, y_sub = X_sub[indices], y_sub[indices]
        
        if sid in seen_subjects:
            n = len(X_sub)
            itrain, ival = int(n * 0.6), int(n * 0.8)
            
            X_train.append(X_sub[:itrain])
            y_train.append(y_sub[:itrain])
            X_val.append(X_sub[itrain:ival])
            y_val.append(y_sub[itrain:ival])
            X_test1.append(X_sub[ival:])
            y_test1.append(y_sub[ival:])
        else:
            # Entire subject held out for zero-shot testing
            X_test2.append(X_sub)
            y_test2.append(y_sub)

    # 3. Final Concatenation
    result = {}
    splits = [
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test1', X_test1, y_test1),
        ('test2', X_test2, y_test2)
    ]

    for name, data_list, label_list in splits:
        if data_list:
            result[name] = (np.concatenate(data_list, axis=0), np.concatenate(label_list, axis=0))
            print(f"Final {name} Split: {result[name][0].shape}")

    return result

# ==================== MI (Motor Imagery) Data Loading ====================

def _MI_load_data_by_session(root_dir: str, subject_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load MI data for one subject"""
    if not MNE_AVAILABLE:
        raise ImportError("mne is required for MI data loading")
    
    fname = f"A{subject_id:02d}T.fif"
    fpath = os.path.join(root_dir, fname)
    
    if not os.path.exists(fpath):
        print(f"Warning: {fpath} not found, skipping subject {subject_id}")
        return torch.empty(0, 0, 0), torch.empty(0, dtype=torch.long)
        
    raw = mne.io.read_raw_fif(fpath, preload=True, verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    motor_keys = ['769', '770', '771', '772']
    motor_event_id = {k: v for k, v in event_id.items() if k in motor_keys}
    
    if len(motor_event_id) < 4:
        print(f"Warning: {fname} missing MI cues. Found: {event_id}")
        return torch.empty(0, 0, 0), torch.empty(0, dtype=torch.long)
        
    events = np.array([e for e in events if e[2] in motor_event_id.values()])
    label_map = {
        motor_event_id['769']: 0,
        motor_event_id['770']: 1,
        motor_event_id['771']: 2,
        motor_event_id['772']: 3,
    }
    labels = np.array([label_map[e[-1]] for e in events])

    epochs = mne.Epochs(
        raw, events,
        tmin=0.0, tmax=4.0,
        baseline=None,
        preload=True,
        verbose=False,
        event_repeated="drop"
    )
    data = epochs.get_data()
    
    return torch.from_numpy(data).float(), torch.from_numpy(labels).long()


def _MI_split_by_class_and_run(Y: torch.Tensor, seed: int = 44, 
                                trials_per_run: int = 12, n_classes: int = 4, n_runs: int = 6,
                                train_count: int = 8, val_count: int = 2, test_count: int = 2) -> Tuple[List, List, List]:
    """Split MI data by class and run"""
    random.seed(seed)
    train_idx, val_idx, test_idx = [], [], []
    
    for cls in range(n_classes):
        cls_indices = (Y == cls).nonzero(as_tuple=True)[0]
        expected_count = trials_per_run * n_runs
        
        if len(cls_indices) < expected_count:
            trials_per_run = len(cls_indices) // n_runs if n_runs > 0 else len(cls_indices)

        for run in range(n_runs):
            start_idx = run * trials_per_run
            end_idx = (run + 1) * trials_per_run
            if end_idx > len(cls_indices):
                end_idx = len(cls_indices)
            if start_idx >= end_idx:
                break
                
            run_trials = cls_indices[start_idx:end_idx].tolist()
            random.shuffle(run_trials)
            
            actual_train = min(train_count, len(run_trials))
            actual_val = min(val_count, len(run_trials) - actual_train)
            
            train_idx.extend(run_trials[:actual_train])
            val_idx.extend(run_trials[actual_train:actual_train + actual_val])
            test_idx.extend(run_trials[actual_train + actual_val:])
            
    return train_idx, val_idx, test_idx


def load_mi_data(data_dir: str, num_seen: int = 7, seed: int = 44) -> Dict:
    """
    Load MI dataset with proper split
    
    Returns:
        Dictionary with 'train', 'val', 'test1', 'test2' splits
    """
    if not MNE_AVAILABLE:
        raise ImportError("mne is required for MI data loading")
    
    random.seed(seed)
    all_subjects = list(range(1, 10))  # A01T.fif to A09T.fif
    seen_subjects = random.sample(all_subjects, num_seen)
    unseen_subjects = [sid for sid in all_subjects if sid not in seen_subjects]

    print(f"[MI Split] Seen: {seen_subjects}")
    print(f"[MI Split] Unseen: {unseen_subjects}")

    X_train_all, Y_train_all = [], []
    X_val_all, Y_val_all = [], []
    X_test1_all, Y_test1_all = [], []

    for sid in seen_subjects:
        X, Y = _MI_load_data_by_session(data_dir, sid)
        
        if len(X) == 0:
            print(f"Skipping MI subject {sid} - no data loaded")
            continue
            
        train_idx, val_idx, test_idx = _MI_split_by_class_and_run(Y, seed=seed)

        X_train_all.append(X[train_idx])
        Y_train_all.append(Y[train_idx])
        X_val_all.append(X[val_idx])
        Y_val_all.append(Y[val_idx])
        X_test1_all.append(X[test_idx])
        Y_test1_all.append(Y[test_idx])

    X_test2_all, Y_test2_all = [], []
    for sid in unseen_subjects:
        X, Y = _MI_load_data_by_session(data_dir, sid)
        if len(X) > 0:
            X_test2_all.append(X)
            Y_test2_all.append(Y)

    datasets = {}
    if X_train_all:
        datasets['train'] = (torch.cat(X_train_all, dim=0).numpy(), torch.cat(Y_train_all, dim=0).numpy())
    if X_val_all:
        datasets['val'] = (torch.cat(X_val_all, dim=0).numpy(), torch.cat(Y_val_all, dim=0).numpy())
    if X_test1_all:
        datasets['test1'] = (torch.cat(X_test1_all, dim=0).numpy(), torch.cat(Y_test1_all, dim=0).numpy())
    if X_test2_all:
        datasets['test2'] = (torch.cat(X_test2_all, dim=0).numpy(), torch.cat(Y_test2_all, dim=0).numpy())

    for split in datasets:
        print(f"  {split}: {datasets[split][0].shape}")

    return datasets


# ==================== Imagined Speech Data Loading ====================

def _ImaginedSpeech_load_subject_data(subject_id: str, root_dir: str) -> Dict:
    """Load Imagined Speech data for one subject"""
    x_path = os.path.join(root_dir, f"{subject_id}.npy")
    y_path = os.path.join(root_dir, f"{subject_id}_labels.npy")

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Data file missing: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Label file missing: {y_path}")

    X = np.load(x_path)  # (n_trials, C, T)
    Y_raw = np.load(y_path, allow_pickle=True).ravel()

    label_set = sorted(set(Y_raw.tolist()))
    label2idx = {lab: i for i, lab in enumerate(label_set)}
    Y = np.array([label2idx[lab] for lab in Y_raw])

    # Track repetitions for each label
    cnt = defaultdict(int)
    repetition = []
    for lab in Y_raw:
        repetition.append(cnt[lab])
        cnt[lab] += 1

    return {"X": X, "Y": Y, "repetition": repetition, "label2idx": label2idx}


def _ImaginedSpeech_split_repetitions(subject_data: Dict) -> Tuple[List, List, List]:
    """Split Imagined Speech data by repetitions"""
    repetition_arr = np.array(subject_data["repetition"])
    
    train_idx, val_idx, test_idx = [], [], []
    
    for rep in range(max(repetition_arr) + 1):
        rep_indices = np.where(repetition_arr == rep)[0]
        if len(rep_indices) > 0:
            n_train = int(len(rep_indices) * 0.6)
            n_val = int(len(rep_indices) * 0.2)
            
            train_idx.extend(rep_indices[:n_train])
            val_idx.extend(rep_indices[n_train:n_train + n_val])
            test_idx.extend(rep_indices[n_train + n_val:])
    
    return train_idx, val_idx, test_idx


def load_imagined_speech_data(data_dir: str, num_seen: int = 8, seed: int = 44) -> Dict:
    """
    Load Imagined Speech dataset with proper split
    
    Returns:
        Dictionary with 'train', 'val', 'test1', 'test2' splits
    """
    random.seed(seed)

    files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    data_bases = {os.path.splitext(f)[0] for f in files if not f.endswith("_labels.npy")}
    label_bases = {f[:-len("_labels.npy")] for f in files if f.endswith("_labels.npy")}
    all_subjects = sorted(data_bases & label_bases)

    print(f"Available Imagined Speech subjects: {all_subjects}")
    n = len(all_subjects)

    if n < 3:
        print(f"Error: Found only {n} subject(s). Need >= 3")
        return {}

    num_seen = min(int(num_seen), n - 2)
    if num_seen < 1:
        num_seen = 1

    seen_subjects = random.sample(all_subjects, num_seen)
    unseen_subjects = [s for s in all_subjects if s not in seen_subjects]

    print(f"[Imagined Speech Split] Seen: {seen_subjects}")
    print(f"[Imagined Speech Split] Unseen: {unseen_subjects}")

    X_train_all, Y_train_all = [], []
    X_val_all, Y_val_all = [], []
    X_test1_all, Y_test1_all = [], []

    for sid in seen_subjects:
        try:
            data = _ImaginedSpeech_load_subject_data(sid, data_dir)
            X = torch.tensor(data['X']).float()
            Y = torch.tensor(data['Y']).long()
            train_idx, val_idx, test1_idx = _ImaginedSpeech_split_repetitions(data)
            X_train_all.append(X[train_idx])
            Y_train_all.append(Y[train_idx])
            X_val_all.append(X[val_idx])
            Y_val_all.append(Y[val_idx])
            X_test1_all.append(X[test1_idx])
            Y_test1_all.append(Y[test1_idx])
        except Exception as e:
            print(f"Error loading Imagined Speech subject {sid}: {e}")

    X_test2_all, Y_test2_all = [], []
    for sid in unseen_subjects:
        try:
            data = _ImaginedSpeech_load_subject_data(sid, data_dir)
            X = torch.tensor(data['X']).float()
            Y = torch.tensor(data['Y']).long()
            X_test2_all.append(X)
            Y_test2_all.append(Y)
        except Exception as e:
            print(f"Error loading Imagined Speech subject {sid}: {e}")

    datasets = {}
    if X_train_all:
        datasets['train'] = (torch.cat(X_train_all, dim=0).numpy(),
                             torch.cat(Y_train_all, dim=0).numpy())
    if X_val_all:
        datasets['val'] = (torch.cat(X_val_all, dim=0).numpy(),
                           torch.cat(Y_val_all, dim=0).numpy())
    if X_test1_all:
        datasets['test1'] = (torch.cat(X_test1_all, dim=0).numpy(),
                             torch.cat(Y_test1_all, dim=0).numpy())
    if X_test2_all:
        datasets['test2'] = (torch.cat(X_test2_all, dim=0).numpy(),
                             torch.cat(Y_test2_all, dim=0).numpy())

    for split in datasets:
        print(f"  {split}: {datasets[split][0].shape}")

    return datasets


# ==================== Unified Data Loader ====================

def load_dataset(task: str, data_dir: Optional[str] = None, 
                 num_seen: Optional[int] = None, seed: int = 44) -> Dict:
    """
    Unified data loader for all EEG tasks
    
    Args:
        task: One of 'SSVEP', 'P300', 'MI', 'Imagined_speech'
        data_dir: Data directory (uses default from TASK_CONFIGS if None)
        num_seen: Number of seen subjects (uses default if None)
        seed: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test1', 'test2' splits
    """
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASK_CONFIGS.keys())}")
    
    config = TASK_CONFIGS[task]
    data_dir = data_dir or config['data_dir']
    num_seen = num_seen or config['num_seen']
    
    print(f"\n{'='*60}")
    print(f"Loading {task} dataset")
    print(f"{'='*60}")
    
    if task == "SSVEP":
        return load_ssvep_data(data_dir, num_seen, seed)
    elif task == "P300":
        return load_p300_data(data_dir, num_seen, seed)
    elif task == "MI":
        return load_mi_data(data_dir, num_seen, seed)
    elif task == "Imagined_speech":
        return load_imagined_speech_data(data_dir, num_seen, seed)
    elif task == "Lee2019_MI":
        return load_lee2019_data(data_dir, num_seen, seed)
    elif task == "Lee2019_SSVEP":
        return load_lee2019_data(data_dir, num_seen, seed)
    elif task == "BNCI2014_P300":
        return load_bnci_p300(data_dir, num_seen, seed)
    else:
        raise ValueError(f"Unknown task: {task}")


# ==================== EEG Dataset Class ====================

class EEGDataset(Dataset):
    """
    EEG Dataset with STFT transform
    
    Supports all EEG paradigms: SSVEP, P300, MI, Imagined Speech
    """
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 stft_config: Optional[Dict] = None, 
                 normalize: bool = True, 
                 augment: bool = False):
        """
        Args:
            data: (N, C, T) raw EEG data
            labels: (N,) integer labels
            stft_config: STFT parameters dict with 'fs', 'nperseg', 'noverlap', 'nfft'
            normalize: Whether to z-score normalize after STFT
            augment: Whether to apply data augmentation
        """
        self.data = data
        self.labels = torch.LongTensor(labels)
        self.normalize = normalize
        self.augment = augment
        
        # Default STFT config (works well for most EEG tasks)
        self.stft_config = stft_config or {
            'fs': 250,
            'nperseg': 128,
            'noverlap': 112,
            'nfft': 512
        }
    
    def __len__(self):
        return len(self.data)
    
    def _augment_raw(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentation on raw EEG"""
        # Gaussian noise injection
        if np.random.random() < 0.5:
            x = x + np.random.randn(*x.shape) * 0.05 * np.std(x)
        # Amplitude scaling
        if np.random.random() < 0.5:
            x = x * np.random.uniform(0.8, 1.2)
        # Time shift
        if np.random.random() < 0.3:
            x = np.roll(x, np.random.randint(-5, 6), axis=-1)
        return x
    
    def _augment_stft(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentation on STFT (SpecAugment-style)"""
        # Frequency masking
        if np.random.random() < 0.3:
            n_freq = x.shape[1]
            f_mask = np.random.randint(1, max(2, n_freq // 10))
            f_start = np.random.randint(0, n_freq - f_mask)
            x[:, f_start:f_start+f_mask, :] = 0
        # Time masking
        if np.random.random() < 0.3:
            n_time = x.shape[2]
            t_mask = np.random.randint(1, max(2, n_time // 5))
            t_start = np.random.randint(0, n_time - t_mask)
            x[:, :, t_start:t_start+t_mask] = 0
        return x
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_raw = self.data[idx].copy()
        y = self.labels[idx]
        
        # RAW Augmentation (applied to x_raw before STFT)
        if self.augment:
            x_raw = self._augment_raw(x_raw)
        
        # STFT transform
        x_stft = apply_stft_transform(x_raw, **self.stft_config)
        
        # STFT augmentation
        if self.augment:
            x_stft = self._augment_stft(x_stft)
        
        # Normalize STFT (per-channel z-score)
        if self.normalize:
            mean = x_stft.mean(axis=(1, 2), keepdims=True)
            std = x_stft.std(axis=(1, 2), keepdims=True) + 1e-8
            x_stft = (x_stft - mean) / std
        
        # Return a tuple of (Raw, STFT) data
        return (torch.FloatTensor(x_raw), torch.FloatTensor(x_stft)), y


# ==================== Helper Functions ====================

def get_stft_dimensions(data_sample: np.ndarray, stft_config: Dict) -> Tuple[int, int, int]:
    """
    Get STFT output dimensions for a single sample
    
    Args:
        data_sample: (C, T) single EEG sample
        stft_config: STFT parameters
        
    Returns:
        (n_channels, freq_bins, time_bins)
    """
    stft_sample = apply_stft_transform(data_sample, **stft_config)
    return stft_sample.shape


def create_dataloaders(datasets: Dict, stft_config: Dict, batch_size: int = 32, 
                       num_workers: int = 4, augment_train: bool = True, seed: int = 44) -> Dict:
    """
    Create DataLoaders for all splits
    
    Args:
        datasets: Dictionary from load_dataset()
        stft_config: STFT parameters
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        augment_train: Whether to augment training data
        seed: seed for shuffling
        
    Returns:
        Dictionary of DataLoaders
    """
    from torch.utils.data import DataLoader
    from seed_utils import worker_init_fn, get_generator
    
    loaders = {}
    
    for split, (X, y) in datasets.items():
        augment = augment_train if split == 'train' else False
        shuffle = (split == 'train')
        
        ds = EEGDataset(X, y, stft_config, normalize=True, augment=augment)
        loaders[split] = DataLoader(
            ds, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers, 
            pin_memory=True,
            worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed),
            generator=get_generator(seed) if shuffle else None
        )
    
    return loaders


if __name__ == "__main__":
    # Test loading each dataset
    print("Testing dataset loaders...")
    
    # Test SSVEP
    try:
        ssvep_data = load_dataset("SSVEP")
        print(f"SSVEP loaded successfully!")
    except Exception as e:
        print(f"SSVEP loading failed: {e}")
    
    # Test P300
    try:
        p300_data = load_dataset("P300")
        print(f"P300 loaded successfully!")
    except Exception as e:
        print(f"P300 loading failed: {e}")
    
    # Test MI
    try:
        mi_data = load_dataset("MI")
        print(f"MI loaded successfully!")
    except Exception as e:
        print(f"MI loading failed: {e}")
    
    # Test Imagined Speech
    try:
        imgspeech_data = load_dataset("Imagined_speech")
        print(f"Imagined Speech loaded successfully!")
    except Exception as e:
        print(f"Imagined Speech loading failed: {e}")

    # Test Lee2019 MI
    try:
        lee2019_mi_data = load_dataset("Lee2019_MI")
        print(f"Lee2019 MI loaded successfully!")
    except Exception as e:
        print(f"Lee2019 MI loading failed: {e}")

    # Test Lee2019 SSVEP
    try:
        lee2019_ssvep_data = load_dataset("Lee2019_SSVEP")
        print(f"Lee2019 SSVEP loaded successfully!")
    except Exception as e:
        print(f"Lee2019 SSVEP loading failed: {e}")

    # Test BNCI2014 P300
    try:
        bnci2014_p300_data = load_dataset("BNCI2014_P300")
        print(f"BNCI2014 P300 loaded successfully!")
    except Exception as e:
        print(f"BNCI2014 P300 loading failed: {e}")

    # from moabb.datasets import Lee2019_SSVEP
    # import mne

    # # Force the configuration here
    # shared_data_path = '/ocean/projects/cis250213p/shared/mne_data'
    # mne.set_config('MNE_DATA', shared_data_path)
    # dataset = Lee2019_SSVEP()
    # dataset.download(path=shared_data_path)

    # from moabb.datasets import BNCI2014_009
    # import mne

    # # Force the configuration here
    # shared_data_path = '/ocean/projects/cis250213p/shared/mne_data'
    # mne.set_config('MNE_DATA', shared_data_path)
    # dataset = BNCI2014_009()
    # dataset.download(path=shared_data_path)


