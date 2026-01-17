import os
import numpy as np
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt

def preprocess_lee2019(raw_dir, save_dir, subjects=range(1, 55)):
    """
    Standardizes Lee2019 data to the MI paradigm: 22 channels, 250Hz, 4s epochs.
    """
    os.makedirs(save_dir, exist_ok=True)
    fs_new = 250
    
    # Specific indices for FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, 
    # CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz, and Fz
    chan_idx = [32, 8, 4, 9, 33, 34, 12, 35, 13, 36, 14, 37, 38, 18, 39, 19, 40, 41, 24, 42, 43, 44]

    for sid in subjects:
        sub_x, sub_y = [], []
        print(f"Processing Subject {sid}...")
        
        for sess in [1, 2]:
            file_path = os.path.join(raw_dir, f'session{sess}', f's{sid}', f'sess{sess:02d}_subj{sid:02d}_EEG_MI.mat')
            if not os.path.exists(file_path): continue
            
            mat = scipy.io.loadmat(file_path)
            for key in ['EEG_MI_train', 'EEG_MI_test']:
                if key not in mat: continue
                struct = mat[key][0, 0]
                raw_x = struct['x']          # Shape: (Time, 62)
                labels = struct['y_dec'].flatten() - 1 # 0=Right, 1=Left
                timestamps = struct['t'].flatten() # Trial start samples
                
                for i, start_sample in enumerate(timestamps):
                    start = int(start_sample) - 1
                    end = start + 4000 # 4s window at 1000Hz
                    
                    if end <= raw_x.shape[0]:
                        # Select only the 22 motor-relevant channels
                        trial = raw_x[start:end, chan_idx].T # Shape: (22, 4000)
                        
                        # 1. Downsample (1000Hz -> 250Hz)
                        trial = signal.resample(trial, 1000, axis=-1)
                        
                        # 2. Bandpass filter (4-40Hz) for MI mu/beta rhythms
                        b, a = signal.butter(4, [4, 40], btype='bandpass', fs=fs_new)
                        trial = signal.filtfilt(b, a, trial, axis=-1)
                        
                        # 3. Z-score normalization
                        trial = (trial - np.mean(trial)) / (np.std(trial) + 1e-8)
                        
                        sub_x.append(trial.astype(np.float32)) # Save as float32 to save space
                        sub_y.append(labels[i])
        
        if sub_x:
            save_path = os.path.join(save_dir, f"S{sid}_preprocessed.npz")
            np.savez(save_path, X=np.array(sub_x), y=np.array(sub_y))
            print(f"  Saved: {save_path} | Final Shape: {np.array(sub_x).shape}")

def preprocess_lee2019_ssvep(raw_dir, save_dir, subjects=range(1, 55), skip_existing=True):
    os.makedirs(save_dir, exist_ok=True)
    fs_orig = 1000
    fs_new = 250
    chan_idx = list(range(62)) 

    for sid in subjects:
        # Check if file already exists to avoid redundant processing
        save_path = os.path.join(save_dir, f"S{sid}_SSVEP_preprocessed.npz")
        if skip_existing and os.path.exists(save_path):
            print(f"Skipping Subject {sid}: File already exists.")
            continue
            
        sub_x, sub_y = [], []
        print(f"Processing Subject {sid} (SSVEP)...")
        
        for sess in [1, 2]:
            file_path = os.path.join(raw_dir, f'session{sess}', f's{sid}', f'sess{sess:02d}_subj{sid:02d}_EEG_SSVEP.mat')
            if not os.path.exists(file_path): continue
            
            mat = scipy.io.loadmat(file_path)
            for key in ['EEG_SSVEP_train', 'EEG_SSVEP_test']:
                if key not in mat: continue
                struct = mat[key][0, 0]
                raw_x = struct['x']          
                labels = struct['y_dec'].flatten() - 1 
                timestamps = struct['t'].flatten() 
                
                for i, start_sample in enumerate(timestamps):
                    start = int(start_sample) - 1
                    end = start + 4000 
                    
                    if end <= raw_x.shape[0]:
                        trial = raw_x[start:end, chan_idx].T 
                        
                        # Bandpass filter (6-90 Hz) per Wang2016 standard
                        b, a = signal.butter(4, [6, 90], btype='bandpass', fs=fs_orig)
                        trial = signal.filtfilt(b, a, trial, axis=-1)

                        # Downsample (1000Hz -> 250Hz) to 1000 timepoints
                        num_samples_new = int(trial.shape[-1] * (fs_new / fs_orig))
                        trial = signal.resample(trial, num_samples_new, axis=-1)
                        
                        # Z-score normalization
                        trial = (trial - np.mean(trial)) / (np.std(trial) + 1e-8)
                        
                        sub_x.append(trial.astype(np.float32))
                        sub_y.append(labels[i])
        
        if sub_x:
            np.savez(save_path, X=np.array(sub_x), y=np.array(sub_y))
            print(f"  Saved Subject {sid} | Shape: {np.array(sub_x).shape}")

# --- RUN EXECUTION ---

RAW_SSVEP = '/ocean/projects/cis250213p/shared/mne_data/MNE-lee2019-ssvep-data/gigadb-datasets/live/pub/10.5524/100001_101000/100542'
PROCESSED_SSVEP = '/ocean/projects/cis250213p/shared/lee2019_ssvep_processed'

preprocess_lee2019_ssvep(RAW_SSVEP, PROCESSED_SSVEP)

# RAW = '/ocean/projects/cis250213p/shared/lee2019_mi'
# PROCESSED = '/ocean/projects/cis250213p/shared/lee2019_processed'
# PLOTS = '/ocean/projects/cis250213p/shared/lee2019_plots'

# preprocess_lee2019(RAW, PROCESSED)
