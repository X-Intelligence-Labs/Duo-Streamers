import os
import numpy as np
from numpy import genfromtxt
import logging
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from config import MAIN_FOLDER, INPUT_DIM, MAX_LENGTH

# Setup logging
_logger = logging.getLogger(__name__)

def read_skeleton_file(file_path):
    """
    Read a single skeleton file (skeletons_world.txt)
    """
    try:
        data = genfromtxt(file_path, delimiter=' ')
        return data
    except:
        _logger.error(f"Could not read file: {file_path}")
        return None

def load_gesture_data(gesture_id, finger_type, subject_id, trial_id):
    """
    Load skeleton data for a specific gesture, finger type, subject, and trial
    """
    path = os.path.join(MAIN_FOLDER, f"gesture_{gesture_id}", f"finger_{finger_type}", 
                        f"subject_{subject_id}", f"essai_{trial_id}", "skeletons_world.txt")
    return read_skeleton_file(path)

def read_labels_file(file_path):
    """
    Read a labels file (train_gestures.txt or test_gestures.txt)
    Returns: A list of dictionaries containing information for all samples
    """
    labels = []
    label_counts = {}  # For tracking label distribution
    
    _logger.info(f"Reading labels file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) >= 7:
                    try:
                        label = int(parts[5])
                        # Count label occurrences
                        if label not in label_counts:
                            label_counts[label] = 0
                        label_counts[label] += 1
                        
                        sample = {
                            'gesture_id': int(parts[0]),
                            'finger_type': int(parts[1]),
                            'subject_id': int(parts[2]),
                            'trial_id': int(parts[3]),
                            'gesture_class': int(parts[4]),
                            'label': label,
                            'num_frames': int(parts[6])
                        }
                        labels.append(sample)
                    except ValueError as e:
                        _logger.error(f"Parsing error (line {line_num}): {e} - {line.strip()}")
                else:
                    _logger.warning(f"Skipped malformatted line {line_num}: {line.strip()}")
    except Exception as e:
        _logger.error(f"Error reading file {file_path}: {e}")
    
    # Output label distribution information
    _logger.info(f"Label distribution: {sorted(label_counts.items())}")
    _logger.info(f"Total samples read: {len(labels)}")
    
    return labels

def load_gesture_dataset(labels_file, max_samples=None, mode='full'):
    """
    Load the dataset
    labels_file: Path to the labels file (train_gestures.txt or test_gestures.txt)
    max_samples: Limit the number of samples loaded (for testing)
    mode: 'single_finger' to use only single-finger gestures (finger_1), 'full' to use all gestures
    """
    labels = read_labels_file(labels_file)
    if max_samples:
        labels = labels[:max_samples]
    
    gesture_data = []
    gesture_labels = []
    skipped_samples = 0
    
    for i, sample in enumerate(tqdm(labels, desc="Loading dataset")):
        # If mode is 'single_finger', only load finger_1 samples
        if mode == 'single_finger' and sample['finger_type'] != 1:
            skipped_samples += 1
            continue
            
        data = load_gesture_data(
            sample['gesture_id'],
            sample['finger_type'],
            sample['subject_id'],
            sample['trial_id']
        )
        
        if data is not None:
            gesture_data.append(data)
            
            # Calculate label: (gesture_id - 1) * 2 + (finger_type - 1)
            # This maps 14 gestures × 2 finger configurations to the range 0-27
            if mode == 'full':
                adjusted_label = (sample['gesture_id'] - 1) * 2 + (sample['finger_type'] - 1)
                gesture_labels.append(adjusted_label)
            else:  # single_finger mode
                # In single-finger mode, the label range is 0-13
                gesture_labels.append(sample['gesture_id'] - 1)
    
    if skipped_samples > 0:
        _logger.info(f"Skipped {skipped_samples} samples based on selected mode '{mode}'")
    
    _logger.info(f"Successfully loaded {len(gesture_data)} samples")
    
    return gesture_data, gesture_labels

def normalize_skeleton(skeleton_data):
    """
    Normalize skeleton data
    """
    # Simple mean-std normalization
    mean = np.mean(skeleton_data, axis=0)
    std = np.std(skeleton_data, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (skeleton_data - mean) / std

def pad_data(data, input_dim=INPUT_DIM, max_length=MAX_LENGTH):
    """
    Pad or truncate skeleton sequences to the specified length
    """
    data_padded = np.zeros([len(data), max_length, input_dim])
    for i in range(len(data)):
        if len(data[i]) <= max_length:
            data_padded[i, :len(data[i])] = data[i][:, :input_dim]  # Only take the first input_dim elements
        else:
            data_padded[i] = data[i][:max_length, :input_dim]
    
    return data_padded

class SHREC2017Dataset(Dataset):
    def __init__(self, data, labels, augment=False, time_warp_prob=0.7):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
        self.time_warp_prob = time_warp_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx].numpy()
        
        if self.augment:
            # Import here to avoid circular import
            from data_augmentation import augment_skeleton, time_warp
            
            # Apply augmentation
            if np.random.random() < 0.9:  # 90% chance to apply skeleton augmentation
                d = augment_skeleton(d)
            
            # Apply time warping
            if np.random.random() < self.time_warp_prob:
                speed_factor = np.random.uniform(0.8, 1.2)  # Speed factor
                d = time_warp(d, speed_factor)
        
        return torch.tensor(d, dtype=torch.float32), self.labels[idx] 