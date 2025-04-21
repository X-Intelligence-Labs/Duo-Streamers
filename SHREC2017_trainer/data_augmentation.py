import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from config import MAX_LENGTH

def translate_joints(joints, translation_range):
    """
    Translation augmentation
    
    Args:
        joints: Joint coordinates with shape [num_frames, num_joints, 3]
        translation_range: Maximum translation range
        
    Returns:
        Translated joint coordinates
    """
    translation = np.random.uniform(-translation_range, translation_range, size=(1, 3))
    translated_joints = joints + translation.reshape(1, 1, 3)
    return translated_joints

def rotate_joints(joints, rotation_range):
    """
    Rotation augmentation
    
    Args:
        joints: Joint coordinates with shape [num_frames, num_joints, 3]
        rotation_range: Maximum rotation angle in radians
        
    Returns:
        Rotated joint coordinates
    """
    angle = np.random.uniform(-rotation_range, rotation_range)
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    r = R.from_rotvec(angle * axis)
    rotated_joints = r.apply(joints.reshape(-1, 3)).reshape(joints.shape)
    return rotated_joints

def scale_joints(joints, scale_range):
    """
    Scaling augmentation
    
    Args:
        joints: Joint coordinates with shape [num_frames, num_joints, 3]
        scale_range: Maximum scale factor deviation from 1.0
        
    Returns:
        Scaled joint coordinates
    """
    scale = np.random.uniform(1 - scale_range, 1 + scale_range)
    return joints * scale

def add_noise(joints, noise_std):
    """
    Add Gaussian noise
    
    Args:
        joints: Joint coordinates with shape [num_frames, num_joints, 3]
        noise_std: Standard deviation of the noise
        
    Returns:
        Joint coordinates with added noise
    """
    return joints + np.random.normal(0, noise_std, size=joints.shape)

def time_warp(sequence, speed_factor):
    """
    Time warping augmentation
    
    Args:
        sequence: Skeleton sequence with shape [num_frames, num_features]
        speed_factor: Factor to speed up or slow down the sequence
        
    Returns:
        Time-warped sequence
    """
    num_frames, num_features = sequence.shape
    new_num_frames = int(num_frames / speed_factor)
    
    if new_num_frames <= 1:  # Avoid extreme warping
        return sequence
    
    # Interpolation indices
    original_indices = np.arange(num_frames)
    new_indices = np.linspace(0, num_frames - 1, new_num_frames)
    
    # Interpolate each feature
    warped_sequence = np.zeros((new_num_frames, num_features))
    for i in range(num_features):
        interp_function = interp1d(original_indices, sequence[:, i], kind='linear', bounds_error=False, fill_value="extrapolate")
        warped_sequence[:, i] = interp_function(new_indices)
    
    # Pad or truncate to the specified length
    data_padded = np.zeros((MAX_LENGTH, num_features))
    data_padded[:min(warped_sequence.shape[0], MAX_LENGTH)] = warped_sequence[:min(warped_sequence.shape[0], MAX_LENGTH)]
    
    return data_padded.astype(np.float32)

def augment_skeleton(skeleton, translation_range=0.1, rotation_range=np.pi/8, scale_range=0.1, noise_std=0.01):
    """
    Combine multiple augmentation methods
    
    Args:
        skeleton: Skeleton sequence with shape [num_frames, num_features]
        translation_range: Maximum translation range
        rotation_range: Maximum rotation angle in radians
        scale_range: Maximum scale factor deviation from 1.0
        noise_std: Standard deviation of the noise
        
    Returns:
        Augmented skeleton sequence
    """
    # Reshape skeleton to [num_frames, num_joints, 3]
    num_frames, num_features = skeleton.shape
    joints_per_frame = num_features // 3
    skeleton_reshaped = skeleton.reshape(num_frames, joints_per_frame, 3)
    
    # Apply augmentations
    skeleton_augmented = translate_joints(skeleton_reshaped, translation_range)
    skeleton_augmented = rotate_joints(skeleton_augmented, rotation_range)
    skeleton_augmented = scale_joints(skeleton_augmented, scale_range)
    skeleton_augmented = add_noise(skeleton_augmented, noise_std)
    
    # Reshape back to original shape
    return skeleton_augmented.reshape(num_frames, num_features).astype(np.float32)

def generate_position_weights(seq_length):
    """
    Generate position-based weight vector, with higher weights for middle positions
    and lower weights for edge positions
    
    Args:
        seq_length: Length of the sequence
        
    Returns:
        Weight vector of shape [seq_length]
    """
    # Use Gaussian distribution to generate weights
    mid_point = seq_length / 2
    # Standard deviation controls the "width" of the weight distribution
    sigma = seq_length / 4
    
    weights = np.zeros(seq_length)
    for i in range(seq_length):
        weights[i] = np.exp(-0.5 * ((i - mid_point) / sigma) ** 2)
    
    # Normalize weights so they sum to 1
    weights = weights / np.sum(weights)
    
    return weights 