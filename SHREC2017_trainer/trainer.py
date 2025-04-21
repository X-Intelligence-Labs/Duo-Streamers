import os
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torcheval.metrics import MulticlassConfusionMatrix

from models import StreamingSightMu, StreamingSightBi, StreamingSightMuOriginal, TCRN, MSTCRN
from data_augmentation import generate_position_weights
from config import MODEL_SAVE_DIR, SELECTED_GESTURE_NAME

# Setup logging
_logger = logging.getLogger(__name__)

def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch
    
    Args:
        model: Neural network model
        device: Device (CPU or GPU) to train on
        train_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        epoch: Current epoch number
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        try:
            # Validate label range
            max_label = len(SELECTED_GESTURE_NAME) - 1  # -1 because we use 0-27 not 1-28
            if torch.max(target).item() > max_label or torch.min(target).item() < 0:
                _logger.error(f"Invalid labels found in batch {batch_idx}: min={torch.min(target).item()}, max={torch.max(target).item()}, valid range=[0,{max_label}]")
                continue
                
            data, target = data.to(device), target.to(device)
            batch_size, max_length, input_dim = data.size()
            
            # Generate position-based weights
            position_weights = generate_position_weights(max_length)
            position_weights = torch.tensor(position_weights, dtype=torch.float32).to(device)
            
            # Initialize external memory
            if isinstance(model, StreamingSightMu):
                # Initialize hidden states
                hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                loss = 0
                frame_outputs = []
                
                # Process each frame with position-based weighting
                for i in range(max_length):
                    frames = data[:, i, :].unsqueeze(1)
                    output, hPt = model(frames, hPt)
                    
                    # Weighted loss
                    frame_loss = criterion(output, target) * position_weights[i]
                    loss += frame_loss
                    
                    # Store each frame's output
                    frame_outputs.append(output)
                
                # Weight-combine all frame outputs
                outputs = torch.zeros_like(frame_outputs[0])
                for i, output in enumerate(frame_outputs):
                    outputs += output * position_weights[i]
                
            elif isinstance(model, StreamingSightBi):
                hPt = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                cPt = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                
                # Process entire sequence at once
                outputs, (hPt, cPt) = model(data, hPt, cPt)
                loss = criterion(outputs, target)
                
            elif isinstance(model, StreamingSightMuOriginal):
                # Initialize hidden states
                hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                loss = 0
                frame_outputs = []
                
                # Process each frame with position-based weighting
                for i in range(max_length):
                    frames = data[:, i, :].unsqueeze(1)
                    output, hPt = model(frames, hPt)
                    
                    # Weighted loss
                    frame_loss = criterion(output, target) * position_weights[i]
                    loss += frame_loss
                    
                    # Store each frame's output
                    frame_outputs.append(output)
                
                # Weight-combine all frame outputs
                outputs = torch.zeros_like(frame_outputs[0])
                for i, output in enumerate(frame_outputs):
                    outputs += output * position_weights[i]
                    
            elif isinstance(model, TCRN):
                # Initialize hidden states
                hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                loss = 0
                frame_outputs = []
                
                # Calculate number of complete segments (3 frames each)
                frames_per_segment = model.frames_per_segment
                complete_segments = max_length // frames_per_segment
                
                # Process each segment (3 frames)
                for i in range(complete_segments):
                    start_idx = i * frames_per_segment
                    end_idx = start_idx + frames_per_segment
                    segment = data[:, start_idx:end_idx, :]
                    
                    # Process current segment
                    output, hPt = model(segment, hPt)
                    
                    # Use segment's middle position weight
                    middle_idx = start_idx + frames_per_segment // 2
                    segment_weight = position_weights[middle_idx]
                    
                    # Weighted loss
                    segment_loss = criterion(output, target) * segment_weight
                    loss += segment_loss
                    
                    # Store output, using same output for all frames in segment
                    for j in range(start_idx, end_idx):
                        # Adjust weights to fit per-frame storage, but keep sum constant
                        frame_weight = position_weights[j] / sum(position_weights[start_idx:end_idx])
                        frame_outputs.append((output, frame_weight))
                
                # Process remaining frames (less than a complete segment)
                remaining_frames = max_length % frames_per_segment
                if remaining_frames > 0:
                    start_idx = complete_segments * frames_per_segment
                    
                    # Use padding to create a complete segment
                    padded_segment = torch.zeros(batch_size, frames_per_segment, input_dim).to(device)
                    padded_segment[:, :remaining_frames, :] = data[:, start_idx:, :]
                    
                    # Process padded segment
                    output, hPt = model(padded_segment, hPt)
                    
                    # Only weight real frames
                    for j in range(remaining_frames):
                        idx = start_idx + j
                        frame_loss = criterion(output, target) * position_weights[idx] / remaining_frames
                        loss += frame_loss
                        
                        # Store output
                        frame_outputs.append((output, position_weights[idx]))
                
                # Weight-combine all frame outputs
                outputs = torch.zeros_like(frame_outputs[0][0])
                total_weight = 0
                for output, weight in frame_outputs:
                    outputs += output * weight
                    total_weight += weight
                
                # Normalize output
                if total_weight > 0:
                    outputs = outputs / total_weight
            
            elif isinstance(model, MSTCRN):
                # Initialize multi-scale states
                fast_h = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                slow_h = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                state_dict = {'fast': fast_h, 'slow': slow_h, 'step_count': 0}
                
                loss = 0
                frame_outputs = []
                
                # Calculate number of complete segments
                frames_per_segment = model.frames_per_segment
                complete_segments = max_length // frames_per_segment
                
                # Process each segment
                for i in range(complete_segments):
                    start_idx = i * frames_per_segment
                    end_idx = start_idx + frames_per_segment
                    segment = data[:, start_idx:end_idx, :]
                    
                    # Process current segment
                    output, state_dict = model(segment, state_dict)
                    
                    # Use segment's middle position weight
                    middle_idx = start_idx + frames_per_segment // 2
                    segment_weight = position_weights[middle_idx]
                    
                    # Weighted loss
                    segment_loss = criterion(output, target) * segment_weight
                    loss += segment_loss
                    
                    # Store output, using same output for all frames in segment
                    for j in range(start_idx, end_idx):
                        # Adjust weights to fit per-frame storage, but keep sum constant
                        frame_weight = position_weights[j] / sum(position_weights[start_idx:end_idx])
                        frame_outputs.append((output, frame_weight))
                
                # Process remaining frames (less than a complete segment)
                remaining_frames = max_length % frames_per_segment
                if remaining_frames > 0:
                    start_idx = complete_segments * frames_per_segment
                    
                    # Use padding to create a complete segment
                    padded_segment = torch.zeros(batch_size, frames_per_segment, input_dim).to(device)
                    padded_segment[:, :remaining_frames, :] = data[:, start_idx:, :]
                    
                    # Process padded segment
                    output, state_dict = model(padded_segment, state_dict)
                    
                    # Only weight real frames
                    for j in range(remaining_frames):
                        idx = start_idx + j
                        frame_loss = criterion(output, target) * position_weights[idx] / remaining_frames
                        loss += frame_loss
                        
                        # Store output
                        frame_outputs.append((output, position_weights[idx]))
                
                # Weight-combine all frame outputs
                outputs = torch.zeros_like(frame_outputs[0][0])
                total_weight = 0
                for output, weight in frame_outputs:
                    outputs += output * weight
                    total_weight += weight
                
                # Normalize output
                if total_weight > 0:
                    outputs = outputs / total_weight
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        except Exception as e:
            _logger.error(f"Error in training batch {batch_idx}: {str(e)}")
            import traceback
            _logger.error(traceback.format_exc())
            continue
    
    train_loss /= max(1, len(train_loader))  # Avoid division by zero
    accuracy = 100. * correct / max(1, total)  # Avoid division by zero
    
    _logger.info(f'Train set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    
    return train_loss, accuracy

def test(model, device, test_loader, criterion):
    """
    Evaluate the model on test data
    
    Args:
        model: Neural network model
        device: Device (CPU or GPU) to evaluate on
        test_loader: DataLoader for test data
        criterion: Loss function
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
            try:
                # Validate label range
                max_label = len(SELECTED_GESTURE_NAME) - 1  # -1 because we use 0-27 not 1-28
                if torch.max(target).item() > max_label or torch.min(target).item() < 0:
                    _logger.error(f"Invalid labels found in test batch {batch_idx}: min={torch.min(target).item()}, max={torch.max(target).item()}, valid range=[0,{max_label}]")
                    continue
                    
                data, target = data.to(device), target.to(device)
                batch_size, max_length, input_dim = data.size()
                
                # Generate position-based weights
                position_weights = generate_position_weights(max_length)
                position_weights = torch.tensor(position_weights, dtype=torch.float32).to(device)
                
                # Initialize external memory based on model type
                if isinstance(model, StreamingSightMu):
                    # Initialize hidden states
                    hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    test_batch_loss = 0
                    frame_outputs = []
                    
                    # Process each frame with position-based weighting
                    for i in range(max_length):
                        frames = data[:, i, :].unsqueeze(1)
                        output, hPt = model(frames, hPt)
                        
                        # Weighted loss
                        frame_loss = criterion(output, target) * position_weights[i]
                        test_batch_loss += frame_loss
                        
                        # Store each frame's output
                        frame_outputs.append(output)
                    
                    # Weight-combine all frame outputs
                    outputs = torch.zeros_like(frame_outputs[0])
                    for i, output in enumerate(frame_outputs):
                        outputs += output * position_weights[i]
                    
                    test_loss += test_batch_loss.item()
                    
                elif isinstance(model, StreamingSightBi):
                    hPt = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                    cPt = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                    
                    # Process entire sequence at once
                    outputs, _ = model(data, hPt, cPt)
                    test_loss += criterion(outputs, target).item()
                    
                elif isinstance(model, StreamingSightMuOriginal):
                    # Initialize hidden states
                    hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    test_batch_loss = 0
                    frame_outputs = []
                    
                    # Process each frame with position-based weighting
                    for i in range(max_length):
                        frames = data[:, i, :].unsqueeze(1)
                        output, hPt = model(frames, hPt)
                        
                        # Weighted loss
                        frame_loss = criterion(output, target) * position_weights[i]
                        test_batch_loss += frame_loss
                        
                        # Store each frame's output
                        frame_outputs.append(output)
                    
                    # Weight-combine all frame outputs
                    outputs = torch.zeros_like(frame_outputs[0])
                    for i, output in enumerate(frame_outputs):
                        outputs += output * position_weights[i]
                    
                    test_loss += test_batch_loss.item()
                    
                elif isinstance(model, TCRN):
                    # Initialize hidden states
                    hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    test_batch_loss = 0
                    frame_outputs = []
                    
                    # Calculate number of complete segments (3 frames each)
                    frames_per_segment = model.frames_per_segment
                    complete_segments = max_length // frames_per_segment
                    
                    # Process each segment (3 frames)
                    for i in range(complete_segments):
                        start_idx = i * frames_per_segment
                        end_idx = start_idx + frames_per_segment
                        segment = data[:, start_idx:end_idx, :]
                        
                        # Process current segment
                        output, hPt = model(segment, hPt)
                        
                        # Use segment's middle position weight
                        middle_idx = start_idx + frames_per_segment // 2
                        segment_weight = position_weights[middle_idx]
                        
                        # Weighted loss
                        segment_loss = criterion(output, target) * segment_weight
                        test_batch_loss += segment_loss
                        
                        # Store output, using same output for all frames in segment
                        for j in range(start_idx, end_idx):
                            # Adjust weights to fit per-frame storage, but keep sum constant
                            frame_weight = position_weights[j] / sum(position_weights[start_idx:end_idx])
                            frame_outputs.append((output, frame_weight))
                    
                    # Process remaining frames (less than a complete segment)
                    remaining_frames = max_length % frames_per_segment
                    if remaining_frames > 0:
                        start_idx = complete_segments * frames_per_segment
                        
                        # Use padding to create a complete segment
                        padded_segment = torch.zeros(batch_size, frames_per_segment, input_dim).to(device)
                        padded_segment[:, :remaining_frames, :] = data[:, start_idx:, :]
                        
                        # Process padded segment
                        output, hPt = model(padded_segment, hPt)
                        
                        # Only weight real frames
                        for j in range(remaining_frames):
                            idx = start_idx + j
                            frame_loss = criterion(output, target) * position_weights[idx] / remaining_frames
                            test_batch_loss += frame_loss
                            
                            # Store output
                            frame_outputs.append((output, position_weights[idx]))
                    
                    # Weight-combine all frame outputs
                    outputs = torch.zeros_like(frame_outputs[0][0])
                    total_weight = 0
                    for output, weight in frame_outputs:
                        outputs += output * weight
                        total_weight += weight
                    
                    # Normalize output
                    if total_weight > 0:
                        outputs = outputs / total_weight
                        
                    test_loss += test_batch_loss.item()
                
                elif isinstance(model, MSTCRN):
                    # Initialize multi-scale states
                    fast_h = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    slow_h = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    state_dict = {'fast': fast_h, 'slow': slow_h, 'step_count': 0}
                    
                    test_batch_loss = 0
                    frame_outputs = []
                    
                    # Calculate number of complete segments
                    frames_per_segment = model.frames_per_segment
                    complete_segments = max_length // frames_per_segment
                    
                    # Process each segment
                    for i in range(complete_segments):
                        start_idx = i * frames_per_segment
                        end_idx = start_idx + frames_per_segment
                        segment = data[:, start_idx:end_idx, :]
                        
                        # Process current segment
                        output, state_dict = model(segment, state_dict)
                        
                        # Use segment's middle position weight
                        middle_idx = start_idx + frames_per_segment // 2
                        segment_weight = position_weights[middle_idx]
                        
                        # Weighted loss
                        segment_loss = criterion(output, target) * segment_weight
                        test_batch_loss += segment_loss
                        
                        # Store output
                        for j in range(start_idx, end_idx):
                            frame_weight = position_weights[j] / sum(position_weights[start_idx:end_idx])
                            frame_outputs.append((output, frame_weight))
                    
                    # Process remaining frames (less than a complete segment)
                    remaining_frames = max_length % frames_per_segment
                    if remaining_frames > 0:
                        start_idx = complete_segments * frames_per_segment
                        
                        # Use padding to create a complete segment
                        padded_segment = torch.zeros(batch_size, frames_per_segment, input_dim).to(device)
                        padded_segment[:, :remaining_frames, :] = data[:, start_idx:, :]
                        
                        # Process padded segment
                        output, state_dict = model(padded_segment, state_dict)
                        
                        # Only weight real frames
                        for j in range(remaining_frames):
                            idx = start_idx + j
                            frame_loss = criterion(output, target) * position_weights[idx] / remaining_frames
                            test_batch_loss += frame_loss
                            
                            # Store output
                            frame_outputs.append((output, position_weights[idx]))
                    
                    # Weight-combine all frame outputs
                    outputs = torch.zeros_like(frame_outputs[0][0])
                    total_weight = 0
                    for output, weight in frame_outputs:
                        outputs += output * weight
                        total_weight += weight
                    
                    # Normalize output
                    if total_weight > 0:
                        outputs = outputs / total_weight
                    
                    test_loss += test_batch_loss.item()
                
                # Calculate accuracy
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            except Exception as e:
                _logger.error(f"Error in test batch {batch_idx}: {str(e)}")
                import traceback
                _logger.error(traceback.format_exc())
                continue
    
    test_loss /= max(1, len(test_loader))  # Avoid division by zero
    accuracy = 100. * correct / max(1, total)  # Avoid division by zero
    
    _logger.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    
    return test_loss, accuracy

def save_model(model, optimizer, epoch, accuracy, model_type, path=MODEL_SAVE_DIR):
    """
    Save model checkpoint
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        epoch: Current epoch number
        accuracy: Validation accuracy
        model_type: Type of model (mu, bi, mu_original, tcrn, mstcrn)
        path: Directory to save model to
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    
    filename = os.path.join(path, f"{model_type}_checkpoint_epoch{epoch}_acc{accuracy:.2f}.pt")
    torch.save(checkpoint, filename)
    _logger.info(f"Model saved to {filename}")

def compute_confusion_matrix(model, device, test_loader):
    """
    Compute confusion matrix on test data
    
    Args:
        model: Neural network model
        device: Device (CPU or GPU) to evaluate on
        test_loader: DataLoader for test data
        
    Returns:
        numpy.ndarray: Confusion matrix
    """
    model.eval()
    confmat = MulticlassConfusionMatrix(num_classes=len(SELECTED_GESTURE_NAME))
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            try:
                data, target = data.to(device), target.to(device)
                batch_size, max_length, input_dim = data.size()
                
                # Generate position-based weights
                position_weights = generate_position_weights(max_length)
                position_weights = torch.tensor(position_weights, dtype=torch.float32).to(device)
                
                # Process data based on model type
                if isinstance(model, StreamingSightMu):
                    # Initialize hidden states
                    hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    frame_outputs = []
                    
                    # Process each frame with position-based weighting
                    for i in range(max_length):
                        frames = data[:, i, :].unsqueeze(1)
                        output, hPt = model(frames, hPt)
                        frame_outputs.append(output)
                    
                    # Weight-combine all frame outputs
                    outputs = torch.zeros_like(frame_outputs[0])
                    for i, output in enumerate(frame_outputs):
                        outputs += output * position_weights[i]
                    
                elif isinstance(model, StreamingSightBi):
                    hPt = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                    cPt = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                    outputs, _ = model(data, hPt, cPt)
                    
                elif isinstance(model, StreamingSightMuOriginal):
                    # Initialize hidden states
                    hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    frame_outputs = []
                    
                    # Process each frame with position-based weighting
                    for i in range(max_length):
                        frames = data[:, i, :].unsqueeze(1)
                        output, hPt = model(frames, hPt)
                        frame_outputs.append(output)
                    
                    # Weight-combine all frame outputs
                    outputs = torch.zeros_like(frame_outputs[0])
                    for i, output in enumerate(frame_outputs):
                        outputs += output * position_weights[i]
                
                elif isinstance(model, TCRN):
                    # Initialize hidden states
                    hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    frame_outputs = []
                    
                    # Calculate number of complete segments
                    frames_per_segment = model.frames_per_segment
                    complete_segments = max_length // frames_per_segment
                    
                    # Process each segment
                    for i in range(complete_segments):
                        start_idx = i * frames_per_segment
                        end_idx = start_idx + frames_per_segment
                        segment = data[:, start_idx:end_idx, :]
                        
                        # Process current segment
                        output, hPt = model(segment, hPt)
                        
                        # Store output, using same output for all frames in segment
                        for j in range(start_idx, end_idx):
                            frame_weight = position_weights[j] / sum(position_weights[start_idx:end_idx])
                            frame_outputs.append((output, frame_weight))
                    
                    # Process remaining frames (less than a complete segment)
                    remaining_frames = max_length % frames_per_segment
                    if remaining_frames > 0:
                        start_idx = complete_segments * frames_per_segment
                        
                        # Use padding to create a complete segment
                        padded_segment = torch.zeros(batch_size, frames_per_segment, input_dim).to(device)
                        padded_segment[:, :remaining_frames, :] = data[:, start_idx:, :]
                        
                        # Process padded segment
                        output, hPt = model(padded_segment, hPt)
                        
                        # Only weight real frames
                        for j in range(remaining_frames):
                            idx = start_idx + j
                            frame_outputs.append((output, position_weights[idx]))
                    
                    # Weight-combine all frame outputs
                    outputs = torch.zeros_like(frame_outputs[0][0])
                    total_weight = 0
                    for output, weight in frame_outputs:
                        outputs += output * weight
                        total_weight += weight
                    
                    # Normalize output
                    if total_weight > 0:
                        outputs = outputs / total_weight
                
                elif isinstance(model, MSTCRN):
                    # Initialize multi-scale states
                    fast_h = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    slow_h = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    state_dict = {'fast': fast_h, 'slow': slow_h, 'step_count': 0}
                    
                    frame_outputs = []
                    
                    # Calculate number of complete segments
                    frames_per_segment = model.frames_per_segment
                    complete_segments = max_length // frames_per_segment
                    
                    # Process each segment
                    for i in range(complete_segments):
                        start_idx = i * frames_per_segment
                        end_idx = start_idx + frames_per_segment
                        segment = data[:, start_idx:end_idx, :]
                        
                        # Process current segment
                        output, state_dict = model(segment, state_dict)
                        
                        # Store output
                        for j in range(start_idx, end_idx):
                            frame_weight = position_weights[j] / sum(position_weights[start_idx:end_idx])
                            frame_outputs.append((output, frame_weight))
                    
                    # Process remaining frames (less than a complete segment)
                    remaining_frames = max_length % frames_per_segment
                    if remaining_frames > 0:
                        start_idx = complete_segments * frames_per_segment
                        
                        # Use padding to create a complete segment
                        padded_segment = torch.zeros(batch_size, frames_per_segment, input_dim).to(device)
                        padded_segment[:, :remaining_frames, :] = data[:, start_idx:, :]
                        
                        # Process padded segment
                        output, state_dict = model(padded_segment, state_dict)
                        
                        # Only weight real frames
                        for j in range(remaining_frames):
                            idx = start_idx + j
                            frame_outputs.append((output, position_weights[idx]))
                    
                    # Weight-combine all frame outputs
                    outputs = torch.zeros_like(frame_outputs[0][0])
                    total_weight = 0
                    for output, weight in frame_outputs:
                        outputs += output * weight
                        total_weight += weight
                    
                    # Normalize output
                    if total_weight > 0:
                        outputs = outputs / total_weight
                
                pred = outputs.argmax(dim=1)
                confmat.update(pred, target)
                
            except Exception as e:
                _logger.error(f"Error computing confusion matrix: {str(e)}")
                continue
    
    return confmat.compute().cpu().numpy()

def plot_confusion_matrix(cm, class_names):
    """
    Plot the confusion matrix as a heatmap
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        
    Returns:
        matplotlib.figure.Figure: Figure containing the heatmap
    """
    # Calculate accuracy
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm_norm, annot=False, fmt='.2f', xticklabels=class_names, yticklabels=class_names, cmap='Blues', ax=ax)
    
    # Set title and labels
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title('Normalized confusion matrix')
    
    # Rotate x-axis labels to prevent overlap
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig 