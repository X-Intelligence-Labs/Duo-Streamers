import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

from config import BATCH_SIZE, INPUT_DIM, OUTPUT_DIM, SELECTED_GESTURE_NAME, FULL_GESTURE_NAME, GESTURE_NAME
from data_loader import load_gesture_dataset, normalize_skeleton, pad_data, SHREC2017Dataset
from models import StreamingSightMu, StreamingSightBi, StreamingSightMuOriginal, TCRN, MSTCRN
from trainer import train, test, save_model, compute_confusion_matrix, plot_confusion_matrix

# Setup logging
_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    # Parameter setup
    parser = argparse.ArgumentParser(description='SHREC2017 Hand Gesture Recognition')
    parser.add_argument('--model-type', type=str, default='tcrn',
                        choices=['mu', 'bi', 'mu_original', 'tcrn', 'mstcrn'],
                        help='Model type: mu (StreamingSightMu), bi (StreamingSightBi), mu_original (StreamingSightMuOriginal), tcrn (TCRN) or mstcrn (MS-TCRN) (default: tcrn)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Training batch size (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=BATCH_SIZE,
                        help='Testing batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='How many batches to log training info (default: 10)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='Whether to save the model')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit the number of samples loaded (for testing)')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'single_finger'],
                        help='Training mode: full (all gestures) or single_finger (only one-finger gestures) (default: full)')
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau',
                        choices=['step', 'reduce_on_plateau', 'cosine', 'none'],
                        help='Learning rate scheduler type (default: reduce_on_plateau)')
    parser.add_argument('--step-size', type=int, default=30,
                        help='Step size for StepLR (default: 30)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Learning rate decay factor (default: 0.1)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for ReduceLROnPlateau (default: 10)')
    parser.add_argument('--min-lr', type=float, default=1e-5,
                        help='Minimum learning rate (default: 1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs (default: 5)')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Whether to use TensorBoard for logging training progress')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--start-epoch', type=int, default=1,
                        help='Starting epoch number if not resuming from checkpoint (default: 1)')
    parser.add_argument('--additional-epochs', type=int, default=100,
                        help='Additional epochs to train when resuming from checkpoint (default: 100)')
    parser.add_argument('--frames-per-segment', type=int, default=3,
                        help='Number of frames per segment for TCRN model (default: 3)')
    parser.add_argument('--hidden-size', type=int, default=768,
                        help='Hidden layer size for model (default: 768)')
    parser.add_argument('--slow-update-rate', type=int, default=3,
                        help='Slow layer update frequency for MS-TCRN model (default: 3)')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='Weight decay coefficient for L2 regularization (default: 0.00001)')
    args = parser.parse_args()
    
    # Set class selection based on chosen mode
    global SELECTED_GESTURE_NAME, OUTPUT_DIM
    if args.mode == 'single_finger':
        SELECTED_GESTURE_NAME = GESTURE_NAME
        _logger.info("Using single finger mode (14 classes)")
    else:
        SELECTED_GESTURE_NAME = FULL_GESTURE_NAME
        _logger.info("Using full gesture mode (28 classes)")
    
    OUTPUT_DIM = len(SELECTED_GESTURE_NAME)
    _logger.info(f"Output dimension: {OUTPUT_DIM}")
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Set up TensorBoard
    if args.tensorboard:
        log_dir = os.path.join("runs", f"{args.model_type}_{args.mode}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        writer = SummaryWriter(log_dir=log_dir)
        _logger.info(f"TensorBoard logs saved to: {log_dir}")
    else:
        writer = None
    
    # Load data
    _logger.info("Loading training data...")
    train_gesture_data, train_gesture_labels = load_gesture_dataset("train_gestures.txt", args.max_samples, args.mode)
    
    _logger.info("Loading test data...")
    test_gesture_data, test_gesture_labels = load_gesture_dataset("test_gestures.txt", args.max_samples, args.mode)
    
    # Preprocess data
    _logger.info("Preprocessing training data...")
    # Apply normalization to each skeleton sequence
    normalized_train_data = [normalize_skeleton(sequence) for sequence in train_gesture_data]
    # Pad/truncate sequences
    train_data = pad_data(normalized_train_data)
    train_label = np.array(train_gesture_labels)
    
    _logger.info("Preprocessing test data...")
    normalized_test_data = [normalize_skeleton(sequence) for sequence in test_gesture_data]
    test_data = pad_data(normalized_test_data)
    test_label = np.array(test_gesture_labels)
    
    _logger.info(f"Training data shape: {train_data.shape}")
    _logger.info(f"Test data shape: {test_data.shape}")
    
    # Create datasets and data loaders
    train_dataset = SHREC2017Dataset(train_data, train_label, augment=True)
    test_dataset = SHREC2017Dataset(test_data, test_label, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    # Create model
    input_dim = INPUT_DIM
    output_dim = OUTPUT_DIM
    hidden_size = args.hidden_size
    
    if args.model_type == 'mu':
        model = StreamingSightMu(input_dim, output_dim, hidden_size=hidden_size).to(device)
        _logger.info("Using StreamingSightMu model")
    elif args.model_type == 'bi':
        model = StreamingSightBi(input_dim, output_dim, hidden_size=hidden_size).to(device)
        _logger.info("Using StreamingSightBi model")
    elif args.model_type == 'tcrn':
        model = TCRN(input_dim, output_dim, hidden_size=hidden_size, frames_per_segment=args.frames_per_segment).to(device)
        _logger.info(f"Using TCRN model, processing {args.frames_per_segment} frames at a time")
    elif args.model_type == 'mstcrn':
        model = MSTCRN(input_dim, output_dim, hidden_size=hidden_size, 
                      frames_per_segment=args.frames_per_segment,
                      slow_update_rate=args.slow_update_rate).to(device)
        _logger.info(f"Using MS-TCRN model, processing {args.frames_per_segment} frames at a time, slow layer updates every {args.slow_update_rate} steps")
    else:
        model = StreamingSightMuOriginal(input_dim, output_dim, hidden_size=hidden_size).to(device)
        _logger.info("Using StreamingSightMuOriginal model")
    
    # Output model structure information
    _logger.info(f"Model input dimension: {input_dim}")
    _logger.info(f"Model output dimension: {output_dim}")
    _logger.info(f"Model hidden layer size: {hidden_size}")
    
    # Optimizer and loss function
    if args.model_type in ['tcrn', 'mstcrn']:
        # Add weight decay for TCRN and MS-TCRN models
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        _logger.info(f"Using Adam optimizer with lr={args.lr}, weight_decay={args.weight_decay}")
    else:
        # Other models use original optimizer configuration
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        _logger.info(f"Using Adam optimizer with lr={args.lr}, no weight decay")
    
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        _logger.info(f"Using StepLR scheduler with step_size={args.step_size}, gamma={args.gamma}")
    elif args.scheduler == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma, 
                                    patience=args.patience, min_lr=args.min_lr)
        _logger.info(f"Using ReduceLROnPlateau scheduler with patience={args.patience}, gamma={args.gamma}")
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        _logger.info(f"Using CosineAnnealingLR scheduler with T_max={args.epochs}, min_lr={args.min_lr}")
    else:
        scheduler = None
        _logger.info("Not using a learning rate scheduler")
    
    # Initialize training parameters
    start_epoch = args.start_epoch
    best_accuracy = 0.0
    
    # Resume training from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            _logger.info(f"Loading model from checkpoint {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Only load if keys exist
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'accuracy' in checkpoint:
                best_accuracy = checkpoint['accuracy']
                
            _logger.info(f"Successfully loaded checkpoint: epoch={start_epoch-1}, best_accuracy={best_accuracy:.2f}%")
            
            # Set total training epochs
            total_epochs = start_epoch + args.additional_epochs - 1
            _logger.info(f"Will train for {args.additional_epochs} more epochs, starting from epoch {start_epoch}, for a total of {total_epochs} epochs")
        else:
            _logger.warning(f"No checkpoint found at: {args.resume}")
            total_epochs = args.epochs
    else:
        total_epochs = args.epochs
    
    # Learning rate warmup
    warmup_epochs = args.warmup_epochs
    if warmup_epochs > 0 and start_epoch == 1:  # Only apply warmup when starting from scratch
        _logger.info(f"Using learning rate warmup, warmup_epochs={warmup_epochs}")
    
    # Train model
    _logger.info("Starting training...")
    
    for epoch in range(start_epoch, total_epochs + 1):
        # Learning rate warmup
        if epoch <= warmup_epochs and scheduler is not None and start_epoch == 1:
            # Linear warmup
            warmup_factor = min(1.0, epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * warmup_factor
            _logger.info(f"Warmup phase {epoch}/{warmup_epochs}, lr={optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Log confusion matrix every 10 epochs
            if epoch % 10 == 0:
                try:
                    # Compute confusion matrix
                    confusion_matrix = compute_confusion_matrix(model, device, test_loader)
                    # Add to TensorBoard
                    fig = plot_confusion_matrix(confusion_matrix, class_names=SELECTED_GESTURE_NAME)
                    writer.add_figure(f'Confusion Matrix/epoch_{epoch}', fig, epoch)
                except Exception as e:
                    _logger.error(f"Error logging confusion matrix: {str(e)}")
        
        # Update learning rate
        if scheduler is not None and (epoch > warmup_epochs or start_epoch > 1):
            if args.scheduler == 'reduce_on_plateau':
                scheduler.step(test_loss)
            else:
                scheduler.step()
            
            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            _logger.info(f"Current learning rate: {current_lr:.6f}")
        
        # Save best model
        if test_acc > best_accuracy and args.save_model:
            best_accuracy = test_acc
            save_model(model, optimizer, epoch, test_acc, f"{args.model_type}_{args.mode}")
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    _logger.info(f"Training complete! Best accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main() 