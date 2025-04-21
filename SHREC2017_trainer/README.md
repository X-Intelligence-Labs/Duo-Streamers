# SHREC2017 Hand Gesture Recognition

This package contains a modular implementation of hand gesture recognition models for the SHREC2017 dataset.

## Structure

The code is organized into the following modules:

- `config.py`: Constants and configuration settings
- `data_loader.py`: Data loading and preprocessing
- `data_augmentation.py`: Augmentation techniques for skeleton data
- `models.py`: Neural network model implementations
- `trainer.py`: Training and evaluation functions
- `main.py`: Main program entry point with argument handling

## Models

The implementation includes several models:

1. **StreamingSightMu**: Advanced gated recurrent model with joint attention mechanism
2. **StreamingSightBi**: Bidirectional LSTM model
3. **StreamingSightMuOriginal**: Original model with recurrent fully connected layers
4. **TCRN**: Temporal Convolutional Recurrent Network
5. **MSTCRN**: Multi-Scale Temporal Convolutional Recurrent Network with fast and slow update paths

## Usage

### Training

To train a model, run:

```bash
python main.py --model-type tcrn --mode full
```

Available options:
- `--model-type`: Model architecture (`mu`, `bi`, `mu_original`, `tcrn`, `mstcrn`)
- `--mode`: Training mode (`full` for all 28 classes, `single_finger` for 14 classes)
- `--batch-size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--tensorboard`: Enable TensorBoard logging
- `--hidden-size`: Hidden layer size
- `--frames-per-segment`: Number of frames per segment for TCRN model

For a full list of options, run:

```bash
python main.py --help
```

### Resuming Training

To resume training from a checkpoint:

```bash
python main.py --model-type tcrn --resume ./models/tcrn_full_checkpoint_epoch50_acc85.00.pt --additional-epochs 50
```

## Data

The code expects the SHREC2017 dataset in the following format:
- Training labels: `train_gestures.txt`
- Test labels: `test_gestures.txt`
- Skeleton files in `gesture_X/finger_Y/subject_Z/essai_W/skeletons_world.txt`

## Requirements

- Python 3.6+
- PyTorch 1.7+
- NumPy
- SciPy
- Matplotlib
- TensorBoard (optional for visualization)
- tqdm

## Citation

If you use this code, please cite the original SHREC2017 paper:

```
De Smedt, Q., Wannous, H., Vandeborre, J.P., 
"Skeleton-based dynamic hand gesture recognition", 
IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2016
``` 