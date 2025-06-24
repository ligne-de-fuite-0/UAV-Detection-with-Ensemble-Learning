# UAV Detection with Ensemble Learning

A PyTorch implementation for UAV (Unmanned Aerial Vehicle) detection using ensemble learning with ResNet-20 and ConvNeXt-Tiny models.

## Features

- **Dual Model Architecture**: Combines ResNet-20 and ConvNeXt-Tiny for robust detection
- **Class Imbalance Handling**: Weighted sampling to handle unbalanced datasets
- **Early Stopping**: Prevents overfitting with configurable patience
- **Ensemble Prediction**: Combines predictions from both models
- **Comprehensive Evaluation**: Detailed metrics and analysis
- **Configuration Management**: JSON-based configuration system
- **Command-line Interface**: Easy-to-use training and evaluation scripts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ligne-de-fuite-0/uav-detection.git
cd uav-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Dataset Structure

Organize your dataset as follows:
```
dataset1/
├── train/
│   ├── UAV/
│   └── background/
├── val/
│   ├── UAV/
│   └── background/
└── test/
    ├── UAV/
    └── background/
```

## Usage

### Training

Train both models with default configuration:
```bash
python main.py --data_dir ./dataset1 --config config/default.json
```

Train with custom parameters:
```bash
python main.py --data_dir ./dataset1 --epochs_resnet 100 --epochs_convnext 20 --batch_size_resnet 64
```

### Evaluation

Evaluate trained models:
```bash
python scripts/evaluate.py --resnet_checkpoint checkpoint_resnet.pt --convnext_checkpoint checkpoint_convnext.pt --data_dir ./dataset1
```

### Single Image Inference

Run inference on a single image:
```bash
python scripts/inference.py --image_path path/to/image.jpg --resnet_checkpoint checkpoint_resnet.pt --convnext_checkpoint checkpoint_convnext.pt
```

## Configuration

The project uses JSON configuration files. See `config/default.json` for available parameters:

- Model architectures and hyperparameters
- Training schedules and optimization settings
- Data augmentation parameters
- Early stopping configuration

## Model Architecture

### ResNet-20
- Designed for CIFAR-style 32x32 images
- 3 residual blocks with [3,3,3] layers
- Batch normalization and ReLU activation
- Adaptive average pooling

### ConvNeXt-Tiny
- Pre-trained on ImageNet
- Fine-tuned for binary classification
- Input size: 224x224
- Advanced data augmentation

## Results

The ensemble approach combines predictions from both models to improve overall accuracy and robustness.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{uav-detection-2024,
  title={UAV Detection with Ensemble Learning},
  author={ligne-de-fuite-0},
  year={2024},
  publisher={GitHub},
  url={https://github.com/ligne-de-fuite-0/uav-detection}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Torchvision for pre-trained models and utilities
- Research community for ConvNeXt architecture innovations
