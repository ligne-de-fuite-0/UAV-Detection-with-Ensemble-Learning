"""Main training script for UAV detection."""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.convnext import convnext_tiny

from config.config import Config
from models.resnet_cifar import resnet20_cifar
from utils.dataset import ImageDataset
from utils.early_stopping import EarlyStopping
from utils.trainer import ModelTrainer


def create_transforms(config):
    """Create data transforms for both models."""
    # ResNet transforms
    transform_train_resnet = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(config.get('transforms.resnet.normalize_mean'),
                           config.get('transforms.resnet.normalize_std')),
    ])

    transform_val_test_resnet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.get('transforms.resnet.normalize_mean'),
                           config.get('transforms.resnet.normalize_std')),
    ])

    # ConvNeXt transforms
    transform_train_convnext = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(config.get('transforms.convnext.normalize_mean'),
                           config.get('transforms.convnext.normalize_std')),
    ])

    transform_val_test_convnext = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(config.get('transforms.convnext.normalize_mean'),
                           config.get('transforms.convnext.normalize_std')),
    ])

    return {
        'resnet': {
            'train': transform_train_resnet,
            'val': transform_val_test_resnet,
            'test': transform_val_test_resnet
        },
        'convnext': {
            'train': transform_train_convnext,
            'val': transform_val_test_convnext,
            'test': transform_val_test_convnext
        }
    }


def create_data_loaders(config, transforms):
    """Create data loaders for both models."""
    train_dir = config.get('data.train_dir')
    val_dir = config.get('data.val_dir')
    test_dir = config.get('data.test_dir')
    
    # Create datasets
    datasets = {}
    for model_name in ['resnet', 'convnext']:
        datasets[model_name] = {
            'train': ImageDataset(
                os.path.join(train_dir, 'UAV'),
                os.path.join(train_dir, 'background'),
                transform=transforms[model_name]['train']
            ),
            'val': ImageDataset(
                os.path.join(val_dir, 'UAV'),
                os.path.join(val_dir, 'background'),
                transform=transforms[model_name]['val']
            ),
            'test': ImageDataset(
                os.path.join(test_dir, 'UAV'),
                os.path.join(test_dir, 'background'),
                transform=transforms[model_name]['test']
            )
        }
    
    # Create weighted samplers for training
    samplers = {}
    for model_name in ['resnet', 'convnext']:
        labels = datasets[model_name]['train'].labels
        class_sample_count = np.array([len(np.where(np.array(labels) == t)[0]) 
                                     for t in np.unique(labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight).double()
        samplers[model_name] = torch.utils.data.WeightedRandomSampler(
            samples_weight, len(samples_weight)
        )
    
    # Create data loaders
    data_loaders = {}
    for model_name in ['resnet', 'convnext']:
        batch_size = config.get(f'{model_name}.batch_size')
        num_workers = config.get('data.num_workers')
        
        data_loaders[model_name] = {
            'train': DataLoader(
                datasets[model_name]['train'],
                batch_size=batch_size,
                sampler=samplers[model_name],
                num_workers=num_workers
            ),
            'val': DataLoader(
                datasets[model_name]['val'],
                batch_size=100,
                shuffle=False,
                num_workers=num_workers
            ),
            'test': DataLoader(
                datasets[model_name]['test'],
                batch_size=100,
                shuffle=False,
                num_workers=num_workers
            )
        }
    
    return data_loaders


def main():
    parser = argparse.ArgumentParser(description='UAV Detection Training')
    parser.add_argument('--config', type=str, default='config/default.json',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='./dataset1',
                       help='Path to dataset directory')
    parser.add_argument('--epochs_resnet', type=int, help='Number of epochs for ResNet')
    parser.add_argument('--epochs_convnext', type=int, help='Number of epochs for ConvNeXt')
    parser.add_argument('--batch_size_resnet', type=int, help='Batch size for ResNet')
    parser.add_argument('--batch_size_convnext', type=int, help='Batch size for ConvNeXt')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Update config with command line arguments
    if args.data_dir:
        config.set('data.train_dir', os.path.join(args.data_dir, 'train'))
        config.set('data.val_dir', os.path.join(args.data_dir, 'val'))
        config.set('data.test_dir', os.path.join(args.data_dir, 'test'))
    
    if args.epochs_resnet:
        config.set('resnet.epochs', args.epochs_resnet)
    if args.epochs_convnext:
        config.set('convnext.epochs', args.epochs_convnext)
    if args.batch_size_resnet:
        config.set('resnet.batch_size', args.batch_size_resnet)
    if args.batch_size_convnext:
        config.set('convnext.batch_size', args.batch_size_convnext)
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Create transforms and data loaders
    transforms_dict = create_transforms(config)
    data_loaders = create_data_loaders(config, transforms_dict)
    
    # Initialize models
    model_resnet = resnet20_cifar().to(device)
    model_convnext = convnext_tiny(pretrained=True)
    num_ftrs = model_convnext.classifier[2].in_features
    model_convnext.classifier[2] = nn.Linear(num_ftrs, 2)
    model_convnext = model_convnext.to(device)
    
    # Initialize trainers
    trainer_resnet = ModelTrainer(
        model=model_resnet,
        device=device,
        config=config,
        model_type='resnet'
    )
    
    trainer_convnext = ModelTrainer(
        model=model_convnext,
        device=device,
        config=config,
        model_type='convnext'
    )
    
    # Train models
    print("----- Training ResNet-20 -----")
    trainer_resnet.train(data_loaders['resnet'])
    
    print("----- Training ConvNeXt-Tiny -----")
    trainer_convnext.train(data_loaders['convnext'])
    
    # Test combined model
    print("----- Testing Combined Model -----")
    trainer_resnet.test_combined(
        model_convnext, 
        data_loaders['resnet']['test'], 
        data_loaders['convnext']['test']
    )


if __name__ == '__main__':
    main()