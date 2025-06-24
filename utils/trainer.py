"""
Training utilities for UAV detection models
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from torchvision.models.convnext import convnext_tiny
import json
from datetime import datetime

from .dataset import ImageDataset
from .early_stopping import EarlyStopping
from models.resnet_cifar import resnet20_cifar


class Trainer:
    """Main trainer class for UAV detection models"""
    
    def __init__(self, config, device, data_dir, output_dir):
        self.config = config
        self.device = device
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Initialize models
        self.model_resnet = resnet20_cifar().to(device)
        self.model_convnext = self._get_convnext_model().to(device)
        
        # Initialize data loaders
        self._setup_data_loaders()
        
        # Initialize optimizers and criteria
        self._setup_optimizers()
        
        # Results storage
        self.results = {
            'resnet': {'train_losses': [], 'train_accuracies': [], 'val_losses': [], 'val_accuracies': []},
            'convnext': {'train_losses': [], 'train_accuracies': [], 'val_losses': [], 'val_accuracies': []}
        }
    
    def _get_convnext_model(self):
        """Initialize ConvNeXt model"""
        model = convnext_tiny(pretrained=True)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, 2)
        return model
    
    def _setup_data_loaders(self):
        """Setup data loaders for both models"""
        # Data transforms for ResNet
        transform_train_resnet = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
        
        transform_val_test_resnet = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
        
        # Data transforms for ConvNeXt
        transform_train_convnext = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        transform_val_test_convnext = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Create datasets
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        test_dir = os.path.join(self.data_dir, 'test')
        
        # ResNet datasets
        self.train_dataset_resnet = ImageDataset(
            os.path.join(train_dir, 'UAV'),
            os.path.join(train_dir, 'background'),
            transform=transform_train_resnet
        )
        self.val_dataset_resnet = ImageDataset(
            os.path.join(val_dir, 'UAV'),
            os.path.join(val_dir, 'background'),
            transform=transform_val_test_resnet
        )
        self.test_dataset_resnet = ImageDataset(
            os.path.join(test_dir, 'UAV'),
            os.path.join(test_dir, 'background'),
            transform=transform_val_test_resnet
        )
        
        # ConvNeXt datasets
        self.train_dataset_convnext = ImageDataset(
            os.path.join(train_dir, 'UAV'),
            os.path.join(train_dir, 'background'),
            transform=transform_train_convnext
        )
        self.val_dataset_convnext = ImageDataset(
            os.path.join(val_dir, 'UAV'),
            os.path.join(val_dir, 'background'),
            transform=transform_val_test_convnext
        )
        self.test_dataset_convnext = ImageDataset(
            os.path.join(test_dir, 'UAV'),
            os.path.join(test_dir, 'background'),
            transform=transform_val_test_convnext
        )
        
        # Setup weighted samplers for class imbalance
        self._setup_weighted_samplers()
        
        # Create data loaders
        self.train_loader_resnet = DataLoader(
            self.train_dataset_resnet, 
            batch_size=self.config['resnet']['batch_size'], 
            sampler=self.sampler_resnet, 
            num_workers=4
        )
        self.val_loader_resnet = DataLoader(
            self.val_dataset_resnet, 
            batch_size=100, 
            shuffle=False, 
            num_workers=4
        )
        self.test_loader_resnet = DataLoader(
            self.test_dataset_resnet, 
            batch_size=100, 
            shuffle=False, 
            num_workers=4
        )
        
        self.train_loader_convnext = DataLoader(
            self.train_dataset_convnext, 
            batch_size=self.config['convnext']['batch_size'], 
            sampler=self.sampler_convnext, 
            num_workers=4
        )
        self.val_loader_convnext = DataLoader(
            self.val_dataset_convnext, 
            batch_size=100, 
            shuffle=False, 
            num_workers=4
        )
        self.test_loader_convnext = DataLoader(
            self.test_dataset_convnext, 
            batch_size=100, 
            shuffle=False, 
            num_workers=4
        )
    
    def _setup_weighted_samplers(self):
        """Setup weighted random samplers for handling class imbalance"""
        # ResNet sampler
        labels_resnet = self.train_dataset_resnet.labels
        class_sample_count_resnet = np.array([
            len(np.where(np.array(labels_resnet) == t)[0]) for t in np.unique(labels_resnet)
        ])
        weight_resnet = 1. / class_sample_count_resnet
        samples_weight_resnet = np.array([weight_resnet[t] for t in labels_resnet])
        samples_weight_resnet = torch.from_numpy(samples_weight_resnet).double()
        self.sampler_resnet = torch.utils.data.WeightedRandomSampler(
            samples_weight_resnet, len(samples_weight_resnet)
        )
        
        # ConvNeXt sampler
        labels_convnext = self.train_dataset_convnext.labels
        class_sample_count_convnext = np.array([
            len(np.where(np.array(labels_convnext) == t)[0]) for t in np.unique(labels_convnext)
        ])
        weight_convnext = 1. / class_sample_count_convnext
        samples_weight_convnext = np.array([weight_convnext[t] for t in labels_convnext])
        samples_weight_convnext = torch.from_numpy(samples_weight_convnext).double()
        self.sampler_convnext = torch.utils.data.WeightedRandomSampler(
            samples_weight_convnext, len(samples_weight_convnext)
        )
    
    def _setup_optimizers(self):
        """Setup optimizers and loss criteria"""
        # ResNet optimizer
        self.criterion_resnet = nn.CrossEntropyLoss()
        self.optimizer_resnet = optim.SGD(
            self.model_resnet.parameters(), 
            lr=self.config['resnet']['learning_rate'],
            momentum=0.9, 
            weight_decay=5e-4
        )
        self.scheduler_resnet = optim.lr_scheduler.MultiStepLR(
            self.optimizer_resnet, 
            milestones=[100, 150], 
            gamma=0.1
        )
        
        # ConvNeXt optimizer
        self.criterion_convnext = nn.CrossEntropyLoss()
        self.optimizer_convnext = optim.Adam(
            self.model_convnext.parameters(), 
            lr=self.config['convnext']['learning_rate']
        )
        self.scheduler_convnext = optim.lr_scheduler.StepLR(
            self.optimizer_convnext, 
            step_size=5, 
            gamma=0.1
        )
    
    def train_epoch(self, model, train_loader, optimizer, criterion, model_name):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / len(train_loader.dataset)
        return avg_loss, accuracy
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                total_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
                
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / len(val_loader.dataset)
        return avg_loss, accuracy
    
    def train_resnet(self):
        """Train ResNet model"""
        print("----- Training ResNet-20 -----")
        early_stopping = EarlyStopping(
            patience=self.config['resnet']['patience'], 
            verbose=True, 
            path=os.path.join(self.output_dir, 'checkpoint_resnet.pt')
        )
        
        for epoch in range(1, self.config['resnet']['epochs'] + 1):
            train_loss, train_acc = self.train_epoch(
                self.model_resnet, self.train_loader_resnet, 
                self.optimizer_resnet, self.criterion_resnet, 'ResNet'
            )
            val_loss, val_acc = self.validate_epoch(
                self.model_resnet, self.val_loader_resnet, self.criterion_resnet
            )
            
            self.scheduler_resnet.step()
            
            # Store results
            self.results['resnet']['train_losses'].append(train_loss)
            self.results['resnet']['train_accuracies'].append(train_acc)
            self.results['resnet']['val_losses'].append(val_loss)
            self.results['resnet']['val_accuracies'].append(val_acc)
            
            print(f'ResNet Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            early_stopping(val_loss, self.model_resnet)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
                
        # Load best model
        self.model_resnet.load_state_dict(torch.load(os.path.join(self.output_dir, 'checkpoint_resnet.pt')))
    
    def train_convnext(self):
        """Train ConvNeXt model"""
        print("----- Training ConvNeXt-Tiny -----")
        early_stopping = EarlyStopping(
            patience=self.config['convnext']['patience'], 
            verbose=True, 
            path=os.path.join(self.output_dir, 'checkpoint_convnext.pt')
        )
        
        for epoch in range(1, self.config['convnext']['epochs'] + 1):
            train_loss, train_acc = self.train_epoch(
                self.model_convnext, self.train_loader_convnext, 
                self.optimizer_convnext, self.criterion_convnext, 'ConvNeXt'
            )
            val_loss, val_acc = self.validate_epoch(
                self.model_convnext, self.val_loader_convnext, self.criterion_convnext
            )
            
            self.scheduler_convnext.step()
            
            # Store results
            self.results['convnext']['train_losses'].append(train_loss)
            self.results['convnext']['train_accuracies'].append(train_acc)
            self.results['convnext']['val_losses'].append(val_loss)
            self.results['convnext']['val_accuracies'].append(val_acc)
            
            print(f'ConvNeXt Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            early_stopping(val_loss, self.model_convnext)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
                
        # Load best model
        self.model_convnext.load_state_dict(torch.load(os.path.join(self.output_dir, 'checkpoint_convnext.pt')))
    
    def train_all_models(self):
        """Train both models"""
        self.train_resnet()
        self.train_convnext()
    
    def test_combined(self):
        """Test combined model performance"""
        print("----- Testing Combined Models -----")
        self.model_resnet.eval()
        self.model_convnext.eval()
        
        correct_combined = 0
        total_combined = 0
        all_preds_resnet = []
        all_preds_convnext = []
        all_targets = []
        
        with torch.no_grad():
            for (data_resnet, target_resnet), (data_convnext, target_convnext) in zip(
                self.test_loader_resnet, self.test_loader_convnext
            ):
                # Ensure targets are consistent
                assert torch.all(target_resnet == target_convnext)
                
                data_resnet, target_resnet = data_resnet.to(self.device), target_resnet.to(self.device)
                data_convnext, target_convnext = data_convnext.to(self.device), target_convnext.to(self.device)
                
                # ResNet prediction
                output_resnet = self.model_resnet(data_resnet)
                _, predicted_resnet = output_resnet.max(1)
                
                # ConvNeXt prediction
                output_convnext = self.model_convnext(data_convnext)
                _, predicted_convnext = output_convnext.max(1)
                
                # Collect predictions
                all_preds_resnet.extend(predicted_resnet.cpu().numpy())
                all_preds_convnext.extend(predicted_convnext.cpu().numpy())
                all_targets.extend(target_resnet.cpu().numpy())
                
                # Combined prediction (ensemble)
                combined_preds = (predicted_resnet == predicted_convnext)
                correct_combined += (combined_preds * (predicted_resnet == target_resnet)).sum().item()
                total_combined += combined_preds.sum().item()
        
        # Calculate accuracies
        if total_combined > 0:
            accuracy_combined = 100. * correct_combined / total_combined
            print(f'Combined Model Accuracy (when models agree): {accuracy_combined:.2f}%')
        
        # Analyze disagreements
        disagreement_indices = np.where(np.array(all_preds_resnet) != np.array(all_preds_convnext))[0]
        print(f"Number of disagreements between models: {len(disagreement_indices)} / {len(all_preds_resnet)}")
        
        # Store test results
        self.results['test'] = {
            'combined_accuracy': accuracy_combined if total_combined > 0 else 0,
            'disagreements': len(disagreement_indices),
            'total_samples': len(all_preds_resnet)
        }
    
    def save_results(self):
        """Save training results and model performance"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f'results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {results_file}")