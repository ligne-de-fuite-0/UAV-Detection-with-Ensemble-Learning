"""
Evaluation script for trained UAV detection models
"""

import torch
import argparse
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils.trainer import Trainer
from config.config import get_config


def evaluate_models(config_path, data_dir, model_dir):
    """Evaluate trained models"""
    config = get_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize trainer
    trainer = Trainer(config, device, data_dir, model_dir)
    
    # Load trained models
    resnet_checkpoint = os.path.join(model_dir, 'checkpoint_resnet.pt')
    convnext_checkpoint = os.path.join(model_dir, 'checkpoint_convnext.pt')
    
    if os.path.exists(resnet_checkpoint):
        trainer.model_resnet.load_state_dict(torch.load(resnet_checkpoint))
        print("ResNet model loaded successfully")
    else:
        print(f"ResNet checkpoint not found: {resnet_checkpoint}")
        return
        
    if os.path.exists(convnext_checkpoint):
        trainer.model_convnext.load_state_dict(torch.load(convnext_checkpoint))
        print("ConvNeXt model loaded successfully")
    else:
        print(f"ConvNeXt checkpoint not found: {convnext_checkpoint}")
        return
    
    # Evaluate individual models
    resnet_accuracy = evaluate_single_model(trainer.model_resnet, trainer.test_loader_resnet, device)
    convnext_accuracy = evaluate_single_model(trainer.model_convnext, trainer.test_loader_convnext, device)
    
    print(f"ResNet Test Accuracy: {resnet_accuracy:.2f}%")
    print(f"ConvNeXt Test Accuracy: {convnext_accuracy:.2f}%")
    
    # Evaluate combined model
    trainer.test_combined()


def evaluate_single_model(model, test_loader, device):
    """Evaluate a single model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate UAV Detection Models')
    parser.add_argument('--config', type=str, default='config/default.json')
    parser.add_argument('--data_dir', type=str, default='./dataset1')
    parser.add_argument('--model_dir', type=str, default='./outputs')
    
    args = parser.parse_args()
    evaluate_models(args.config, args.data_dir, args.model_dir)