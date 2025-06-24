"""
Inference script for UAV detection
"""

import torch
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms

from models.resnet_cifar import resnet20_cifar
from torchvision.models.convnext import convnext_tiny
import torch.nn as nn


def load_models(resnet_path, convnext_path, device):
    """Load trained models"""
    # Load ResNet
    model_resnet = resnet20_cifar().to(device)
    if os.path.exists(resnet_path):
        model_resnet.load_state_dict(torch.load(resnet_path, map_location=device))
        model_resnet.eval()
    else:
        raise FileNotFoundError(f"ResNet checkpoint not found: {resnet_path}")
    
    # Load ConvNeXt
    model_convnext = convnext_tiny(pretrained=False)
    num_ftrs = model_convnext.classifier[2].in_features
    model_convnext.classifier[2] = nn.Linear(num_ftrs, 2)
    model_convnext = model_convnext.to(device)
    
    if os.path.exists(convnext_path):
        model_convnext.load_state_dict(torch.load(convnext_path, map_location=device))
        model_convnext.eval()
    else:
        raise FileNotFoundError(f"ConvNeXt checkpoint not found: {convnext_path}")
    
    return model_resnet, model_convnext


def preprocess_image(image_path, model_type):
    """Preprocess image for inference"""
    image = Image.open(image_path).convert('RGB')
    
    if model_type == 'resnet':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
    elif model_type == 'convnext':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError("Model type must be 'resnet' or 'convnext'")
    
    return transform(image).unsqueeze(0)


def predict(model_resnet, model_convnext, image_path, device):
    """Make prediction on a single image"""
    # Preprocess for both models
    image_resnet = preprocess_image(image_path, 'resnet').to(device)
    image_convnext = preprocess_image(image_path, 'convnext').to(device)
    
    with torch.no_grad():
        # ResNet prediction
        output_resnet = model_resnet(image_resnet)
        prob_resnet = torch.softmax(output_resnet, dim=1)
        pred_resnet = output_resnet.argmax(dim=1).item()
        conf_resnet = prob_resnet[0][pred_resnet].item()
        
        # ConvNeXt prediction
        output_convnext = model_convnext(image_convnext)
        prob_convnext = torch.softmax(output_convnext, dim=1)
        pred_convnext = output_convnext.argmax(dim=1).item()
        conf_convnext = prob_convnext[0][pred_convnext].item()
    
    # Class names
    class_names = ['Background', 'UAV']
    
    return {
        'resnet': {
            'prediction': class_names[pred_resnet],
            'confidence': conf_resnet
        },
        'convnext': {
            'prediction': class_names[pred_convnext],
            'confidence': conf_convnext
        },
        'ensemble': {
            'agreement': pred_resnet == pred_convnext,
            'prediction': class_names[pred_resnet] if pred_resnet == pred_convnext else 'Disagreement'
        }
    }


def main():
    parser = argparse.ArgumentParser(description='UAV Detection Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--resnet_model', type=str, default='./outputs/checkpoint_resnet.pt')
    parser.add_argument('--convnext_model', type=str, default='./outputs/checkpoint_convnext.pt')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    model_resnet, model_convnext = load_models(args.resnet_model, args.convnext_model, device)
    
    # Make prediction
    result = predict(model_resnet, model_convnext, args.image, device)
    
    # Print results
    print(f"Image: {args.image}")
    print(f"ResNet Prediction: {result['resnet']['prediction']} (Confidence: {result['resnet']['confidence']:.3f})")
    print(f"ConvNeXt Prediction: {result['convnext']['prediction']} (Confidence: {result['convnext']['confidence']:.3f})")
    print(f"Ensemble Result: {result['ensemble']['prediction']}")
    print(f"Models Agree: {result['ensemble']['agreement']}")


if __name__ == "__main__":
    main()