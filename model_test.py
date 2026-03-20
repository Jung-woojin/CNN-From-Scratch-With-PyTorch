#!/usr/bin/env python3
"""
CNN Architecture Comparison - Unified Testing Framework

This module provides a unified interface to test and compare various CNN architectures
on PyTorch, including:
- EfficientNet, SqueezeNet, DenseNet, NASNet, MobileNetV3
- AlexNet, VGG19, ResNet50, Inception, MobileNetV2, Xception, ResNeXt, RepVGG, GoogLeNet

Usage:
    python model_test.py --model efficientnet --input test_image.jpg --device cpu
    python model_test.py --model all --batch_size 32
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path


# Import all CNN architectures
from EfficientNet import create_efficientnet_b0
from SqueezeNet import create_squeezenet
from DenseNet import densenet121
from NASNet import create_nasnet_mobile
from MobileNetV3 import create_mobilenetv3_large
from Alexnet import create_alexnet
from VGG19 import create_vgg19
from Resnet50 import create_resnet50
from Inception import create_inception_v1
from Mobilenetv2 import create_mobilenet_v2
from Xception import create_xception
from ResNext import create_resnext
from RepVGG import create_repvgg
from GoogLeNet import create_googlenet


# Model registry
MODEL_REGISTRY = {
    'alexnet': {
        'class': create_alexnet,
        'input_size': 224,
        'description': 'First deep CNN with ReLU, Dropout (2012)'
    },
    'vgg19': {
        'class': create_vgg19,
        'input_size': 224,
        'description': 'Thin 3x3 convolutions, 19 layers (2014)'
    },
    'resnet50': {
        'class': create_resnet50,
        'input_size': 224,
        'description': 'Bottleneck, Skip Connections, Residual Learning (2015)'
    },
    'inception': {
        'class': create_inception_v1,
        'input_size': 224,
        'description': 'Multi-Branch, 1x1 dimensionality reduction (2014)'
    },
    'mobilenetv2': {
        'class': create_mobilenet_v2,
        'input_size': 224,
        'description': 'Depthwise Separable, Inverted Residual (2018)'
    },
    'xception': {
        'class': create_xception,
        'input_size': 299,
        'description': 'Depthwise Separable, Extends Inception (2017)'
    },
    'resnext': {
        'class': create_resnext,
        'input_size': 224,
        'description': 'Group Convolution, Cardinality (2016)'
    },
    'repvgg': {
        'class': create_repvgg,
        'input_size': 224,
        'description': 'Reparameterization, Training/Inference separation (2021)'
    },
    'googlenet': {
        'class': create_googlenet,
        'input_size': 224,
        'description': 'Multi-Branch, Auxiliary Classifiers (2014)'
    },
    'efficientnet': {
        'class': create_efficientnet_b0,
        'input_size': 224,
        'description': 'Compound Scaling, MBConv, SOTA accuracy (2019)'
    },
    'squeezenet': {
        'class': create_squeezenet,
        'input_size': 224,
        'description': 'Fire Module, 50x fewer parameters (2016)'
    },
    'densenet': {
        'class': create_densenet121,
        'input_size': 224,
        'description': 'Dense Connection, Feature Reuse (2017)'
    },
    'nasnet': {
        'class': create_nasnet_mobile,
        'input_size': 331,
        'description': 'Neural Architecture Search, Reusable blocks (2018)'
    },
    'mobilenetv3': {
        'class': create_mobilenetv3_large,
        'input_size': 224,
        'description': 'h-swish activation, SE blocks, Mobile optimized (2019)'
    }
}


class CNNTester:
    """Unified CNN Architecture Testing Framework
    
    Provides methods to:
    - Load models by name
    - Process images
    - Run inference
    - Compare architectures
    - Visualize results
    """
    
    def __init__(self, device=None):
        """Initialize the CNN Tester
        
        Args:
            device: PyTorch device ('cpu' or 'cuda')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
    
    def get_model(self, model_name, num_classes=1000, pretrained=False):
        """Get a model by name
        
        Args:
            model_name: Name of the model to load
            num_classes: Number of output classes (default 1000)
            pretrained: Whether to load pretrained weights (not implemented yet)
        
        Returns:
            PyTorch model on specified device
            
        Raises:
            ValueError: If model_name is not in registry
        """
        if model_name.lower() not in MODEL_REGISTRY:
            available_models = ', '.join(sorted(MODEL_REGISTRY.keys()))
            raise ValueError(f"Unknown model '{model_name}'. Available: {available_models}")
        
        model_info = MODEL_REGISTRY[model_name.lower()]
        model_fn = model_info['class']
        
        # Create model
        model = model_fn(num_classes=num_classes)
        
        # Move to device
        model = model.to(self.device)
        
        # Set to eval mode
        model.eval()
        
        return model
    
    def load_image(self, image_path, size):
        """Load and preprocess an image
        
        Args:
            image_path: Path to image file
            size: Input size for the model
        
        Returns:
            Preprocessed tensor
        """
        from torchvision import transforms
        from PIL import Image
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocessing transforms (ImageNet standardized)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Resize based on model requirements
        if size != 224:
            preprocess = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # Convert to tensor and add batch dimension
        input_tensor = preprocess(image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        return input_tensor
    
    def classify(self, model, image_path, top_k=5):
        """Run inference on an image
        
        Args:
            model: PyTorch model
            image_path: Path to image file
            top_k: Return top-k predictions (default 5)
        
        Returns:
            List of (class_idx, probability) tuples
        """
        # Load and preprocess image
        input_tensor = self.load_image(image_path, MODEL_REGISTRY[model.__class__.__name__.lower()]['input_size'])
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get top-k predictions
        top_probs, top_idx = torch.topk(probabilities, top_k, dim=1)
        
        results = []
        for i in range(top_k):
            results.append((top_idx[0][i].item(), top_probs[0][i].item()))
        
        return results
    
    def print_results(self, results, top_k=5):
        """Print classification results in a formatted way
        
        Args:
            results: List of (class_idx, probability) tuples
            top_k: Number of results to print
        """
        print(f"\nTop {top_k} predictions:")
        for i, (idx, prob) in enumerate(results[:top_k]):
            print(f"  {i+1}. Class #{idx}: {prob:.4f} ({prob*100:.2f}%)")
    
    def compare_architectures(self, image_path, top_k=3):
        """Compare all architectures on an image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to compare
        
        Returns:
            Dictionary of results for each model
        """
        results = {}
        
        print("\n" + "="*80)
        print("CNN ARCHITECTURE COMPARISON")
        print("="*80)
        print(f"Image: {image_path}\n")
        print(f"{'Model':<15} {'Input Size':<12} {'Top 1':<15} {'Time (ms)':<15}")
        print("-"*65)
        
        import time
        
        for model_name in sorted(MODEL_REGISTRY.keys()):
            try:
                model = self.get_model(model_name)
                input_size = MODEL_REGISTRY[model_name]['input_size']
                
                # Measure inference time
                torch.cuda.synchronize() if 'cuda' in str(self.device) else None
                start_time = time.time()
                
                output = self.classify(model, image_path, top_k=top_k)
                
                torch.cuda.synchronize() if 'cuda' in str(self.device) else None
                end_time = time.time()
                
                elapsed_ms = (end_time - start_time) * 1000
                top_prob = output[0][1] * 100
                
                print(f"{model_name:<15} {input_size:<12} {top_prob:<15.2f}% {elapsed_ms:<15.2f}")
                
                results[model_name] = {
                    'top_k': output[:top_k],
                    'inference_time_ms': elapsed_ms
                }
                
                # Clean up model to free memory
                del model
                torch.cuda.empty_cache() if 'cuda' in str(self.device) else None
                
            except Exception as e:
                print(f"{model_name:<15} {'Error':<12} {'Error':<15} {'Error':<15}")
                print(f"  -> {str(e)}")
                results[model_name] = {'error': str(e)}
        
        print("="*80)
        
        return results
    
    def benchmark(self, image_path, iterations=10):
        """Benchmark all architectures on an image
        
        Args:
            image_path: Path to image file
            iterations: Number of inference iterations to average
        """
        print("\n" + "="*80)
        print("ARCHITECTURE BENCHMARKING")
        print("="*80)
        print(f"Image: {image_path}\n")
        print(f"{'Model':<15} {'Params (M)':<15} {'Avg Time (ms)':<15} {'Throughput (fps)':<15}")
        print("-"*65)
        
        import time
        
        for model_name in sorted(MODEL_REGISTRY.keys()):
            try:
                model = self.get_model(model_name)
                
                # Warmup
                for _ in range(5):
                    _ = self.classify(model, image_path, top_k=1)
                
                # Benchmark
                times = []
                for _ in range(iterations):
                    torch.cuda.synchronize() if 'cuda' in str(self.device) else None
                    start_time = time.time()
                    _ = self.classify(model, image_path, top_k=1)
                    torch.cuda.synchronize() if 'cuda' in str(self.device) else None
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)
                
                avg_time = sum(times) / len(times)
                fps = 1000 / avg_time
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                params_m = total_params / 1_000_000
                
                print(f"{model_name:<15} {params_m:<15.2f} {avg_time:<15.2f} {fps:<15.2f}")
                
                del model
                torch.cuda.empty_cache() if 'cuda' in str(self.device) else None
                
            except Exception as e:
                print(f"{model_name:<15} {'Error':<15} {'Error':<15} {'Error':<15}")
                print(f"  -> {str(e)}")
        
        print("="*80)


def main():
    """Main entry point for CNN Tester"""
    
    parser = argparse.ArgumentParser(
        description='CNN Architecture Comparison and Testing Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Test a single model
  python model_test.py --model efficientnet --image cat.jpg

  # Compare all models on an image
  python model_test.py --image dog.jpg --compare

  # Benchmark all models
  python model_test.py --image test.jpg --benchmark

  # Custom settings
  python model_test.py --model densenet --image test.jpg --top_k 10 --device cuda
        '''
    )
    
    parser.add_argument('--model', '-m', type=str, default='efficientnet',
                       help='Model to test (default: efficientnet)')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--top_k', '-k', type=int, default=5,
                       help='Number of top predictions to show (default: 5)')
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use (cpu or cuda, default: auto)')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Compare all architectures')
    parser.add_argument('--benchmark', '-b', action='store_true',
                       help='Benchmark all architectures')
    parser.add_argument('--iterations', '-n', type=int, default=10,
                       help='Number of iterations for benchmarking (default: 10)')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = CNNTester(device=args.device)
    
    if args.compare:
        # Compare all models
        results = tester.compare_architectures(args.image, top_k=args.top_k)
        
        # Print top results
        print("\n" + "="*80)
        print("BEST PERFORMING MODELS")
        print("="*80)
        
        best_results = sorted(
            [(name, info['top_k'][0][1] if 'top_k' in info else 0) 
             for name, info in results.items() 
             if 'top_k' in info],
            key=lambda x: x[1],
            reverse=True
        )[:args.top_k]
        
        for rank, (model_name, top_prob) in enumerate(best_results, 1):
            print(f"{rank}. {model_name}: {top_prob:.2f}% accuracy")
        
        print("="*80)
    
    elif args.benchmark:
        # Benchmark all models
        tester.benchmark(args.image, iterations=args.iterations)
    
    else:
        # Test single model
        try:
            model = tester.get_model(args.model)
            results = tester.classify(model, args.image, top_k=args.top_k)
            tester.print_results(results, top_k=args.top_k)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
