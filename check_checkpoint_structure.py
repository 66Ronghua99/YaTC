#!/usr/bin/env python3
"""
Script to check checkpoint structure and compare with model initialization
"""

import torch
import argparse
from models_YaTC import TraFormer_YaTC

def check_checkpoint_structure(checkpoint_path):
    """Check the structure of a checkpoint file"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Try loading with weights_only=True first
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        print("✓ Successfully loaded checkpoint with weights_only=True")
    except Exception as e:
        print(f"✗ Loading with weights_only=True failed: {e}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            print("✓ Successfully loaded checkpoint with weights_only=False")
        except Exception as e2:
            print(f"✗ Loading with weights_only=False also failed: {e2}")
            return None
    
    print(f"\nCheckpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check if it has model state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"✓ Found 'model' key in checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"✓ Found 'state_dict' key in checkpoint")
        else:
            state_dict = checkpoint
            print("✓ Using checkpoint directly as state dict")
    else:
        state_dict = checkpoint
        print("✓ Using checkpoint directly as state dict")
    
    print(f"\nState dict keys: {list(state_dict.keys())}")
    
    # Check specific layers that might have size mismatches
    key_layers = ['head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias']
    for key in key_layers:
        if key in state_dict:
            shape = state_dict[key].shape
            print(f"  {key}: {shape}")
        else:
            print(f"  {key}: Not found")
    
    return state_dict

def check_model_structure(num_classes=1000):
    """Check the structure of a model with given parameters"""
    print(f"\nInitializing model with num_classes={num_classes}")
    
    try:
        model = TraFormer_YaTC(num_classes=num_classes)
        print("✓ Successfully initialized model")
        
        # Check specific layers
        print(f"  head.weight: {model.head.weight.shape}")
        print(f"  head.bias: {model.head.bias.shape}")
        print(f"  fc_norm.weight: {model.fc_norm.weight.shape}")
        print(f"  fc_norm.bias: {model.fc_norm.bias.shape}")
        
        return model
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return None

def test_loading(checkpoint_path, num_classes=20):
    """Test loading checkpoint into model with specific num_classes"""
    print(f"\nTesting loading with num_classes={num_classes}")
    
    # Load checkpoint
    state_dict = check_checkpoint_structure(checkpoint_path)
    if state_dict is None:
        return False
    
    # Initialize model
    model = check_model_structure(num_classes)
    if model is None:
        return False
    
    # Try loading state dict
    try:
        model.load_state_dict(state_dict)
        print("✓ Successfully loaded checkpoint into model!")
        return True
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Check checkpoint structure')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the checkpoint file')
    parser.add_argument('--test_loading', action='store_true',
                       help='Test loading the checkpoint into a model')
    parser.add_argument('--num_classes', type=int, default=20,
                       help='Number of classes for model initialization')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CHECKPOINT STRUCTURE ANALYSIS")
    print("="*60)
    
    # Check checkpoint structure
    state_dict = check_checkpoint_structure(args.checkpoint_path)
    
    if state_dict is None:
        print("Failed to load checkpoint")
        return
    
    # Check model structure with default parameters
    print("\n" + "="*60)
    print("MODEL STRUCTURE ANALYSIS")
    print("="*60)
    
    model_default = check_model_structure(num_classes=1000)
    model_custom = check_model_structure(num_classes=args.num_classes)
    
    if args.test_loading:
        print("\n" + "="*60)
        print("LOADING TEST")
        print("="*60)
        
        success = test_loading(args.checkpoint_path, args.num_classes)
        if success:
            print("\n✓ Checkpoint can be loaded successfully!")
        else:
            print("\n✗ Checkpoint loading failed!")

if __name__ == "__main__":
    main() 