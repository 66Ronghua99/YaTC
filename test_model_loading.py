#!/usr/bin/env python3
"""
Test script to verify model loading works correctly
"""

import torch
import os
from models_YaTC import MAE_YaTC

def test_model_loading():
    """Test loading the MAE model from checkpoint"""
    
    model_path = "output_dir/checkpoint-step150000.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    print(f"Testing model loading from: {model_path}")
    print("="*50)
    
    try:
        # Initialize model
        print("1. Initializing model...")
        model = MAE_YaTC(norm_pix_loss=False)
        print("   ✓ Model initialized successfully")
        
        # Load checkpoint
        print("\n2. Loading checkpoint...")
        try:
            # Try weights_only=True first
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            print("   ✓ Loaded with weights_only=True")
        except Exception as e:
            print(f"   ⚠ weights_only=True failed: {e}")
            # Try weights_only=False
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            print("   ✓ Loaded with weights_only=False")
        
        # Load state dict
        print("\n3. Loading state dict...")
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                print("   ✓ Loaded from checkpoint['model']")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("   ✓ Loaded from checkpoint['state_dict']")
            else:
                model.load_state_dict(checkpoint)
                print("   ✓ Loaded from checkpoint directly")
        else:
            model.load_state_dict(checkpoint)
            print("   ✓ Loaded from checkpoint directly")
        
        # Test forward pass
        print("\n4. Testing forward pass...")
        model.eval()
        with torch.no_grad():
            # Create dummy input
            dummy_input = torch.randn(2, 1, 40, 40)  # batch_size=2, channels=1, height=40, width=40
            loss, pred, mask = model(dummy_input, mask_ratio=0.75)
            print(f"   ✓ Forward pass successful")
            print(f"   ✓ Loss: {loss.item():.4f}")
            print(f"   ✓ Prediction shape: {pred.shape}")
            print(f"   ✓ Mask shape: {mask.shape}")
        
        print("\n" + "="*50)
        print("✓ Model loading test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n✗ Model loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encoder_extraction():
    """Test encoder representation extraction"""
    
    from rp_clustering import MAEEncoderExtractor
    
    model_path = "output_dir/checkpoint-step150000.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    print(f"\nTesting encoder extraction from: {model_path}")
    print("="*50)
    
    try:
        # Initialize extractor
        print("1. Initializing encoder extractor...")
        extractor = MAEEncoderExtractor(model_path)
        print("   ✓ Encoder extractor initialized")
        
        # Test representation extraction
        print("\n2. Testing representation extraction...")
        dummy_data = torch.randn(4, 1, 40, 40)  # 4 samples
        representations = extractor.extract_representations(dummy_data, mask_ratio=0.0)
        print(f"   ✓ Extracted representations shape: {representations.shape}")
        print(f"   ✓ Expected shape: (4, 192)")
        
        print("\n" + "="*50)
        print("✓ Encoder extraction test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n✗ Encoder extraction test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("MAE Model Loading Test")
    print("="*50)
    
    # Test basic model loading
    success1 = test_model_loading()
    
    # Test encoder extraction
    success2 = test_encoder_extraction()
    
    if success1 and success2:
        print("\n🎉 All tests PASSED! The model loading is working correctly.")
    else:
        print("\n❌ Some tests FAILED. Please check the error messages above.") 