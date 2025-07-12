#!/usr/bin/env python3
"""
Simple script to run contrastive clustering using the main clustering pipeline
"""

import subprocess
import sys
import os

def run_contrastive_clustering():
    """Run contrastive clustering with default parameters"""
    
    # Configuration
    dataset_path = "YaTC_datasets/USTC-TFC2016_MFR"
    model_path = "contrastive_output_dir/checkpoint-199.pth"  # Update this path
    output_dir = "./contrastive_clustering_results"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please update the model_path variable with your actual MAEContrast model path")
        return
    
    # Command to run
    cmd = [
        "python", "rp_clustering.py",
        "--model_path", model_path,
        "--model_type", "contrastive",
        "--data_path", dataset_path,
        "--output_dir", output_dir,
        "--n_clusters", "10",
        "--algorithm", "kmeans",
        "--save_representations",
        "--max_samples_per_class", "500"
    ]
    
    print("Running contrastive clustering with command:")
    print(" ".join(cmd))
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\nContrastive clustering completed successfully!")
        print(f"Results saved to: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error running contrastive clustering: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_contrastive_clustering() 