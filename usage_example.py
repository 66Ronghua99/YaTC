#!/usr/bin/env python3
"""
Usage examples for clustering with MAE and TrafficTransformer representations
"""

import os
import sys

def example_mae_clustering():
    """Example of clustering with MAE encoder representations"""
    print("="*60)
    print("MAE CLUSTERING EXAMPLE")
    print("="*60)
    
    # Example command for MAE clustering
    cmd = """python rp_clustering.py \\
    --model_path "pre-trained-model/YaTC_pretrained_model.pth" \\
    --model_type mae \\
    --data_path "YaTC_datasets/USTC-TFC2016_MFR" \\
    --output_dir "./mae_clustering_results" \\
    --n_clusters 10 \\
    --algorithm kmeans \\
    --mask_ratio 0.0 \\
    --save_representations \\
    --max_samples_per_class 500"""
    
    print("Command:")
    print(cmd)
    print("\nThis will:")
    print("1. Load the pre-trained MAE model")
    print("2. Extract encoder representations from the YaTC dataset")
    print("3. Perform K-means clustering with 10 clusters")
    print("4. Save results and visualizations")
    print("5. Save the extracted representations for further analysis")


def example_traffic_transformer_clustering():
    """Example of clustering with TrafficTransformer flow-level representations"""
    print("\n" + "="*60)
    print("TRAFFIC TRANSFORMER CLUSTERING EXAMPLE")
    print("="*60)
    
    # Example command for TrafficTransformer clustering
    cmd = """python rp_clustering.py \\
    --model_path "fine_tuned_models/traffic_transformer_finetuned.pth" \\
    --model_type traffic_transformer \\
    --data_path "YaTC_datasets/USTC-TFC2016_MFR" \\
    --output_dir "./traffic_transformer_clustering_results" \\
    --n_clusters 10 \\
    --algorithm kmeans \\
    --save_representations \\
    --max_samples_per_class 500"""
    
    print("Command:")
    print(cmd)
    print("\nThis will:")
    print("1. Load the fine-tuned TrafficTransformer model")
    print("2. Extract flow-level representations (before final layernorm)")
    print("3. Perform K-means clustering with 10 clusters")
    print("4. Save results and visualizations")
    print("5. Save the extracted representations for further analysis")


def example_dbscan_clustering():
    """Example of DBSCAN clustering with TrafficTransformer representations"""
    print("\n" + "="*60)
    print("DBSCAN CLUSTERING EXAMPLE")
    print("="*60)
    
    # Example command for DBSCAN clustering
    cmd = """python rp_clustering.py \\
    --model_path "fine_tuned_models/traffic_transformer_finetuned.pth" \\
    --model_type traffic_transformer \\
    --data_path "YaTC_datasets/USTC-TFC2016_MFR" \\
    --output_dir "./dbscan_clustering_results" \\
    --algorithm dbscan \\
    --eps 0.5 \\
    --min_samples 5 \\
    --save_representations \\
    --max_samples_per_class 500"""
    
    print("Command:")
    print(cmd)
    print("\nThis will:")
    print("1. Load the fine-tuned TrafficTransformer model")
    print("2. Extract flow-level representations")
    print("3. Perform DBSCAN clustering with eps=0.5 and min_samples=5")
    print("4. Save results and visualizations")
    print("5. Note: DBSCAN automatically determines the number of clusters")


def example_hierarchical_clustering():
    """Example of hierarchical clustering with MAE representations"""
    print("\n" + "="*60)
    print("HIERARCHICAL CLUSTERING EXAMPLE")
    print("="*60)
    
    # Example command for hierarchical clustering
    cmd = """python rp_clustering.py \\
    --model_path "pre-trained-model/YaTC_pretrained_model.pth" \\
    --model_type mae \\
    --data_path "YaTC_datasets/USTC-TFC2016_MFR" \\
    --output_dir "./hierarchical_clustering_results" \\
    --n_clusters 5 \\
    --algorithm hierarchical \\
    --mask_ratio 0.0 \\
    --save_representations \\
    --max_samples_per_class 500"""
    
    print("Command:")
    print(cmd)
    print("\nThis will:")
    print("1. Load the pre-trained MAE model")
    print("2. Extract encoder representations")
    print("3. Perform hierarchical clustering with 5 clusters")
    print("4. Save results and visualizations")


def example_with_custom_data():
    """Example with custom data file"""
    print("\n" + "="*60)
    print("CUSTOM DATA CLUSTERING EXAMPLE")
    print("="*60)
    
    # Example command for custom data
    cmd = """python rp_clustering.py \\
    --model_path "fine_tuned_models/traffic_transformer_finetuned.pth" \\
    --model_type traffic_transformer \\
    --data_path "path/to/your/custom_data.npy" \\
    --output_dir "./custom_data_clustering_results" \\
    --n_clusters 8 \\
    --algorithm kmeans \\
    --save_representations"""
    
    print("Command:")
    print(cmd)
    print("\nThis will:")
    print("1. Load the fine-tuned TrafficTransformer model")
    print("2. Extract flow-level representations from your custom data")
    print("3. Perform K-means clustering with 8 clusters")
    print("4. Save results and visualizations")
    print("\nNote: Your custom data should be in shape (N, 1, H, W) where:")
    print("  - N: number of samples")
    print("  - 1: single channel (grayscale)")
    print("  - H, W: height and width (e.g., 40x200 for YaTC format)")


def main():
    """Show all usage examples"""
    print("CLUSTERING USAGE EXAMPLES")
    print("="*60)
    print("This script demonstrates how to use the updated rp_clustering.py")
    print("for clustering with both MAE and TrafficTransformer representations.")
    print("\nKey differences:")
    print("- MAE: Uses encoder representations with optional masking")
    print("- TrafficTransformer: Uses flow-level representations before final layernorm")
    print("- Both: Support K-means, DBSCAN, and hierarchical clustering")
    
    example_mae_clustering()
    example_traffic_transformer_clustering()
    example_dbscan_clustering()
    example_hierarchical_clustering()
    example_with_custom_data()
    
    print("\n" + "="*60)
    print("ADDITIONAL NOTES")
    print("="*60)
    print("1. Make sure your model paths are correct")
    print("2. The TrafficTransformer extractor gets representations before the final layernorm")
    print("3. You can compare results between MAE and TrafficTransformer representations")
    print("4. Use --save_representations to save extracted features for further analysis")
    print("5. Adjust --max_samples_per_class for faster processing on large datasets")
    print("6. For DBSCAN, experiment with --eps and --min_samples parameters")


if __name__ == "__main__":
    main() 