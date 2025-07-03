#!/usr/bin/env python3
"""
Simple command-line script to run clustering on YaTC dataset
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Run clustering on YaTC dataset using MAE representations')
    parser.add_argument('--model_path', type=str, default="output_dir/checkpoint-step150000.pth", help='Path to the trained MAE model checkpoint')
    parser.add_argument('--dataset', type=str, default='CICIoT2022_MFR',
                       choices=['CICIoT2022_MFR', 'ISCXVPN2016_MFR', 'USTC-TFC2016_MFR', 'ISCXTor2016_MFR'],
                       help='YaTC dataset to use')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--max_samples_per_class', type=int, default=1000,
                       help='Maximum number of samples per class (use -1 for all)')
    parser.add_argument('--algorithm', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan', 'hierarchical'],
                       help='Clustering algorithm')
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of clusters (for kmeans and hierarchical)')
    parser.add_argument('--eps', type=float, default=0.5,
                       help='Epsilon parameter for DBSCAN')
    parser.add_argument('--min_samples', type=int, default=5,
                       help='Min samples parameter for DBSCAN')
    parser.add_argument('--output_dir', type=str, default='./clustering_results',
                       help='Output directory for results')
    parser.add_argument('--save_representations', action='store_true',
                       help='Save extracted representations')
    parser.add_argument('--compare_algorithms', action='store_true',
                       help='Compare multiple clustering algorithms')
    
    args = parser.parse_args()
    
    # Set max_samples_per_class to None if -1
    if args.max_samples_per_class == -1:
        args.max_samples_per_class = None
    
    # Construct dataset path
    dataset_path = f"YaTC_datasets/{args.dataset}"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Available datasets:")
        for dataset in ['CICIoT2022_MFR', 'ISCXVPN2016_MFR', 'USTC-TFC2016_MFR', 'ISCXTor2016_MFR']:
            path = f"YaTC_datasets/{dataset}"
            if os.path.exists(path):
                print(f"  ✓ {dataset}")
            else:
                print(f"  ✗ {dataset}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("YaTC DATASET CLUSTERING")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model_path}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    
    if args.compare_algorithms:
        # Run comparison of multiple algorithms
        from yaTC_clustering_example import run_yaTC_clustering_example
        
        # Temporarily modify the function to use our arguments
        import yaTC_clustering_example
        yaTC_clustering_example.dataset_path = dataset_path
        yaTC_clustering_example.model_path = args.model_path
        yaTC_clustering_example.output_dir = args.output_dir
        yaTC_clustering_example.max_samples_per_class = args.max_samples_per_class
        
        run_yaTC_clustering_example()
        
    else:
        # Run single algorithm clustering
        from rp_clustering import main as run_clustering
        
        # Construct command line arguments for rp_clustering
        sys.argv = [
            'rp_clustering.py',
            '--model_path', args.model_path,
            '--data_path', dataset_path,
            '--output_dir', args.output_dir,
            '--algorithm', args.algorithm,
            '--n_clusters', str(args.n_clusters),
            '--eps', str(args.eps),
            '--min_samples', str(args.min_samples),
            '--max_samples_per_class', str(args.max_samples_per_class) if args.max_samples_per_class else 'None'
        ]
        
        if args.save_representations:
            sys.argv.append('--save_representations')
        
        # Run clustering
        run_clustering()


if __name__ == "__main__":
    main() 