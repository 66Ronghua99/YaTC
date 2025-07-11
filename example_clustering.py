#!/usr/bin/env python3
"""
Example script demonstrating how to use the MAE clustering pipeline
"""

import numpy as np
import torch
from rp_clustering import MAEEncoderExtractor, ClusteringPipeline, VisualizationHelper

def create_sample_data(n_samples=1000, img_size=40):
    """Create sample data for demonstration"""
    # Generate random traffic-like data
    data = torch.randn(n_samples, 1, img_size, img_size)
    
    # Add some structure to make clustering meaningful
    for i in range(n_samples):
        # Add different patterns based on sample index
        if i < n_samples // 3:
            # Pattern 1: horizontal lines
            data[i, 0, ::4, :] += 2.0
        elif i < 2 * n_samples // 3:
            # Pattern 2: vertical lines
            data[i, 0, :, ::4] += 2.0
        else:
            # Pattern 3: diagonal pattern
            for j in range(img_size):
                data[i, 0, j, j] += 2.0
    
    return data

def main():
    # Example usage of the clustering pipeline
    
    # 1. Create sample data (replace with your actual data loading)
    print("Creating sample data...")
    data = create_sample_data(n_samples=500, img_size=40)
    print(f"Sample data shape: {data.shape}")
    
    # 2. Initialize encoder extractor (replace with your model path)
    model_path = "./output_dir/checkpoint-step150000.pth"  # Replace with actual path
    
    try:
        extractor = MAEEncoderExtractor(model_path)
        print("MAE model loaded successfully!")
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        print("Please update the model_path variable with your actual model path")
        return
    
    # 3. Extract representations
    print("Extracting encoder representations...")
    representations = extractor.extract_representations(data, mask_ratio=0.0)
    print(f"Representations shape: {representations.shape}")
    
    # 4. Try different clustering algorithms
    
    algorithms = [
        ('kmeans', {'n_clusters': 3}),
        ('dbscan', {'eps': 0.5, 'min_samples': 5}),
        ('hierarchical', {'n_clusters': 3})
    ]
    
    for algorithm_name, params in algorithms:
        print(f"\n{'='*50}")
        print(f"Testing {algorithm_name.upper()} clustering")
        print(f"{'='*50}")
        
        # Initialize clustering pipeline
        pipeline = ClusteringPipeline(
            algorithm=algorithm_name,
            random_state=42,
            **params
        )
        
        # Perform clustering
        labels = pipeline.fit_predict(representations)
        
        # Evaluate clustering
        metrics = pipeline.evaluate_clustering(representations, labels)
        
        # Print results
        print(f"Number of clusters: {len(np.unique(labels[labels != -1]))}")
        print(f"Number of noise points: {np.sum(labels == -1)}")
        print("Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Create visualizations
        viz_helper = VisualizationHelper()
        
        # Plot clusters
        viz_helper.plot_clusters_2d(
            representations, 
            labels, 
            title=f"{algorithm_name.upper()} Clustering Results"
        )
        
        # Plot cluster statistics
        viz_helper.plot_cluster_statistics(labels)

def compare_clustering_algorithms():
    """Compare different clustering algorithms on the same data"""
    
    # Load your data and model here
    # data = load_your_data()
    # extractor = MAEEncoderExtractor("your_model_path.pth")
    # representations = extractor.extract_representations(data)
    
    # For demonstration, we'll use random data
    representations = np.random.randn(500, 192)
    
    # Define different clustering configurations
    configs = [
        {
            'name': 'K-means (3 clusters)',
            'algorithm': 'kmeans',
            'params': {'n_clusters': 3}
        },
        {
            'name': 'K-means (5 clusters)',
            'algorithm': 'kmeans',
            'params': {'n_clusters': 5}
        },
        {
            'name': 'DBSCAN (eps=0.5)',
            'algorithm': 'dbscan',
            'params': {'eps': 0.5, 'min_samples': 5}
        },
        {
            'name': 'DBSCAN (eps=1.0)',
            'algorithm': 'dbscan',
            'params': {'eps': 1.0, 'min_samples': 5}
        },
        {
            'name': 'Hierarchical (3 clusters)',
            'algorithm': 'hierarchical',
            'params': {'n_clusters': 3}
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        
        pipeline = ClusteringPipeline(
            algorithm=config['algorithm'],
            random_state=42,
            **config['params']
        )
        
        labels = pipeline.fit_predict(representations)
        metrics = pipeline.evaluate_clustering(representations, labels)
        
        results.append({
            'name': config['name'],
            'n_clusters': len(np.unique(labels[labels != -1])),
            'n_noise': np.sum(labels == -1),
            'metrics': metrics
        })
        
        print(f"  Clusters: {results[-1]['n_clusters']}")
        print(f"  Noise points: {results[-1]['n_noise']}")
        print(f"  Silhouette: {metrics['silhouette']:.4f}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("CLUSTERING ALGORITHM COMPARISON")
    print("="*80)
    print(f"{'Algorithm':<25} {'Clusters':<10} {'Noise':<8} {'Silhouette':<12} {'Calinski':<12} {'Davies':<12}")
    print("-" * 80)
    
    for result in results:
        metrics = result['metrics']
        print(f"{result['name']:<25} {result['n_clusters']:<10} {result['n_noise']:<8} "
              f"{metrics['silhouette']:<12.4f} {metrics['calinski_harabasz']:<12.4f} "
              f"{metrics['davies_bouldin']:<12.4f}")

if __name__ == "__main__":
    print("MAE Clustering Example")
    print("=" * 50)
    
    # Uncomment the function you want to run:
    
    # main()  # Basic usage example
    compare_clustering_algorithms()  # Algorithm comparison
    
    print("\nExample completed!") 