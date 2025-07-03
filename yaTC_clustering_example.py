#!/usr/bin/env python3
"""
Example script for clustering YaTC dataset using MAE encoder representations
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from rp_clustering import MAEEncoderExtractor, ClusteringPipeline, VisualizationHelper
from data_loader import load_all_data_as_tensor, get_dataset_info, visualize_dataset_samples


def run_yaTC_clustering_example():
    """Run clustering on YaTC dataset"""
    
    # Configuration
    dataset_path = "YaTC_datasets/CICIoT2022_MFR"
    model_path = "output_dir/checkpoint-step150000.pth"  # Replace with your model path
    output_dir = "./yaTC_clustering_results"
    max_samples_per_class = 500  # Limit samples per class for faster processing
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("YaTC DATASET CLUSTERING EXAMPLE")
    print("="*60)
    
    # 1. Get dataset information
    print("\n1. Dataset Information:")
    info = get_dataset_info(dataset_path)
    for split, split_info in info.items():
        if 'error' not in split_info:
            print(f"  {split.upper()}: {split_info['num_samples']} samples, {split_info['num_classes']} classes")
            print(f"    Classes: {split_info['classes']}")
            print(f"    Distribution: {split_info['class_distribution']}")
    
    # 2. Visualize sample images
    print("\n2. Visualizing sample images...")
    sample_viz_path = os.path.join(output_dir, "sample_images.png")
    visualize_dataset_samples(dataset_path, 'train', samples_per_class=3, save_path=sample_viz_path)
    
    # 3. Load data
    print(f"\n3. Loading data (max {max_samples_per_class} samples per class)...")
    data, labels = load_all_data_as_tensor(dataset_path, 'train', max_samples_per_class=max_samples_per_class)
    print(f"   Data shape: {data.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Unique labels: {torch.unique(labels).tolist()}")
    
    # 4. Load MAE model and extract representations
    print("\n4. Loading MAE model and extracting representations...")
    try:
        extractor = MAEEncoderExtractor(model_path)
        print("   MAE model loaded successfully!")
    except FileNotFoundError:
        print(f"   Model not found at {model_path}")
        print("   Please update the model_path variable with your actual model path")
        return
    
    # Extract representations
    representations = extractor.extract_representations(data, mask_ratio=0.0)
    print(f"   Representations shape: {representations.shape}")
    
    # Save representations
    rep_path = os.path.join(output_dir, "mae_representations.npy")
    np.save(rep_path, representations)
    print(f"   Saved representations to: {rep_path}")
    
    # 5. Perform clustering with different algorithms
    print("\n5. Performing clustering analysis...")
    
    # Define clustering configurations
    clustering_configs = [
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
            'name': 'K-means (10 clusters)',
            'algorithm': 'kmeans',
            'params': {'n_clusters': 10}
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
            'name': 'Hierarchical (5 clusters)',
            'algorithm': 'hierarchical',
            'params': {'n_clusters': 5}
        }
    ]
    
    results = []
    
    for config in clustering_configs:
        print(f"\n   Testing: {config['name']}")
        
        # Initialize clustering pipeline
        pipeline = ClusteringPipeline(
            algorithm=config['algorithm'],
            random_state=42,
            **config['params']
        )
        
        # Perform clustering
        cluster_labels = pipeline.fit_predict(representations)
        
        # Evaluate clustering
        metrics = pipeline.evaluate_clustering(representations, cluster_labels)
        
        # Store results
        result = {
            'name': config['name'],
            'algorithm': config['algorithm'],
            'n_clusters': len(np.unique(cluster_labels[cluster_labels != -1])),
            'n_noise': np.sum(cluster_labels == -1),
            'labels': cluster_labels,
            'metrics': metrics
        }
        results.append(result)
        
        print(f"     Clusters: {result['n_clusters']}")
        print(f"     Noise points: {result['n_noise']}")
        print(f"     Silhouette: {metrics['silhouette']:.4f}")
        print(f"     Calinski-Harabasz: {metrics['calinski_harabasz']:.4f}")
        print(f"     Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
        
        # Create visualizations for this clustering
        viz_helper = VisualizationHelper()
        
        # Plot clusters
        cluster_plot_path = os.path.join(output_dir, f"clusters_{config['algorithm']}_{config['params'].get('n_clusters', 'dbscan')}.png")
        viz_helper.plot_clusters_2d(
            representations, 
            cluster_labels, 
            title=f"{config['name']} - YaTC Dataset",
            save_path=cluster_plot_path
        )
        
        # Plot cluster statistics
        stats_plot_path = os.path.join(output_dir, f"stats_{config['algorithm']}_{config['params'].get('n_clusters', 'dbscan')}.png")
        viz_helper.plot_cluster_statistics(cluster_labels, save_path=stats_plot_path)
    
    # 6. Compare clustering results
    print("\n6. Clustering Results Comparison:")
    print("-" * 80)
    print(f"{'Algorithm':<25} {'Clusters':<10} {'Noise':<8} {'Silhouette':<12} {'Calinski':<12} {'Davies':<12}")
    print("-" * 80)
    
    for result in results:
        metrics = result['metrics']
        print(f"{result['name']:<25} {result['n_clusters']:<10} {result['n_noise']:<8} "
              f"{metrics['silhouette']:<12.4f} {metrics['calinski_harabasz']:<12.4f} "
              f"{metrics['davies_bouldin']:<12.4f}")
    
    # 7. Analyze clustering vs true labels
    print("\n7. Analyzing clustering vs true labels...")
    analyze_clustering_vs_true_labels(results, labels.numpy(), output_dir)
    
    # 8. Save all results
    print("\n8. Saving results...")
    save_clustering_results(results, representations, labels.numpy(), output_dir)
    
    print(f"\nClustering analysis completed! Results saved to: {output_dir}")


def analyze_clustering_vs_true_labels(results, true_labels, output_dir):
    """Analyze how well clustering aligns with true labels"""
    
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    print("   Clustering vs True Labels Analysis:")
    print("   " + "-" * 50)
    
    for result in results:
        cluster_labels = result['labels']
        
        # Remove noise points for evaluation
        valid_mask = cluster_labels != -1
        if not np.any(valid_mask):
            continue
            
        cluster_labels_valid = cluster_labels[valid_mask]
        true_labels_valid = true_labels[valid_mask]
        
        # Calculate metrics
        ari = adjusted_rand_score(true_labels_valid, cluster_labels_valid)
        nmi = normalized_mutual_info_score(true_labels_valid, cluster_labels_valid)
        
        result['true_label_metrics'] = {
            'adjusted_rand_score': ari,
            'normalized_mutual_info': nmi
        }
        
        print(f"   {result['name']}:")
        print(f"     Adjusted Rand Score: {ari:.4f}")
        print(f"     Normalized Mutual Info: {nmi:.4f}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    algorithms = [r['name'] for r in results if 'true_label_metrics' in r]
    ari_scores = [r['true_label_metrics']['adjusted_rand_score'] for r in results if 'true_label_metrics' in r]
    nmi_scores = [r['true_label_metrics']['normalized_mutual_info'] for r in results if 'true_label_metrics' in r]
    
    # Plot ARI scores
    bars1 = ax1.bar(range(len(algorithms)), ari_scores, color='skyblue')
    ax1.set_title('Adjusted Rand Score vs True Labels')
    ax1.set_ylabel('Adjusted Rand Score')
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars1, ari_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # Plot NMI scores
    bars2 = ax2.bar(range(len(algorithms)), nmi_scores, color='lightcoral')
    ax2.set_title('Normalized Mutual Information vs True Labels')
    ax2.set_ylabel('Normalized Mutual Information')
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars2, nmi_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "clustering_vs_true_labels.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_clustering_results(results, representations, true_labels, output_dir):
    """Save all clustering results to files"""
    
    import json
    
    # Save summary results
    summary = []
    for result in results:
        summary.append({
            'name': result['name'],
            'algorithm': result['algorithm'],
            'n_clusters': result['n_clusters'],
            'n_noise': result['n_noise'],
            'metrics': result['metrics'],
            'true_label_metrics': result.get('true_label_metrics', {})
        })
    
    summary_path = os.path.join(output_dir, "clustering_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    detailed_results = {
        'representations_shape': representations.shape,
        'true_labels_shape': true_labels.shape,
        'n_samples': len(true_labels),
        'n_true_classes': len(np.unique(true_labels)),
        'clustering_results': results
    }
    
    detailed_path = os.path.join(output_dir, "detailed_results.json")
    with open(detailed_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for result in results:
            json_result = result.copy()
            json_result['labels'] = result['labels'].tolist()
            json_results.append(json_result)
        
        detailed_results['clustering_results'] = json_results
        json.dump(detailed_results, f, indent=2)
    
    print(f"   Summary saved to: {summary_path}")
    print(f"   Detailed results saved to: {detailed_path}")


if __name__ == "__main__":
    run_yaTC_clustering_example() 