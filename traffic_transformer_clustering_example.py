#!/usr/bin/env python3
"""
Example script for clustering YaTC dataset using TrafficTransformer flow-level representations
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from rp_clustering import TrafficTransformerExtractor, ClusteringPipeline, VisualizationHelper
from data_loader import load_all_data_as_tensor, get_dataset_info, visualize_dataset_samples


def create_tsne_visualization(representations, true_labels, output_dir, perplexity=30, max_iter=1000):
    """Create T-SNE visualization of the representations"""
    
    print("\n4.5. Creating T-SNE visualization of TrafficTransformer representations...")
    
    # Apply T-SNE dimensionality reduction
    print(f"   Applying T-SNE (perplexity={perplexity}, max_iter={max_iter})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, random_state=42, verbose=1)
    representations_2d = tsne.fit_transform(representations)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Get unique labels and create color map
    unique_labels = np.unique(true_labels)
    # Generate colors for different labels
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each class with different color
    for i, label in enumerate(unique_labels):
        mask = true_labels == label
        plt.scatter(representations_2d[mask, 0], representations_2d[mask, 1], 
                   c=[colors[i]], s=50, alpha=0.7, label=f'Class {label}')
    
    plt.title(f'T-SNE Visualization of TrafficTransformer Flow-Level Representations\n'
              f'(perplexity={perplexity}, iterations={max_iter})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    tsne_path = os.path.join(output_dir, "tsne_traffic_transformer_representations.png")
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   T-SNE visualization saved to: {tsne_path}")
    
    return representations_2d


def run_traffic_transformer_clustering_example():
    """Run clustering on YaTC dataset using TrafficTransformer representations"""
    
    # Configuration
    dataset_path = "YaTC_datasets/USTC-TFC2016_MFR"
    # Update these paths to your actual fine-tuned TrafficTransformer model
    model_path = "3t1t_output_dir/finetune_model/traffic_transformer_final.pth"  # Replace with your model path
    output_dir = "./finetune_yaTC_clustering_results"
    max_samples_per_class = 500  # Limit samples per class for faster processing
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("TRAFFIC TRANSFORMER CLUSTERING EXAMPLE")
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
    
    # 4. Load TrafficTransformer model and extract representations
    print("\n4. Loading TrafficTransformer model and extracting flow-level representations...")
    try:
        extractor = TrafficTransformerExtractor(model_path)
        print("   TrafficTransformer model loaded successfully!")
    except FileNotFoundError:
        print(f"   Model not found at {model_path}")
        print("   Please update the model_path variable with your actual fine-tuned TrafficTransformer model path")
        return
    
    # Extract representations
    representations = extractor.extract_representations(data)
    print(f"   Representations shape: {representations.shape}")
    
    # Save representations
    rep_path = os.path.join(output_dir, "traffic_transformer_representations.npy")
    np.save(rep_path, representations)
    print(f"   Saved representations to: {rep_path}")
    
    # 4.5. Create T-SNE visualization
    tsne_representations = create_tsne_visualization(
        representations, 
        labels.numpy(), 
        output_dir,
        perplexity=30,  # Adjust based on dataset size
        max_iter=1000
    )
    
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
            'name': 'K-means (20 clusters)',
            'algorithm': 'kmeans',
            'params': {'n_clusters': 20}
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
            title=f"{config['name']} - TrafficTransformer Flow-Level Representations",
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
    
    print(f"\nTrafficTransformer clustering analysis completed! Results saved to: {output_dir}")


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
    ax1.set_title('Adjusted Rand Score vs True Labels (TrafficTransformer)')
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
    ax2.set_title('Normalized Mutual Information vs True Labels (TrafficTransformer)')
    ax2.set_ylabel('Normalized Mutual Information')
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars2, nmi_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "traffic_transformer_clustering_vs_true_labels.png")
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
            'n_noise': int(result['n_noise']),
            'metrics': result['metrics'],
            'true_label_metrics': result.get('true_label_metrics', {})
        })
    
    summary_path = os.path.join(output_dir, "traffic_transformer_clustering_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_traffic_transformer_clustering_example() 