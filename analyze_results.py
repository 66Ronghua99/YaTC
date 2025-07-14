#!/usr/bin/env python3
"""
Analyze clustering results from JSON output
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_results(json_path):
    """Load clustering results from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def print_summary(results):
    """Print a summary of the clustering results"""
    print("="*60)
    print("CLUSTERING RESULTS SUMMARY")
    print("="*60)
    
    # Dataset info
    dataset_info = results['dataset_info']
    print(f"Dataset: {dataset_info['dataset_path']}")
    print(f"Total samples: {dataset_info['total_samples']}")
    print(f"Number of classes: {dataset_info['n_classes']}")
    print(f"Number of clusters: {dataset_info['n_clusters']}")
    print(f"Classes: {', '.join(dataset_info['class_names'])}")
    print()
    
    # Method comparison
    print("METHOD COMPARISON:")
    print("-"*80)
    print(f"{'Method':<30} {'Silhouette':<12} {'ARI':<12} {'NMI':<12} {'Purity':<12}")
    print("-"*80)
    
    for method in ['simple_features', 'pretrained_features']:
        method_data = results[method]
        metrics = method_data['metrics']
        print(f"{method_data['name']:<30} {metrics['silhouette_score']:<12.4f} "
              f"{metrics['adjusted_rand_index']:<12.4f} {metrics['normalized_mutual_info']:<12.4f} "
              f"{metrics['average_cluster_purity']:<12.4f}")
    
    print()
    
    # Best method analysis
    comparison = results['comparison']
    print("BEST PERFORMING METHOD:")
    print(f"  Silhouette Score: {comparison['best_silhouette']}")
    print(f"  Adjusted Rand Index: {comparison['best_ari']}")
    print(f"  Normalized Mutual Info: {comparison['best_nmi']}")
    print(f"  Cluster Purity: {comparison['best_purity']}")

def analyze_cluster_quality(results, method='pretrained_features'):
    """Analyze cluster quality for a specific method"""
    method_data = results[method]
    cluster_analysis = method_data['cluster_analysis']
    
    print(f"\n{method_data['name'].upper()} - CLUSTER QUALITY ANALYSIS:")
    print("-"*60)
    
    # Calculate statistics
    purities = [cluster['purity'] for cluster in cluster_analysis]
    sample_counts = [cluster['total_samples'] for cluster in cluster_analysis]
    
    print(f"Number of clusters: {len(cluster_analysis)}")
    print(f"Average cluster purity: {np.mean(purities):.4f}")
    print(f"Standard deviation of purity: {np.std(purities):.4f}")
    print(f"Min purity: {np.min(purities):.4f}")
    print(f"Max purity: {np.max(purities):.4f}")
    print(f"Total samples: {sum(sample_counts)}")
    print(f"Average samples per cluster: {np.mean(sample_counts):.1f}")
    
    # Find best and worst clusters
    best_cluster = max(cluster_analysis, key=lambda x: x['purity'])
    worst_cluster = min(cluster_analysis, key=lambda x: x['purity'])
    
    print(f"\nBest cluster (ID {best_cluster['cluster_id']}):")
    print(f"  Purity: {best_cluster['purity']:.4f}")
    print(f"  Dominant class: {best_cluster['dominant_class']}")
    print(f"  Samples: {best_cluster['total_samples']}")
    
    print(f"\nWorst cluster (ID {worst_cluster['cluster_id']}):")
    print(f"  Purity: {worst_cluster['purity']:.4f}")
    print(f"  Dominant class: {worst_cluster['dominant_class']}")
    print(f"  Samples: {worst_cluster['total_samples']}")

def plot_cluster_purities(results):
    """Plot cluster purities for both methods"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Simple features
    simple_data = results['simple_features']
    simple_purities = [cluster['purity'] for cluster in simple_data['cluster_analysis']]
    simple_cluster_ids = [cluster['cluster_id'] for cluster in simple_data['cluster_analysis']]
    
    bars1 = ax1.bar(simple_cluster_ids, simple_purities, color='skyblue', alpha=0.7)
    ax1.set_title(f'{simple_data["name"]}\nCluster Purity Distribution')
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Purity')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add average line
    avg_purity = simple_data['metrics']['average_cluster_purity']
    ax1.axhline(y=avg_purity, color='red', linestyle='--', label=f'Average: {avg_purity:.3f}')
    ax1.legend()
    
    # Pretrained features
    pretrained_data = results['pretrained_features']
    pretrained_purities = [cluster['purity'] for cluster in pretrained_data['cluster_analysis']]
    pretrained_cluster_ids = [cluster['cluster_id'] for cluster in pretrained_data['cluster_analysis']]
    
    bars2 = ax2.bar(pretrained_cluster_ids, pretrained_purities, color='lightcoral', alpha=0.7)
    ax2.set_title(f'{pretrained_data["name"]}\nCluster Purity Distribution')
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Purity')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add average line
    avg_purity = pretrained_data['metrics']['average_cluster_purity']
    ax2.axhline(y=avg_purity, color='red', linestyle='--', label=f'Average: {avg_purity:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('cluster_purity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_comparison(results):
    """Plot comparison of all metrics between methods"""
    methods = ['simple_features', 'pretrained_features']
    method_names = [results[method]['name'] for method in methods]
    
    # Extract metrics
    metrics_names = ['silhouette_score', 'adjusted_rand_index', 'normalized_mutual_info', 'average_cluster_purity']
    metrics_labels = ['Silhouette', 'ARI', 'NMI', 'Purity']
    
    # Create data for plotting
    data = []
    for method in methods:
        method_metrics = []
        for metric in metrics_names:
            method_metrics.append(results[method]['metrics'][metric])
        data.append(method_metrics)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['skyblue', 'lightcoral']
    
    for i, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
        ax = axes[i]
        
        # Create bar plot
        bars = ax.bar(method_names, [data[0][i], data[1][i]], color=colors, alpha=0.7)
        ax.set_title(f'{label} Comparison')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, [data[0][i], data[1][i]]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Set y-axis limits based on metric
        if metric in ['silhouette_score', 'adjusted_rand_index', 'normalized_mutual_info']:
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_distribution_heatmap(results, method='pretrained_features'):
    """Plot heatmap of class distribution across clusters"""
    method_data = results[method]
    cluster_analysis = method_data['cluster_analysis']
    class_names = results['dataset_info']['class_names']
    
    # Create matrix
    n_clusters = len(cluster_analysis)
    n_classes = len(class_names)
    matrix = np.zeros((n_clusters, n_classes))
    
    for cluster in cluster_analysis:
        cluster_id = cluster['cluster_id']
        for class_name, count in cluster['class_distribution'].items():
            class_idx = class_names.index(class_name)
            matrix[cluster_id, class_idx] = count
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, annot=True, fmt='g', 
                xticklabels=class_names, 
                yticklabels=[f'Cluster {i}' for i in range(n_clusters)],
                cmap='Blues')
    plt.title(f'Class Distribution Across Clusters - {method_data["name"]}')
    plt.xlabel('True Classes')
    plt.ylabel('Clusters')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{method}_class_distribution_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function"""
    json_path = "quick_demo_clustering_results/quick_demo_clustering_summary.json"
    
    print("Loading clustering results...")
    results = load_results(json_path)
    
    # Print summary
    print_summary(results)
    
    # Analyze cluster quality for both methods
    for method in ['simple_features', 'pretrained_features']:
        analyze_cluster_quality(results, method)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_cluster_purities(results)
    plot_metrics_comparison(results)
    plot_class_distribution_heatmap(results, 'simple_features')
    plot_class_distribution_heatmap(results, 'pretrained_features')
    
    print("\nAnalysis completed! Check the generated plots for detailed visualizations.")

if __name__ == "__main__":
    main() 