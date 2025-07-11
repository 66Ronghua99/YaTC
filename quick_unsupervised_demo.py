#!/usr/bin/env python3
"""
Quick Demo: Unsupervised Learning for Image Classification
Simplified version for quick testing and demonstration
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import models
import warnings
warnings.filterwarnings('ignore')

class SimpleImageDataset(Dataset):
    """Simplified dataset for loading images"""
    def __init__(self, data_dir, max_samples_per_class=50):
        self.data_dir = data_dir
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Get all class directories
        class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        
        for class_idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(data_dir, class_name)
            image_files = [f for f in os.listdir(class_path) if f.endswith('.png')][:max_samples_per_class]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                self.images.append(img_path)
                self.labels.append(class_idx)
                self.class_names.append(class_name)
        
        print(f"Loaded {len(self.images)} images from {len(class_dirs)} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and resize image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (64, 64))  # Resize for faster processing
        
        return image, label, img_path

def extract_simple_features(dataset):
    """Extract simple features using histogram and basic statistics"""
    features = []
    labels = []
    
    print("Extracting simple features...")
    for i in tqdm(range(len(dataset))):
        img, label, _ = dataset[i]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Extract features
        # 1. Histogram features
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        hist = hist / np.sum(hist)  # Normalize
        
        # 2. Statistical features
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        skewness = np.mean(((gray - mean_val) / std_val) ** 3) if std_val > 0 else 0
        
        # 3. Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Combine all features
        feature = np.concatenate([hist, [mean_val, std_val, skewness, edge_density]])
        features.append(feature)
        labels.append(label)
    
    return np.array(features), np.array(labels)

def extract_pretrained_features(dataset):
    """Extract features using pretrained ResNet18"""
    # Load pretrained model
    model = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove last layer
    model.eval()
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features = []
    labels = []
    
    print("Extracting pretrained features...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            img, label, _ = dataset[i]
            img_tensor = transform(img).unsqueeze(0)
            
            # Extract features
            feature = model(img_tensor)
            feature = feature.flatten().numpy()
            
            features.append(feature)
            labels.append(label)
    
    return np.array(features), np.array(labels)

def perform_clustering(features, n_clusters=20):
    """Perform K-means clustering"""
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels

def evaluate_clustering(features, true_labels, cluster_labels):
    """Evaluate clustering performance"""
    silhouette_avg = silhouette_score(features, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    
    return silhouette_avg, ari, nmi

def visualize_clustering(features, true_labels, cluster_labels, class_names, title="Clustering Results", output_dir="./"):
    """Visualize clustering results using t-SNE"""
    print("Creating visualization...")
    
    # Reduce dimensionality
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot true labels
    unique_labels = np.unique(true_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = true_labels == label
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=class_names[label], alpha=0.7, s=20)
    
    ax1.set_title('True Labels')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    
    # Plot cluster labels
    unique_clusters = np.unique(cluster_labels)
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        ax2.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[cluster_colors[i]], label=f'Cluster {cluster}', alpha=0.7, s=20)
    
    ax2.set_title('Cluster Labels')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save to output directory
    filename = f'{title.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.show()

def analyze_clusters(true_labels, cluster_labels, class_names):
    """Analyze cluster quality"""
    n_clusters = len(np.unique(cluster_labels))
    
    print(f"\nCluster Analysis:")
    total_purity = 0
    cluster_analysis = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_true_labels = true_labels[cluster_mask]
        
        if len(cluster_true_labels) > 0:
            unique, counts = np.unique(cluster_true_labels, return_counts=True)
            dominant_class = unique[np.argmax(counts)]
            dominant_count = np.max(counts)
            total_in_cluster = len(cluster_true_labels)
            purity = dominant_count / total_in_cluster
            total_purity += purity
            
            # Store analysis for JSON output
            cluster_info = {
                'cluster_id': int(cluster_id),
                'total_samples': int(total_in_cluster),
                'dominant_class': class_names[dominant_class],
                'dominant_class_id': int(dominant_class),
                'dominant_count': int(dominant_count),
                'purity': float(purity),
                'class_distribution': {}
            }
            
            # Add class distribution
            for class_id, count in zip(unique, counts):
                cluster_info['class_distribution'][class_names[class_id]] = int(count)
            
            cluster_analysis.append(cluster_info)
            
            print(f"Cluster {cluster_id}: {total_in_cluster} samples")
            print(f"  Dominant: {class_names[dominant_class]} ({dominant_count}/{total_in_cluster})")
            print(f"  Purity: {purity:.3f}")
    
    avg_purity = total_purity / n_clusters
    print(f"\nAverage Cluster Purity: {avg_purity:.3f}")
    
    return cluster_analysis, avg_purity

def main():
    """Main demo function"""
    import json
    import os
    
    # Configuration
    data_dir = "YaTC_datasets/USTC-TFC2016_MFR/train"
    max_samples_per_class = 50  # Small sample for quick demo
    n_clusters = 20
    output_dir = "./quick_demo_clustering_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Quick Unsupervised Learning Demo ===")
    print(f"Dataset: {data_dir}")
    print(f"Max samples per class: {max_samples_per_class}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = SimpleImageDataset(data_dir, max_samples_per_class)
    
    # Get unique class names
    class_names = list(set(dataset.class_names))
    class_names.sort()
    
    print(f"Classes: {class_names}")
    print()
    
    # Method 1: Simple features
    print("=== Method 1: Simple Features ===")
    features_simple, labels_simple = extract_simple_features(dataset)
    
    cluster_labels_simple = perform_clustering(features_simple, n_clusters)
    
    print("\nSimple Features Results:")
    silhouette_simple, ari_simple, nmi_simple = evaluate_clustering(features_simple, labels_simple, cluster_labels_simple)
    
    visualize_clustering(features_simple, labels_simple, cluster_labels_simple, 
                        class_names, "Simple Features Clustering", output_dir)
    
    cluster_analysis_simple, avg_purity_simple = analyze_clusters(labels_simple, cluster_labels_simple, class_names)
    
    # Method 2: Pretrained features
    print("\n=== Method 2: Pretrained Features ===")
    features_pretrained, labels_pretrained = extract_pretrained_features(dataset)
    
    cluster_labels_pretrained = perform_clustering(features_pretrained, n_clusters)
    
    print("\nPretrained Features Results:")
    silhouette_pretrained, ari_pretrained, nmi_pretrained = evaluate_clustering(features_pretrained, labels_pretrained, cluster_labels_pretrained)
    
    visualize_clustering(features_pretrained, labels_pretrained, cluster_labels_pretrained, 
                        class_names, "Pretrained Features Clustering", output_dir)
    
    cluster_analysis_pretrained, avg_purity_pretrained = analyze_clusters(labels_pretrained, cluster_labels_pretrained, class_names)
    
    # Compare methods
    print("\n=== Method Comparison ===")
    print(f"Simple Features - Silhouette: {silhouette_simple:.4f}, ARI: {ari_simple:.4f}, NMI: {nmi_simple:.4f}")
    print(f"Pretrained Features - Silhouette: {silhouette_pretrained:.4f}, ARI: {ari_pretrained:.4f}, NMI: {nmi_pretrained:.4f}")
    
    # Prepare results for JSON output
    results_summary = {
        'dataset_info': {
            'dataset_path': data_dir,
            'max_samples_per_class': max_samples_per_class,
            'n_clusters': n_clusters,
            'total_samples': len(labels_simple),
            'n_classes': len(class_names),
            'class_names': class_names
        },
        'simple_features': {
            'name': 'Simple Features (Histogram + Statistics + Edge)',
            'feature_dimension': features_simple.shape[1],
            'n_clusters': n_clusters,
            'metrics': {
                'silhouette_score': float(silhouette_simple),
                'adjusted_rand_index': float(ari_simple),
                'normalized_mutual_info': float(nmi_simple),
                'average_cluster_purity': float(avg_purity_simple)
            },
            'cluster_analysis': cluster_analysis_simple
        },
        'pretrained_features': {
            'name': 'Pretrained ResNet18 Features',
            'feature_dimension': features_pretrained.shape[1],
            'n_clusters': n_clusters,
            'metrics': {
                'silhouette_score': float(silhouette_pretrained),
                'adjusted_rand_index': float(ari_pretrained),
                'normalized_mutual_info': float(nmi_pretrained),
                'average_cluster_purity': float(avg_purity_pretrained)
            },
            'cluster_analysis': cluster_analysis_pretrained
        },
        'comparison': {
            'best_silhouette': 'pretrained_features' if silhouette_pretrained > silhouette_simple else 'simple_features',
            'best_ari': 'pretrained_features' if ari_pretrained > ari_simple else 'simple_features',
            'best_nmi': 'pretrained_features' if nmi_pretrained > nmi_simple else 'simple_features',
            'best_purity': 'pretrained_features' if avg_purity_pretrained > avg_purity_simple else 'simple_features'
        }
    }
    
    # Save JSON results
    json_path = os.path.join(output_dir, "quick_demo_clustering_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nJSON results saved to: {json_path}")
    
    # Save numpy results
    results = {
        'simple': {
            'silhouette': silhouette_simple,
            'ari': ari_simple,
            'features': features_simple,
            'cluster_labels': cluster_labels_simple,
            'true_labels': labels_simple
        },
        'pretrained': {
            'silhouette': silhouette_pretrained,
            'ari': ari_pretrained,
            'features': features_pretrained,
            'cluster_labels': cluster_labels_pretrained,
            'true_labels': labels_pretrained
        },
        'class_names': class_names
    }
    
    np_path = os.path.join(output_dir, "quick_demo_results.npy")
    np.save(np_path, results)
    print(f"NumPy results saved to: {np_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("CLUSTERING RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'Silhouette':<12} {'ARI':<12} {'NMI':<12} {'Purity':<12}")
    print("-"*80)
    print(f"{'Simple Features':<25} {silhouette_simple:<12.4f} {ari_simple:<12.4f} {nmi_simple:<12.4f} {avg_purity_simple:<12.4f}")
    print(f"{'Pretrained Features':<25} {silhouette_pretrained:<12.4f} {ari_pretrained:<12.4f} {nmi_pretrained:<12.4f} {avg_purity_pretrained:<12.4f}")
    print("="*80)

if __name__ == "__main__":
    main() 