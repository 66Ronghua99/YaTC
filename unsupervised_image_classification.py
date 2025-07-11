#!/usr/bin/env python3
"""
Unsupervised Learning for Image Classification
Using USTC-TFC2016_MFR dataset with 20 classes
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import warnings
warnings.filterwarnings('ignore')

class ImageDataset(Dataset):
    """Custom dataset for loading images"""
    def __init__(self, data_dir, transform=None, max_samples_per_class=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Get all class directories
        class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        
        for class_idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(data_dir, class_name)
            image_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
            
            # Limit samples per class if specified
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
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
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

class Autoencoder(nn.Module):
    """Simple Autoencoder for feature learning"""
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=64):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

def extract_features_pretrained(model_name='resnet18', dataset=None):
    """Extract features using pretrained model"""
    # Load pretrained model
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        # Remove the last classification layer
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    
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
    
    print("Extracting features using pretrained model...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            img, label, _ = dataset[i]
            img = transform(img).unsqueeze(0)
            
            # Extract features
            feature = model(img)
            feature = feature.flatten().numpy()
            
            features.append(feature)
            labels.append(label)
    
    return np.array(features), np.array(labels)

def train_autoencoder(dataset, input_dim=784, hidden_dim=128, latent_dim=64, epochs=50):
    """Train autoencoder for feature learning"""
    # Data preprocessing for autoencoder
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    
    # Prepare data
    data = []
    labels = []
    for i in tqdm(range(len(dataset)), desc="Preparing data"):
        img, label, _ = dataset[i]
        img = transform(img).flatten().numpy()
        data.append(img)
        labels.append(label)
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Convert to PyTorch tensors
    data_tensor = torch.FloatTensor(data)
    
    # Initialize model and training
    model = Autoencoder(input_dim, hidden_dim, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Training autoencoder...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data_tensor)
        loss = criterion(outputs, data_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Extract features
    model.eval()
    with torch.no_grad():
        features = model.encode(data_tensor).numpy()
    
    return features, labels

def perform_clustering(features, n_clusters=20, method='kmeans'):
    """Perform clustering on features"""
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        raise ValueError(f"Method {method} not implemented")
    
    cluster_labels = clusterer.fit_predict(features)
    return cluster_labels, clusterer

def evaluate_clustering(true_labels, cluster_labels):
    """Evaluate clustering performance"""
    # Silhouette score
    silhouette_avg = silhouette_score(features, cluster_labels)
    
    # Adjusted Rand Index (if we have true labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    
    return silhouette_avg, ari

def visualize_results(features, true_labels, cluster_labels, class_names, method='tsne'):
    """Visualize clustering results"""
    # Reduce dimensionality for visualization
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Method {method} not implemented")
    
    features_2d = reducer.fit_transform(features)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot true labels
    unique_labels = np.unique(true_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = true_labels == label
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=class_names[label], alpha=0.7, s=20)
    
    ax1.set_title('True Labels')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xlabel(f'{method.upper()} 1')
    ax1.set_ylabel(f'{method.upper()} 2')
    
    # Plot cluster labels
    unique_clusters = np.unique(cluster_labels)
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        ax2.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[cluster_colors[i]], label=f'Cluster {cluster}', alpha=0.7, s=20)
    
    ax2.set_title('Cluster Labels')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xlabel(f'{method.upper()} 1')
    ax2.set_ylabel(f'{method.upper()} 2')
    
    plt.tight_layout()
    plt.savefig('clustering_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_cluster_distribution(true_labels, cluster_labels, class_names):
    """Analyze the distribution of true labels within each cluster"""
    n_clusters = len(np.unique(cluster_labels))
    n_classes = len(np.unique(true_labels))
    
    # Create confusion matrix
    confusion_matrix = np.zeros((n_clusters, n_classes))
    
    for i in range(len(true_labels)):
        confusion_matrix[cluster_labels[i], true_labels[i]] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', 
                xticklabels=class_names, yticklabels=[f'Cluster {i}' for i in range(n_clusters)],
                cmap='Blues')
    plt.title('Cluster vs True Label Distribution')
    plt.xlabel('True Labels')
    plt.ylabel('Clusters')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print cluster analysis
    print("\nCluster Analysis:")
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_true_labels = true_labels[cluster_mask]
        
        if len(cluster_true_labels) > 0:
            unique, counts = np.unique(cluster_true_labels, return_counts=True)
            dominant_class = unique[np.argmax(counts)]
            dominant_count = np.max(counts)
            total_in_cluster = len(cluster_true_labels)
            
            print(f"Cluster {cluster_id}: {total_in_cluster} samples")
            print(f"  Dominant class: {class_names[dominant_class]} ({dominant_count}/{total_in_cluster})")
            print(f"  Purity: {dominant_count/total_in_cluster:.3f}")
            print()

def main():
    """Main function"""
    # Configuration
    data_dir = "YaTC_datasets/USTC-TFC2016_MFR/train"
    max_samples_per_class = 100  # Limit samples per class for faster processing
    n_clusters = 20
    
    print("=== Unsupervised Learning for Image Classification ===")
    print(f"Dataset: {data_dir}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Max samples per class: {max_samples_per_class}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = ImageDataset(data_dir, max_samples_per_class=max_samples_per_class)
    
    # Get unique class names
    class_names = list(set(dataset.class_names))
    class_names.sort()
    
    print(f"Classes: {class_names}")
    print()
    
    # Method 1: Extract features using pretrained model
    print("=== Method 1: Pretrained Model Features ===")
    features_pretrained, labels_pretrained = extract_features_pretrained('resnet18', dataset)
    
    # Perform clustering
    cluster_labels_pretrained, clusterer_pretrained = perform_clustering(
        features_pretrained, n_clusters, 'kmeans'
    )
    
    # Evaluate clustering
    print("\nPretrained Model Results:")
    silhouette_pretrained, ari_pretrained = evaluate_clustering(labels_pretrained, cluster_labels_pretrained)
    
    # Visualize results
    visualize_results(features_pretrained, labels_pretrained, cluster_labels_pretrained, 
                     class_names, 'tsne')
    
    # Analyze cluster distribution
    analyze_cluster_distribution(labels_pretrained, cluster_labels_pretrained, class_names)
    
    # Method 2: Autoencoder features
    print("\n=== Method 2: Autoencoder Features ===")
    features_autoencoder, labels_autoencoder = train_autoencoder(dataset)
    
    # Perform clustering
    cluster_labels_autoencoder, clusterer_autoencoder = perform_clustering(
        features_autoencoder, n_clusters, 'kmeans'
    )
    
    # Evaluate clustering
    print("\nAutoencoder Results:")
    silhouette_autoencoder, ari_autoencoder = evaluate_clustering(labels_autoencoder, cluster_labels_autoencoder)
    
    # Visualize results
    visualize_results(features_autoencoder, labels_autoencoder, cluster_labels_autoencoder, 
                     class_names, 'tsne')
    
    # Analyze cluster distribution
    analyze_cluster_distribution(labels_autoencoder, cluster_labels_autoencoder, class_names)
    
    # Compare methods
    print("\n=== Method Comparison ===")
    print(f"Pretrained Model - Silhouette: {silhouette_pretrained:.4f}, ARI: {ari_pretrained:.4f}")
    print(f"Autoencoder - Silhouette: {silhouette_autoencoder:.4f}, ARI: {ari_autoencoder:.4f}")
    
    # Save results
    results = {
        'pretrained': {
            'silhouette': silhouette_pretrained,
            'ari': ari_pretrained,
            'features': features_pretrained,
            'cluster_labels': cluster_labels_pretrained,
            'true_labels': labels_pretrained
        },
        'autoencoder': {
            'silhouette': silhouette_autoencoder,
            'ari': ari_autoencoder,
            'features': features_autoencoder,
            'cluster_labels': cluster_labels_autoencoder,
            'true_labels': labels_autoencoder
        },
        'class_names': class_names
    }
    
    np.save('unsupervised_results.npy', results)
    print("\nResults saved to 'unsupervised_results.npy'")

if __name__ == "__main__":
    main() 