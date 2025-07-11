import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
import pickle
import json

from models_YaTC import MAE_YaTC, TraFormer_YaTC


class MAEEncoderExtractor:
    """Extract encoder representations from MAE model"""
    
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained MAE model"""
        # Initialize model with same parameters as training
        model = MAE_YaTC()
        
        # Load trained weights with robust error handling
        print(f"Loading model from: {model_path}")
        
        try:
            # Try loading with weights_only=True first (PyTorch 2.6+ default)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            print("Successfully loaded checkpoint with weights_only=True")
        except Exception as e:
            print(f"Loading with weights_only=True failed: {e}")
            try:
                # If that fails, try with weights_only=False (for older checkpoints)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                print("Successfully loaded checkpoint with weights_only=False")
            except Exception as e2:
                print(f"Loading with weights_only=False also failed: {e2}")
                raise RuntimeError(f"Failed to load model checkpoint: {e2}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                print("Loaded model state from checkpoint['model']")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("Loaded model state from checkpoint['state_dict']")
            else:
                # Assume the entire checkpoint is the state dict
                model.load_state_dict(checkpoint)
                print("Loaded model state from checkpoint directly")
        else:
            # Assume checkpoint is directly the state dict
            model.load_state_dict(checkpoint)
            print("Loaded model state from checkpoint directly")
            
        model.to(self.device)
        model.eval()
        return model
    
    def extract_representations(self, data: torch.Tensor, mask_ratio: float = 0.0) -> np.ndarray:
        """
        Extract encoder representations from input data
        
        Args:
            data: Input tensor of shape (N, 1, H, W)
            mask_ratio: Masking ratio for encoder (0.0 for no masking)
            
        Returns:
            Encoder representations of shape (N, embed_dim)
        """
        with torch.no_grad():
            # Use the encoder part of MAE
            latent, _, _ = self.model.forward_encoder(data.to(self.device), mask_ratio)
            
            # Extract CLS token representation (first token)
            # This is the global representation of the input
            representations = latent[:, 0, :].cpu().numpy()  # (N, embed_dim)
            
        return representations


class TrafficTransformerExtractor:
    """Extract flow-level representations from fine-tuned TrafficTransformer model"""
    
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu', num_classes: int = 20):
        self.device = device
        self.num_classes = num_classes
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained TrafficTransformer model"""
        # Initialize model with same parameters as training
        model = TraFormer_YaTC(num_classes=self.num_classes)
        
        # Load trained weights with robust error handling
        print(f"Loading TrafficTransformer model from: {model_path}")
        
        try:
            # Try loading with weights_only=True first (PyTorch 2.6+ default)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            print("Successfully loaded checkpoint with weights_only=True")
        except Exception as e:
            print(f"Loading with weights_only=True failed: {e}")
            try:
                # If that fails, try with weights_only=False (for older checkpoints)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                print("Successfully loaded checkpoint with weights_only=False")
            except Exception as e2:
                print(f"Loading with weights_only=False also failed: {e2}")
                raise RuntimeError(f"Failed to load model checkpoint: {e2}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                print("Loaded model state from checkpoint['model']")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("Loaded model state from checkpoint['state_dict']")
            else:
                # Assume the entire checkpoint is the state dict
                model.load_state_dict(checkpoint)
                print("Loaded model state from checkpoint directly")
        else:
            # Assume checkpoint is directly the state dict
            model.load_state_dict(checkpoint)
            print("Loaded model state from checkpoint directly")
            
        model.to(self.device)
        model.eval()
        return model
    
    def extract_representations(self, data: torch.Tensor) -> np.ndarray:
        """
        Extract flow-level representations from input data before final layernorm
        
        Args:
            data: Input tensor of shape (N, 1, H, W)
            
        Returns:
            Flow-level representations of shape (N, embed_dim)
        """
        with torch.no_grad():
            # Use the forward_features method which returns representations before the head
            representations = self.model.forward_features(data.to(self.device))
            representations = representations.cpu().numpy()  # (N, embed_dim)
            
        return representations
    



class ClusteringPipeline:
    """Pipeline for unsupervised clustering with multiple algorithms"""
    
    def __init__(self, n_clusters: int = 5, algorithm: str = 'kmeans', 
                 random_state: int = 42, **kwargs):
        self.n_clusters = n_clusters
        self.algorithm = algorithm.lower()
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = self._initialize_clustering_model()
        self.scaler = StandardScaler()
        
    def _initialize_clustering_model(self):
        """Initialize clustering model based on algorithm choice"""
        if self.algorithm == 'kmeans':
            return KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **self.kwargs
            )
        elif self.algorithm == 'dbscan':
            return DBSCAN(
                eps=self.kwargs.get('eps', 0.5),
                min_samples=self.kwargs.get('min_samples', 5),
                **{k: v for k, v in self.kwargs.items() if k not in ['eps', 'min_samples']}
            )
        elif self.algorithm == 'hierarchical':
            return AgglomerativeClustering(
                n_clusters=self.n_clusters,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit clustering model and return cluster labels"""
        # Scale the features
        # X_scaled = self.scaler.fit_transform(X)
        
        # Fit and predict
        # labels = self.model.fit_predict(X_scaled)
        labels = self.model.fit_predict(X)
        return labels
    
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality using multiple metrics"""
        # Remove noise points for evaluation (DBSCAN can assign -1)
        valid_mask = labels != -1
        if not np.any(valid_mask):
            return {'silhouette': -1, 'calinski_harabasz': -1, 'davies_bouldin': -1}
        
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]
        
        # Calculate metrics
        metrics = {}
        
        # Silhouette score (higher is better, range: -1 to 1)
        if len(np.unique(labels_valid)) > 1:
            metrics['silhouette'] = silhouette_score(X_valid, labels_valid)
        else:
            metrics['silhouette'] = -1
            
        # Calinski-Harabasz score (higher is better)
        if len(np.unique(labels_valid)) > 1:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_valid, labels_valid)
        else:
            metrics['calinski_harabasz'] = -1
            
        # Davies-Bouldin score (lower is better)
        if len(np.unique(labels_valid)) > 1:
            metrics['davies_bouldin'] = davies_bouldin_score(X_valid, labels_valid)
        else:
            metrics['davies_bouldin'] = -1
            
        return metrics


class VisualizationHelper:
    """Helper class for clustering visualization"""
    
    @staticmethod
    def plot_clusters_2d(X: np.ndarray, labels: np.ndarray, 
                        title: str = "Clustering Results", 
                        save_path: Optional[str] = None):
        """Plot clustering results in 2D using PCA"""
        # Reduce dimensionality to 2D using PCA
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot clusters
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # Noise points
                mask = labels == label
                plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                           c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                mask = labels == label
                plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                           c=[colors[i]], s=50, alpha=0.7, label=f'Cluster {label}')
        
        plt.title(f"{title}\nPCA projection")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_cluster_statistics(labels: np.ndarray, save_path: Optional[str] = None):
        """Plot cluster size distribution"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(unique_labels)), counts, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(unique_labels))))
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                    str(count), ha='center', va='bottom')
        
        plt.title("Cluster Size Distribution")
        plt.xlabel("Cluster ID")
        plt.ylabel("Number of Samples")
        plt.xticks(range(len(unique_labels)), [f"Cluster {l}" if l != -1 else "Noise" for l in unique_labels])
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def load_data(data_path: str, max_samples_per_class: Optional[int] = None) -> torch.Tensor:
    """Load data from file or YaTC dataset"""
    # Check if it's a YaTC dataset path
    if os.path.isdir(data_path) and ('YaTC_datasets' in data_path or 'MFR' in data_path):
        from data_loader import load_all_data_as_tensor
        print(f"Loading YaTC dataset from: {data_path}")
        data, labels = load_all_data_as_tensor(data_path, split='train', max_samples_per_class=max_samples_per_class)
        return data
    else:
        # Load from file
        if data_path.endswith('.npy'):
            data = np.load(data_path)
        elif data_path.endswith('.pt') or data_path.endswith('.pth'):
            data = torch.load(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Ensure data is in the correct format (N, 1, H, W)
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        if len(data.shape) == 3:
            data = data.unsqueeze(1)  # Add channel dimension
        
        return data


def main():
    parser = argparse.ArgumentParser(description='Unsupervised clustering using MAE or TrafficTransformer representations')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='mae',
                       choices=['mae', 'traffic_transformer'],
                       help='Type of model to use for representation extraction')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the input data file (.npy or .pt) or YaTC dataset directory')
    parser.add_argument('--output_dir', type=str, default='./clustering_results',
                       help='Directory to save clustering results')
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of clusters for K-means and hierarchical clustering')
    parser.add_argument('--algorithm', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan', 'hierarchical'],
                       help='Clustering algorithm to use')
    parser.add_argument('--mask_ratio', type=float, default=0.0,
                       help='Masking ratio for encoder (0.0 for no masking, only for MAE)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing data')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--eps', type=float, default=0.5,
                       help='Epsilon parameter for DBSCAN')
    parser.add_argument('--min_samples', type=int, default=5,
                       help='Min samples parameter for DBSCAN')
    parser.add_argument('--save_representations', action='store_true',
                       help='Save extracted representations to file')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                       help='Maximum number of samples per class (for YaTC datasets)')
    parser.add_argument('--num_classes', type=int, default=20,
                       help='Number of classes in the fine-tuned model (default: 20)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data = load_data(args.data_path, args.max_samples_per_class)
    print(f"Data shape: {data.shape}")
    
    # Initialize encoder extractor based on model type
    if args.model_type == 'mae':
        print("Loading MAE model...")
        extractor = MAEEncoderExtractor(args.model_path, device)
        
        # Extract representations
        print("Extracting MAE encoder representations...")
        representations = extractor.extract_representations(data, args.mask_ratio)
        rep_type = 'mae_encoder'
        
    elif args.model_type == 'traffic_transformer':
        print("Loading TrafficTransformer model...")
        extractor = TrafficTransformerExtractor(args.model_path, device, num_classes=args.num_classes)
        
        # Extract representations
        print("Extracting TrafficTransformer flow-level representations...")
        representations = extractor.extract_representations(data)
        rep_type = 'traffic_transformer_flow'
        
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    print(f"Representations shape: {representations.shape}")
    
    # Save representations if requested
    if args.save_representations:
        rep_path = os.path.join(args.output_dir, f'{rep_type}_representations.npy')
        np.save(rep_path, representations)
        print(f"Saved representations to: {rep_path}")
    
    # Initialize clustering pipeline
    clustering_kwargs = {}
    if args.algorithm == 'dbscan':
        clustering_kwargs.update({'eps': args.eps, 'min_samples': args.min_samples})
    
    pipeline = ClusteringPipeline(
        n_clusters=args.n_clusters,
        algorithm=args.algorithm,
        random_state=args.random_state,
        **clustering_kwargs
    )
    
    # Perform clustering
    print(f"Performing {args.algorithm} clustering...")
    labels = pipeline.fit_predict(representations)
    
    # Evaluate clustering
    print("Evaluating clustering quality...")
    metrics = pipeline.evaluate_clustering(representations, labels)
    
    # Print results
    print("\n" + "="*50)
    print("CLUSTERING RESULTS")
    print("="*50)
    print(f"Model type: {args.model_type}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Number of clusters: {len(np.unique(labels[labels != -1]))}")
    print(f"Number of noise points: {np.sum(labels == -1)}")
    print(f"Total samples: {len(labels)}")
    
    print("\nClustering Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    results = {
        'model_type': args.model_type,
        'algorithm': args.algorithm,
        'n_clusters': args.n_clusters,
        'labels': labels.tolist(),
        'metrics': metrics,
        'parameters': vars(args)
    }
    
    results_path = os.path.join(args.output_dir, 'clustering_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")
    
    # Visualizations
    print("Creating visualizations...")
    viz_helper = VisualizationHelper()
    
    # Plot clusters
    cluster_plot_path = os.path.join(args.output_dir, 'cluster_plot.png')
    viz_helper.plot_clusters_2d(representations, labels, 
                               title=f"{args.algorithm.upper()} Clustering",
                               save_path=cluster_plot_path)
    
    # Plot cluster statistics
    stats_plot_path = os.path.join(args.output_dir, 'cluster_statistics.png')
    viz_helper.plot_cluster_statistics(labels, save_path=stats_plot_path)
    
    print(f"Saved visualizations to: {args.output_dir}")
    print("\nClustering completed successfully!")


if __name__ == "__main__":
    main()
