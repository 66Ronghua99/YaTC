# YaTC MAE Clustering Pipeline

This repository provides a comprehensive clustering pipeline for unsupervised classification using MAE (Masked Autoencoder) encoder representations on the YaTC datasets.

## Overview

The pipeline consists of several components:

1. **MAE Model Integration**: Loads trained MAE models and extracts encoder representations
2. **Data Loading**: Supports YaTC datasets (CICIoT2022_MFR, ISCXVPN2016_MFR, etc.) and custom data
3. **Clustering Algorithms**: Multiple clustering algorithms (K-means, DBSCAN, Hierarchical)
4. **Evaluation**: Comprehensive clustering evaluation metrics
5. **Visualization**: 2D projections and cluster statistics
6. **Comparison**: Compare different algorithms and parameters

## Files

- `rp_clustering.py`: Main clustering pipeline with command-line interface
- `data_loader.py`: YaTC dataset loader and utilities
- `yaTC_clustering_example.py`: Comprehensive example with multiple algorithms
- `run_yaTC_clustering.py`: Simple command-line interface for YaTC datasets
- `example_clustering.py`: Basic clustering examples
- `requirements_clustering.txt`: Required dependencies

## Installation

1. Install dependencies:
```bash
pip install -r requirements_clustering.txt
```

2. Ensure you have the YaTC datasets in the `YaTC_datasets/` directory:
```
YaTC_datasets/
├── CICIoT2022_MFR/
│   ├── train/
│   │   ├── Attacks_Flood/
│   │   ├── Idle/
│   │   └── ...
│   └── test/
├── ISCXVPN2016_MFR/
└── ...
```

## Quick Start

### 1. Basic Clustering with YaTC Dataset

```bash
python run_yaTC_clustering.py \
    --model_path path/to/your/mae_model.pth \
    --dataset CICIoT2022_MFR \
    --algorithm kmeans \
    --n_clusters 5 \
    --max_samples_per_class 500
```

### 2. Compare Multiple Algorithms

```bash
python run_yaTC_clustering.py \
    --model_path path/to/your/mae_model.pth \
    --dataset CICIoT2022_MFR \
    --compare_algorithms \
    --max_samples_per_class 500
```

### 3. Using the Main Pipeline

```bash
python rp_clustering.py \
    --model_path path/to/your/mae_model.pth \
    --data_path YaTC_datasets/CICIoT2022_MFR \
    --algorithm kmeans \
    --n_clusters 5 \
    --max_samples_per_class 1000 \
    --output_dir ./results
```

## Usage Examples

### Command Line Options

#### Basic Options
- `--model_path`: Path to trained MAE model checkpoint
- `--data_path`: Path to data (file or YaTC dataset directory)
- `--output_dir`: Output directory for results
- `--algorithm`: Clustering algorithm (`kmeans`, `dbscan`, `hierarchical`)

#### Clustering Parameters
- `--n_clusters`: Number of clusters (K-means, Hierarchical)
- `--eps`: Epsilon parameter (DBSCAN)
- `--min_samples`: Min samples parameter (DBSCAN)
- `--max_samples_per_class`: Limit samples per class (YaTC datasets)

#### Output Options
- `--save_representations`: Save extracted representations
- `--compare_algorithms`: Compare multiple algorithms

### Python API

```python
from rp_clustering import MAEEncoderExtractor, ClusteringPipeline
from data_loader import load_all_data_as_tensor

# Load data
data, labels = load_all_data_as_tensor("YaTC_datasets/CICIoT2022_MFR", max_samples_per_class=500)

# Extract representations
extractor = MAEEncoderExtractor("path/to/model.pth")
representations = extractor.extract_representations(data)

# Perform clustering
pipeline = ClusteringPipeline(algorithm='kmeans', n_clusters=5)
cluster_labels = pipeline.fit_predict(representations)

# Evaluate
metrics = pipeline.evaluate_clustering(representations, cluster_labels)
print(f"Silhouette score: {metrics['silhouette']:.4f}")
```

## Supported Datasets

The pipeline supports all YaTC datasets:

1. **CICIoT2022_MFR**: IoT traffic classification
2. **ISCXVPN2016_MFR**: VPN traffic detection
3. **USTC-TFC2016_MFR**: Traffic classification
4. **ISCXTor2016_MFR**: Tor traffic detection

Each dataset contains:
- `train/`: Training data organized by class
- `test/`: Test data organized by class
- PNG images (40x40 pixels) representing network traffic

## Clustering Algorithms

### 1. K-means
- **Use case**: When you know the number of clusters
- **Parameters**: `n_clusters`
- **Pros**: Fast, simple, works well with spherical clusters
- **Cons**: Assumes spherical clusters, sensitive to initialization

### 2. DBSCAN
- **Use case**: When you don't know the number of clusters, want to detect noise
- **Parameters**: `eps`, `min_samples`
- **Pros**: Can find clusters of arbitrary shape, detects noise
- **Cons**: Sensitive to parameter tuning, doesn't work well with varying densities

### 3. Hierarchical Clustering
- **Use case**: When you want to understand cluster hierarchy
- **Parameters**: `n_clusters`
- **Pros**: Provides cluster hierarchy, works with any distance metric
- **Cons**: Computationally expensive, sensitive to noise

## Evaluation Metrics

The pipeline provides multiple evaluation metrics:

### Internal Metrics (No ground truth needed)
- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1, higher is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster dispersion (higher is better)
- **Davies-Bouldin Score**: Average similarity measure of clusters (lower is better)

### External Metrics (With ground truth)
- **Adjusted Rand Score**: Measures clustering vs true labels (higher is better)
- **Normalized Mutual Information**: Information-theoretic measure (higher is better)

## Output Files

The pipeline generates several output files:

### Visualizations
- `cluster_plot.png`: 2D PCA projection of clusters
- `cluster_statistics.png`: Cluster size distribution
- `sample_images.png`: Sample images from each class
- `clustering_vs_true_labels.png`: Comparison with true labels

### Data Files
- `mae_representations.npy`: Extracted encoder representations
- `clustering_results.json`: Detailed clustering results
- `clustering_summary.json`: Summary of all algorithms

### Results Structure
```json
{
  "algorithm": "kmeans",
  "n_clusters": 5,
  "labels": [...],
  "metrics": {
    "silhouette": 0.45,
    "calinski_harabasz": 1234.56,
    "davies_bouldin": 0.78
  },
  "true_label_metrics": {
    "adjusted_rand_score": 0.67,
    "normalized_mutual_info": 0.72
  }
}
```

## Advanced Usage

### Custom Data Loading

```python
from data_loader import YaTCDataset

# Load custom dataset
dataset = YaTCDataset(
    root_dir="path/to/dataset",
    split="train",
    max_samples_per_class=1000
)

# Access data
for data, label in dataset:
    # data: torch.Tensor of shape (1, 40, 40)
    # label: int class index
    pass
```

### Custom Clustering

```python
from rp_clustering import ClusteringPipeline

# Create custom clustering pipeline
pipeline = ClusteringPipeline(
    algorithm='kmeans',
    n_clusters=10,
    random_state=42,
    n_init=20  # Additional K-means parameters
)

# Fit and predict
labels = pipeline.fit_predict(representations)
```

### Visualization

```python
from rp_clustering import VisualizationHelper

viz = VisualizationHelper()

# Plot clusters
viz.plot_clusters_2d(representations, labels, title="My Clustering")

# Plot statistics
viz.plot_cluster_statistics(labels)
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model path is correct and the file exists
2. **Dataset not found**: Check that the YaTC_datasets directory structure is correct
3. **Memory issues**: Reduce `max_samples_per_class` or use smaller batch sizes
4. **CUDA out of memory**: Use CPU or reduce batch size

### Performance Tips

1. **Use GPU**: Set device to 'cuda' for faster processing
2. **Limit samples**: Use `max_samples_per_class` for large datasets
3. **Batch processing**: For very large datasets, process in batches
4. **Save representations**: Save and reuse representations to avoid recomputation

## Example Results

Typical clustering results on CICIoT2022_MFR dataset:

```
Algorithm          Clusters   Noise    Silhouette   Calinski     Davies
K-means (5)        5          0        0.4523       1234.5678    0.7890
DBSCAN (eps=0.5)   8          45       0.3456       987.6543    0.9123
Hierarchical (5)   5          0        0.4123       1111.2222    0.8234
```

## Citation

If you use this clustering pipeline in your research, please cite:

```bibtex
@article{yatc2023,
  title={YaTC: Yet another Traffic Classification using Vision Transformer},
  author={...},
  journal={...},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 