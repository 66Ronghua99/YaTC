# YaTC Clustering with MAE and TrafficTransformer Representations

This repository provides comprehensive clustering capabilities for the YaTC (Yet another Traffic Classification) dataset using representations from both MAE (Masked Autoencoder) and TrafficTransformer models.

## Overview

The clustering pipeline supports two types of model representations:

1. **MAE Encoder Representations**: Extracted from pre-trained MAE models using the encoder part
2. **TrafficTransformer Flow-Level Representations**: Extracted from fine-tuned TrafficTransformer models before the final layernorm

## Key Features

- **Multiple Model Support**: Both MAE and TrafficTransformer models
- **Multiple Clustering Algorithms**: K-means, DBSCAN, and Hierarchical clustering
- **Comprehensive Evaluation**: Silhouette score, Calinski-Harabasz score, Davies-Bouldin score
- **Visualization**: PCA projections, T-SNE visualizations, cluster statistics
- **True Label Analysis**: Compare clustering results with ground truth labels
- **Flexible Data Loading**: Support for YaTC datasets and custom data files

## Files Structure

```
├── rp_clustering.py                    # Main clustering pipeline
├── traffic_transformer_clustering_example.py  # TrafficTransformer-specific example
├── yaTC_clustering_example.py          # MAE-specific example
├── usage_example.py                    # Usage examples and commands
├── models_YaTC.py                      # Model definitions
└── README_clustering.md                # This documentation
```

## Installation

Ensure you have the required dependencies:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn numpy
```

## Usage

### Basic Usage

The main clustering script supports both model types:

```bash
python rp_clustering.py --model_path <path> --model_type <type> --data_path <path>
```

### Model Types

- `--model_type mae`: Use MAE encoder representations
- `--model_type traffic_transformer`: Use TrafficTransformer flow-level representations

### Examples

#### MAE Clustering
```bash
python rp_clustering.py \
    --model_path "pre-trained-model/YaTC_pretrained_model.pth" \
    --model_type mae \
    --data_path "YaTC_datasets/USTC-TFC2016_MFR" \
    --output_dir "./mae_clustering_results" \
    --n_clusters 10 \
    --algorithm kmeans \
    --mask_ratio 0.0 \
    --save_representations \
    --max_samples_per_class 500
```

#### TrafficTransformer Clustering
```bash
python rp_clustering.py \
    --model_path "fine_tuned_models/traffic_transformer_finetuned.pth" \
    --model_type traffic_transformer \
    --data_path "YaTC_datasets/USTC-TFC2016_MFR" \
    --output_dir "./traffic_transformer_clustering_results" \
    --n_clusters 10 \
    --algorithm kmeans \
    --save_representations \
    --max_samples_per_class 500
```

#### DBSCAN Clustering
```bash
python rp_clustering.py \
    --model_path "fine_tuned_models/traffic_transformer_finetuned.pth" \
    --model_type traffic_transformer \
    --data_path "YaTC_datasets/USTC-TFC2016_MFR" \
    --output_dir "./dbscan_clustering_results" \
    --algorithm dbscan \
    --eps 0.5 \
    --min_samples 5 \
    --save_representations
```

## Command Line Arguments

### Required Arguments
- `--model_path`: Path to the trained model checkpoint
- `--data_path`: Path to input data (YaTC dataset directory or .npy/.pt file)

### Optional Arguments
- `--model_type`: Model type (`mae` or `traffic_transformer`, default: `mae`)
- `--output_dir`: Output directory for results (default: `./clustering_results`)
- `--n_clusters`: Number of clusters for K-means/hierarchical (default: 5)
- `--algorithm`: Clustering algorithm (`kmeans`, `dbscan`, `hierarchical`, default: `kmeans`)
- `--mask_ratio`: Masking ratio for MAE encoder (default: 0.0, only for MAE)
- `--eps`: Epsilon parameter for DBSCAN (default: 0.5)
- `--min_samples`: Min samples parameter for DBSCAN (default: 5)
- `--save_representations`: Save extracted representations to file
- `--max_samples_per_class`: Maximum samples per class for YaTC datasets
- `--random_state`: Random state for reproducibility (default: 42)

## Model Representations

### MAE Encoder Representations
- **Source**: Pre-trained MAE model encoder
- **Extraction**: Uses `forward_encoder()` method
- **Features**: CLS token representation from encoder output
- **Dimensions**: (N, embed_dim) where embed_dim=192

### TrafficTransformer Flow-Level Representations
- **Source**: Fine-tuned TrafficTransformer model
- **Extraction**: Flow-level representations before final layernorm
- **Features**: Average of packet-level CLS tokens across 5 packets
- **Dimensions**: (N, embed_dim) where embed_dim=192

## Clustering Algorithms

### K-means
- **Parameters**: `n_clusters`
- **Use Case**: When you know the expected number of clusters
- **Advantages**: Fast, simple, works well with spherical clusters

### DBSCAN
- **Parameters**: `eps`, `min_samples`
- **Use Case**: When you don't know the number of clusters
- **Advantages**: Handles noise, finds clusters of arbitrary shapes

### Hierarchical Clustering
- **Parameters**: `n_clusters`
- **Use Case**: When you want to understand cluster hierarchy
- **Advantages**: Provides dendrogram, can handle non-spherical clusters

## Evaluation Metrics

### Internal Metrics (No Ground Truth Required)
- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1, higher is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster dispersion (higher is better)
- **Davies-Bouldin Score**: Average similarity measure of clusters (lower is better)

### External Metrics (With Ground Truth)
- **Adjusted Rand Score**: Measures clustering vs true labels (-1 to 1, higher is better)
- **Normalized Mutual Information**: Information-theoretic measure (0 to 1, higher is better)

## Output Files

The clustering pipeline generates several output files:

### Results
- `clustering_results.json`: Main results with labels and metrics
- `*_representations.npy`: Extracted representations (if `--save_representations`)

### Visualizations
- `cluster_plot.png`: PCA projection of clustering results
- `cluster_statistics.png`: Cluster size distribution
- `tsne_*_representations.png`: T-SNE visualization (from example scripts)

### Example Scripts Output
- `clustering_summary.json`: Summary of all clustering algorithms
- `clustering_vs_true_labels.png`: Comparison with ground truth labels

## Example Scripts

### TrafficTransformer Clustering Example
```bash
python traffic_transformer_clustering_example.py
```

This script:
1. Loads YaTC dataset
2. Extracts TrafficTransformer flow-level representations
3. Tests multiple clustering algorithms
4. Compares results with true labels
5. Generates comprehensive visualizations

### MAE Clustering Example
```bash
python yaTC_clustering_example.py
```

This script:
1. Loads YaTC dataset
2. Extracts MAE encoder representations
3. Tests multiple clustering algorithms
4. Compares results with true labels
5. Generates comprehensive visualizations

### Usage Examples
```bash
python usage_example.py
```

Shows various command-line examples for different use cases.

## Data Format

### YaTC Dataset
- **Structure**: Directory with train/val/test splits
- **Format**: Images in YaTC format (40x200 grayscale)
- **Loading**: Automatically handled by `data_loader.py`

### Custom Data
- **Format**: NumPy array (.npy) or PyTorch tensor (.pt/.pth)
- **Shape**: (N, 1, H, W) where:
  - N: number of samples
  - 1: single channel (grayscale)
  - H, W: height and width (e.g., 40x200 for YaTC format)

## Model Loading

The pipeline supports various checkpoint formats:
- Direct state dict
- Checkpoint with 'model' key
- Checkpoint with 'state_dict' key
- PyTorch 2.6+ weights_only format

## Performance Tips

1. **Limit Samples**: Use `--max_samples_per_class` for faster processing
2. **GPU Usage**: Automatically uses CUDA if available
3. **Batch Processing**: Representations are extracted efficiently
4. **Memory Management**: Uses `torch.no_grad()` for inference

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Check checkpoint format and model architecture
2. **Memory Issues**: Reduce `--max_samples_per_class` or use CPU
3. **Data Format Errors**: Ensure data is in correct shape (N, 1, H, W)
4. **Matplotlib Errors**: Use compatible colormaps (rainbow, tab10, etc.)

### Debugging

- Enable verbose output in T-SNE visualization
- Check representation shapes and statistics
- Verify model loading with print statements
- Monitor memory usage during processing

## Comparison Between Models

| Aspect | MAE | TrafficTransformer |
|--------|-----|-------------------|
| **Representation Type** | Encoder CLS token | Flow-level average |
| **Training Stage** | Pre-trained | Fine-tuned |
| **Masking Support** | Yes | No |
| **Packet Processing** | Single image | 5-packet sequence |
| **Use Case** | Unsupervised learning | Supervised learning |

## Future Enhancements

- Support for other model architectures
- Additional clustering algorithms (Spectral, Gaussian Mixture)
- Interactive visualizations
- Batch processing for large datasets
- Cross-validation for clustering evaluation

## Citation

If you use this clustering pipeline in your research, please cite the original YaTC paper and this implementation.

## License

This code is provided for research purposes. Please check the original repository licenses for model weights and datasets. 