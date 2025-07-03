import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from collections import Counter


class YaTCDataset(Dataset):
    """Dataset class for YaTC datasets (CICIoT2022_MFR, ISCXVPN2016_MFR, etc.)"""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None, 
                 max_samples_per_class: Optional[int] = None):
        """
        Args:
            root_dir: Path to dataset root (e.g., 'YaTC_datasets/CICIoT2022_MFR')
            split: 'train' or 'test'
            transform: Optional transform to apply to images
            max_samples_per_class: Maximum number of samples to load per class (None for all)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        
        self.data_path = os.path.join(root_dir, split)
        self.classes = self._get_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
        
    def _get_classes(self) -> List[str]:
        """Get list of class names"""
        if not os.path.exists(self.data_path):
            raise ValueError(f"Split directory not found: {self.data_path}")
        
        classes = [d for d in os.listdir(self.data_path) 
                  if os.path.isdir(os.path.join(self.data_path, d))]
        classes.sort()  # Ensure consistent ordering
        return classes
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load sample paths and their class indices"""
        samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_path, class_name)
            image_files = glob.glob(os.path.join(class_dir, "*.png"))
            
            if self.max_samples_per_class:
                image_files = image_files[:self.max_samples_per_class]
            
            class_idx = self.class_to_idx[class_name]
            samples.extend([(img_path, class_idx) for img_path in image_files])
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, class_idx = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).float()
        
        # Normalize to [0, 1]
        image_tensor = image_tensor / 255.0
        
        # Add channel dimension: (H, W) -> (1, H, W)
        image_tensor = image_tensor.unsqueeze(0)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, class_idx
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of samples across classes"""
        class_counts = Counter([self.samples[i][1] for i in range(len(self.samples))])
        return {self.classes[idx]: count for idx, count in class_counts.items()}


def load_yaTC_data(dataset_path: str, split: str = 'train', 
                   max_samples_per_class: Optional[int] = None,
                   batch_size: int = 32, shuffle: bool = True,
                   num_workers: int = 4) -> Tuple[DataLoader, YaTCDataset]:
    """
    Load YaTC dataset and return DataLoader and Dataset objects
    
    Args:
        dataset_path: Path to dataset (e.g., 'YaTC_datasets/CICIoT2022_MFR')
        split: 'train' or 'test'
        max_samples_per_class: Maximum samples per class
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (DataLoader, Dataset)
    """
    dataset = YaTCDataset(
        root_dir=dataset_path,
        split=split,
        max_samples_per_class=max_samples_per_class
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset


def load_all_data_as_tensor(dataset_path: str, split: str = 'train',
                           max_samples_per_class: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load all data from dataset as tensors (useful for clustering)
    
    Args:
        dataset_path: Path to dataset
        split: 'train' or 'test'
        max_samples_per_class: Maximum samples per class
        
    Returns:
        Tuple of (data_tensor, labels_tensor)
    """
    dataset = YaTCDataset(
        root_dir=dataset_path,
        split=split,
        max_samples_per_class=max_samples_per_class
    )
    
    # Load all data
    all_data = []
    all_labels = []
    
    print(f"Loading {len(dataset)} samples...")
    for i in range(len(dataset)):
        data, label = dataset[i]
        all_data.append(data)
        all_labels.append(label)
        
        if (i + 1) % 1000 == 0:
            print(f"Loaded {i + 1}/{len(dataset)} samples")
    
    # Stack into tensors
    data_tensor = torch.stack(all_data)
    labels_tensor = torch.tensor(all_labels)
    
    print(f"Final data shape: {data_tensor.shape}")
    print(f"Final labels shape: {labels_tensor.shape}")
    
    return data_tensor, labels_tensor


def visualize_dataset_samples(dataset_path: str, split: str = 'train', 
                             samples_per_class: int = 5, save_path: Optional[str] = None):
    """
    Visualize sample images from each class in the dataset
    
    Args:
        dataset_path: Path to dataset
        split: 'train' or 'test'
        samples_per_class: Number of samples to show per class
        save_path: Optional path to save the visualization
    """
    dataset = YaTCDataset(dataset_path, split, max_samples_per_class=samples_per_class)
    
    classes = dataset.classes
    n_classes = len(classes)
    
    fig, axes = plt.subplots(samples_per_class, n_classes, figsize=(2*n_classes, 2*samples_per_class))
    if n_classes == 1:
        axes = axes.reshape(-1, 1)
    if samples_per_class == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx, class_name in enumerate(classes):
        # Get samples for this class
        class_samples = [(i, data, label) for i, (data, label) in enumerate(dataset) if label == class_idx]
        
        for sample_idx in range(min(samples_per_class, len(class_samples))):
            _, data, _ = class_samples[sample_idx]
            
            # Convert tensor to numpy for plotting
            img = data.squeeze().numpy()
            
            ax = axes[sample_idx, class_idx]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{class_name}\nSample {sample_idx+1}')
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def get_dataset_info(dataset_path: str) -> Dict:
    """
    Get information about the dataset
    
    Args:
        dataset_path: Path to dataset
        
    Returns:
        Dictionary with dataset information
    """
    info = {}
    
    for split in ['train', 'test']:
        try:
            dataset = YaTCDataset(dataset_path, split)
            info[split] = {
                'num_samples': len(dataset),
                'num_classes': len(dataset.classes),
                'classes': dataset.classes,
                'class_distribution': dataset.get_class_distribution()
            }
        except Exception as e:
            info[split] = {'error': str(e)}
    
    return info


if __name__ == "__main__":
    # Example usage
    dataset_path = "YaTC_datasets/CICIoT2022_MFR"
    
    # Get dataset information
    print("Dataset Information:")
    info = get_dataset_info(dataset_path)
    for split, split_info in info.items():
        print(f"\n{split.upper()}:")
        if 'error' in split_info:
            print(f"  Error: {split_info['error']}")
        else:
            print(f"  Samples: {split_info['num_samples']}")
            print(f"  Classes: {split_info['num_classes']}")
            print(f"  Class distribution: {split_info['class_distribution']}")
    
    # Visualize samples
    print("\nVisualizing sample images...")
    visualize_dataset_samples(dataset_path, 'train', samples_per_class=3)
    
    # Load data as tensors
    print("\nLoading data as tensors...")
    data, labels = load_all_data_as_tensor(dataset_path, 'train', max_samples_per_class=100)
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {torch.unique(labels).tolist()}") 