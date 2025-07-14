import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import random

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_YaTC

# ===================== Contrastive Dataset =====================
def add_noise(arr, noise_ratio=0.5):  # 从0.3增加到0.5
    arr = arr.copy()
    for i in range(5):
        payload_start = 80 + i * 320
        payload_end = payload_start + 240
        payload = arr[payload_start:payload_end]
        last_nonzero_idx = None
        for j in range(len(payload)-1, -1, -1):
            if payload[j] != 0:
                last_nonzero_idx = j
                break
        if last_nonzero_idx is not None:
            for idx in range(240):
                payload[idx] = random.randint(0, 255)
            
        arr[payload_start:payload_end] = payload
    return arr

class ContrastiveDataset(Dataset):
    def __init__(self, root, transform=None, noise_ratio=0.5):  # 从0.3增加到0.5
        self.samples = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith('.png'):
                    self.samples.append(os.path.join(dirpath, f))
        self.transform = transform
        self.noise_ratio = noise_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path)
        
        # Original image (anchor)
        if self.transform:
            img = self.transform(img)
        
        # Create positive sample with noise
        arr = np.array(Image.open(img_path)).flatten()
        arr_noised = add_noise(arr, self.noise_ratio)
        img_noised = Image.fromarray(arr_noised.reshape((40, 40)).astype(np.uint8))
        
        # Apply additional augmentations to positive sample
        if self.transform:
            img_noised = self.transform(img_noised)
        return img, img_noised

# ===================== Triplet Loss with Unsupervised Clustering =====================
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

def find_hardest_negative(anchor_features, positive_features, negative_features, k_clusters=2):
    """
    Use unsupervised clustering to find the hardest negative sample.
    With 2 clusters, use samples from the other cluster as negatives.
    
    Args:
        anchor_features: [B, dim] - anchor samples
        positive_features: [B, dim] - positive samples (perturbed versions)
        negative_features: [B, dim] - negative samples (other samples in batch)
        k_clusters: int - number of clusters for K-means (default=2)
    
    Returns:
        hardest_negative_indices: [B] - indices of hardest negative for each anchor
    """
    B = anchor_features.shape[0]
    device = anchor_features.device
    
    # Move to CPU for sklearn clustering
    anchor_cpu = anchor_features.detach().cpu().numpy()
    positive_cpu = positive_features.detach().cpu().numpy()
    negative_cpu = negative_features.detach().cpu().numpy()
    
    hardest_negative_indices = []
    
    for i in range(B):
        # Combine anchor, positive, and negative features for clustering
        combined_features = np.vstack([
            anchor_cpu[i:i+1],      # [1, dim]
            positive_cpu[i:i+1],    # [1, dim]
            negative_cpu            # [B, dim]
        ])
        
        # Perform K-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(combined_features)
        
        # Find which cluster contains the anchor
        anchor_cluster = cluster_labels[0]
        
        # Find negatives that are in the OTHER cluster (not the anchor's cluster)
        # other_cluster_mask = (cluster_labels[2:] != anchor_cluster)  # Skip anchor and positive
        # other_cluster_indices = np.where(other_cluster_mask)[0]
        same_cluster_mask = (cluster_labels[2:] == anchor_cluster)  # Skip anchor and positive
        same_cluster_indices = np.where(same_cluster_mask)[0]
        
        if len(same_cluster_indices) > 0:
            # Among other cluster negatives, find the one closest to anchor
            # (this will be the hardest negative from the other cluster)
            anchor_feat = anchor_cpu[i:i+1]  # [1, dim]
            other_cluster_negatives = negative_cpu[same_cluster_indices]  # [N, dim]
            
            # Calculate distances
            distances = np.linalg.norm(other_cluster_negatives - anchor_feat, axis=1)
            closest_idx = np.argmin(distances)
            hardest_negative_idx = same_cluster_indices[closest_idx]
        else:
            # If no negatives in other cluster, pick the closest one overall
            anchor_feat = anchor_cpu[i:i+1]  # [1, dim]
            distances = np.linalg.norm(negative_cpu - anchor_feat, axis=1)
            hardest_negative_idx = np.argmin(distances)
        
        hardest_negative_indices.append(hardest_negative_idx)
    
    return torch.tensor(hardest_negative_indices, device=device)

def triplet_loss(anchor_features, positive_features, margin=1.0, k_clusters=2, debug=False):
    """
    Triplet loss with semi-hard negative mining.
    For each anchor, select a negative with distance > pos_dist and < pos_dist+margin (semi-hard); if none, use the hardest negative.
    Args:
        anchor_features: [B, dim] - anchor samples
        positive_features: [B, dim] - positive samples (perturbed versions)
        margin: float - margin for triplet loss
        k_clusters: int - number of clusters for K-means (unused here)
        debug: bool - whether to print debug info
    Returns:
        loss: scalar - triplet loss
    """
    B = anchor_features.shape[0]
    anchor_features = F.normalize(anchor_features, dim=1)
    positive_features = F.normalize(positive_features, dim=1)
    total_loss = torch.tensor(0.0, device=anchor_features.device, requires_grad=True)
    valid_triplets = 0
    pos_distances = []
    neg_distances = []
    triplet_losses = []
    semi_hard_found = 0
    for i in range(B):
        anchor = anchor_features[i:i+1]
        positive = positive_features[i:i+1]
        negatives = torch.cat([anchor_features[:i], anchor_features[i+1:]], dim=0)
        if negatives.shape[0] == 0:
            continue
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)  # [1]
        neg_dists = torch.sum((anchor - negatives) ** 2, dim=1)  # [B-1]
        # Semi-hard mask
        mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + margin)
        semi_hard_negatives = neg_dists[mask]
        if semi_hard_negatives.numel() > 0:
            neg_dist = semi_hard_negatives.min().unsqueeze(0)
            semi_hard_found += 1
        else:
            neg_dist = neg_dists.min().unsqueeze(0)
        pos_distances.append(pos_dist.item())
        neg_distances.append(neg_dist.item())
        triplet_loss_i = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        triplet_losses.append(triplet_loss_i.item())
        total_loss = total_loss + triplet_loss_i
        valid_triplets += 1
    feature_norms = torch.norm(anchor_features, dim=1)
    regularization_loss = torch.mean(torch.clamp(feature_norms - 0.8, min=0.0))
    total_loss = total_loss + 0.01 * regularization_loss
    if debug and valid_triplets > 0:
        avg_pos_dist = np.mean(pos_distances)
        avg_neg_dist = np.mean(neg_distances)
        avg_triplet_loss = np.mean(triplet_losses)
        print(f"[DEBUG] Avg pos_dist: {avg_pos_dist:.4f}, Avg neg_dist: {avg_neg_dist:.4f}, Avg triplet_loss: {avg_triplet_loss:.4f}")
        print(f"[DEBUG] Semi-hard negatives found: {semi_hard_found}/{valid_triplets}")
        active_triplets = sum(1 for loss in triplet_losses if loss > 0)
        print(f"[DEBUG] Active triplets: {active_triplets}/{len(triplet_losses)}")
        if np.isnan(avg_pos_dist) or np.isnan(avg_neg_dist):
            print("[DEBUG] WARNING: NaN detected in distances!")
    if valid_triplets > 0:
        return total_loss / valid_triplets
    else:
        return torch.tensor(0.0, device=anchor_features.device, requires_grad=True)

def find_hardest_negative_single(anchor, positive, negatives, k_clusters=2, debug=False):
    """
    Find hardest negative for a single anchor using clustering.
    Args:
        anchor: [1, dim] - single anchor sample
        positive: [1, dim] - single positive sample
        negatives: [N, dim] - negative samples
        k_clusters: int - number of clusters for K-means
        debug: bool - whether to print debug info
    Returns:
        hardest_negative_idx: int - index of hardest negative
    """
    device = anchor.device
    anchor_cpu = anchor.detach().cpu().numpy()  # [1, dim]
    positive_cpu = positive.detach().cpu().numpy()  # [1, dim]
    negatives_cpu = negatives.detach().cpu().numpy()  # [N, dim]
    combined_features = np.vstack([anchor_cpu, positive_cpu, negatives_cpu])  # [2+N, dim]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(combined_features)
    anchor_cluster = cluster_labels[0]
    positive_cluster = cluster_labels[1]
    negatives_clusters = cluster_labels[2:]
    if debug:
        print(f"[DEBUG] Anchor cluster: {anchor_cluster}, Positive cluster: {positive_cluster}")
        print(f"[DEBUG] Negatives clusters: {negatives_clusters}")
        print(f"[DEBUG] All cluster labels: {cluster_labels}")
    other_cluster_mask = (cluster_labels[2:] != anchor_cluster)
    other_cluster_indices = np.where(other_cluster_mask)[0]
    if len(set(cluster_labels)) == 1 and debug:
        print("[DEBUG] WARNING: All samples fell into one cluster!")
    if len(other_cluster_indices) > 0:
        anchor_feat = anchor_cpu  # [1, dim]
        other_cluster_negatives = negatives_cpu[other_cluster_indices]  # [M, dim]
        distances = np.linalg.norm(other_cluster_negatives - anchor_feat, axis=1)
        closest_idx = np.argmin(distances)
        hardest_negative_idx = other_cluster_indices[closest_idx]
        if debug:
            print(f"[DEBUG] Distances to other-cluster negatives: {distances}")
            print(f"[DEBUG] Selected hardest negative idx (other cluster): {hardest_negative_idx}, distance: {distances[closest_idx]}")
    else:
        anchor_feat = anchor_cpu  # [1, dim]
        distances = np.linalg.norm(negatives_cpu - anchor_feat, axis=1)
        hardest_negative_idx = np.argmin(distances)
        if debug:
            print(f"[DEBUG] Distances to all negatives: {distances}")
            print(f"[DEBUG] Selected hardest negative idx (all): {hardest_negative_idx}, distance: {distances[hardest_negative_idx]}")
    return hardest_negative_idx

# ===================== NT-Xent Loss =====================
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    # print(z1.shape, z2.shape)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2N, dim]
    sim = torch.mm(z, z.t()) / temperature  # [2N, 2N]
    mask = torch.eye(2*N, dtype=torch.bool, device=z1.device)
    # Use a smaller value that fits in half precision (float16)
    # float16 range is approximately [-65504, 65504]
    sim = sim.masked_fill(mask, -1e4)
    positives = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)], dim=0).to(z1.device)
    loss = F.cross_entropy(sim, positives)
    return loss

# ===================== Argument Parser =====================
def get_args_parser():
    parser = argparse.ArgumentParser('YaTC contrastive finetuning', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--input_size', default=40, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=5e-5)
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--finetune', default='./3t1t_output_dir/checkpoint-step150000.pth')
    parser.add_argument('--data_path', default='./YaTC_datasets/USTC-TFC2016_MFR', type=str)
    parser.add_argument('--output_dir', default='./contrastive_output_dir', type=str)
    parser.add_argument('--log_dir', default='./contrastive_output_dir', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--noise_ratio', type=float, default=0.7)  # 从1.0调整到0.7
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--margin', type=float, default=1, help='Margin for triplet loss (ensures continued learning)')
    parser.add_argument('--k_clusters', type=int, default=3, help='Number of clusters for K-means')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--debug', action='store_true')
    return parser

# ===================== Main Training Loop =====================
def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    mean = [0.5]
    std = [0.5]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = ContrastiveDataset(os.path.join(args.data_path, 'train'), transform=transform, noise_ratio=args.noise_ratio)

    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # type: ignore
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # ========== Model ========== #
    model = models_YaTC.TraFormer_YaTC(num_classes=20)  # Specify num_classes for YaTC dataset
    # 加载MAE预训练权重
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = model.state_dict()
        # 只加载encoder部分
        for k in list(checkpoint_model.keys()):
            if k.startswith('decoder') or k.startswith('mask_token'):
                del checkpoint_model[k]
        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=[],
                                        layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    print("Start contrastive training for {} epochs".format(args.epochs))
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(epoch)
        model.train(True)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 20
        optimizer.zero_grad()
        for data_iter_step, (img1, img2) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                # 使用MAEContrast模型进行对比学习
                # 直接调用forward方法，返回投影后的特征
                # anchor_features = model_without_ddp(img1)  # [B, projection_dim] - anchor samples
                # positive_features = model_without_ddp(img2)  # [B, projection_dim] - positive samples (perturbed)
                anchor_features = model_without_ddp.forward_features(img1)
                positive_features = model_without_ddp.forward_features(img2)
                # print(anchor_features.shape, positive_features.shape)
                # Use triplet loss with other samples in batch as negatives
                loss = triplet_loss(anchor_features, positive_features, 
                                  margin=args.margin, k_clusters=args.k_clusters, debug=args.debug)
            loss_value = loss.item()
            if not np.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                exit(1)
            loss /= args.accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % args.accum_iter == 0)
            if (data_iter_step + 1) % args.accum_iter == 0:
                optimizer.zero_grad()
            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            if log_writer is not None and (data_iter_step + 1) % args.accum_iter == 0:
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        if args.output_dir and (epoch+1) % 5 == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model_without_ddp.state_dict(), os.path.join(args.output_dir, f'contrastive_model_epoch{epoch+1}.pth'))
            print(f"✓ Model state dict saved to: {os.path.join(args.output_dir, f'contrastive_model_epoch{epoch+1}.pth')}")
    # Save the final model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model_without_ddp.state_dict(), os.path.join(args.output_dir, 'contrastive_model.pth'))
        print(f"✓ Model state dict saved to: {os.path.join(args.output_dir, 'contrastive_model.pth')}")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('YaTC contrastive finetuning', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args) 