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
def add_noise(arr, noise_ratio=0.1):
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
        if last_nonzero_idx is not None and last_nonzero_idx > 0:
            num_bytes_to_noise = int(last_nonzero_idx * noise_ratio)
            if num_bytes_to_noise > 0:
                noise_indices = random.sample(range(last_nonzero_idx), num_bytes_to_noise)
                for idx in noise_indices:
                    payload[idx] = random.randint(0, 255)
        arr[payload_start:payload_end] = payload
    return arr

class ContrastiveDataset(Dataset):
    def __init__(self, root, transform=None, noise_ratio=0.1):
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
        arr = np.array(img).flatten()
        arr_noised = add_noise(arr, self.noise_ratio)
        img_noised = Image.fromarray(arr_noised.reshape((40, 40)).astype(np.uint8))
        if self.transform:
            img = self.transform(img)
            img_noised = self.transform(img_noised)
        return img, img_noised

# ===================== NT-Xent Loss =====================
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2N, dim]
    sim = torch.mm(z, z.t()) / temperature  # [2N, 2N]
    mask = torch.eye(2*N, dtype=torch.bool, device=z1.device)
    sim = sim.masked_fill(mask, -9e15)
    positives = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)], dim=0)
    loss = F.cross_entropy(sim, positives)
    return loss

# ===================== Argument Parser =====================
def get_args_parser():
    parser = argparse.ArgumentParser('YaTC contrastive finetuning', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--input_size', default=40, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=2e-3)
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--finetune', default='./output_dir/pretrained-model.pth')
    parser.add_argument('--data_path', default='./data/ISCXVPN2016_MFR', type=str)
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
    parser.add_argument('--noise_ratio', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.5)
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
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
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
    model = models_YaTC.TraFormer_YaTC(drop_path_rate=args.drop_path)
    # 加载MAE预训练权重
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
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
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    print("Start contrastive training for {} epochs".format(args.epochs))
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
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
            with torch.cuda.amp.autocast():
                # 去掉分类头，直接用主干输出
                feat1 = model.forward_features(img1)
                feat2 = model.forward_features(img2)
                loss = nt_xent_loss(feat1, feat2, temperature=args.temperature)
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