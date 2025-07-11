import os
from PIL import Image
import numpy as np
import random

def add_noise_to_all_payloads(arr, noise_ratio=0.1):
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

def process_dataset(input_dir, output_dir, noise_ratio=0.1):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.png'):
                rel_dir = os.path.relpath(root, input_dir)
                out_dir = os.path.join(output_dir, rel_dir)
                os.makedirs(out_dir, exist_ok=True)
                in_path = os.path.join(root, file)
                out_path = os.path.join(out_dir, file)

                img = Image.open(in_path)
                arr = np.array(img).flatten()
                arr_noised = add_noise_to_all_payloads(arr, noise_ratio)
                arr_noised = arr_noised.reshape((40, 40)).astype(np.uint8)
                Image.fromarray(arr_noised).save(out_path)
                print(f"Processed: {in_path} -> {out_path}")

# 用法示例
process_dataset(
    input_dir='YaTC_datasets/CICIoT2022_MFR/train',
    output_dir='YaTC_datasets/CICIoT2022_MFR/train_noised',
    noise_ratio=0.1
)