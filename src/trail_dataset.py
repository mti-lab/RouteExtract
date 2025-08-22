import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random
import torch
from dataset.download_maps import load_config_file

class PerTrailDataset(Dataset):
    def __init__(self, image_dir, mask_dir, manifest_file, split_file, split_name, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.manifest = load_config_file(manifest_file, return_dict=True)
        self.transform = transform

        self.entries = []
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                split, map_filename, trail_id = parts
                if split == split_name:
                    self.entries.append((map_filename, trail_id))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        map_filename, trail_id = self.entries[idx]
        base_name = os.path.splitext(map_filename)[0]

        image_path = os.path.join(self.image_dir, map_filename)
        mask_path = os.path.join(self.mask_dir, f"{base_name}_{trail_id}.npz")

        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))

        # Load binary mask
        mask = np.load(mask_path)["mask"]

        if self.transform:
            augmented = self.transform(image=image, mask=mask.astype(np.uint8))
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask.astype(np.uint8))

        # Load trail RGB and normalize
        trail_color = self.manifest[base_name]["trails"][int(trail_id[-2:])-1]["color"]
        trail_rgb = torch.tensor(trail_color, dtype=torch.float32) / 255.0  # shape: (3,)
        trail_rgb = trail_rgb.view(3, 1, 1).expand(-1, image.shape[1], image.shape[2])

        # If image was normalized using ImageNet stats, do the same to trail_rgb
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        trail_rgb = (trail_rgb - imagenet_mean) / imagenet_std

        # Combine image and trail RGB
        combined = torch.cat([image, trail_rgb], dim=0)  # shape: (6, H, W)

        return combined, mask


def generate_split_entries(image_dir, manifest_file, output_file_path, train_ratio=0.7, val_ratio=0.1,
                           test_ratio=0.2, seed=42):
    random.seed(seed)

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    base_names = [os.path.splitext(f)[0] for f in image_files]
    random.shuffle(base_names)

    total = len(base_names)
    n_train = int(total * train_ratio / (train_ratio + val_ratio + test_ratio))
    n_val = int(total * val_ratio / (train_ratio + val_ratio + test_ratio))
    n_test = int(total * test_ratio / (train_ratio + val_ratio + test_ratio))

    split_assignments = {
        "train": base_names[:n_train],
        "val": base_names[n_train:n_train + n_val],
        "test": base_names[n_train + n_val:]
    }

    with open(output_file_path, "w") as f:
        for split_name, base_list in split_assignments.items():
            for base in base_list:
                num_trails = len(load_config_file(manifest_file, return_dict=True)[base]["trails"])

                for trail_id in range(num_trails):
                    map_filename = base + ".png"  # adjust extension if needed
                    f.write(f"{split_name} {map_filename} trail{trail_id+1:02d}\n")

    print(f"Split entries saved to {output_file_path}")
    for split_name in ["train", "val", "test"]:
        count = sum(1 for line in open(output_file_path) if line.startswith(split_name))
        print(f"{split_name.capitalize()}: {count} trail entries")


# Example usage:
if __name__ == "__main__":
    image_dir = "../dataset/images_final"
    manifest_file = "../dataset/maps_manifest.yaml"
    output_file_path = "../dataset/split.txt"
    generate_split_entries(image_dir, manifest_file, output_file_path)
