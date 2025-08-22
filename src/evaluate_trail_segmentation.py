import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np
from trail_dataset import PerTrailDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from train_trail_segmentation import visualize_binary_prediction
from PIL import Image
import os

# Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "../checkpoints/trail_segmentation_new_masks_best.pth"
OUTPUT_DIR = "../outputs/"
IMG_SIZE = 1024
BATCH_SIZE = 1
VISUALIZE_DEBUG = False
SAVE_PNG = True
SAVE_NPZ = True

if SAVE_PNG:
    os.makedirs(os.path.join(OUTPUT_DIR, "predicted_masks_png"), exist_ok=True)
if SAVE_NPZ:
    os.makedirs(os.path.join(OUTPUT_DIR, "predicted_masks_npz"), exist_ok=True)

# Load model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=6,
    classes=1,
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Dataset & Dataloader
test_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

image_dir = "../dataset/images_final"
mask_dir = "../dataset/masks"
manifest_file = "../dataset/maps_manifest.yaml"

test_dataset = PerTrailDataset(image_dir=image_dir, mask_dir=mask_dir, manifest_file=manifest_file, split_file='../dataset/split.txt', split_name='test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load map IDs and trail IDs for test split
split_txt_path = "../dataset/split.txt"
map_trail_ids = []

with open(split_txt_path, "r", encoding="utf-8") as f:
    for line in f:
        split, filename, trail_id = line.strip().split()
        if split == "test":
            map_id = os.path.splitext(filename)[0]  # removes .png
            map_trail_ids.append((map_id, trail_id))

# IoU computation
def compute_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection
    iou = intersection / (union + 1e-6)
    return iou.item()



ious = []
with torch.no_grad():
    j = 0
    for images, masks in test_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE).unsqueeze(1).float()
        logits = model(images)
        probs = torch.sigmoid(logits)

        for i in range(images.size(0)):
            iou = compute_iou(probs[i], masks[i])
            ious.append(iou)

            # Get correct map/trail name
            map_id, trail_id = map_trail_ids[j]
            out_name = f"{map_id}_{trail_id}"

            prediction = (probs[i] > 0.5).int()
            if SAVE_PNG:
                Image.fromarray((255 - (prediction.squeeze().cpu().numpy() * 255)).astype(np.uint8)
                                ).save(os.path.join(OUTPUT_DIR, "predicted_masks_png", f"{out_name}.png"))
            if SAVE_NPZ:
                np.savez_compressed(os.path.join(OUTPUT_DIR, "predicted_masks_npz", f"{out_name}.npz"),
                                    mask=prediction.squeeze().cpu().numpy().astype(np.uint8))
            if VISUALIZE_DEBUG:
                visualize_binary_prediction(images[i][:3], prediction, masks[i], title=f"IoU: {iou:.4f}")
            j += 1

# --- Final results ---
ious_array = np.array(ious)
mean_iou = np.mean(ious_array)
median_iou = np.median(ious_array)
std_iou = np.std(ious_array)
q1 = np.percentile(ious_array, 25)
q3 = np.percentile(ious_array, 75)
iqr = q3 - q1

# Define outliers using 1.5 * IQR rule
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = ious_array[(ious_array < lower_bound) | (ious_array > upper_bound)]

print(f"Results for checkpoint epoch")
print(f"Mean IoU: {mean_iou:.4f}")
print(f"Median IoU: {median_iou:.4f}")
print(f"Standard Deviation: {std_iou:.4f}")
print(f"IQR: {iqr:.4f} (Q1 = {q1:.4f}, Q3 = {q3:.4f})")
print(f"Outliers ({len(outliers)}): {outliers}")
