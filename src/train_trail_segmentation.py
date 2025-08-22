import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import segmentation_models_pytorch as smp
from trail_dataset import PerTrailDataset
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=6,
    classes=1,
)


def compute_iou_matrix(gt_mask, pred_mask, num_classes):
    iou_matrix = np.zeros((num_classes, num_classes))
    for gt_class in range(1, num_classes):
        gt_bin = (gt_mask == gt_class)
        for pred_class in range(1, num_classes):
            pred_bin = (pred_mask == pred_class)
            intersection = torch.logical_and(gt_bin, pred_bin).sum()
            union = torch.logical_or(gt_bin, pred_bin).sum()
            if union > 0:
                iou_matrix[gt_class-1, pred_class-1] = intersection / union
    return iou_matrix

def visualize_binary_prediction(image, pred_mask, gt_mask, title=None):
    plt.figure(figsize=(7, 3.5))

    if title:
        plt.suptitle(title)

    plt.subplot(1, 3, 1)
    plt.imshow(torch.sigmoid(image.permute(1, 2, 0)).cpu().numpy())
    plt.title("Input")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask.squeeze().cpu().numpy(), cmap="gray")
    plt.title("Prediction")

    plt.subplot(1, 3, 3)
    plt.imshow(gt_mask.squeeze().cpu().numpy(), cmap="gray")
    plt.title("Ground Truth")

    plt.show()

if __name__ == "__main__":
    # Paths to your image and mask folders
    image_dir = "../dataset/images_final"
    mask_dir = "../dataset/masks"
    manifest_file = "../dataset/maps_manifest.yaml"

    val_every_n = 1

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Create dataset
    train_dataset = PerTrailDataset(image_dir=image_dir, mask_dir=mask_dir, manifest_file=manifest_file, split_file='../dataset/split.txt', split_name='train', transform=train_transform)
    val_dataset = PerTrailDataset(image_dir=image_dir, mask_dir=mask_dir, manifest_file=manifest_file, split_file='../dataset/split.txt', split_name='val', transform=val_transform)
    test_dataset = PerTrailDataset(image_dir=image_dir, mask_dir=mask_dir, manifest_file=manifest_file, split_file='../dataset/split.txt', split_name='test', transform=val_transform)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)


    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    writer = SummaryWriter("../runs/trail_segmentation_new_masks")

    os.makedirs('../checkpoints', exist_ok=True)

    best_val_loss = float('inf')

    for epoch in tqdm(range(600)):
        model.train()
        total_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1).float()

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        if epoch % val_every_n == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    masks = masks.unsqueeze(1).float()
                    preds = model(images)
                    pred_classes = (torch.sigmoid(preds) > 0.5).float()
                    loss = criterion(preds, masks)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            torch.save(model.state_dict(), f'../checkpoints/trail_segmentation_new_masks_epoch_{epoch}.pth')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), '../checkpoints/trail_segmentation_new_masks_best.pth')

