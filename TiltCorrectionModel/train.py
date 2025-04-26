#!/usr/bin/env python3
"""
train.py: Training script for EfficientNet-B0 with super‑charged data augmentations.
"""
import os
import time
import glob
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

# Enable benchmark mode in cuDNN for optimized performance
cudnn.benchmark = True
# Turn on interactive plotting
plt.ion()


def imshow(tensor, title=None):
    """Un-normalize and display a batch of images."""
    image = tensor.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    if title:
        plt.title(title, fontsize=10)
    plt.pause(0.001)


class ResizePad:
    """
    Resize an image to fit within a fixed size and pad the remaining area with a solid background.
    """
    def __init__(self, size, fill_mean=(0.485, 0.456, 0.406)):
        self.W, self.H = size
        r, g, b = (int(c * 255) for c in fill_mean)
        self.fill = (r, g, b)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(self.W / w, self.H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.BICUBIC)
        background = Image.new("RGB", (self.W, self.H), self.fill)
        paste_x = (self.W - new_w) // 2
        paste_y = (self.H - new_h) // 2
        background.paste(img_resized, (paste_x, paste_y))
        return background


def get_data_transforms():
    """
    Return data augmentation pipelines for training, validation, and test sets.
    """
    # Photometric augmentations (color jitter, grayscale, blur)
    photo_augs = [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ]

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])

    train = transforms.Compose([
        ResizePad((224, 400)),
        *photo_augs,
        transforms.RandomAffine(degrees=5,
                                translate=(0.1, 0.1),
                                scale=(0.8, 1.2),
                                fill=(int(0.485*255), int(0.456*255), int(0.406*255))),
        transforms.RandomPerspective(distortion_scale=0.2,
                                     p=0.5,
                                     fill=(int(0.485*255), int(0.456*255), int(0.406*255))),
        transforms.ToTensor(),
        normalize,
    ])

    val_test = transforms.Compose([
        ResizePad((224, 400)),
        transforms.ToTensor(),
        normalize,
    ])

    return {'train': train, 'val': val_test, 'test': val_test}


def load_dataloaders(data_dir, batch_size=16, num_workers=6):
    """
    Create PyTorch dataloaders for train, val, and test splits.
    """
    transforms_dict = get_data_transforms()
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transforms_dict[x])
        for x in ['train', 'val', 'test']
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size,
            shuffle=(x == 'train'), num_workers=num_workers
        ) for x in ['train', 'val', 'test']
    }
    sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    classes = image_datasets['train'].classes
    return dataloaders, sizes, classes


def train_model(model, criterion, optimizer, scheduler,
                dataloaders, dataset_sizes, device,
                num_epochs=100, save_dir='saved_models'):
    """
    Train and validate a model, saving the best weights based on validation accuracy.
    """
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, 'best_model.pt')
    best_acc = 0.0
    start_time = time.time()

    # Optional: save initial weights
    torch.save(model.state_dict(), best_path)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 20)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}  "
                  f"Acc: {epoch_acc:.4f}")

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_path)
        print()

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"Best validation Acc: {best_acc:.4f}")

    # Load best weights
    model.load_state_dict(torch.load(best_path))
    return model


def visualize_dataset(model, dataloaders, classes, device,
                      phase='val', num_images=6):
    """Visualize predictions on a few samples from the dataset."""
    model.eval()
    images_shown = 0
    cols = 3
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=(cols * 5, rows * 5))

    with torch.no_grad():
        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                if images_shown >= num_images:
                    plt.tight_layout()
                    plt.show()
                    return
                images_shown += 1
                ax = plt.subplot(rows, cols, images_shown)
                ax.axis('off')
                title = f"Pred: {classes[preds[i]]}\nTrue: {classes[labels[i]]}"
                color = 'red' if preds[i] != labels[i] else 'green'
                ax.set_title(title, color=color)
                imshow(inputs.cpu().data[i])

    plt.tight_layout()
    plt.show()


def main():
    # Configuration
    data_dir   = '8apr_dataset_split'
    batch_size = 16
    num_epochs_head = 44
    num_epochs_full = 90
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataloaders, dataset_sizes, classes = load_dataloaders(data_dir, batch_size)
    print("Classes:", classes)

    # Initialize EfficientNet-B0 and replace classifier
    model = models.efficientnet_b0(weights='DEFAULT')
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, len(classes))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase 1: Train classifier head only
    print("\n=== Phase 1: Training classifier head only ===")
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer_head = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler_head = lr_scheduler.ReduceLROnPlateau(optimizer_head, mode='min', patience=3, factor=0.5)
    model = train_model(model, criterion, optimizer_head, scheduler_head,
                        dataloaders, dataset_sizes, device,
                        num_epochs=num_epochs_head,
                        save_dir='saved_models/head_only')

    # Phase 2: Fine-tune entire model
    print("\n=== Phase 2: Fine-tuning full model ===")
    for param in model.parameters():
        param.requires_grad = True
    optimizer_full = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler_full = lr_scheduler.ReduceLROnPlateau(optimizer_full, mode='min', patience=5, factor=0.5)
    model = train_model(model, criterion, optimizer_full, scheduler_full,
                        dataloaders, dataset_sizes, device,
                        num_epochs=num_epochs_full,
                        save_dir='saved_models/full_finetune')

    # Visualize a few validation examples
    visualize_dataset(model, dataloaders, classes, device, phase='val')


if __name__ == '__main__':
    main()
