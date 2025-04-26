import os
import time
import glob
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image

# Enable benchmark mode in cudnn for potential speedup
cudnn.benchmark = True
plt.ion()   # interactive mode for visualization


def get_data_transforms():
    """
    Create data augmentation and normalization transforms for train, val, and test sets.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((400, 224)),
            transforms.Pad((0, 88)),  # Pad width to square
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((400, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((400, 224)),
            transforms.Pad((0, 88)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


def get_dataloaders(data_dir, batch_size=16, num_workers=4):
    """
    Create ImageFolder datasets and DataLoaders for train, val, and test splits.
    """
    transforms = get_data_transforms()
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transforms[x])
        for x in ['train', 'val', 'test']
    }
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': torch.utils.data.DataLoader(
            image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': torch.utils.data.DataLoader(
            image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names


def imshow(inp, title=None):
    """
    Display a tensor as an image after un-normalizing.
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title, fontsize=10)
    plt.pause(0.001)


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
                device, num_epochs=70, save_dir='saved_models'):
    """
    Train and validate the model, saving the best weights based on validation accuracy.
    """
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_model_params.pt')
    best_acc = 0.0
    since = time.time()

    # Save initial weights (optional)
    torch.save(model.state_dict(), best_model_path)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}\n' + '-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_path)
        print()

    elapsed = time.time() - since
    print(f'Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(torch.load(best_model_path))
    return model


def visualize_results(model, dataloader, class_names, device, num_images=6):
    """
    Display a few predictions from a dataloader side-by-side.
    """
    model.eval()
    images_so_far = 0
    cols = 3
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=(cols * 6, rows * 6))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                images_so_far += 1
                ax = plt.subplot(rows, cols, images_so_far)
                ax.axis('off')
                title_color = 'green' if preds[j] == labels[j] else 'red'
                ax.set_title(f'Pred: {class_names[preds[j]]}\nTrue: {class_names[labels[j]]}',
                             color=title_color)
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.show()
                    return
    plt.tight_layout()
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train MobileNetV3 on a custom dataset')
    parser.add_argument('--data-dir', type=str, default='Dataset5AprV1_split',
                        help='Root directory of dataset with train/val/test subfolders')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--num-workers', type=int, default=6, help='Number of dataloader workers')
    parser.add_argument('--epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=9.5e-4, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='saved_models', help='Directory to save models')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample predictions')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders, dataset_sizes, class_names = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # Load pretrained MobileNetV3 and modify classifier
    model = models.mobilenet_v3_small(weights='DEFAULT')
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Train and evaluate
    best_model = train_model(
        model, dataloaders, dataset_sizes,
        criterion, optimizer, scheduler,
        device, num_epochs=args.epochs,
        save_dir=args.output_dir
    )

    # Visualize predictions
    if args.visualize:
        visualize_results(best_model, dataloaders['val'], class_names, device)


if __name__ == '__main__':
    main()
