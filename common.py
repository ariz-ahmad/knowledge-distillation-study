"""Shared ImageNette data loading and evaluation utilities.

ImageNette is a 10-class subset of ImageNet, so a pretrained (ImageNet-1k)
model's logits must be remapped onto ImageNette's class indices to be
evaluated zero-shot. This mapping, and the evaluate/latency helpers built
on it, are shared by teacher.py and distill.py.
"""

import time

import torch
import torchvision.datasets as D
import torchvision.transforms as T

IMAGENETTE_TO_IMAGENET = {
    'n01440764': 0,   'n02102040': 217, 'n02979186': 482,
    'n03000684': 491, 'n03028079': 497, 'n03394916': 566,
    'n03417042': 569, 'n03425413': 571, 'n03445777': 574,
    'n03888257': 701,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VAL_TRANSFORM = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

TRAIN_TRANSFORM = T.Compose([
    T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _loader(path: str, transform, batch_size: int, shuffle: bool):
    dataset = D.ImageFolder(path, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle, pin_memory=False
    )
    idx_to_imagenet = {
        v: IMAGENETTE_TO_IMAGENET[k] for k, v in dataset.class_to_idx.items()
    }
    return loader, idx_to_imagenet


def get_val_loader(path: str = "imagenette2-320/val", batch_size: int = 64):
    return _loader(path, VAL_TRANSFORM, batch_size, shuffle=False)


def get_train_loader(path: str = "imagenette2-320/train", batch_size: int = 32):
    return _loader(path, TRAIN_TRANSFORM, batch_size, shuffle=True)


def evaluate(model, loader, idx_to_imagenet, device) -> float:
    """Zero-shot ImageNet-1k accuracy of `model` on ImageNette, via label remapping."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            imagenet_labels = torch.tensor(
                [idx_to_imagenet[i] for i in labels.tolist()]
            ).to(device)
            correct += (outputs.argmax(1) == imagenet_labels).sum().item()
            total += len(labels)
    return correct / total


def measure_latency(model, device, runs: int = 100) -> float:
    """Mean single-image inference latency in milliseconds."""
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t = time.perf_counter()
            model(dummy)
            times.append(time.perf_counter() - t)
    return sum(times) / len(times) * 1000
