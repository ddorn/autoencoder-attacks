# %%

import dataclasses
from functools import cached_property
import os
from typing import TypeVar

from diffusers import AutoencoderKL
import torch.nn as nn
from torchvision.models import resnet50

from lib.utils import *
from lib.datasets import *


from diffusers import AutoencoderKL
from diffusers.models.vae import DecoderOutput
from torchinfo import summary
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Lambda
from tqdm import tqdm, trange
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50
from datasets import load_dataset, Dataset

from lib.utils import *

__ALL__ = [
    "Task", "Scale",
    "get_mnist", "get_cifar10", "get_imagenet",
    "get_imagenet_networks", "get_mnist_networks", "get_cifar10_networks"]


@dataclasses.dataclass
class Task:
    dataset: Dataset
    labels: list[str]
    preprocess: Compose
    vae: nn.Module | None
    classifier: nn.Module
    name: str

    @cached_property
    def device(self) -> torch.device:
        return next(self.classifier.parameters()).device

    @cached_property
    def sample_images(self) -> torch.tensor:
        return torch.stack([self.preprocess(self.dataset[i]['image']) for i in range(9)]).to(self.device)

    @cached_property
    def sample_labels(self) -> torch.tensor:
        return torch.tensor([self.dataset[i]['label'] for i in range(9)]).to(self.device)

    @cached_property
    def sample_labels_str(self) -> list[str]:
        return [self.labels[label] for label in self.sample_labels]

    @cached_property
    def vae_and_classifier(self) -> nn.Module:
        assert self.vae is not None
        return nn.Sequential(self.vae, self.classifier)

    def __repr__(self) -> str:
        return f"Task({self.name})"

    def iter(self, batch_size: int = 100) -> tuple[torch.tensor, torch.tensor]:
        for batch in self.dataset.iter(batch_size=batch_size):
            images = torch.stack([self.preprocess(image) for image in batch['image']]).to(self.device)
            labels = torch.tensor(batch['label']).to(self.device)
            yield images, labels






class Scale(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self):
        return Scale(-self.mean / self.std, 1 / self.std)



# MNIST

def get_mnist(split="test") -> tuple[torch.utils.data.Dataset, list[str], Compose]:
    """Get MNIST dataset and labels.

    Returns:
        - mnist: MNIST dataset
        - LABELS: list of str labels for each class
        - preprocess: preprocessing pipeline for images. Returns a 1x28x28 image, normalized to [0, 1] on the GPU.
    """
    mnist = load_dataset("mnist", split=split)
    LABELS = mnist.features['label'].names
    preprocess = Compose([ToTensor(), Lambda(lambda x: x.cuda())])

    return mnist, LABELS, preprocess


from lib.mnist import MnistNet, MnistVAE, train_MNIST_classifier, train_MNIST_VAE

def get_mnist_networks(bottleneck_size: int = 3) -> tuple[nn.Module, nn.Module]:
    """Get MNIST autoencoder and classifier.

    Returns:
        - vae: MNIST autoencoder, takes images in [0, 1] and outputs images in [0, 1]
        - classifier: MNIST classifier, takes images in [0, 1] and outputs logits
    """

    try:
        vae = MnistVAE.load(bottleneck_size).cuda().eval()
    except FileNotFoundError as e:
        print(f"No MNIST VAE found at {e.filename}, training a new one...")
        vae = train_MNIST_VAE(bottleneck_size).cuda().eval()

    try:
        classifier = MnistNet.load().cuda().eval()
    except FileNotFoundError as e:
        print(f"No MNIST classifier found at {e.filename}, training a new one...")
        classifier = train_MNIST_classifier().cuda().eval()

    scale = Scale(0.1307, 0.3081)
    vae = nn.Sequential(scale, vae, scale.inverse()).eval()
    classifier = nn.Sequential(scale, classifier).eval()

    return vae, classifier






# CIFAR10

from lib.cifar_classifier import load_classifier as load_cifar10_classifier, train as train_cifar10_classifier
from lib.cifar_vae import Cifar10VAE, train_cifar_vae


def get_cifar10(split="test") -> tuple[torch.utils.data.Dataset, list[str], Compose]:
    """Get CIFAR10 dataset and labels.

    Returns:
        - cifar10: CIFAR10 dataset
        - LABELS: list of str labels for each class
        - preprocess: preprocessing pipeline for images. Returns a 3x32x32 image, normalized to [0, 1] on the GPU.
    """
    cifar10 = load_dataset("cifar10", split=split)
    LABELS = cifar10.features['label'].names
    # Rename img to image
    cifar10 = cifar10.rename_column("img", "image")
    preprocess = Compose([ToTensor(), Lambda(lambda x: x.cuda().to(torch.float16))])

    return cifar10, LABELS, preprocess

def get_cifar10_networks(bottleneck_size: int = 128) -> tuple[nn.Module, nn.Module]:
    """Get CIFAR10 autoencoder and classifier.

    Returns:
        - vae: CIFAR10 autoencoder, takes images in [0, 1] and outputs images in [0, 1]
        - classifier: CIFAR10 classifier, takes images in [0, 1] and outputs logits
    """

    try:
        vae = Cifar10VAE.load(bottleneck_size).cuda().eval()
    except FileNotFoundError as e:
        print(f"No CIFAR10 VAE found at {e.filename}, training a new one...")
        vae = train_cifar_vae(bottleneck_size).cuda().eval()

    try:
        classifier = load_cifar10_classifier().cuda().eval()
    except FileNotFoundError as e:
        print(f"No CIFAR10 classifier found at {e.filename}, training a new one...")
        classifier = train_cifar10_classifier().cuda().eval()

    device = next(classifier.parameters()).device
    # dtype = next(classifier.parameters()).dtype
    dtype = torch.float16

    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device, dtype=dtype).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device, dtype=dtype).view(3, 1, 1)

    scale = Scale(mean, std)
    classifier = nn.Sequential(scale, classifier).eval()
    vae = nn.Sequential(scale, vae, scale.inverse()).eval()

    return vae.to(dtype), classifier.to(dtype)




# ------------ #
# ImageNet     #
# ------------ #

def get_imagenet(split="validation[:10000]") -> tuple[torch.utils.data.Dataset, list[str], Compose]:
    """Get ImageNet dataset and labels.

    Returns:
        - imagenet: ImageNet dataset
        - LABELS: list of str labels for each class
        - preprocess: preprocessing pipeline for images. Returns a 3x256x256 image, normalized to [0, 1] on the GPU.
    """
    imagenet = load_dataset("imagenet-1k", split=split)
    LABELS = imagenet.features['label'].names
    preprocess = Compose([
        Resize(256),
        CenterCrop(256),
        ToTensor(),
        Lambda(lambda x: x.cuda()),
        Lambda(lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1)),
    ])

    return imagenet, LABELS, preprocess


class ImageNetVae(AutoencoderKL):
    """Patch the AutoencoderKL class to return the sample instead of an DecoderOutput object."""
    def forward(self, *args, **kwargs) -> Float[Tensor, "batch channels height width"]:
        out = super().forward(*args, **kwargs)
        return out.sample


def get_imagenet_networks() -> tuple[ImageNetVae, nn.Module]:
    """Get ImageNet autoencoder and classifier.

    Returns:
        - vae: ImageNet autoencoder
        - classifier: ImageNet classifier
    """
    vae = ImageNetVae.from_pretrained("stabilityai/sd-vae-ft-mse").cuda().eval()
    classifier = resnet50(pretrained=True).cuda().eval()

    device = next(classifier.parameters()).device

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    scale = Scale(mean, std)
    # vae = nn.Sequential(scale, vae, scale.inverse())
    classifier = nn.Sequential(scale, classifier)

    return vae.eval(), classifier.eval()


def get_tasks(*skip: Literal["MNIST", "CIFAR10", "ImageNet"]) -> list[Task]:
    """Get all tasks."""
    tasks = []
    if "MNIST" not in skip:
        tasks.append(Task(*get_mnist(), *get_mnist_networks(), "MNIST"))
    if "CIFAR10" not in skip:
        tasks.append(Task(*get_cifar10(), *get_cifar10_networks(), "CIFAR10"))
    if "ImageNet" not in skip:
        tasks.append(Task(*get_imagenet(), *get_imagenet_networks(), "ImageNet"))
    return tasks


def test():
    tasks = get_tasks()

    # Test that the datasets are working
    for task in tasks:
        assert len(task.dataset) > 0
        image, label = task.dataset[0].values()
        image = task.preprocess(image)
        assert image.shape[0] == 3 or image.shape[0] == 1
        assert image.shape[1] == image.shape[2]
        assert image.shape[1] in [28, 32, 256]
        assert image.min() >= 0 and image.max() <= 1

        print(f"Test for dataset {task.name} passed.")

    # Check accuracy of the classifiers
    for task in tasks:
        correct = 0
        total = 0
        with torch.inference_mode():
            for batch in tqdm(task.dataset.iter(batch_size=100), total=len(task.dataset) // 100):
                images = torch.stack([task.preprocess(image) for image in batch['image']]).cuda()
                labels = torch.tensor(batch['label']).cuda()
                predictions = task.classifier(images).argmax(-1)
                correct += (predictions == labels).sum()
                total += len(labels)

        accuracy = correct / total
        assert accuracy > 0.75
        print(f"Accuracy for {task.name}: {accuracy:.1%}")

    print("No tests for the autoencoders yet.")
    print("All tests passed.")

if __name__ == "__main__":
    pass
    # test()
