# %%

import torch
from torch import nn, optim
from tqdm import tqdm

from lib.cifar_classifier import load_cifar10
from lib.mnist import BaseVAE
from lib.utils import LossRecord, batch, channel_last, to_plotly

import plotly.graph_objects as go


class Cifar10VAE(BaseVAE):
    def __init__(self, bottleneck_size=128):
        super().__init__(bottleneck_size,
                            encoder=nn.Sequential(
                                nn.Conv2d(3, 16, 3, 1, 1),
                                nn.ReLU(),
                                nn.Conv2d(16, 32, 3, 1, 1),
                                nn.MaxPool2d(2),
                                nn.ReLU(),
                                nn.Conv2d(32, 64, 3, 1, 1),
                                nn.MaxPool2d(2),
                                nn.ReLU(),
                                nn.Flatten(),
                                nn.LazyLinear(1024),
                                nn.ReLU(),
                                nn.LazyLinear(2 * bottleneck_size),
                            ),
                            decoder=nn.Sequential(
                                nn.LazyLinear(1024),
                                nn.ReLU(),
                                nn.LazyLinear(64 * 8 * 8),
                                nn.ReLU(),
                                nn.Unflatten(-1, (64, 8, 8)),
                                nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(16, 3, 3, 1, 1),
                            ))

    def mk_trace(self, x, bounds):
        """Create a trace from an image tensor."""
        return to_plotly(x)


# Function to train CIFAR VAE
def train_cifar_vae(bottleneck_size: int = 128):
    # Set device and data type
    device = "cuda"
    dtype = torch.float32

    # Load CIFAR10 Data
    train_data, _, test_data, _ = load_cifar10(device, dtype, False)

    # Instantiate the VAE, Optimizer, and Logger
    vae = Cifar10VAE(bottleneck_size).to(device)
    optimizer = optim.AdamW(vae.parameters(), lr=0.001)
    logger = LossRecord()

    # Training Loop
    beta = 0.1
    noise = 0.02 * (train_data.max() - train_data.min())
    epochs = 80
    for epoch in range(epochs):
        running_loss = 0.0
        for i, inputs in enumerate(tqdm(batch(train_data), desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            loss = vae.forward_loss(inputs, noise=noise, beta=beta, logger=logger)
            loss.backward()
            optimizer.step()
            logger.step()
            running_loss += loss.detach()
        logger.log("train_loss_epoch", running_loss / (i + 1), verbose=True)

        with torch.no_grad():
            out = vae.forward_loss(test_data, noise=0, beta=beta)
            logger.log("test_loss", out, verbose=True)

        if epoch % min(20, epochs // 4) == 0 or epoch == epochs - 1:
            # Plot the training curve
            logger.plot()

            # Show some reconstructions
            vae.show(test_data[:4])

    vae.save()

    # Return the trained model
    return vae

# %%
