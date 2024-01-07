# Goal train a performant MNIST classifier

# %%

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import plotly.express as px
from tqdm import tqdm
from lib.utils import smooth, LossRecord, batch
from einops import rearrange

from jaxtyping import Float, Int
from torch import Tensor
from plotly.subplots import make_subplots
import plotly.graph_objects as go


DATA_FOLDER = Path(__file__).parent.parent / "data"


MNIST_PATH = DATA_FOLDER / "mnist.pt"


def prepare_mnist(save_path=MNIST_PATH):
    """Prepare the MNIST dataset and save it to a single file."""

    # Load MNIST train+test data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = torchvision.datasets.MNIST(
        root=DATA_FOLDER, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=DATA_FOLDER, train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=10, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=10, pin_memory=True
    )

    # Save the transformed MNIST data to a single tensor file

    train_list = list(tqdm(trainloader, desc="Train"))
    test_list = list(tqdm(testloader, desc="Test"))
    train_data = torch.cat([x[0] for x in train_list], dim=0)
    train_targets = torch.cat([x[1] for x in train_list], dim=0)
    test_data = torch.cat([x[0] for x in test_list], dim=0)
    test_targets = torch.cat([x[1] for x in test_list], dim=0)

    print(train_data.shape, train_targets.shape)
    print(test_data.shape, test_targets.shape)

    # Save the data to a single file
    mnist = [t.to(device) for t in [train_data, train_targets, test_data, test_targets]]
    torch.save(mnist, save_path)


def load_mnist(path=MNIST_PATH) -> tuple[Float[Tensor, 'train=60000 c=1 w=28 h=28'], Int[Tensor, 'train'], Float[Tensor, 'test=10000 c w h'], Int[Tensor, 'test']]:
    """Load the MNIST dataset from a single file 'mnist.pt'.

    Returns: train_data, train_targets, test_data, test_targets.
        The dataset has mean 0 and std 1."""

    try:
        return torch.load(path)
    except FileNotFoundError:
        print("MNIST not found, preparing it...")
        prepare_mnist(path)
        return torch.load(path)


# Define the architecture

class MnistNet(nn.Module):
    """
    Taken from https://github.com/tuomaso/train_mnist_fast/blob/master/8_Final_00s76.ipynb
    """
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, 5, 1)
        self.conv2 = nn.Conv2d(24, 32, 3, 1)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def save(self, save_path=DATA_FOLDER / "mnist_classifier.pt"):
        torch.save(self.state_dict(), save_path)

    @classmethod
    def load(cls, save_path=DATA_FOLDER / "mnist_classifier.pt"):
        classifier = cls()
        classifier.load_state_dict(torch.load(save_path))
        return classifier

# Define VAE model

# The importants parts of the VAE are:
# - The encoder, which maps the input image to a latent vector
# - The decoder, which maps the latent vector to an output image
# - The loss function, which is a combination of a reconstruction loss and a
#   regularization loss
# - The sampling function, which samples a latent vector from the encoder's
#   output
# - The reparameterization trick, which allows the model to be trained with
#   backpropagation
# - The bottleneck size, which is the size of the latent vector

class BaseVAE(nn.Module):
    def __init__(self, bottleneck_size, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.encoder = encoder
        self.decoder = decoder
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_latent=False):
        shape = x.shape
        latent = self.encoder(x)
        x = self.sample(latent)
        x = self.decoder(x)
        x = x.reshape(shape)

        if return_latent:
            return x, latent
        else:
            return x

    def sample(self, x):
        """Sample a latent vector from the encoder's output."""
        mu, logvar = x.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss(self, input, output, latent, beta=1.0, logger=None):
        """Compute the loss of the VAE.

        The loss is the sum of the reconstruction loss and the regularization
        loss.
        """

        scale = self.log_scale.exp()
        distribution = torch.distributions.Normal(output, scale)
        reconstruction_loss = -distribution.log_prob(input).sum() / input.shape[0]
        reconstruction_loss = F.mse_loss(input, output, reduction="sum") / input.shape[0]

        mu, logvar = latent.chunk(2, dim=-1)
        regularization_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]


        if logger is not None:
            logger.log("Reconstruction loss", reconstruction_loss)
            logger.log("Regularization loss", regularization_loss * beta)

        return reconstruction_loss + regularization_loss * beta

    def forward_loss(self, input, return_output=False, noise=0.1, beta=1.0, logger=None):
        """Compute the loss of the VAE. Adds noise to the input."""
        noisy = input + torch.randn_like(input) * noise

        output, latent = self(noisy, return_latent=True)
        loss = self.loss(input, output, latent, beta=beta, logger=logger)

        if return_output:
            return loss, output
        else:
            return loss

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def show(self, input, row_titles: list[str] | None = None, **plotly_layout):
        """Show the input and output of the model side by side."""

        if isinstance(input, (list, tuple)):
            # Ensure each is 4D, then cat along axis 0
            input = torch.cat([x[(None,) * (4 - x.ndim)] for x in input], dim=0)

        if input.ndim == 3:
            input = input.unsqueeze(0)
        assert input.ndim == 4

        loss, output = self.forward_loss(input.to(self.device), return_output=True)
        # output = torch.distributions.Normal(output, self.log_scale.exp()).sample()

        assert input.shape == output.shape
        bounds = dict(
            zmin = min(input.min(), output.min()).item(),
            zmax = max(input.max(), output.max()).item(),
        )

        rows = input.shape[0]
        fig = make_subplots(rows=rows, cols=2,
                            column_titles=("Input", "Output"),
                            row_titles=row_titles,
                            horizontal_spacing=0.05, vertical_spacing=0.1/rows,
                            shared_xaxes=True, shared_yaxes=True,
                            )

        for row, (in_, out) in enumerate(zip(input, output)):
            fig.add_trace(self.mk_trace(in_, bounds), row=row + 1, col=1)
            fig.add_trace(self.mk_trace(out, bounds), row=row + 1, col=2)

        default_layout = dict(
            height=300 * input.shape[0],
            width=600,
            title_text=f"Loss: {loss.item():.3f}",
        )
        fig.update_layout(**{**default_layout, **plotly_layout})
        # Remove the axis labels
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        # Remove all colorbars
        for trace in fig.data:
            try:
                trace.colorbar = None
            except ValueError:
                pass  # Does not have a colorbar

        fig.show()

    def mk_trace(self, x, bounds):
        """Create a trace from an image tensor."""
        raise NotImplementedError

    def save(self, save_path: str = None):
        if save_path is None:
            save_path = DATA_FOLDER / f"{self.__class__.__name__}_{self.bottleneck_size}.pt"
        torch.save(self.state_dict(), save_path)
        print(f"Saved {self.__class__.__name__} to {save_path}")

    @classmethod
    def load(cls, bottleneck_size: int, save_path: str = None):
        if save_path is None:
            save_path = DATA_FOLDER / f"{cls.__name__}_{bottleneck_size}.pt"
        model = cls(bottleneck_size)
        model.load_state_dict(torch.load(save_path))
        return model


class MnistVAE(BaseVAE):

    def __init__(self, bottleneck_size=12):
        super().__init__(bottleneck_size,
            encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 1, 1),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(128),
                nn.ReLU(),
                nn.LazyLinear(2 * bottleneck_size),
            ),
            decoder = nn.Sequential(
                nn.LazyLinear(128),
                nn.ReLU(),
                nn.LazyLinear(32 * 7 * 7),
                nn.ReLU(),
                nn.LazyLinear(28 * 28),
            ),
        )

    def mk_trace(self, x, bounds):
        """Create a trace from an image tensor."""
        return go.Heatmap(z=x.squeeze().cpu().flip(0), colorscale="gray", showscale=False, **bounds)



def train_MNIST_classifier():
    # Seed everything
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST train+test data
    train_data, train_targets, test_data, test_targets = load_mnist()

    classifier = MnistNet().to(device)
    optimizer = optim.AdamW(classifier.parameters(), lr=0.003)
    losses = []

    # Train the classifier
    from lib.utils import batch

    for epoch in range(1):
        running_loss = 0.0
        progress = tqdm(batch(train_data, train_targets), desc=f"Epoch {epoch+1}")
        for i, (inputs, labels) in enumerate(progress):
            optimizer.zero_grad()
            loss = F.cross_entropy(classifier(inputs), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach()
            # progress.set_postfix(loss=running_loss / (i + 1))
            losses.append(loss.detach())

    fig = px.line(y=smooth(losses), log_y=True)
    fig.show()

    # Check the classifier's accuracy on the test set
    with torch.no_grad():
        out = classifier(test_data)
        loss = F.cross_entropy(out, test_targets)
        accuracy = (out.argmax(-1) == test_targets).float().mean()

    print(f"Test loss: {loss.item():.4f}")
    print(f"Test accuracy: {accuracy.item():.4f}")

    #
    classifier.save()
    return classifier


def train_MNIST_VAE(latent_size: int = 2, noise: float = 0.1, beta: float = 1.0, epochs: int = 48):
    # Load MNIST train+test data
    train_data, train_targets, test_data, test_targets = load_mnist()

    vae = MnistVAE(latent_size).cuda()
    optimizer = optim.AdamW(vae.parameters(), lr=0.001)
    loss_record = LossRecord()


    # Train the VAE
    REGULAR_PLOTS = epochs // 2
    for epoch in range(epochs):
        running_loss = 0.0
        progress = tqdm(batch(train_data), desc=f"Epoch {epoch+1}")
        for i, inputs in enumerate(progress):
            optimizer.zero_grad()
            loss = vae.forward_loss(inputs, noise=noise, beta=beta, logger=loss_record)
            loss.backward()
            optimizer.step()

            loss_record.log("Total loss", loss)
            loss_record.log("Scale", vae.log_scale.exp())
            loss_record.step()

            running_loss += loss.detach()

        if epoch % REGULAR_PLOTS == 0 or epoch == epochs - 1:
            vae.show(test_data[4:8])
        loss_record.log("Train loss", running_loss / (i + 1), verbose=True)

        # Check the VAE's reconstruction loss on the test set
        with torch.no_grad():
            loss = vae.forward_loss(test_data, noise=noise, beta=beta).item()
            loss_record.log("Test loss", loss, verbose=True)

    loss_record.plot()

    vae.show(test_data[4].unsqueeze(0).repeat(4, 1, 1, 1))

    vae.save()
    return vae


def vae_manifold(vae: MnistVAE, n: int = 20):
    """Plot the manifold of the latent space"""
    assert vae.bottleneck_size == 2
    device = next(vae.parameters()).device

    with torch.no_grad():
        z = torch.stack(torch.meshgrid(torch.linspace(0, 1, n), torch.linspace(0, 1, n)), dim=-1).reshape(-1, 2).to(device)
        # Inverse the gaussion CDF
        z = torch.erfinv(2 * z - 1) * (2 ** 0.5)
        x = vae.decoder(z).clamp(0, 1).cpu()
        nice = rearrange(x, "(row col) (h w) -> (row h) (col w)", row=n, h=28)
        fig = px.imshow(nice)
        fig.show()
