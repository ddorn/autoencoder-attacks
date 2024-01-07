# Implementation of a neural cellular automata (NCA) model

# %%

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from einops import rearrange, repeat, reduce, einsum

from jaxtyping import Float, Int, Bool
from tqdm import trange

ALIVE_CHANNEL = 1
VIEW_CHANNEL = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def pad2d_circular(x, padding=1):
    """Pad a 2d tensor circularly."""

    out = F.pad(x, (padding, padding, padding, padding), mode="circular")
    return out

SOBEL = torch.tensor([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=torch.float32, device=device)

def sobel_extend(x: Float[Tensor, 'batch height width channel']) -> Float[Tensor, 'batch height width 3*channel']:
    """Concatenate the input with the Sobel_x and Sobel_y filters applied to it."""
    b, h, w, n_channels = x.shape
    assert h == w
    # There is no mixing of the channels
    x = rearrange(x, "b h w c -> (b c) () h w")
    x_padded = pad2d_circular(x)
    sobel_x = F.conv2d(x_padded, SOBEL[None, None])
    sobel_y = F.conv2d(x_padded, SOBEL.T[None, None])

    return rearrange([x, sobel_x, sobel_y], "new (b c) () h w -> b h w (new c)", c=n_channels)

# Test the sobel convolution
x = torch.rand(1, 4, 4, 1, device=device)
x = torch.tensor([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
], dtype=torch.float32, device=device)[None, :, :, None]
# print(x.shape)
out = sobel_extend(x)
# print(out.shape)
# print(out.permute(0, 3, 1, 2))

assert out[..., 0].allclose(x[..., 0])
assert out[0, ..., 1].allclose(-SOBEL)
assert out[0, ..., 2].allclose(-SOBEL.T)

# %%

def next_to_alive(x: Float[Tensor, 'batch height width channel']) -> Bool[Tensor, 'batch height width']:
    """Return a boolean mask of the alive cells and the ones next to them."""

    live_padded = pad2d_circular(x[None, ..., ALIVE_CHANNEL])
    # A cell is dead if all of its neighbors have a value < 0.1 (max pool)
    dead = F.max_pool2d(live_padded, kernel_size=3, stride=1) < 0.1
    return ~dead.squeeze(0)

# Test the next_to_alive function
x = (torch.rand(1, 10, 10, 16) > 0.9).float()
out = next_to_alive(x)
assert out.shape == (1, 10, 10), out.shape
assert out.dtype == torch.bool, out.dtype
# Plot x[..., 0] and out
fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Heatmap(z=x[0, :, :, 0]), row=1, col=1)
fig.add_trace(go.Heatmap(z=out[0].float()), row=1, col=2)
fig.show()


# %%

N_CHANNELS = 8
WORLD_SIZE = 16
ALIVE_CHANNEL = 0
VIEW_CHANNEL = 1

class NCA(nn.Sequential):
    def __init__(self, n_channels: int = N_CHANNELS):
        super().__init__(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(n_channels),
        )

        # Set the last linear layer to zero
        self[-1].weight.data.zero_()
        self[-1].bias.data.zero_()

    def forward(self, x):
        x = sobel_extend(x)
        for module in self:
            x = module(x)
        # Sample a boolean for each position
        dropout = torch.rand(x.shape[:3], device=x.device) < 0.8  # (batch, h, w)
        x = x * dropout[..., None]
        return x


class World:
    world: Float[Tensor, "batch height width channel"]
    def __init__(self,
                 n_channels: int = N_CHANNELS,
                 world_size: int = WORLD_SIZE,
                 update_strength: float = 1,
                 n_worlds: int = 1,
                 device=device,
                 ):
        self.n_channels = n_channels
        self.world_size = world_size
        self.update_strength = update_strength
        self.n_worlds = n_worlds
        self.world = torch.zeros((n_worlds, world_size, world_size, n_channels), device=device)
        self.reset()

    def new_nca(self):
        """Return a new NCA"""
        nca = NCA(n_channels=self.n_channels)
        nca.to(self.world.device)
        return nca

    def reset(self, n_worlds=None):
        """Reset the world to its initial state"""
        if n_worlds is None:
            n_worlds = self.n_worlds
        self.world = torch.zeros((n_worlds, self.world_size, self.world_size, self.n_channels), device=self.world.device)
        # Set a random cell to 1
        # x, y = np.random.randint(0, self.world_size, size=2)
        x = y = self.world_size // 2
        self.world[:, x, y, :] = 1
        return x, y

    def image(self):
        """Return the world as an image."""
        return self.world[..., VIEW_CHANNEL] * self.world[..., ALIVE_CHANNEL]

    def show(self):
        """Plot the world with plotly"""
        fig = px.imshow(self.image().detach().cpu())
        fig.show()

    def step(self, nca: NCA):
        """Step the world by one iteration."""

        live_mask = next_to_alive(self.world)
        update = nca(self.world)[0] * self.update_strength
        self.world = (self.world + update) * live_mask[..., None]
        self.world = self.world.clamp(0, 1)

    def video(self, nca: NCA, n_iter: int = 1) -> Float[Tensor, "n_iter_plus_1 batch height width"]:
        """Return the image of the world by n_iter iteration(s)."""
        evolution = [self.image()]
        for _ in range(n_iter):
            self.step(nca)
            evolution.append(self.image())
        return torch.stack(evolution)

    def evolution(self, nca: NCA, n_iter: int = 1,
                  plot_every: int = -1,
                  all_channels: bool = False,
                  goal: None | Tensor = None,
                  **plotly_kwargs):
        """Return the evolution of the world by n_iter iteration(s)."""
        evolution = [self.world.clone()]
        for _ in range(n_iter):
            self.step(nca)
            evolution.append(self.world.clone())
        video = torch.stack(evolution)

        if plot_every > 0:
            to_plot = video[::plot_every, 0].detach().cpu()
            image = to_plot[..., VIEW_CHANNEL] * to_plot[..., ALIVE_CHANNEL]
            if not all_channels:
                to_plot = image
            else:
                print(image.shape, to_plot.shape)
                to_plot = torch.cat([image[..., None], to_plot], dim=-1)
                plotly_kwargs.setdefault("animation_frame", 3)
                assert goal is None
            if goal is not None:
                to_plot = torch.cat([to_plot, goal[None].cpu()])
            fig = px.imshow(to_plot,
                            facet_col=0,
                            facet_col_wrap=4,
                            **plotly_kwargs,
                            )
            fig.show()

        return video

# %%


world = World()
nca = world.new_nca()
world.evolution(nca, 100, 10, all_channels=True);

# %%

# Heart bitmap
heart_str = """
0000000000000000
0000000000000000
0000000000000000
0000110001100000
0001111111110000
0001111111110000
0001111111110000
0000111111100000
0000011111000000
0000001110000000
0000000100000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
"""

heart = torch.tensor([
    [int(c) for c in line]
    for line in heart_str.strip().splitlines()],
    device=device).float()
print(heart)
# inflate 2x
# heart = F.interpolate(heart[None, None], (heart.shape[0] * 2, heart.shape[1] * 2))[0, 0]
heart = F.interpolate(heart[None, None], (WORLD_SIZE, WORLD_SIZE))[0, 0]
# pad with zeros instead
# heart = F.pad(heart, (WORLD_SIZE // 2, WORLD_SIZE // 2, WORLD_SIZE // 2, WORLD_SIZE // 2))
print(heart)
# plot
fig = px.imshow(heart.detach().cpu())
fig.show()

# %% Training an NCA to draw a heart
from lib.utils import LossRecord


world = World()
nca = world.new_nca()
optimizer = torch.optim.SGD(nca.parameters(), lr=0, weight_decay=0)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

losses = LossRecord()

def set_lr(lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# %%
epochs = 500
plot_every = epochs // 5
lr = 0.0001
n_worlds = 500

set_lr(lr)

progress = trange(epochs)
for i in progress:
    x, y = world.reset(n_worlds)
    shifted_heart = heart.roll((y - WORLD_SIZE // 2, x - WORLD_SIZE // 2), dims=(0, 1))

    video = world.video(nca, 42)

    loss = (video[10:] - heart * 0.5).abs().mean() - video.mean() * 0.1
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()

    losses.log("Loss", loss)
    losses.step()

    if i % 20 == 0:
        progress.set_description(f"Loss: {loss.item():.4f}")
    if i % epochs // 5 == 0:
        lr *= 4
        set_lr(lr)

    if i % plot_every == 0:
        losses.plot()
        with torch.no_grad():
            world.reset(1)
            world.evolution(nca, 42, plot_every=4, all_channels=True)
losses.plot()

# %%
torch.save(nca.state_dict(), "nca-heart-16x16.pt")

# %%
nca.load_state_dict(torch.load("data/nca-heart-16x16.pt"))

# %% Make a gif
world = World()

with torch.no_grad():
    video = world.evolution(nca, 100, plot_every=4)[:, 0, :, :, VIEW_CHANNEL]

# %%
# assert video.shape == (51, WORLD_SIZE, WORLD_SIZE)
video = video.cpu().numpy()
# Upscale with nearest neighbor to x8
import numpy as np
video = np.kron(video, np.ones((8, 8)))
# Apply Inferno colormap
import matplotlib.cm as cm
video = cm.inferno(video)
# Convert to uint8
video = (video * 255).astype(np.uint8)
# video = video[:30]  # trim
# video = np.concatenate([video, video[::-1], video[:1].repeat(10, axis=0)])  # Reverse
# Save as gif, loop forever, 10 fps
from imageio import mimsave
mimsave("heart.gif", video, loop=0, fps=10)






# %%

# %%
x, y = world.reset()
shifted_heart = heart.roll((y - WORLD_SIZE // 2, x - WORLD_SIZE // 2), dims=(0, 1))
world.evolution(
    nca,
    42, plot_every=4,
    all_channels=True,
);
# %%
