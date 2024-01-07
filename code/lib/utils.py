# %%
from collections import defaultdict
import math
from pathlib import Path
import time
from contextlib import contextmanager
from turtle import width
from diffusers.models.vae import DecoderOutput
from typing import Callable, Literal
from jaxtyping import Float, Int
import numpy as np
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib import cm

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

IMAGES_FOLDER = (Path(__file__).parent.parent.parent / "images").resolve()
print("Images folder:", IMAGES_FOLDER)


def save(fig, name: str):
    path = IMAGES_FOLDER / f"{name}.png"
    fig.write_image(path, scale=2)
    print("Saved to", path)
    return path


def show_top5(probas: Float[Tensor, "classes"], correct: int, labels: list[str]):
    """Pretty print the top 5 predictions."""
    # Print the top 5 predictions
    correct_printed = False
    top5 = probas.topk(5)
    for i, (p, l) in enumerate(zip(top5.values.tolist(), top5.indices.tolist())):
        text =f"Top {i+1}:{p: 7.2%} {labels[l]}"
        if l == correct: # print in green
            print(f"\033[92m{text}\033[0m")
            correct_printed = True
        else:
            print(text)
    if not correct_printed:
        correct_index = probas.sort(descending=True).indices.tolist().index(correct)
        print(f"\033[92mTop {correct_index+1}:{probas[correct]: 7.2%} {labels[correct]}\033[0m")


def shorten(label: str, max_length=15):
    """Shorten a label to max_length characters."""
    if len(label) > max_length:
        return label[:max_length - 1] + "…"
    return label

@torch.no_grad()
def mk_top5_trace(image, task):
    probas, indices = (task.classifier(image[None])
            .softmax(-1)
            .squeeze(0)
            .float()
            .cpu()
            .topk(5))
    labels = [shorten(task.labels[i]) for i in indices]
    return go.Bar(x=labels, y=probas, orientation="v", textposition="auto", #texttemplate="%{y:.1%}",
                   text=labels)


def to_plotly(x: Tensor) -> Float[Tensor, "height width channels=3"]:
    """Convert a tensor to a plotly-compatible tensor."""
    if x.ndim == 4:
        assert x.shape[0] == 1
        x = x.squeeze(0)

    x = x.detach().cpu().float()
    x = channel_last(x)
    if x.shape[-1] == 1:
        # Apply a nice purple/orange/yellow colormap
        x = x.squeeze(-1)
        x = x.numpy()
        x = cm.get_cmap("plasma")(x)
    return go.Image(z=x * 255)


def show(images, labels, unnormalize: Literal["cifar", "256", None]=None,
    save: str | None = None,
    **plotly_kwargs):
    """
    Show a grid of images with their labels.

    Saves to images/{save}.png if save is not None.
    """
    if isinstance(images, list):
        images = torch.stack(images)
    images = images.detach()
    batch_size = images.shape[0]

    # Unnormalize
    if unnormalize == "cifar":
        std = torch.tensor([62.99, 62.09, 66.70], device=images.device, dtype=images.dtype)[:, None, None]
        mean = torch.tensor([125.31, 122.95, 113.87], device=images.device, dtype=images.dtype)[:, None, None]
        images = images * std + mean
        # images = images[:, :, 4:-4, 4:-4]
    elif unnormalize == "256":
        images = images * 256
    elif unnormalize is None:
        pass
    else:
        raise ValueError(f"Unknown unnormalize value {unnormalize}")

    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()

    cols = math.ceil(math.sqrt(batch_size))
    images = images.cpu().float().permute(0, 2, 3, 1)
    if images.shape[-1] == 1:
        images = images.squeeze(-1)
    fig = px.imshow(images,
                    facet_col=0, facet_col_wrap=cols)

    # Plotly orders the annotations with the last row first,
    # so we need to reorder the labels
    rows = math.ceil(batch_size / cols)
    new_labels = []
    for i in range(rows):
        new_labels = labels[i * cols: (i + 1) * cols] + new_labels
    # Set the labels
    for i, label in enumerate(new_labels):
        fig.layout.annotations[i].text = label

    # Remove margins
    # fig.update_layout(margin=dict(l=0, r=0, b=0, t=30, pad=0))
    margin_kwargs = plotly_kwargs.setdefault("margin", {})
    margin_kwargs.setdefault("l", 5)
    margin_kwargs.setdefault("r", 5)
    margin_kwargs.setdefault("b", 5)
    margin_kwargs.setdefault("t", 30)
    margin_kwargs.setdefault("pad", 0)

    # Remove axes
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    # Set all fonts to 15pt
    plotly_kwargs.setdefault("font", {}).setdefault("size", 18)
    fig.update_layout(**plotly_kwargs)
    fig.show()

    if save is not None:
        fig.write_image(IMAGES_FOLDER / f"{save}.png", scale=3)


def report(image: Float[Tensor, "channels=3 height width"],
           label: int,
           vae: nn.Module,
           classifier: nn.Module,
           labels: list[str],
           perturbation: Float[Tensor, "channels height width"]=None,
           ):
    """
    Show a report on the image, its reconstruction, the adversarial image and
    its classification.
    """
    if image.ndim == 3:
        image = image[None]
    if perturbation is not None:
        if perturbation.ndim == 3:
            perturbation = perturbation[None]
        adversarial_image = (image + perturbation)
        inputs = torch.cat([image, adversarial_image])
    else:
        inputs = image

    inputs = inputs.clamp(0, 1)

    out = vae(inputs)
    if isinstance(out, DecoderOutput):
        out = out.sample
    out = out.clamp(0, 1)
    to_classify = torch.cat([inputs, out])
    probas = F.softmax(classifier(to_classify), dim=-1)

    titles = ["Original image", "Adversarial image", "Original reconstruction", "Adversarial reconstruction"]
    if perturbation is None:
        titles = titles[::2]

    # Show top 5 predictions
    for title, probs in zip(titles, probas):
        print("\033[4m" + title + "\033[0m")
        show_top5(probs, label, labels)
        print()

    # Show the 2 or 4 images
    show(to_classify.clamp(0, 1), titles, width=600, height=600)

    # Show the perturbation as heatmap
    if perturbation is not None:
        if perturbation.shape[1] == 3:
            # Cannot do heatmap on RGB image, so we shift to around 0.5 to see.
            perturbation = perturbation + 0.5
        show(perturbation, ["Perturbation"], width=600, height=600)


class batch:
    def __init__(self, *tensors, batch_size: int = 128, shuffle: bool = True, drop_last: bool = True):
        """Iterate over batches of data and targets."""
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.nb_datapoints = tensors[0].shape[0]
        assert all(t.shape[0] == self.nb_datapoints for t in tensors)


    def __len__(self):
        if self.drop_last:
            return self.nb_datapoints // self.batch_size
        return math.ceil(self.nb_datapoints / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.nb_datapoints)
            tensors = [t[indices] for t in self.tensors]
        else:
            tensors = self.tensors

        if self.drop_last:
            end = self.nb_datapoints - self.nb_datapoints % self.batch_size
        else:
            end = self.nb_datapoints

        starts = range(0, end, self.batch_size)

        if len(tensors) == 1:
            for i in starts:
                yield tensors[0][i : i + self.batch_size]
        else:
            for i in starts:
                yield [t[i : i + self.batch_size] for t in tensors]


@contextmanager
def timeit(name="Time"):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start:.3f}s")


def smooth(x, window_size=None):
    """Smooth a 1D tensor by averaging over a window."""
    if window_size is None:
        window_size = 1 + int(len(x) ** 0.5 / 2)

    return torch.tensor(x).unfold(0, window_size, 1).mean(-1)

def channel_last(x):
    """Ensure that the channel dimension is the last dimension."""
    if x.shape[-1] in (1, 3):
        return x
    assert x.shape[-3] in (1, 3)
    if x.ndim == 4:
        return x.permute(0, 2, 3, 1)
    elif x.ndim == 3:
        return x.permute(1, 2, 0)
    else:
        raise ValueError(f"Invalid number shape: {x.shape}")

class LossRecord:
    def __init__(self) -> None:
        self.buffers = {}
        self.buffers_x = {}
        self._step = 0

    def log(self, name: str, value: float | Float[Tensor, ""] | dict, verbose=False):
        """Log a value."""
        if isinstance(value, dict):
            for sub_name, value in value.items():
                self.log(f"{name}/{sub_name}", value, verbose=verbose)
            return

        if name not in self.buffers:
            self.buffers[name] = []
            self.buffers_x[name] = []

        if isinstance(value, Tensor):
            value = value.detach()
        if verbose:
            print(f"{name}: {value:.4f} (step {self._step})")

        self.buffers[name].append(value)
        self.buffers_x[name].append(self._step)

    def step(self):
        """Increment the step counter."""
        self._step += 1

    def plot(self, **kwargs):
        """Plot the losses."""
        if not self.buffers:
            print("Nothing to plot")
            return

        losses_fig = go.Figure()
        for name, values in self.buffers.items():
            try:
                y = torch.tensor(values)
            except Exception:
                print(f"Could not convert {name} to tensor")
                continue

            if y.ndim == 1:
                losses_fig.add_trace(
                    go.Scatter(x=self.buffers_x[name], y=torch.tensor(values, dtype=float), name=name,
                               legendgroup=name.rpartition("/")[0])
                )
            elif y.ndim == 4:
                # plot as a video
                if len(y > 64):
                    step = len(y) // 64
                    y = y[::step]
                    # x = self.buffers_x[name][::step]
                else:
                    step = 1
                y = channel_last(y)
                if step > 1:
                    name += f" (every {step} steps)"
                fig = px.imshow(y.float(), animation_frame=0, title=name)
                fig.show()

        losses_fig.update_layout(
            yaxis_type="log",
            **kwargs)
        losses_fig.show()


class GridPlot:
    def __init__(self, rows: int, cols: int, **kwargs) -> None:
        self.rows = rows
        self.cols = cols
        self.fig = make_subplots(rows=rows, cols=cols, **kwargs)
        self._row = 1
        self._col = 1

    def add(self, trace: go.Scatter | go.Image, title: str = None):
        """Add a trace to the grid."""
        self.fig.add_trace(trace, row=self._row, col=self._col)
        # if title is not None:
            # Subplot title
        self._col += 1
        if self._col > self.cols:
            self.new_row()

    def show(self, **kwargs):
        """Show the grid."""
        self.fig.update_layout(**kwargs)
        self.fig.show()

    def new_row(self):
        """Start a new row."""
        self._row += 1
        self._col = 1


class Cache(dict[str, Tensor]):

    def __repr__(self):
        return "Cached activations:\n" + "\n".join(
            f"- {name}: {tuple(activation.shape)}" for name, activation in self.items()
        )

    __str__ = __repr__

    def __getitem__(self, item: str) -> Tensor:
        # Find the key that matches and make sure it's unique.
        if item in self:
            return super().__getitem__(item)

        keys = [key for key in self.keys() if item in key]
        if len(keys) == 0:
            raise KeyError(item)
        elif len(keys) > 1:
            raise KeyError(f"Multiple keys match {item}: {keys}")
        return super().__getitem__(keys[0])

    def remove_batch_dim(self):
        """Remove the batch dimension from all activations."""
        if any(activation.shape[0] != 1 for activation in self.values()):
            raise ValueError("Not all activations have batch dimension 1.")

        for name, activation in self.items():
            self[name] = activation.squeeze(0)

    def apply(self, func: Callable[[Tensor], Tensor]):
        """Apply a function to all activations."""
        for name, activation in self.items():
            self[name] = func(activation)


@contextmanager
def record_activations(module: nn.Module, *ignore: str,
                       reduce: Literal[None, "mean", "sum"] = None,
                       verbose: bool = True,
                       ) -> Cache:
    """Context manager to record activations from a module and its submodules.

    Args:
        module (nn.Module): Module to record activations from.
        ignore (str): Any submodule whose name contains any of these strings will be ignored.

    Yields:
        dist[str, Tensor]: Dictionary of activations, that will be populated once the
            context manager is exited.
    """

    cache = Cache()
    activations: dict[str, list[Tensor]] = {}
    counts: dict[str, int] = defaultdict(int)  # used for reduce="mean"
    hooks = []

    skipped = set()
    module_to_name = {m: f"{n} {m.__class__.__name__}" for n, m in module.named_modules()}

    def hook(m: nn.Module, input: Tensor, output: Tensor):
        name = module_to_name[m]
        if not isinstance(output, Tensor) or any(s in name for s in ignore):
            skipped.add(name)
        elif name not in activations:
            activations[name] = [output.detach()]
        elif reduce is None:
            activations[name].append(output.detach())
        elif reduce == "sum":
            activations[name][0] += output.detach()
        elif reduce == "mean":
            activations[name][0] += output.detach()
            counts[name] += 1
        else:
            raise ValueError(f"Unknown reduce value {reduce!r}")

    for module in module.modules():
        hooks.append(module.register_forward_hook(hook))

    try:
        yield cache
    finally:
        for hook in hooks:
            hook.remove()

    for name, activation in activations.items():
        if len(activation) == 1:
            cache[name] = activation[0]
        else:
            try:
                cache[name] = torch.stack(activation)
            except RuntimeError:
                print(f"Could not stack activations for {name!r}")
                raise

    if reduce == "mean":
        for name, activation in cache.items():
            cache[name] = activation / counts[name]

    if skipped and verbose:
        print("Skipped:")
        for name in skipped:
            print("-", name)

# %%
def add_line(fig, equation: str, line: dict = None, **add_trace_kwargs):
    # get the x range from the figure
    try:
        minx, maxx = fig.layout.xaxis.range
    except TypeError:
        # Find the min and max x values
        minx = float("inf")
        maxx = float("-inf")
        for trace in fig.data:
            minx = min(minx, min(trace.x))
            maxx = max(maxx, max(trace.x))

    x = np.linspace(minx, maxx, 100)

    # Parse equation
    left, _, right = equation.partition("=")
    if left.strip() == "y":
        y = eval(right)
    elif right.strip() == "y":
        y = eval(left)
    else:
        raise ValueError(f"Equation {equation} should contain y on one side")

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        # line=dict(color="red", width=2),
        line=line,
        # Add to legend
        name=equation,
        legendgroup=equation,
    ),
    **add_trace_kwargs)


def annotate_modules(fig, layer_names, has_classifier: bool = False,
                     encoder_name="encoder", decoder_name="decoder",
                     colorful=False):
    """Annotate the encoder, decoder and classifier on a figure with rectangles."""

    layer_names = list(layer_names)
    last_encoder_layer = max(i for i, name in enumerate(layer_names) if encoder_name in name.lower()) + 1
    last_decoder_layer = max(i for i, name in enumerate(layer_names) if decoder_name in name.lower()) + 1

    positions = [0, last_encoder_layer, last_decoder_layer, len(layer_names)]
    if not has_classifier:
        del positions[2]

    # a rectangle for each
    rectange_style = dict(
        type="rect",
        yref="paper",
        xref="x",
        y0=0,
        y1=1,
        layer="below",
        line_width=0,
        opacity=0.1,
    )
    if colorful:
        colors = ["LightSalmon", "LightSkyBlue", "LightGreen"]
    else:
        colors = ["white", "grey", "white"]

    for x0, x1, color in zip(positions[:-1], positions[1:], colors):
        fig.add_shape(
            x0=x0 - 0.5,
            x1=x1 - 0.5,
            fillcolor=color,
            **rectange_style,
        )

    # Add annotations for each rectangle
    common = dict(
        yref="paper",
        y=1,
        showarrow=False,
        font=dict(size=15),
        yshift=-5,
    )
    fig.add_annotation(
        x=last_encoder_layer / 2,
        text="Encoder", **common,
    )
    fig.add_annotation(
        x=(last_encoder_layer + last_decoder_layer) / 2 + 0.5,
        text="Decoder", **common,
    )
    if has_classifier:
        fig.add_annotation(
            x=(last_decoder_layer + len(layer_names)) / 2 + 0.5,
            text="Classifier", **common,
        )


def find_closest(start, start_label, task, max_checks:int=-1):
    """Find the closest image to start with a different label."""
    target_dist = float('inf')
    target = None
    for i, (image, label) in enumerate(task.iter(batch_size=1)):
        if label == start_label:
            continue
        image = image.squeeze(0)
        dist = (start - image).abs().sum()
        if dist < target_dist:
            target_dist = dist
            target = image
            print(f"Found target at {i} with distance {dist}")

        if i == max_checks:
            break

    return target


@torch.no_grad()
def get_activations(task, image, kind: Literal["norm", "size", "name", "cache"]):
    """Get the activations of the VAE for an image."""
    if image.ndim == 3:
        image = image[None]

    with record_activations(task.vae,
                            "nonlin", "relu", "Dropout", "Norm", "Down", "Block",
                            "Sequential", "ReLu",
                            verbose=False) as cache:
        task.vae(image)

    if kind == "norm":
        return torch.tensor([a.pow(2).sum().sqrt().item() for a in cache.values()], device="cuda")
    elif kind == "size":
        return [a.numel() for a in cache.values()]
    elif kind == "name":
        return list(cache.keys())
    elif kind == "cache":
        return cache
    else:
        raise ValueError(f"Unknown kind {kind}")




# %%
