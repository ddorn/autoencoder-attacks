
# %%
import os
from typing import Type, TypeVar

from diffusers import AutoencoderKL
from diffusers.models.vae import DecoderOutput
from tqdm import tqdm, trange
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50
from datasets import load_dataset

from lib.utils import *
from lib.datasets import *


T = TypeVar("T")
WeightedOptional = T | tuple[T, float] | None

class Attack:
    def __init__(self,
                 start: Float[Tensor, "channels height width"],
                 vae: nn.Module | None,
                 classifier: nn.Module,
                 optimizer: Type[optim.Optimizer] = optim.AdamW,
                 init_noise: float = 0.1,
                 ) -> None:
        assert start.ndim == 3
        assert start.shape[0] in (1, 3)
        assert start.min() >= 0 and start.max() <= 1

        self.start = start
        self.vae = vae if vae is not None else lambda x: x
        self.classifier = classifier
        self.perturbation = nn.Parameter(torch.randn_like(start) * init_noise)
        self.optimizer = optimizer([self.perturbation], lr=0.01, weight_decay=0)
        self.loss_record = LossRecord()

    @property
    def adversarial_image(self):
        return (self.start + self.perturbation).clamp(0, 1)

    def set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self,
        target_class: WeightedOptional[int] = None,
        target_image: WeightedOptional[Float[Tensor, "channels=3 height width"]] = None,
        away_from_label: WeightedOptional[int] = None,
        max_l1_distance: float | None = None,
        lr: float | None = None,
        l2_penalty: Callable[[float], float] | float | None = None,
        noise_std: float | None = None,
        num_steps: int = 1,
        batch_size: int = 1,
        early_stop: float | Callable[[dict[str, float]], bool] | None = None,
        normalize_grad: bool = None,
    ):
        # Add a weight of 1 when not specified
        if isinstance(target_class, tuple):
            target_class, target_class_weight = target_class
        elif target_class is not None:
            target_class_weight = 1
        if isinstance(target_image, tuple):
            target_image, target_image_weight = target_image
        elif target_image is not None:
            target_image_weight = 1
        if isinstance(away_from_label, tuple):
            away_from_label, away_from_label_weight = away_from_label
        elif away_from_label is not None:
            away_from_label_weight = 1

        args = dict(locals())
        args.pop("self")
        args.pop("num_steps")
        args.pop("early_stop")
        args.pop("target_image")
        args = {k: v for k, v in args.items() if v is not None}
        if callable(l2_penalty):
            args["l2_penalty"] = l2_penalty(0)
        self.loss_record.log("args", args)

        assert target_class is not None or target_image is not None or away_from_label is not None, \
            "At least one of target_class, target_image or away_from_label must be specified"

        if lr is not None:
            self.set_lr(lr)

        progress = trange(num_steps, disable=num_steps < 5)
        for step in progress:
            inputs = self.adversarial_image.repeat(batch_size, 1, 1, 1)
            if noise_std is not None:
                inputs = inputs + torch.randn_like(inputs) * noise_std
            out = self.vae(inputs.clamp(0, 1)).clamp(0, 1)
            logits = self.classifier(out)

            # Compute the loss
            parts = {}
            if target_image is not None:
                parts["Target image"] = F.mse_loss(out, target_image[None]) * target_image_weight
            if target_class is not None:
                parts["Target class"] = -logits.log_softmax(-1).mean(0)[target_class] * target_class_weight
            if away_from_label is not None:
                parts["Away from label"] = logits.softmax(-1).mean(0)[away_from_label] * away_from_label_weight
            if l2_penalty is not None:
                if callable(l2_penalty):
                    penalty = l2_penalty(step / num_steps)
                    self.loss_record.log("args/l2_penalty", penalty)
                else:
                    penalty = l2_penalty
                parts["L2 penalty"] = (self.perturbation ** 2).mean() * penalty

            loss = sum(parts.values())

            # Log
            for name, value in parts.items():
                self.loss_record.log(name, value)
            self.loss_record.log("Loss", loss)
            # if step % 25 == 0:
            progress.set_description(f"Loss: {loss:.5f}")

            self.optimizer.zero_grad()
            loss.backward()
            if normalize_grad:
                self.perturbation.grad /= self.perturbation.grad.norm()
            self.step_perturbation()
            self.loss_record.step()
            self.loss_record.log("Max gradient", self.perturbation.grad.abs().max())

            if max_l1_distance is not None:
                self.perturbation.data.clamp_(-max_l1_distance, max_l1_distance)

            if callable(early_stop):
                if early_stop(locals()):
                    break
            elif early_stop is not None:
                if loss < early_stop:
                    break

    def step_perturbation(self):
        self.optimizer.step()

    def show(self, correct_label: int, labels: list[str]):
        self.loss_record.plot()
        report(self.start, correct_label, self.vae, self.classifier, labels, self.perturbation)

    def export_settings(self, *ignore_args: str) -> list[dict[str, object]]:
        """
        Exports the arguments passed to attack.train() as a list of dicts.

        Args:
            *ignore_args: Arguments to ignore when exporting settings. For example, if you want to
                ignore the target image, you can pass "target_image", without any leading "args/".
        """

        names = [name for name in self.loss_record.buffers
                 if name.startswith("args/")
                 and not any(name.startswith(f"args/{arg}") for arg in ignore_args)]
        xs_with_changes = {}
        for name in names:
            for x in self.loss_record.buffers_x[name]:
                xs_with_changes.setdefault(x, []).append(name)
        print(xs_with_changes)

        settings = []
        for x, names in sorted(xs_with_changes.items()):
            setting = {}
            for name in names:
                setting[name[len("args/"):]] = self.loss_record.buffers[name][self.loss_record.buffers_x[name].index(x)]
            settings.append(setting)

        # Set num_steps for each setting
        xs = sorted(xs_with_changes)
        xs.append(self.loss_record._step)
        num_steps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]

        def eq(a, b):
            """Checks if two objects are equal recursively, even if they are tensors"""
            if type(a) != type(b):
                return False
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a.keys())
            if isinstance(a, torch.Tensor):
                return torch.equal(a, b)
            if isinstance(a, list):
                return len(a) == len(b) and all(eq(a[i], b[i]) for i in range(len(a)))
            return a == b

        # Merge settings with same args and set num_steps
        merged_settings: list[dict] = []
        merged_num_steps: list[int] = []
        for setting, num_step in zip(settings, num_steps):
            if merged_settings and eq(setting, merged_settings[-1]):
                merged_num_steps[-1] += num_step
            else:
                merged_settings.append(setting)
                merged_num_steps.append(num_step)
        for setting, num_step in zip(merged_settings, merged_num_steps):
            setting["num_steps"] = num_step

        return merged_settings

class FgsmAttack(Attack):
    def __init__(self, *args, optimizer=optim.SGD, **kwargs):
        super().__init__(*args, optimizer=optimizer, init_noise=0, **kwargs)

    def step_perturbation(self):
        self.perturbation.grad.sign_()
        super().step_perturbation()
