# Goal: Train an VAE on MNIST, then find adversarial examples that transform the
# image 7 into an 8 when put throught the VAE

# %%

import time
from sklearn import manifold
from sympy import per
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, reduce, repeat
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm, trange
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

from lib.mnist import MnistNet, load_mnist, prepare_mnist, MnistVAE
from lib.utils import *

# %% Seed everything
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# %% Load mnist
train_data, train_targets, test_data, test_targets = load_mnist()
LABELS = [str(i) for i in range(10)]

print("Train data", train_data.shape, train_data.device, train_data.dtype)
print("Train targets", train_targets.shape, train_targets.device, train_targets.dtype)

data_min = train_data.min()
data_max = train_data.max()
data_range = data_max - data_min
print("Data range:", data_range, "min:", data_min, "max:", data_max)

# %% Load the classifier
classifier = MnistNet.load().to(device)

# %% Plot some MNIST images
index = 0
fig = px.imshow(train_data[index, 0].cpu(), title=f"Label: {train_targets[index].item()}, predicted: {classifier(train_data[None, index]).argmax()}")
fig.show()


# %%  show the average of dataset images
avg = train_data.mean(0)
(train_data - avg).pow(2).mean()
# %% Save the VAE
torch.save(vae.state_dict(), f"vae-mnist-{LATENT_SIZE}.pt")




# %% Load the VAE
vae = MnistVAE(LATENT_SIZE).to(device)
vae.load_state_dict(torch.load(f"vae-mnist-{LATENT_SIZE}.pt"))


# %% Create an adversarial perturbation to transform
# ------------------

START = 0
TARGET = 11
start_image = test_data[START]
target_image = test_data[TARGET]
target_label = test_targets[TARGET].unsqueeze(0)
start_label = test_targets[START].unsqueeze(0)

other_labels = torch.arange(10, device=device) != start_label

with torch.no_grad():
    target_through_vae, target_latent = vae(target_image.unsqueeze(0), return_latent=True)

vae.show(torch.stack([start_image, target_image]))
print("START label:", start_label.item())
print("TARGET label:", target_label.item())

# %% Find the adversarial perturbation
vae.requires_grad_(False)
classifier.requires_grad_(False)
classifier.to(device)
vae.eval()
classifier.eval()

perturbation = nn.Parameter(torch.randn_like(start_image))
optimizer = optim.AdamW([perturbation], lr=0.01, weight_decay=0)
loss_record = LossRecord()
adversarial_image_record = []

# %%
THRESHOLD = 0.000

target_label = torch.tensor([9], device=device)
max_norm = data_range * 10 / 255
max_norm = data_range * 0.4
max_norm = None
FGSM = False
epsilon = data_range * 0.1

batch_size = 10
steps = 200
penalisation_strength = 5
pen_norm = 1

penalty_dist_to_start = 30

lr = 0.1 ** 1

for g in optimizer.param_groups:
    g['lr'] = lr

for epoch in range(1):
    progress = trange(steps)
    for i in progress:
        adversarial_image = (start_image + perturbation).unsqueeze(0).clamp(
            data_min, data_max)
        out, latent = vae.forward(adversarial_image.repeat(
            batch_size, 1, 1, 1),
                                  return_latent=True)
        predictions = classifier(out).log_softmax(-1).mean(0, keepdim=True)

        loss_record.log("Top prediction", predictions.argmax())

        # loss = F.mse_loss(out, target_through_vae.repeat(batch_size, 1, 1, 1))
        # loss = F.mse_loss(latent, target_latent.repeat(batch_size, 1))

        # loss = F.cross_entropy(predictions, target_label)
        # loss = -predictions[:, target_label]
        # loss = -F.cross_entropy(predictions, start_label)
        vals = predictions[:, other_labels]
        loss = -vals.max()
        # loss = -vals.topk(2).values.sum() - vals.max()
        # loss = torch.clamp(loss, min=0.01)
        loss_record.log("Loss", loss)

        if penalty_dist_to_start:
            penalty = F.mse_loss(start_image,
                                 adversarial_image) * penalty_dist_to_start
            loss += penalty
            loss_record.log("Penalty", penalty)

        if pen_norm == 1:
            penalisation = perturbation.abs().mean()
        elif pen_norm == 2:
            penalisation = perturbation.pow(2).mean()
        else:
            penalisation = 0

        penalisation = penalisation_strength * penalisation
        loss_record.log("Penalisation", penalisation)

        total_loss = loss + penalisation
        loss_record.log("Total loss", total_loss)

        optimizer.zero_grad()
        total_loss.backward()

        if FGSM:
            perturbation.data.add_(perturbation.grad.sign(),
                                   alpha=-epsilon / steps)
        else:
            optimizer.step()

        # Clamp the perturbation to the valid range
        if max_norm is not None:
            perturbation.data = perturbation.data.clamp(-max_norm, max_norm)
        # perturbation.data[:, 3:-3, 3:-3] = 0
        # perturbation.data[:, 1:-1, 1:-1] = 0
        # perturbation.data.renorm_(p=norm_p, dim=0, maxnorm=max_norm)

        if i % 100 == 0:
            progress.set_postfix(loss=loss.item(), pen=penalisation.item())
            if loss < THRESHOLD:
                break

        adversarial_image_record.append(adversarial_image.detach())
        loss_record.step()

    loss_record.plot()

    adversarial_image = (start_image + perturbation).clamp(data_min, data_max)
    # Visualize the attack. Plot (start, start+perturbation, pert, output, target_through_vae)
    # in two rows
    vae.show(
        [start_image, adversarial_image],
        title_text="Targeted adversarial attack on a variational autoencoder",
        row_titles=["Initial image", "Adversarial image"])

    predictions = classifier(vae(adversarial_image[None]))
    probabilities = F.softmax(predictions, dim=-1)[0]
    start_probability = probabilities[start_label].item()
    best = probabilities.argmax().item()
    best_probability = probabilities[best].item()
    print("START label:", start_label.item())
    print("TARGET label:", target_label.item())
    print("Best label:", best)
    print("Best probability:", best_probability)
    print("Start probability:", start_probability)
    print("Target probability:", probabilities[target_label].item())

    # Plot the evolution of the adversarial image
    adversarial_video = torch.cat(adversarial_image_record)
    out_video = vae(adversarial_video)
    to_show = torch.cat([adversarial_video, out_video], dim=1)
    px.imshow(
        to_show.cpu(),
        animation_frame=0,
        facet_col=1,
    ).show()
    # Compare distances
    images = {
        "Start": start_image,
        "Target": target_image,
        "Adversarial": adversarial_image,
        "Adversarial output": vae(adversarial_image[None]).squeeze(0),
        "Start output": vae(start_image[None]).squeeze(0),
    }
    # Plot 2d distance matrix
    distances = torch.zeros(len(images), len(images))
    for i, (name1, image1) in enumerate(images.items()):
        for j, (name2, image2) in enumerate(images.items()):
            distances[i, j] = F.mse_loss(image1, image2).item()
    fig = px.imshow(distances.cpu(),
                    x=list(images.keys()),
                    y=list(images.keys()),
                    title="MSE between images")
    fig.show()

    # %%

    fig.update_layout(
        height=700,
        width=700,
        title_text="Untargeted adversarial attack on a variational autoencoder",
    )
    #  title_text=f"Start ({LABELS[start_label.item()]}): {start_probability:.2%}, "
    # f"Best ({LABELS[best]}): {best_probability:.2%}")
    fig.show()

# %%

vae.show([start_image, adversarial_image],)

diff_original = F.mse_loss(start_image[None], vae(start_image[None]))
diff_adversarial = F.mse_loss(start_image[None], vae(adversarial_image[None]))

print("Original MSE:", diff_original.item())
print("Adversarial MSE:", diff_adversarial.item())

F.mse_loss(start_image, adversarial_image)


# %%

random_image = torch.rand_like(start_image) * data_range + data_min
vae.show([random_image])



# %% Load CIFAR10 data

from lib.cifar_classifier import load_cifar10, load_classifier

train_data, train_targets, valid_data, valid_targets = load_cifar10(device, torch.float16)
# %% Show some images

n = 14
show(train_data[:n], train_targets[:n], unnormalize="cifar")

# %%

from diffusers import AutoencoderKL

vae: nn.Module = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
vae.cuda()

#%%
from torchinfo import summary

summary(vae, (1, 3, 256, 256))

# %%

# Load data/diego-loupe.jpg
from PIL import Image
from torchvision.transforms import ToTensor
image = ToTensor()(Image.open("data/diego-loupe.jpg")).cuda().permute(0, 2, 1)
# Make square
side = min(image.shape[1:])
image = image[:, :side, :side]
# Scale to 256x256
image = F.interpolate(image[None], size=256).squeeze(0)

# %%
out = vae(image[None])

show(256 * torch.stack([image, out[0][0].detach()]).clamp(0, 1), ["Original", "Reconstruction"], False)

# %% Show adversarial_image_record as a video
with torch.no_grad():
    inputs = torch.stack(adversarial_image_record[650:700])
    outs = torch.cat([vae(i[None])[0] for i in inputs])
to_show = torch.stack([inputs, outs], dim=0).cpu().clamp(0, 1)
px.imshow(to_show.permute(0, 1, 3, 4, 2), animation_frame=1, facet_col=0).show()

# %% Clean cuda memory
torch.cuda.empty_cache()

# Load the classifier: resnet-50

from torchvision.models import resnet50
classifier = resnet50(pretrained=True).cuda().eval()

# %% Download imagenet from huggingface if not already downloaded
from datasets import load_dataset
imagenet = load_dataset("imagenet-1k", split="validation[:10]",
                        cache_dir="/scratch/diego/data/")

# %%
LABELS = imagenet.features['label'].names
# %%

image, label = imagenet[1].values()
print(LABELS[label], label)
display(image)

from torchvision.transforms import ToTensor, CenterCrop, Compose, Resize
preprocess = Compose([Resize(256), CenterCrop(256), ToTensor()])

image = preprocess(image).cuda()

print(image.max(), image.min(), image.shape)
show(image[None], [LABELS[label]])


# %% Find a small perturbation that yields a different prediction

perturbation = nn.Parameter(torch.randn_like(image) * 0.01)
optimizer = optim.AdamW([perturbation], lr=0.01, weight_decay=0)
loss_record = LossRecord()
adversarial_image_record = []

# %%
steps = 2000
epsilon = 120 / 255
target = 999
target_image = preprocess(imagenet[2]['image']).cuda()

for step in trange(steps):
    perturbation.data.clamp_(-epsilon, epsilon)
    adversarial_image = (image + perturbation).clamp(0, 1)
    out = vae(adversarial_image[None])[0].clamp(0, 1)

    # in_diff = F.mse_loss(image, adversarial_image)
    # out_diff = F.mse_loss(out, image[None])
    # loss = (gamma + in_diff) / out_diff
    # loss_record.log("In diff", in_diff)
    # loss_record.log("Out diff", out_diff)
    # loss_record.log("Loss", loss)

    probas = classifier(out).softmax(-1)
    norm_penalty = perturbation.pow(2).mean() * 0.4
    # guide = probas[0, label]
    # guide = 1-probas[0, target]
    guide = F.mse_loss(out, target_image[None])
    loss = guide + norm_penalty
    loss_record.log("Guide", guide)
    loss_record.log("Norm penalty", norm_penalty)
    loss_record.log("Loss", loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # adversarial_image_record.append(adversarial_image.detach())
    loss_record.step()

loss_record.plot()
# show(256 * torch.stack([image, adversarial_image, out[0].detach()]).clamp(0, 1), ["Original", "Adversarial", "Reconstruction"], False)

def report(image, label, vae, classifier, perturbation=None, labels=LABELS):
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

    out = vae(inputs)[0].clamp(0, 1)
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
    show(to_classify.clamp(0, 1), titles)

    # Show the perturbation as heatmap
    if perturbation is not None:
        show(perturbation + 0.5, ["Perturbation"])






report(image, label, vae, classifier, perturbation=perturbation)

# %%
show(target_image[None], [LABELS[999]])