# %%

import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn

from lib.utils import *
from lib.datasets import *
from lib.networks import *
from lib.attack import *

# %% Load the networks and dataset - ImageNet
vae, classifier = get_imagenet_networks()
dataset, LABELS, preprocess = get_imagenet("validation[:2000]")

# %% Load the networks and dataset - MNIST
vae, classifier = get_mnist_networks(3)
dataset, LABELS, preprocess = get_mnist("test[:2000]")

# %%
def get(index: int):
    image = preprocess(dataset[index]['image'])
    label = dataset[index]['label']
    show(image[None], [LABELS[label]])
    return image, label

# %% Find a small perturbation that yields a different prediction

image, label = get(5)
target_image, target = get(15)

torch.cuda.empty_cache()
attack = Attack(image, vae, classifier, )

# %%

attack.train(
    target_class=(target, 1),
    # target_class=7,
    # target_image=(target_image, 1),
    # away_from_label=(label, 1),
    max_l1_distance=127/255,
    num_steps=1000,
    lr=1,
    noise_std=0.05,
    l2_penalty=1,
    batch_size=10,
)
# %%
attack.show(label, LABELS)

# %% histogram of perturbation
px.histogram(attack.perturbation.detach().cpu().numpy().flatten(), nbins=1000)

# %%
pert = attack.perturbation.detach().cpu().numpy()
# Generate noise from the same distribution as the perturbation
mu, sigma = pert.mean(), pert.std() / 3


# %% Check the evolution of perturbation inside the vae


vae.eval()
adversarial_image = (image + attack.perturbation).clamp(0, 1)
network = nn.Sequential(
    vae,
    classifier,
)

with record_activations(network, "nonlin", "relu") as cache_clean:
    network(image[None])

with record_activations(network, "nonlin", "relu") as cache_adv:
    network(adversarial_image[None])

with record_activations(network, "nonlin", "relu") as cache_clean_pert:
    noise = torch.randn_like(image) * sigma + mu
    network((image + noise).clamp(0, 1)[None])

with record_activations(network, "nonlin", "relu") as cache_adv_pert:
    noise = torch.randn_like(image) * sigma + mu
    network((adversarial_image + noise).clamp(0, 1)[None])

other_image = preprocess(dataset[52]['image'])
with record_activations(network, "nonlin", "relu") as cache_other:
    network(other_image[None])

with record_activations(network, "nonlin", "relu") as cache_other_pert:
    noise = torch.randn_like(image) * sigma + mu
    network((other_image + noise).clamp(0, 1)[None])


# %% Plot the distance between the two

def plot_diff_evolution(clean: Cache, adv: Cache, **plotly_kwargs):
    # use relative error
    dist = torch.tensor([torch.dist(a, b) / torch.dist(a, torch.zeros_like(a))
                        for a, b in zip(clean.values(), adv.values())])
    norms = torch.tensor([a.pow(2).mean() for a in clean.values()])
    numels = torch.tensor([a.numel() for a in clean.values()])
    x = list(clean.keys())

    # Plot on two different axis, because the scales are very different
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=dist, name="L1 distance"))
    # fig.add_trace(go.Scatter(x=x, y=numels, name="Number of elements", yaxis="y2"))
    fig.add_trace(go.Scatter(x=x, y=norms, name="Norm", yaxis="y2"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right"),
                    height=1200,
                    **plotly_kwargs)
    fig.show()

# plot_diff_evolution(cache_clean, cache_adv, title="Evolution of the perturbation inside the VAE<br>Start vs. adversarial image")
# plot_diff_evolution(cache_clean, cache_other, title="Evolution of the perturbation inside the VAE<br>Start vs. random image")
# plot_diff_evolution(cache_adv, cache_other, title="Evolution of the perturbation inside the VAE<br>Adversarial image vs. random image")
# # %%

# plot_diff_evolution(cache_clean_pert, cache_adv, title="Start+noise vs. adversarial image")
# plot_diff_evolution(cache_clean, cache_clean_pert, title="Start vs. start+noise")
# plot_diff_evolution(cache_adv, cache_adv_pert, title="Adversarial image vs. adversarial image+noise")

# %% Instead make a grid plot with each pair of images
caches = [cache_clean, cache_adv, cache_clean_pert, cache_adv_pert, cache_other, cache_other_pert]
titles = ["Start", "Adversarial image", "Start+noise", "Adversarial image+noise", "Other image", "Other image+noise"]

fig = make_subplots(rows=len(caches), cols=len(caches),
                    column_titles=titles,
                    row_titles=titles,
                    shared_xaxes=True,
                    shared_yaxes=True,
)

for i, cache1 in enumerate(caches):
    for j, cache2 in enumerate(caches):
        dist = torch.tensor([torch.dist(a, b) / torch.dist(a, torch.zeros_like(a))
                            for a, b in zip(cache1.values(), cache2.values())])
        cosine_sim = torch.tensor([torch.cosine_similarity(a.flatten(), b.flatten(), dim=0)
                            for a, b in zip(cache1.values(), cache2.values())])
        y = dist
        fig.add_trace(go.Scatter(x=list(cache1.keys()), y=y, showlegend=False), row=i+1, col=j+1)

fig.update_layout(height=1000, width=1000)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(range=[0, 1 if y is cosine_sim else 2])
fig.show()

# %% Plot the norms for each cache, on one plot
baseline = torch.tensor([a.pow(2).mean() for a in cache_other.values()])
fig = go.Figure()
for cache, title in zip(caches, titles):
    norms = torch.tensor([a.pow(2).mean() for a in cache.values()])
    fig.add_trace(go.Scatter(x=list(cache.keys()), y=(norms / baseline) ** 1, name=title))
fig.update_layout(height=800)
fig.update_layout(title="Norm of the activations inside the VAE, relative to the norm of a random image")
fig.update_xaxes(title_text="Layer",
                 # showticklabels=False
                 )
fig.update_yaxes(title_text="Norm")
fig.show()


# %% report adv vs adv+noise
report(adversarial_image, label, vae, classifier, LABELS, noise)

# %% Print the latent on the adversarial image
with record_activations(vae, "nonlin", "relu") as cache:
    vae(image[None])

print(cache["1.encoder Sequential"])