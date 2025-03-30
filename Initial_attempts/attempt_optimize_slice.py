#%%

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
%matplotlib tk

rng = torch.Generator().manual_seed(13)

def plot_circles(p, radii, c, ax):
    for i in range(p.shape[1]):
        circle = patches.Circle((p[0, i].item(), p[1, i].item()), radii[i].item(), edgecolor='b', facecolor='none')
        ax.add_patch(circle)
    ax.add_patch(patches.Circle((0, 0), c, edgecolor='y', facecolor='none'))
    ax.set_xlim(-c, c)
    ax.set_ylim(-c, c)
    ax.set_aspect('equal')

c = 100  # Domain radius
r0 = 5  # Mean fibre radius
fvf = 60  # Desired fibre volume fraction
N = int(fvf / 100 * (c**2) / (r0**2))  # Number of fibres
radii = torch.randn(N, generator=rng) * 1.5 + r0
radii *= (fvf / 100 * c**2 / (radii**2).sum())**0.5
min_d = radii.unsqueeze(1) + radii.unsqueeze(0)

ri = torch.sqrt(torch.rand(N, generator=rng) * c**2)
ai = torch.rand(N, generator=rng) * 2 * torch.pi

# Define p as a parameter that requires gradients
p = torch.stack((ri * torch.cos(ai), ri * torch.sin(ai)))
p.requires_grad = True

optimizer = torch.optim.Adam([p], lr=0.1)
fig, ax = plt.subplots()
plot_circles(p, radii, c, ax)

delta = 0 # 0.01 * r0  # The desired distance between circles

losses = []

for iter in range(1, 20001):  # Number of optimization steps
    
    optimizer.zero_grad()
    
    d = ((p[0:1] - p[0:1].T)**2 + (p[1:2] - p[1:2].T)**2)
    d.fill_diagonal_(torch.inf)
    d = d**0.5
    
    # Calculate overlap
    overlap = torch.relu(min_d - d).sum()

    # Attraction
    closest_vals, closest_inds = torch.topk(d, 2, largest=False)
    attraction = ((closest_vals[:, 0] - radii - radii[closest_inds[:, 0]] - delta)**2).sum()
    attraction += ((closest_vals[:, 1] - radii - radii[closest_inds[:, 1]] - delta)**2).sum()
    
    # Penalize circles that are outside the domain
    r = (p[0]**2 + p[1]**2)**0.5
    penalty = torch.relu(r + radii - c).sum()
    
    # Total loss
    loss = overlap + 1/N * attraction + N * penalty
    losses.append(loss.item())
    
    loss.backward()
    optimizer.step()
    
    if iter % 100 == 0:
        ax.clear()  # Prevent matplotlib accumulating all it has drawn.
        plot_circles(p, radii, c, ax)
        ax.set_title(f'Iter {iter}, loss {loss.item():0.1f}, overlap {overlap.item():0.1f}')
        plt.pause(0.01)

plt.show()

fig, ax = plt.subplots()
ax.plot(losses)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
plt.show()

# %%