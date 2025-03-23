#%%

import torch
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

def show_slice(p, radii, c, title=None):
    colors = px.colors.qualitative.Plotly
    shapes = []
    for i in range(len(radii)):
        x, y = p[:, i].detach()
        r = radii[i].detach()
        color = colors[i % len(colors)]
        shapes.append(dict(x0=x - r, y0=y - r, x1=x + r, y1=y + r, 
                type="circle", fillcolor=color, opacity=0.5, line_width=0))
    shapes.append(dict(x0=-c, y0=-c, x1=c, y1=c,
            type="circle", line_color='gray', opacity=0.5, line_width=5))
    r = radii.mean()
    fig = go.Figure()
    fig.update_layout(shapes=shapes,
        xaxis=dict(range=[-c - r, c + r]),
        yaxis=dict(range=[-c - r, c + r], scaleanchor="x"))
    if title:
        fig.update_layout(title=title)
    fig.show()

def show_3D_configuration(configuration, title=None):
    Z, _, N = configuration.shape
    x, y = configuration.transpose(0, 1).detach()
    z = torch.arange(Z).view((-1, 1)).repeat((1, N))
    group = torch.arange(N).repeat((Z, 1))
    fig = px.line_3d(x=x.ravel(), y=y.ravel(), z=z.ravel(), 
            line_group=group.ravel())
    if title:
        fig.update_layout(title=title)
    fig.show()

def animate_configuration(configuration, title=None):
    Z, _, N = configuration.shape
    x, y = configuration.transpose(0, 1).detach()
    c = 1.1 * max(x.abs().max(), y.abs().max())
    z = torch.arange(Z).view((-1, 1)).repeat((1, N))
    fig = px.scatter(x=x.ravel(), y=y.ravel(), animation_frame=z.ravel(), 
        range_x=[-c, c], range_y=[-c, c])
    fig.update_layout(
        xaxis=dict(range=[-c, c]),
        yaxis=dict(range=[-c, c], scaleanchor="x"))
    if title:
        fig.update_layout(title=title)
    fig.show()

def animation_controls(Z):
    '''Animation controls for Z slices.'''
    layout = go.Layout(
        updatemenus=[{"buttons": [
            {"args": [None, {"frame": {"duration": 500, "redraw": True}, 
                "fromcurrent": True}], "label": ">", "method": "animate"},
            {"args": [[None], {"frame": {"duration": 0, "redraw": True}, 
                "mode": "immediate", "transition": {"duration": 0}}],
                "label": "||", "method": "animate"}],
            "direction": "left", "pad": {"r": 10, "t": 30}, "showactive": False,
            "type": "buttons", "x": 0, "xanchor": "left", "y": 0, "yanchor": "top"}],
        sliders=[{"steps": [
            {"args": [[str(z)], { "frame": {"duration": 300, "redraw": True},
                "mode": "immediate", "transition": {"duration": 300}}],
                "label": str(z), "method": "animate"} for z in range(Z)],
            "x": 0.2, "len": 0.8, "xanchor": "left", "y": 0, "yanchor": "top",
            "pad": {"b": 10, "t": 10},
            "currentvalue": {"font": {"size": 15}, "prefix": "Slice:",
                "visible": True, "xanchor": "right"},
            "transition": {"duration": 300, "easing": "cubic-in-out"}}])
    return layout

def animate_slices(configuration, radii, c, title=None):
    frames = []
    colors = px.colors.qualitative.Plotly
    for z, p in enumerate(configuration.detach()):
        shapes = []
        for i, (r, (x, y)) in enumerate(zip(radii, p.T)):
            shapes.append(dict(x0=x - r, y0=y - r, x1=x + r, y1=y + r, type="circle",
                fillcolor=colors[i % len(colors)], opacity=0.5, line_width=0))
        shapes.append(dict(x0=-c, y0=-c, x1=c, y1=c, type="circle",
            line_color='gray', opacity=0.5, line_width=5))
        frames.append(go.Frame(layout=dict(shapes=shapes), name=str(z)))
    
    rng = [-c - radii.max(), c + radii.max()]
    layout = animation_controls(configuration.shape[0])
    layout.update( xaxis=dict(range=rng), yaxis=dict(range=rng, scaleanchor="x"))   
    fig = go.Figure(layout=layout, frames=frames)
    fig.update_layout(shapes=frames[0].layout.shapes)
    if title:
        fig.update_layout(title=title)
    fig.show()

def show_losses(loss_contributions):
    fig = go.Figure()
    for k, v in loss_contributions.items():
        fig.add_trace(go.Scatter
            (x=list(range(len(v))), y=v, mode='lines', name=k))
    fig.update_layout(title='Loss contributions', xaxis_title='Iteration',
        yaxis_title='Loss', yaxis_type='log') 
    fig.show()

def initialize_slice_points(c, N):
    ri = torch.sqrt(torch.rand(N, generator=RNG) * c**2)
    ai = torch.rand(N, generator=RNG) * 2 * torch.pi
    p = torch.stack((ri * torch.cos(ai), ri * torch.sin(ai)))
    return p

def initialize_configuration_naively(c, N, Z):
    p = initialize_slice_points(c, N)
    configuration = [p]
    a = 0 * p
    for i in range(1, Z):
        a +=  0.1 * torch.randn_like(p, generator=RNG)
        configuration.append(configuration[-1] + a)
    configuration = torch.stack(configuration)
    r = configuration.norm(dim=1).max(dim=0)[0]
    s = torch.clamp(r/c, 1)
    s = 0.5 * (s + r.max()/c)
    scale = torch.outer(torch.linspace(0, 1, Z), s - 1) + 1
    return configuration / scale.view(Z, 1, N)

def initialize_radii(c, fvf, r_mean, r_sigma=0):
    '''Get radii for a domain radius, FVF and mean (and sigma) for fibre radius.'''
    f = fvf / 100 * (c**2) 
    N = int(f / (r_mean**2))  # This is not correct if r_sigma 
    r = r_mean + r_sigma * torch.randn(N, generator=RNG)
    r = torch.clamp(r, 0.01 * r_mean)
    return r * (f / (r**2).sum()) ** 0.5

def minimal_distance(radii):
    '''Minimal non-overlap distance between circles with given radii.'''
    min_d = radii.unsqueeze(1) + radii.unsqueeze(0)
    min_d.fill_diagonal_(0)
    return min_d

def pairwise_distance(p):
    '''Pairwise distance between circle centers.'''
    d = (p.unsqueeze(-1) - p.unsqueeze(-2)).norm(dim=-3)
    return d

def overlap_penalty(d, min_d, delta=0):
    '''Overlap between circles.'''
    return torch.relu(min_d - d + delta).sum()

def protrusion_penalty(p, radii, c, delta=0):
    '''Protrusion of circles outside the domain.'''
    r = p.norm(dim=-2)
    return torch.relu(r + radii - c + delta).sum()

def separation_penalty(d, radii, n, delta=0):
    '''Separation to n nearest neighbors, part larger than delta.'''
    vals, inds = torch.topk(d, n + 1, largest=False)
    vals, inds = vals[:, 1:], inds[:, 1:]
    return torch.relu(vals - radii[:, None] - radii[inds] - delta).sum()
    #return torch.pow(vals - radii[:, None] - radii[inds] - delta, 2).sum()

def rotate_bundle(p, radii, center, radius, angle):
    '''Rotate a bundle around a center.'''
    center = torch.tensor(center)[:,None]
    angle = torch.tensor(angle)
    bundle = (p - center).norm(dim=0) + radii < radius
    c, s = torch.cos(angle), torch.sin(angle)
    R = torch.tensor([[c, -s], [s, c]])
    p[:, bundle] = R @ (p[:, bundle] - center) + center
    return p

def swap_points(p, K=None):
    '''Swap random (but relatively close) points.'''
    k = 36  # Consider k nearest neighbors for swapping
    N = p.shape[1]  # Number of points
    k = min(k, N//2)
    d = pairwise_distance(p)
    if K is None:
        K = N//5  # 120 percent of points
    inds = torch.topk(d, k + 1, largest=False)[1]
    ri = torch.randint(1, k, (p.shape[1], ), generator=RNG)
    pairs = torch.stack((torch.arange(N), inds[torch.arange(N), ri]), dim=1)
    pairs, _ = torch.sort(pairs, dim=1)
    pairs = torch.unique(pairs, dim=0)
    pairs = pairs[torch.randperm(pairs.shape[0], generator=RNG)]
    pairs = pairs[:K]
    for pair in pairs:
        p[:, pair] = p[:, pair.flip(0)]
    return p

def interpolate_configuration(p0, pZ, Z, type='mixed'):
    if type == 'linear':
        w = torch.linspace(0, 1, Z)
    elif type == 'logistic':
        w = torch.special.expit(torch.linspace(-4, 4, Z))
        w = (w - w[0]) / (w[-1] - w[0])
    else:
        w = torch.special.expit(torch.linspace(-4, 4, Z))
        w = (w - w[0]) / (w[-1] - w[0])
        w = 0.5 * (w + torch.linspace(0, 1, Z))
    w = w.view(Z, 1, 1)
    configuration = (1 - w) * p0 + w * pZ
    return configuration

def stretching_penalty(c):
    s = c[2:] + c[:-2] - 2 * c[1:-1]
    return (1/2) * s.pow(2).sum()

def bending_penalty(c):
    s = (- c[4:] - c[:-4] + 4 * c[3:-1] 
            + 4 * c[1:-3] - 6 * c[2:-2])
    return (1/6) * s.pow(2).sum()

def boundary_penalty(c, p0, pZ):
    return (c[0] - p0).pow(2).sum() + (c[-1] - pZ).pow(2).sum()

#%%        
def optimize_slice_points(p, radii, c, iters=2000):
    delta = 0.01 * radii.mean()
    n = 3
    N = len(radii)
    min_d = minimal_distance(radii)
    p.requires_grad = True
    p.to(device)
    optimizer = torch.optim.Adam([p], lr=0.1)
    loss_contributions = []
    progress_bar = tqdm(range(iters), bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}')
    for iter in progress_bar:  
        optimizer.zero_grad()   
        d = pairwise_distance(p)
        overlap = overlap_penalty(d, min_d, delta)
        protrusion = protrusion_penalty(p, radii, c)
        separation = separation_penalty(d, radii, n, delta)
        loss = overlap + N * protrusion + 1/N * separation
        loss.backward()
        optimizer.step()
        loss_contributions.append((overlap.item(), N * protrusion.item(), 1/N * separation.item()))
        progress_bar.set_description(f"Overlap {overlap:.2f}, " + 
            f"Protrusion {protrusion:.2f}", refresh=True)
    
    p = p.detach()
    overlap = overlap_penalty(d, min_d)
    protrusion = protrusion_penalty(p, radii, c)
    loss_contributions = {k:list(v) for k, v in 
            zip(['overlap', 'protrusion', 'separation'], zip(*loss_contributions))}
    return p, (overlap, protrusion), loss_contributions

#%%

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

RNG = torch.Generator().manual_seed(13)  # Global random number generator

c = 40  # Domain radius
r0 = 2  # Mean fibre radius
fvf = 70  # Desired fibre volume fraction
Z = 20 # Number of slices

radii = initialize_radii(c, fvf, r0, 0.2 * r0)
N = len(radii)

p0 = initialize_slice_points(c - r0, N)
show_slice(p0, radii, c, title='First slice (p0) initial')

p0, (overlap, protrusion), losses = optimize_slice_points(p0, radii, c)
show_slice(p0, radii, c, 
    title=f'First slice (p0) optimized. Overlap {overlap:.2f}, protrusion {protrusion:.2f}')
show_losses(losses)


pZ = p0.clone()
pZ = rotate_bundle(pZ, radii, (c/2, 0), c/2.5, -torch.pi/2)
pZ = rotate_bundle(pZ, radii, (-c/2, 0), c/2, -torch.pi/3)
pZ = rotate_bundle(pZ, radii, (0, 0), c, torch.pi/4)
pz = swap_points(pZ)

show_slice(pZ, radii, c, title='Last slice (pZ) initial')
pZ, (overlap, protrusion), losses = optimize_slice_points(pZ, radii, c)
show_slice(pZ, radii, c, 
    title=f'Last slice (pZ) optimized. Overlap {overlap:.2f}, protrusion {protrusion:.2f}')
show_losses(losses)

configuration = interpolate_configuration(p0, pZ, Z)
show_3D_configuration(configuration, title='Initial configuration')
animate_configuration(configuration, title='Initial configuration')

#%%

min_d = minimal_distance(radii)
delta = 0.01 * radii.mean()
configuration.requires_grad = True
configuration.to(device)
optimizer = torch.optim.Adam([configuration], lr=0.1)
loss_contributions = []
iters = 2000
progress_bar = tqdm(range(iters), bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}')
for iter in progress_bar:  
    optimizer.zero_grad()   
    d = pairwise_distance(configuration)
    overlap = overlap_penalty(d, min_d, delta)
    protrusion = protrusion_penalty(configuration, radii, c)
    stretching = stretching_penalty(configuration)
    bending = bending_penalty(configuration)
    boundary = boundary_penalty(configuration, p0, pZ)
    loss = overlap + N * protrusion + 1/N * stretching + 2/N * bending + N * boundary
    loss.backward()
    optimizer.step()
    loss_contributions.append((overlap.item(), N * protrusion.item(), 
        1/N * stretching.item(), 2/N * bending.item(), N * boundary.item()))
    progress_bar.set_description(f"Over. {overlap.item():.2f}, " + 
            f"Prot. {protrusion.item():.1f}, " +
            f"Str. {stretching.item():.1f}, " +
            f"Bend. {bending.item():.1f}, " +
            f"Boun. {boundary.item():.1f}",
            refresh=True)

loss_contributions = {k:list(v) for k, v in 
        zip(['overlap', 'protrusion', 'stretching', 'bending', 'boundary'], zip(*loss_contributions))}
show_losses(loss_contributions)

show_3D_configuration(configuration, title='Optimized configuration')
animate_slices(configuration, radii, c, title='Optimized configuration')


# %%
