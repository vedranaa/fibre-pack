#%%

import torch
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

RNG = torch.Generator().manual_seed(13)  

def show_slice(p, radii, R, title=None):
    colors = px.colors.qualitative.Plotly
    shapes = []
    for i in range(len(radii)):
        x, y = p[:, i].detach()
        r = radii[i].detach()
        color = colors[i % len(colors)]
        shapes.append(dict(x0=x - r, y0=y - r, x1=x + r, y1=y + r, 
                type="circle", fillcolor=color, opacity=0.5, line_width=0))
    shapes.append(dict(x0=-R, y0=-R, x1=R, y1=R,
            type="circle", line_color='gray', opacity=0.5, line_width=5))
    r = radii.mean()
    fig = go.Figure()
    fig.update_layout(shapes=shapes,
        xaxis=dict(range=[-R - r, R + r]),
        yaxis=dict(range=[-R - r, R + r], scaleanchor="x"))
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

def animate_slices(configuration, radii, R, title=None):
    frames = []
    colors = px.colors.qualitative.Plotly
    for z, p in enumerate(configuration.detach()):
        shapes = []
        for i, (r, (x, y)) in enumerate(zip(radii, p.T)):
            shapes.append(dict(x0=x - r, y0=y - r, x1=x + r, y1=y + r, type="circle",
                fillcolor=colors[i % len(colors)], opacity=0.5, line_width=0))
        shapes.append(dict(x0=-R, y0=-R, x1=R, y1=R, type="circle",
            line_color='gray', opacity=0.5, line_width=5))
        frames.append(go.Frame(layout=dict(shapes=shapes), name=str(z)))
    
    xyrange = [-R - radii.max(), R + radii.max()]
    layout = animation_controls(configuration.shape[0])
    layout.update(xaxis=dict(range=xyrange), yaxis=dict(range=xyrange, scaleanchor="x"))   
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

def show_radii_distribution(radii):
    fig = px.histogram(radii, title='Radii distribution', nbins=50)
    fig.update_layout(showlegend=False) 
    fig.show()

def initialize_slice_points(R, N, generator=RNG):
    ri = torch.sqrt(torch.rand(N, generator=generator) * R**2)
    ai = torch.rand(N, generator=generator) * 2 * torch.pi
    p = torch.stack((ri * torch.cos(ai), ri * torch.sin(ai)))
    return p

def initialize_configuration_naively(R, N, Z, generator=RNG):
    p = initialize_slice_points(R, N)
    configuration = [p]
    a = 0 * p
    for i in range(1, Z):
        a +=  0.1 * torch.randn_like(p, generator=generator)
        configuration.append(configuration[-1] + a)
    configuration = torch.stack(configuration)
    r = configuration.norm(dim=1).max(dim=0)[0]
    s = torch.clamp(r/R, 1)
    s = 0.5 * (s + r.max()/R)
    scale = torch.outer(torch.linspace(0, 1, Z), s - 1) + 1
    return configuration / scale.view(Z, 1, N)

def initialize_radii(R, fvf, r_mean, r_sigma=0, generator=RNG):
    '''Get radii for a domain radius, FVF and mean (and sigma) for fibre radius.'''
    # This is a bit hacky, if radii follow the normal distribution, the FVF is
    # not depending only on the mean, but also on the sigma.
    f = fvf / 100 * (R**2) 
    N = int(f / (r_mean**2))   
    r = r_mean + r_sigma * torch.randn(N, generator=generator)
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

def overlap_penalty(R, min_d, delta=0):
    '''Overlap between circles.'''
    return torch.relu(min_d - R + delta).sum()

def protrusion_penalty(p, radii, R, delta=0):
    '''Protrusion of circles outside the domain.'''
    r = p.norm(dim=-2)
    return torch.relu(r + radii - R + delta).sum()

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

def swap_points(p, K=None, generator=RNG):
    '''Swap random (but relatively close) points.'''
    k = 36  # Consider k nearest neighbors for swapping
    N = p.shape[1]  # Number of points
    k = min(k, N//2)
    d = pairwise_distance(p)
    if K is None:
        K = N//5  # 120 percent of points
    inds = torch.topk(d, k + 1, largest=False)[1]
    ri = torch.randint(1, k, (p.shape[1], ), generator=generator)
    pairs = torch.stack((torch.arange(N), inds[torch.arange(N), ri]), dim=1)
    pairs, _ = torch.sort(pairs, dim=1)
    pairs = torch.unique(pairs, dim=0)
    pairs = pairs[torch.randperm(pairs.shape[0], generator=generator)]
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

def stretching_penalty(conf):
    s = conf[2:] + conf[:-2] - 2 * conf[1:-1]
    return (1/2) * s.pow(2).sum()

def bending_penalty(conf):
    s = (- conf[4:] - conf[:-4] + 4 * conf[3:-1] 
            + 4 * conf[1:-3] - 6 * conf[2:-2])
    return (1/6) * s.pow(2).sum()

def boundary_penalty(conf, p0, pZ):
    return (conf[0] - p0).pow(2).sum() + (conf[-1] - pZ).pow(2).sum()

def select_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device {device}")
    return device

