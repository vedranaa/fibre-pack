#%%

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from tqdm.auto import tqdm  # problems with container width
# from tqdm import tqdm

RNG = torch.Generator().manual_seed(13)

## HELPING FUNCTIONS
def minimal_distance(radii):
    '''Minimal non-overlap distance between circles with given radii.'''
    min_d = radii.unsqueeze(1) + radii.unsqueeze(0)
    min_d.fill_diagonal_(0)
    return min_d

def pairwise_distance(p, q = None):
    '''Pairwise distance between point-clouds.'''
    if q is None:
        q = p
    d = (p.unsqueeze(-1) - q.unsqueeze(-2)).norm(dim=-3)
    return d

def overlap_penalty(d, min_d, delta=0):
    '''Overlap between circles.'''
    return torch.relu(torch.triu(min_d - d + delta,  diagonal=1)).pow(2).sum()

def protrusion_penalty(p, radii, R, delta=0):
    '''Protrusion of circles outside the domain.'''
    r = p.norm(dim=-2)
    return torch.relu(r + radii - R + delta).pow(2).sum()

def separation_penalty(d, radii, n, delta=0):
    '''Separation to n nearest neighbors, part larger than delta.'''
    n = min(n, len(radii) - 1)
    if n < 1:
        return torch.tensor(0)
    vals, inds = torch.topk(d, n + 1, largest=False)
    vals, inds = vals[:, 1:], inds[:, 1:]
    return torch.relu(vals - radii.unsqueeze(1) - radii[inds] - delta).pow(2).sum()

def stretching_penalty(conf):
    '''Penalty for change in direction.'''
    s = conf[2:] - 2 * conf[1:-1] + conf[:-2] 
    return (1/2) * s.pow(2).sum()

def bending_penalty(conf):
    '''Penalty for change in curvature.'''
    s = - conf[4:] + 4 * conf[3:-1] - 6 * conf[2:-2] + 4 * conf[1:-3] - conf[:-4]
    return (1/6) * s.pow(2).sum()

def boundary_penalty(conf, p0, pZ):
    '''Penalty for deviation from boundary conditions.'''
    return (conf[0] - p0).pow(2).sum() + (conf[-1] - pZ).pow(2).sum()

def animation_controls(Z, prefix='Slice '):
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
            "currentvalue": {"font": {"size": 15}, "prefix": prefix,
                "visible": True, "xanchor": "right"},
            "transition": {"duration": 300, "easing": "cubic-in-out"}}])
    return layout

def animate(volume, prefix='Slice ', figsize=600):
    '''Animate slices of a given volume.'''
    if type(volume) is torch.Tensor:
        volume = volume.numpy()
    vmin = volume.min()
    vrange = volume.max() - vmin
    nr_slices = volume.shape[0]
    frames = []
    for i in range(nr_slices):
        s = volume[i]
        s = (255 * (s - vmin) / vrange).astype(np.uint8)
        image = np.stack([s, s, s], axis=-1)  # plotly wants rgb image
        frames.append(go.Frame(data=[go.Image(z=image)], name=str(i)))
    layout = animation_controls(nr_slices, prefix=prefix)
    layout.update({'width': figsize, 'height': figsize})
    fig = go.Figure(layout=layout, frames=frames)
    fig.add_trace(frames[0].data[0])
    fig.show()

def t2s(tensor):
    '''Tensor to string, for exporting mesh.'''
    return ' '.join([str(x.item()) for x in tensor])

def save_obj(filename, vertices, faces):
    with open(filename, 'w') as f:
        for i in range(vertices.shape[0]):
            f.write(f'v {t2s(vertices[i])}\n')
        for i in range(faces.shape[0]):
            f.write(f'f {t2s(faces[i])} \n')

def append_faces(filename, faces):
    '''Appending faces to obj file, to close tube ends with polygons.'''
    with open(filename, 'a') as f:
        for i in range(faces.shape[0]):
            f.write(f'f {t2s(faces[i])} \n')

def make_tube_faces(z, n=8):
    '''Zero-indexed faces of tube for exporting as mesh.'''
    k = torch.arange(n)
    strip = torch.stack((k, (k + 1)%n, (k + 1)%n + n, n + k), dim=1)
    tube = (torch.cat([strip + i * n for i in range(z - 1)], dim=0))
    return tube

def make_tube_vertices(x, y, z, r, n=8):
    '''Vertices of tube for exporting as mesh.'''
    if r.ndim == 0:
        r = r.unsqueeze(0)
    x, y, z, r = x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1), r.unsqueeze(0)
    a = torch.arange(n)* 2 * torch.pi / n
    sa = torch.sin(a).unsqueeze(0)
    ca = torch.cos(a).unsqueeze(0)
    X = x + r * ca
    Y = y + r * sa
    Z = z.repeat(1, n)
    vertices = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1)
    return vertices

def indices_to_coordinates(summary, configuration):
    '''From indices of problematic fibres to coordinates.'''
    x, y = configuration[summary[:,0], :, summary[:,1]].T
    z = summary[:,0]
    if summary.shape[1] == 3:  # when dealing with overlap we have 2 points
        x2, y2 = configuration[summary[:,0], :, summary[:,2]].T
        x, y = 0.5 * (x + x2) , 0.5 * (y + y2)
    return x, y, z

def overlap_for_configuration(configuration, min_d, return_coordinates=True):
    '''Overlap values and coordinates given a configuration, 
    interpolated configuration, or a slice'''
    if configuration.ndim == 2:  # Single slice case
        configuration = configuration.unsqueeze(0)  # Add a Z dimension to handle slices
    d = pairwise_distance(configuration)
    overlap_matrix = torch.relu(torch.triu(torch.relu(min_d - d), diagonal=1))
    overlap_indices = torch.nonzero(overlap_matrix)
    overlap_values = overlap_matrix[torch.unbind(overlap_indices, dim=1)]
    if not return_coordinates:
        return overlap_values
    overlap_coordinates = indices_to_coordinates(overlap_indices, configuration)
    return overlap_values, overlap_coordinates

def get_mapping_function(sigma=None, edge=None):
    '''For mapping the signed distance field to intensity.'''
    def f(x):
        if sigma is None:
            return 1 - torch.heaviside(x, torch.tensor(0.5))
        hg = 1 - 0.5 * (1 + torch.special.erf(x / (2**0.5 * sigma)))
        if edge is None:
            return hg
        else:
            return hg - edge * x / sigma * torch.exp(0.5 * (1 - (x / sigma)**2))
    return f


## FIBRE PACKER CLASS

def from_n(R, N, r_mean, r_sigma=0, rng=None):
    fib = FibrePacker()
    fib.R = R
    fib.N = N
    if rng is None:
        rng = torch.Generator().manual_seed(13)
    fib.rng = rng
    if r_sigma:
        radii = r_mean + r_sigma * torch.randn(N, generator=fib.rng)
        radii = torch.clamp(radii, 0.01 * r_mean)
    else:
        radii = r_mean * torch.ones(N)
    fib.set_radii(radii)
    fib.set_weights()
    return fib

def from_fvf(R, fvf, r_mean, r_sigma=0, rng=None):
    N = int(fvf / 100 * (R**2) / (r_mean**2))   
    fib = from_n(R, N, r_mean, r_sigma, rng)
    fix = (fvf / 100 * (R**2) / fib.radii.pow(2).sum()).pow(0.5)
    fib.set_radii(fib.radii * fix)
    return fib

class FibrePacker():
    def __init__(self):
        self.R = None
        self.radii = None
        self.N = None
        self.Z = None
        
        self.boundaries = {'start': None, 'end': None}
        self.configuration = None
    
        self.device = None
        self.rng = torch.Generator().manual_seed(13)
        self.learning_rate = 0.05
        
        self.slice_weights = {
                'overlap': None, 
                'separation': None, 
                'protrusion': None
                }
        self.configuration_weights = {
            'overlap': None,
            'protrusion': None,
            'crossover': None,
            'stretching': None,
            'bending': None,
            'boundary': None
        }
        self.separation_neighbors = 3
        self.overlap_delta = 0.01
        
        self.colors = qualitative.Plotly
        self.figsize = 600  # in pixels
        self.tqdmformat = '{desc:<50}{bar}{n_fmt}/{total_fmt}'
        self.tqdmwidth = 600  # in pixels for tqdm.notebook, else in characters
        
    ### INITIALIZATION METHODS

    def set_weights(self):
        
        self.slice_weights = {
            'overlap': 1, 
            'separation': 1/1000,
            'protrusion': 1,
            }
        self.configuration_weights = {
            'overlap': 1, 
            'protrusion': 1,
            'crossover': 1,
            'stretching': 1/100,
            'bending': 1/100,
            'boundary': 1
            }

    def set_radii(self, radii):
        self.radii = radii
        self.N = len(radii)
        self.min_d = minimal_distance(radii)

    def initialize_start_slice(self):
        max_R = self.R - self.radii.max()
        ri = max_R * torch.sqrt(torch.rand(self.N, generator=self.rng))
        ai = torch.rand(self.N, generator=self.rng) * 2 * torch.pi
        self.boundaries['start'] = torch.stack((ri * torch.cos(ai), ri * torch.sin(ai)))
    
    def initialize_end_slice_first_try(self):
        if self.boundaries['start'] is None:
            print("Aborting. Start slice not initialized.")
            return
        self.boundaries['end'] = self.boundaries['start'].clone()
        self.rotate_bundle((1/2, 0), 1/2.5, torch.tensor(-torch.pi/2))
        self.rotate_bundle((1/2, 0), 1/2, torch.tensor(-torch.pi/3))
        self.rotate_bundle((0, 0), 1, torch.tensor(torch.pi/4))
        self.swap_points(1/5)

    def initialize_end_slice(self, misalignment, k=3):
        # k is the number of clusters
        if self.boundaries['start'] is None:
            print("Aborting. Start slice not initialized.")
            return
        self.boundaries['end'] = self.boundaries['start'].clone()
    
        if misalignment == 'none':
            return
        elif misalignment=='very low':
            angle_range = (0, 5)
            swap = (0.01, 3)
            noise = 0.1
        elif misalignment=='low':
            angle_range = (5, 10)
            swap = (0.02, 6)
            noise = 0.2
        elif misalignment=='moderate':
            angle_range = (10, 15)
            swap = (0.04, 12)
            noise = 0.4
        elif misalignment=='high':
            angle_range = (15, 20)
            swap = (0.08, 24)
            noise = 0.8
        elif misalignment=='very high':
            angle_range = (20, 25)
            swap = (0.16, 48)
            noise = 1.6

        clusters = self.cluster(k)
        da = angle_range[1] - angle_range[0]
        angles = - da + 2 * da * torch.rand(k + 1)
        angles = torch.sign(angles) * angle_range[0] + angles
        angles = angles/ 180 * torch.pi
        for i, cluster in enumerate(clusters):
            self.rotate_cluster(cluster, angles[i])
        self.rotate_bundle((0, 0), 1, angles[-1])
        self.swap_points(swap[0], knn=swap[1])
        self.perturb_points('end', noise)

    # TODO add staggered type where each fibre has different profile 
    # with different z-position of the fastest  movement
    def interpolate_configuration(self, Z, z_multiplier=1, type='mixed'):
        if (self.boundaries['start'] is None) or (self.boundaries['end'] is None):
            print("Aborting. Boundary slices not initialized.")
            return
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
        self.configuration = (1 - w) * self.boundaries['start'] + w * self.boundaries['end']
        self.Z = Z
        self.z_multiplier = z_multiplier
    
    def cluster(self, k):
        '''Very rough clustering of points.'''
        kappa = 0.5  # if 1 all points are clustered, if 0 many are not
        c = self.boundaries['start'][:, torch.randperm(self.N, generator=self.rng)[:k] ] # initial centers
        for step in range(10): 
            d = pairwise_distance(self.boundaries['start'], c).min(dim=1).indices
            c = torch.stack([self.boundaries['start'][:, d==i].mean(dim=1) for i in range(k)], dim=1)

        v, d = pairwise_distance(self.boundaries['start'], c).min(dim=1)
        r = kappa * v.max() + (1 - kappa) * v.mean()  # somewhere between the mean and the max
        d[v > r] = -1  # remove points that are too far 
        clusters = [d==i for i in range(k)]
        return clusters

    def rotate_bundle(self, center, radius, angle):
        '''Rotate a bundle around a center.'''
        center = self.R * torch.tensor(center).unsqueeze(1)
        bundle = (self.boundaries['end'] - center).norm(dim=0) + self.radii < self.R * radius
        c, s = np.cos(angle), np.sin(angle)
        R = torch.tensor([[c, -s], [s, c]], dtype=torch.float)
        self.boundaries['end'][:, bundle] = R @ (self.boundaries['end'][:, bundle] - center) + center

    def rotate_cluster(self, cluster, angle):
        '''Rotate a bundle around a center.'''
        center = self.boundaries['end'][:, cluster].mean(dim=1, keepdim=True)
        c, s = np.cos(angle), np.sin(angle)
        R = torch.tensor([[c, -s], [s, c]], dtype=torch.float)
        self.boundaries['end'][:, cluster] = R @ (self.boundaries['end'][:, cluster] - center) + center

    def swap_points(self, fraction, knn=36):
        '''Swap random (but relatively close) points.'''
        knn = min(knn, self.N//2) # consider k nearest neighbors for swapping
        if knn < 2:
            return
        d = pairwise_distance(self.boundaries['end'])
        inds = torch.topk(d, knn + 1, largest=False)[1]
        ri = torch.randint(1, knn, (self.N, ), generator=self.rng)
        pairs = torch.stack((torch.arange(self.N), 
                inds[torch.arange(self.N), ri]), dim=1)
        pairs, _ = torch.sort(pairs, dim=1)
        pairs = torch.unique(pairs, dim=0)
        pairs = pairs[torch.randperm(pairs.shape[0], generator=self.rng)]
        for pair in pairs[:int(fraction * self.N)]:
            self.boundaries['end'][:, pair] = self.boundaries['end'][:, pair.flip(0)]

    def perturb_points(self, id, sigma):
        sigma *= self.radii.mean()
        self.boundaries[id] += sigma * torch.randn(size=self.boundaries[id].shape, generator=self.rng)

    ### VISUALIZATION METHODS

    def show_radii_distribution(self, nbins=100):
        # I don't know exactly how plotly computes number of bins from nbinsx
        data = [go.Histogram(x=self.radii, nbinsx=nbins)]
        fig = go.Figure(data=data)
        fvf = self.radii.pow(2).sum() / self.R**2 * 100
        title = f"Radii distribution, N = {self.N}, mean = {self.radii.mean():.2f}, fvf = {fvf:.1f}"
        fig.update_layout(title=title, xaxis_title='Radii', yaxis_title="Count", 
                width=self.figsize, height=self.figsize//2)
        fig.show()
    
    def get_slice_circles(self, id, show_issues=False):
        '''A helping function for show_slice, turns fibres to circles.'''
        p = None
        if (id=='start') or (id=='end'):
            p = self.boundaries[id]
        else:
            if self.configuration is not None:
                p = self.configuration[id]
        if p is None:
            return
        shapes = []   
        for i, r in enumerate(self.radii):
            x, y = p[:, i]
            shapes.append(dict(x0=x - r, y0=y - r, x1=x + r, y1=y + r, 
                    type="circle", fillcolor=self.color(i), opacity=0.5, line_width=0))
        shapes.append(dict(x0=-self.R, y0=-self.R, x1=self.R, y1=self.R,
                type="circle", line_color='gray', opacity=0.5, line_width=5))

        if show_issues:
            overlap_values, overlap_coordinates = overlap_for_configuration(p, self.min_d)
            xo, yo, _ = overlap_coordinates
            for v, x, y in zip(overlap_values, xo, yo):
                shapes.append(dict(x0=x - 2*v, y0=y - 2*v, x1=x + 2*v, y1=y + 2*v,
                        type="rect", line_color='gray', opacity=0.5, line_width=1))  
        return shapes

    def show_slice(self, id, title=None, show_issues=False):
        fig = go.Figure()
        shapes = self.get_slice_circles(id, show_issues)
        if shapes is None:
            print(f"Aborting. Slice {id} not found.")
            return
        fig.update_layout(shapes=shapes)
        fig.update_layout(self.get_layout())
        if title:
            fig.update_layout(title=title)
        fig.show()
    
    def color(self, i):
        '''Color for i-th fibre.'''
        return self.colors[i % len(self.colors)]

    def show_losses(self, losses):
        fig = go.Figure()
        for k, v in losses.items():
            fig.add_trace(go.Scatter(x=list(range(len(v))), y=v, 
                    mode='lines', name=k))
        fig.update_layout(title='Loss contributions (log scale)', 
                xaxis_title='Iteration', yaxis_title='Loss', yaxis_type='log', 
                width=self.figsize, height=self.figsize//2) 
        fig.show()

    def show_3D_configuration(self, title=None):
        if self.configuration is None:
            print("Aborting. No configuration.")
            return
        x, y = self.configuration.transpose(0, 1)
        z = torch.arange(self.Z)
        fig = go.Figure()
        for i, (xi, yi) in enumerate(zip(x.T, y.T)):
            fig.add_trace(go.Scatter3d(x=xi, y=yi, z=z, mode='lines', 
                line_color = self.color(i), line=dict(width=4)))
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y',
            zaxis_title='Z', aspectmode='cube'), showlegend=False, 
            width=self.figsize, height=self.figsize)
        if title:
            fig.update_layout(title=title)
        fig.show()

    def get_layout(self):
        layout = {'xaxis': dict(range=[-self.R, self.R]),
                'yaxis': dict(range=[-self.R, self.R], scaleanchor="x"),
                'width': self.figsize, 'height': self.figsize}
        return layout

    def animate_slices(self, title=None):
        if self.configuration is None:
            print("Aborting. No configuration.")
            return
        frames = []
        for z in range(self.Z):
            shapes = self.get_slice_circles(z)
            frames.append(go.Frame(layout=dict(shapes=shapes), name=str(z)))
        d = self.R + self.radii.max()
        layout = animation_controls(self.Z)
        layout.update(self.get_layout())   
        fig = go.Figure(layout=layout, frames=frames)
        fig.update_layout(shapes=frames[0].layout.shapes)
        if title:
            fig.update_layout(title=title)
        fig.show()
    
    ### OPTIMIZATION METHODS

    def select_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using device {self.device}")
        return self.device

    def optimize_slice(self, id, iters=200):        
        p = self.boundaries[id]
        if p is None:
            print("Aborting. Slice not initialized.")
            return
        
        delta = self.overlap_delta * self.radii.mean()
        weights = self.slice_weights
        k = self.separation_neighbors
        
        if self.device is None:
            self.select_device()

        p = p.to(self.device)
        p.requires_grad = True
        min_d = self.min_d.to(self.device)
        radii = self.radii.to(self.device)

        optimizer = torch.optim.Adam([p], lr=self.learning_rate)
        losses = {key: [] for key in weights.keys()}
        progress_bar = tqdm(range(iters), bar_format=self.tqdmformat, 
                ncols=self.tqdmwidth)
        for iter in progress_bar:  
            optimizer.zero_grad()   
            d = pairwise_distance(p)
            overlap = overlap_penalty(d, min_d, delta)
            overlap = overlap * weights['overlap']
            separation = separation_penalty(d, radii, k, delta)
            separation = separation * weights['separation']
            protrusion = protrusion_penalty(p, radii, self.R)
            protrusion = protrusion * weights['protrusion']
            loss = overlap + protrusion + separation
            loss.backward()
            optimizer.step()
            losses['overlap'].append(overlap.item())
            losses['separation'].append(separation.item())
            losses['protrusion'].append(protrusion.item())
            
            progress_bar.set_description(
                f"Over. {overlap:.2f}, sep. {separation:.2f}, prot. {protrusion:.1f}",
                refresh=True)

        self.boundaries[id] = p.detach().to('cpu')

        return losses


    def optimize_configuration(self, iters=200):
        if self.configuration is None:
            print("Aborting. No configuration.")
            return
        
        delta = self.overlap_delta * self.radii.mean()

        if self.device is None:
            self.select_device()

        configuration = self.configuration.to(self.device)
        configuration.requires_grad = True
        min_d = self.min_d.to(self.device)
        radii = self.radii.to(self.device)
        p0 = self.boundaries['start'].to(self.device)
        pZ = self.boundaries['end'].to(self.device)

        weights = self.configuration_weights

        optimizer = torch.optim.Adam([configuration], lr=self.learning_rate)
        losses = {key: [] for key in weights.keys()}
        progress_bar = tqdm(range(iters), bar_format=self.tqdmformat, 
                ncols=self.tqdmwidth)
        for iter in progress_bar:  
            optimizer.zero_grad()   
            d = pairwise_distance(configuration)
            overlap = overlap_penalty(d, min_d, delta)
            overlap = overlap * weights['overlap']
            crossover_configuration = 0.5*(configuration[1:] + configuration[:-1])
            crossover_d = pairwise_distance(crossover_configuration)
            crossover = overlap_penalty(crossover_d, min_d, delta)
            crossover = crossover * weights['crossover']
            protrusion = protrusion_penalty(configuration, radii, self.R)
            protrusion = protrusion * weights['protrusion']
            stretching = stretching_penalty(configuration)
            stretching = stretching * weights['stretching']
            bending = bending_penalty(configuration)
            bending = bending * weights['bending']
            boundary = boundary_penalty(configuration, p0, pZ)
            boundary = boundary * weights['boundary']
            loss = overlap + crossover + protrusion + stretching + bending + boundary
            loss.backward()
            optimizer.step()

            losses['overlap'].append(overlap.item())
            losses['crossover'].append(crossover.item())
            losses['protrusion'].append(protrusion.item())
            losses['stretching'].append(stretching.item())
            losses['bending'].append(bending.item())
            losses['boundary'].append(boundary.item())

            progress_bar.set_description(
                f"Over. {overlap:.2f}, cros. {crossover:.2f}, prot. {protrusion:.1f}, "
                f"stre. {stretching:.1f}, bend. {bending:.1f}, boun. {boundary:.1f}",
                refresh=True
            )

        self.configuration = configuration.detach().to('cpu')
        return losses


    def optimize_slice_heuristic(self, id, iters, repetitions=10, 
            change_every=3, show_figures=False):
        '''Temporary reduces separation.'''
        sn = self.separation_neighbors
        sw = self.slice_weights['separation']
        quality = -1, 'unknown'
        best_quality = quality
        iter = 0
        while iter < repetitions:
            if iter > 0: 
                self.perturb_points(id, sigma=0.1)
                if iter % change_every == 0:
                    self.separation_neighbors = max(self.separation_neighbors - 1, 1)  
                    self.slice_weights['separation'] /= 100
            losses = self.optimize_slice(id, iters=iters)
            if show_figures:
                self.show_losses(losses)
                self.show_slice(id, f'Optimized {id} slice', show_issues=False)
            quality = self.assess_analysis_summary(id)
            print(f'Quality {quality[1]}, repetition {iter}/{repetitions}')
            if quality[1] == 'perfect':
                self.separation_neighbors = sn
                self.slice_weights['separation'] = sw
                return losses
            elif quality[0] > best_quality[0]:
                best_quality = quality
                best_slice = self.boundaries[id]
                best_losses = losses
            iter += 1
        else:
            print(f'Using best quality {best_quality[1]}')
            self.boundaries[id] = best_slice
            self.separation_neighbors = sn
            self.slice_weights['separation'] = sw
            return losses
            
    def optimize_configuration_heuristic(self, iters, repetitions=10, 
            change_every=3, show_figures=False):
        '''Temporary reduces stretching, bending and boundary.'''
        quality = -1, 'unknown'
        cw = self.configuration_weights.copy()
        best_quality = quality
        iter = 0
        while iter < repetitions:
            if (iter > 0) and (iter % change_every == 0):
                    self.configuration_weights['stretching'] /= 2
                    self.configuration_weights['bending'] /= 2
                    self.configuration_weights['boundary'] /= 2
            losses = self.optimize_configuration(iters=iters)
            if show_figures:
                self.show_losses(losses)
            quality = self.assess_analysis_summary(id)
            print(f'Quality {quality[1]}, repetition {iter}/{repetitions}')
            if quality[1] == 'perfect':
                self.configuration_weights = cw
                return losses
            elif quality[0] > best_quality[0]:
                best_quality = quality
                best_configuration = self.configuration
                best_losses = losses
            iter += 1
        else:
            print(f'Using best quality {best_quality[1]}')
            self.configuration = best_configuration
            self.configuration_weights = cw
            return losses

    ### ANALYSIS METHODS

    def get_full_analysis(self):
        '''Various measures about the result.'''

        if self.configuration is None:
            print("Aborting. No configuration.")
            return
                
        def get_topk_indices(a, k=5):
            '''Used for finding fibres that stretch or bend the most.'''
            k = min(k, a.shape[1]//2)
            _, j = torch.topk(a.flatten(), k)
            j = j.unsqueeze(1)
            return torch.cat((j//a.shape[1], j%a.shape[1]), dim=1)

        overlap_values, overlap_coordinates = overlap_for_configuration(self.configuration, self.min_d)
        overlap_mid_values, omc = overlap_for_configuration(0.5 * (self.configuration[1:] 
                    + self.configuration[:-1]), self.min_d)
        overlap_mid_coordinates = (omc[0], omc[1], omc[2] + 0.5) 
    
        protrusion_matrix = torch.relu(self.configuration.norm(dim=-2) + self.radii - self.R)
        protrusion_indices = torch.nonzero(protrusion_matrix)
        protrusion_values = protrusion_matrix[torch.unbind(protrusion_indices, dim=1)]
        protrusion_coordinates = indices_to_coordinates(protrusion_indices, self.configuration)

        stretching_matrix = (1/2)**2 * (self.configuration[2:] 
            - 2 * self.configuration[1:-1] + self.configuration[:-2]).pow(2).sum(dim=1)
        stretching_summary = get_topk_indices(stretching_matrix) + torch.tensor([1, 0])
        
        bending_matrix = (1/6)**2 * (- self.configuration[4:] 
            + 4 * self.configuration[3:-1] - 6 * self.configuration[2:-2] 
            + 4 * self.configuration[1:-3] - self.configuration[:-4]).pow(2).sum(dim=1)
        bending_summary = get_topk_indices(bending_matrix) + torch.tensor([2, 0])

        analysis = {
            'overlap_values': overlap_values,
            'overlap_coordinates': overlap_coordinates,
            'overlap_mid_values': overlap_mid_values,
            'overlap_mid_coordinates': overlap_mid_coordinates,
            'protrusion_values': protrusion_values,
            'protrusion_coordinates': protrusion_coordinates,
            'stretching_summary': stretching_summary,
            'bending_summary': bending_summary
        } 
        return analysis
    
    def get_analysis_summary(self, id = None):

        if (id == 'start') or (id == 'end'):
            p = self.boundaries[id]
        else:
            p = self.configuration
        if p is None:
            print("Aborting. No slice or configuration.")
            return
                
        overlap = overlap_for_configuration(p, self.min_d, return_coordinates=False)
        protrusion = torch.relu(p.norm(dim=-2) + self.radii - self.R)
        protrusion = protrusion[torch.unbind(torch.nonzero(protrusion), dim=1)]

        if id is not None:
            return overlap, protrusion
        
        else:
            overlap_mid_values = overlap_for_configuration(0.5 * (p[1:] 
                        + p[:-1]), self.min_d, return_coordinates = False)
            return overlap, protrusion, overlap_mid_values

    def assess_analysis_summary(self, id = None):
        
        if (id=='start') or (id=='end'):
            overlap, protrusion = self.get_analysis_summary(id)
            overlap_mid = torch.tensor([])
            n = self.N 
        else:
            overlap, protrusion, overlap_mid =self.get_analysis_summary()
            n = self.N * self.Z

        no = overlap.nelement()
        np = protrusion.nelement()
        nom = overlap_mid.nelement()

        if (no == 0) and (np == 0) and (nom == 0):
            return 5, 'perfect'
        mo = overlap.max() if no>0 else 0
        mp = protrusion.max() if np>0 else 0
        mom = overlap_mid.max() if nom>0 else 0

        mean_r = self.radii.mean()
        if (no < 0.01 * n and mo < 0.05 * mean_r and
            nom < 0.01 * n and mom < 0.05 * mean_r and 
            mp < 0.001 * n and mp < 0.01 * mean_r):
            return 4, 'great'
        if (no < 0.02 * n and mo < 0.1 * mean_r and
            nom < 0.02 * n and mom < 0.1 * mean_r and 
            mp < 0.002 * n and mp < 0.02 * mean_r):
            return 3, 'ok'
        # mediocre
        return 2, 'bad'
        
    def show_analysis_distribution(self, analysis, title, nbins=50):
        fig = go.Figure()
        for id in ['overlap', 'overlap_mid', 'protrusion']:
            fig.add_trace(go.Histogram(x=analysis[id + '_values'], nbinsx=nbins, 
                    name=id, opacity=0.5))        
        fig.update_layout(xaxis_title='Values (length)', yaxis_title="Count", 
                title=title, width=self.figsize, height=self.figsize//2)
        fig.show()

        
    def show_3D_configuration_analysis(self, analysis, title=None):
        x, y = self.configuration.transpose(0, 1)
        z = torch.arange(self.Z)
        fig = go.Figure()
        for i, (xi, yi) in enumerate(zip(x.T, y.T)):
            fig.add_trace(go.Scatter3d(x=xi, y=yi, z=z, opacity=0.25,
                line_color='gray', mode='lines', line=dict(width=4)))

        for id in ['overlap', 'overlap_mid']:    
            cx, cy, cz = analysis[id + '_coordinates']
            sv = analysis[id + '_values']
            fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode='markers', marker={'color':'black', 'size':2, 'symbol':'circle'}))
            fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode='markers', opacity=0.5, marker={'color':sv, 'size':sv*5, 'symbol':'circle'}))

        cx, cy, cz = analysis['protrusion_coordinates']
        sv = analysis['protrusion_values']
        fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode='markers', marker={'color':'black', 'size':2, 'symbol':'square'}))
        fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode='markers', opacity=0.5, marker={'color':sv, 'size':sv*100, 'symbol':'square'}))

        c = {'stretching': 'blue', 'bending': 'red'}
        for id in ['stretching', 'bending']:
            for i in analysis[id + '_summary'][:, 1]:
                fig.add_trace(go.Scatter3d(x=x[:, i], y=y[:, i], z=z, 
                    line_color = c[id], opacity=0.25, mode='lines', line=dict(width=6)))

        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y',
            zaxis_title='Z', aspectmode='cube'), showlegend=False,
            width=self.figsize, height=self.figsize)
        if title:
            fig.update_layout(title=title)
        fig.show()
    
    def fix_radii(self, epsilon=1e-3):
        if self.configuration is None:
            print("Aborting. No configuration.")
            return

        d = pairwise_distance(self.configuration)
        overlap_matrix = torch.relu(self.min_d - d + epsilon).max(dim=0).values
        overlap_matrix = overlap_matrix.max(dim=0).values/2 

        cross_configuration = 0.5*(self.configuration[1:] + self.configuration[:-1])
        d = pairwise_distance(cross_configuration)
        cross_matrix = torch.relu(self.min_d - d + epsilon).max(dim=0).values
        cross_matrix = cross_matrix.max(dim=0).values/2

        fix = torch.maximum(overlap_matrix, cross_matrix) 

        protrusion_matrix = torch.relu(self.configuration.norm(dim=-2) + 
                self.radii - self.R + epsilon).max(dim=0).values
        fix = torch.maximum(fix, protrusion_matrix)
        fixed_radii = self.radii - fix
        return fixed_radii

    def get_fvp(self, radii=None):
        if radii is None:
            return self.radii.pow(2).sum() / self.R**2 * 100
        else:
            return radii.pow(2).sum() / self.R**2 * 100

    def show_fixed_radii(self, fixed_radii):     
        before = self.get_fvp()
        after = self.get_fvp(fixed_radii)

        fig = go.Figure()
        x = torch.arange(self.N)
        fig.add_trace(go.Bar(x=x, y=self.radii, name='radii', opacity=0.5))
        fig.add_trace(go.Bar(x=x, y=fixed_radii, name='fixed_radii', opacity=0.5))
        fig.update_layout(title=f'Radii and fixed radii. FVF before {before:.1f}, after {after:.1f}', 
                xaxis_title='Fibre ID', yaxis_title="Radii", 
                width=self.figsize, height=self.figsize//2)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=self.radii, name='radii', opacity=0.5))
        fig.add_trace(go.Histogram(x=fixed_radii, name='fix', opacity=0.5))
        fig.update_layout(title=f'Radii and fixed radii distribution. FVF before {before:.1f}, after {after:.1f}', 
            xaxis_title='Radii', yaxis_title="Count", 
            width=self.figsize, height=self.figsize//2)
        fig.show()

    ### I/O METHODS

    def save_result(self, filename):
        
        if self.configuration is None:
            print("Aborting. No configuration.")
            return

        x, y = self.configuration.transpose(0, 1)
        with open(filename, 'w') as f:
            for xi, yi, ri in zip(x.T, y.T, self.radii):
                f.write(f"{ri:.6f}\n")
                f.write(" ".join(f"{i:.6f}" for i in xi) + "\n")
                f.write(" ".join(f"{i:.6f}" for i in yi) + "\n")
        print(f"Saved to {filename}")

    
    def save_mesh(self, filename, close_ends=False, n=16):

        if self.configuration is None:
            print("Aborting. No configuration.")
            return
        
        x, y = self.configuration.transpose(0, 1)
        z = torch.arange(self.Z) * self.z_multiplier
        z = z - z[-1]/2

        tube = make_tube_faces(self.Z, n)
            
        faces = []
        vertices = []
        for i, (xi, yi, ri) in enumerate(zip(x.T, y.T, self.radii)):

            vertices.append(make_tube_vertices(xi, yi, z, ri, n))
            faces.append(tube + self.Z * n * i)
        
        vertices = torch.cat(vertices, dim=0)
        faces = torch.cat(faces, dim=0)
        save_obj(filename, vertices, faces + 1)

        if close_ends:
            end = torch.stack((torch.arange(n), 
                    torch.arange((self.Z - 1) * n, self.Z * n)))
            ends = torch.cat([end + i * n * self.Z for i in range(self.N)], dim=0)
            append_faces(filename, ends + 1)

        print(f"Saved to {filename}")

    ###  PROJECT AND VOXELIZE METHODS

    def resample(self, new_z):
        if type(new_z) is int:
            new_z = torch.linspace(0, self.Z - 1, new_z)
        new_size = (len(new_z), 2, self.N)
        new_configuration = torch.empty(size=new_size, dtype=self.configuration.dtype)
        for i, s in enumerate(new_z):
            f, c = np.floor(s), np.ceil(s)
            if f == c:
                new_configuration[i] = self.configuration[int(f)]
            else:
                new_configuration[i] = (self.configuration[int(f)] * (c - s) 
                        + self.configuration[int(c)] * (s - f))
        return new_configuration

    def project(self, thetas, bins, new_z=None):
        if new_z is None:
            configuration = self.configuration
        else:
            configuration = self.resample(new_z)
        if type(bins) is int:
            bins = torch.linspace(-self.R, self.R, bins)
        if type(thetas) is int:
            thetas = torch.arange(0, thetas) * torch.pi / thetas
        bins = bins.unsqueeze(0).unsqueeze(2)
        radii = self.radii.unsqueeze(0).unsqueeze(0)
        x, y = configuration.transpose(0, 1)
        x, y = x.unsqueeze(1), y.unsqueeze(1)
        p = []
        for theta in thetas:
            f = np.cos(theta) * x + np.sin(theta) * y
            g = 2 * (torch.relu(radii**2 - (bins - f)**2)).pow(0.5)
            p.append(g.sum(dim=2))
        return torch.stack(p, dim=0)
    
    def select_mapping_function(self, transition=None):
        if transition is None:
            mapping_function = get_mapping_function()
        elif type(transition) is str:
            r = self.radii.mean()
            if transition == 'smooth':
                mapping_function = get_mapping_function(sigma=0.1 * r)
            elif tranistion == 'enhanced':
                mapping_function = get_mapping_function(sigma=0.1 * r, edge=0.5)
        elif type(transition) is tuple:
            mapping_function = get_mapping_function(sigma=transition[0], edge=transition[1])
        return mapping_function
     
    def voxelize(self, pixels=None, new_z=None, transition=None):
        if new_z is None:
            configuration = self.configuration
        else:
            configuration = self.resample(new_z)
        if pixels is None:
            pixels = torch.arange(-self.R, self.R, 1)
        elif type(pixels) is int:
            pixels = torch.linspace(-self.R, self.R, pixels)
        elif type(pixels) is not torch.Tensor:
            pixels = torch.tensor(pixels)
        X, Y = torch.meshgrid(pixels, pixels, indexing='xy')
        X, Y = X.unsqueeze(0).unsqueeze(3), Y.unsqueeze(0).unsqueeze(3)
        radii = self.radii.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        x, y = configuration.transpose(0, 1)
        x, y = x.unsqueeze(1).unsqueeze(1), y.unsqueeze(1).unsqueeze(1)
        df = ((X - x).pow(2) + (Y - y).pow(2)).pow(0.5) - radii
        df = df.min(dim=3)[0]

        mapping_function = self.select_mapping_function(transition)
        df = mapping_function(df)

        return df

