#%%

import torch
import plotly.graph_objects as go
from plotly.colors import qualitative
from tqdm.notebook import tqdm

RNG = torch.Generator().manual_seed(13)

## Helping functions    
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

def t2s(tensor):
    return ' '.join([str(x.item()) for x in tensor])

def save_obj(filename, vertices, faces):
    with open(filename, 'w') as f:
        for i in range(vertices.shape[0]):
            f.write(f'v {t2s(vertices[i])}\n')
        for i in range(faces.shape[0]):
            f.write(f'f {t2s(faces[i])} \n')

def make_tube_faces(z, n=8):
    '''Zero-indexed faces'''
    k = torch.arange(n)
    strip = torch.stack((k, (k + 1)%n, (k + 1)%n + n, n + k), dim=1)
    tube = (torch.cat([strip + i * n for i in range(z - 1)], dim=0))
    return tube

def make_tube_vertices(x, y, z, r, n=8):
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


# %% FibrePacker class

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
        
        self.p0 = None
        self.pZ = None
        self.configuration = None
    
        self.device = None
        self.rng = torch.Generator().manual_seed(13)
        self.colors = qualitative.Plotly
        self.figsize = 600
        self.tqdmformat = '{desc:<50}{bar}{n_fmt}/{total_fmt}'
        self.tqdmwidth = 600
        
    def set_radii(self, radii):
        self.radii = radii
        self.N = len(radii)
        self.min_d = minimal_distance(radii)

    ## MAIN WORKFLOW METHODS
    
    def initialize_start_slice(self):
        max_R = self.R - self.radii.max()
        ri = max_R * torch.sqrt(torch.rand(self.N, generator=self.rng))
        ai = torch.rand(self.N, generator=self.rng) * 2 * torch.pi
        self.p0 = torch.stack((ri * torch.cos(ai), ri * torch.sin(ai)))
    
    def initialize_end_slice_original(self):
        self.pZ = self.p0.clone()
        self.rotate_bundle((1/2, 0), 1/2.5, torch.tensor(-torch.pi/2))
        self.rotate_bundle((1/2, 0), 1/2, torch.tensor(-torch.pi/3))
        self.rotate_bundle((0, 0), 1, torch.tensor(torch.pi/4))
        self.swap_points(1/5)

    def initialize_end_slice(self, misalignment, k=3):
        # k is the number of clusters

        if misalignment=='minimal':
            angle_range = (0, 5)
            swap = (0.01, 3)
        elif misalignment=='tiny':
            angle_range = (5, 10)
            swap = (0.02, 6)
        elif misalignment=='low':
            angle_range = (10, 15)
            swap = (0.04, 12)
        elif misalignment=='medium':
            angle_range = (15, 20)
            swap = (0.08, 24)
        elif misalignment=='high':
            angle_range = (20, 25)
            swap = (0.16, 48)

        self.pZ = self.p0.clone()
        clusters = self.cluster(k)

        da = angle_range[1] - angle_range[0]
        angles = - da + 2 * da * torch.rand(k + 1)
        angles = torch.sign(angles) * angle_range[0] + angles
        angles = angles/ 180 * torch.pi
        for i, cluster in enumerate(clusters):
            self.rotate_cluster(cluster, angles[i])
        self.rotate_bundle((0, 0), 1, angles[-1])
        self.swap_points(swap[0], knn=swap[1])



    def interpolate_configuration(self, Z, z_multiplier=1, type='mixed'):
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
        self.configuration = (1 - w) * self.p0 + w * self.pZ
        self.Z = Z
        self.z_multiplier = z_multiplier
    

    def cluster(self, k):
        '''Very rough clustering of points.'''
        kappa = 0.5  # if 1 all points are clustered, if 0 many are not
        c = self.p0[:, torch.randperm(self.N, generator=self.rng)[:k] ] # initial centers
        for step in range(10): 
            d = pairwise_distance(self.p0, c).min(dim=1).indices
            c = torch.stack([self.p0[:, d==i].mean(dim=1) for i in range(k)], dim=1)

        v, d = pairwise_distance(self.p0, c).min(dim=1)
        r = kappa * v.max() + (1 - kappa) * v.mean()  # somewhere between the mean and the max
        d[v > r] = -1  # remove points that are too far 
        clusters = [d==i for i in range(k)]
        return clusters

    def rotate_bundle(self, center, radius, angle):
        '''Rotate a bundle around a center.'''
        center = self.R * torch.tensor(center).unsqueeze(1)
        bundle = (self.pZ - center).norm(dim=0) + self.radii < self.R * radius
        c, s = torch.cos(angle), torch.sin(angle)
        R = torch.tensor([[c, -s], [s, c]])
        self.pZ[:, bundle] = R @ (self.pZ[:, bundle] - center) + center

    def rotate_cluster(self, cluster, angle):
        '''Rotate a bundle around a center.'''
        center = self.pZ[:, cluster].mean(dim=1, keepdim=True)
        c, s = torch.cos(angle), torch.sin(angle)
        R = torch.tensor([[c, -s], [s, c]])
        self.pZ[:, cluster] = R @ (self.pZ[:, cluster] - center) + center

    def swap_points(self, fraction, knn=36):
        '''Swap random (but relatively close) points.'''
        knn = min(knn, self.N//2) # consider k nearest neighbors for swapping
        d = pairwise_distance(self.pZ)
        inds = torch.topk(d, knn + 1, largest=False)[1]
        ri = torch.randint(1, knn, (self.N, ), generator=self.rng)
        pairs = torch.stack((torch.arange(self.N), 
                inds[torch.arange(self.N), ri]), dim=1)
        pairs, _ = torch.sort(pairs, dim=1)
        pairs = torch.unique(pairs, dim=0)
        pairs = pairs[torch.randperm(pairs.shape[0], generator=self.rng)]
        for pair in pairs[:int(fraction * self.N)]:
            self.pZ[:, pair] = self.pZ[:, pair.flip(0)]

    ## VISUALIZATION METHODS

    def show_radii_distribution(self, nbins=100):
        # I don't know exactly how plotly computes number of bins from nbinsx
        data = [go.Histogram(x=self.radii, nbinsx=nbins)]
        fig = go.Figure(data=data)
        fvf = self.radii.pow(2).sum() / self.R**2 * 100
        title = f"Radii distribution, N = {self.N}, mean = {self.radii.mean():.2f}, fvf = {fvf:.1f}"
        fig.update_layout(title=title, xaxis_title='Radii', yaxis_title="Count", 
                width=self.figsize, height=self.figsize//2)
        fig.show()
    
    def get_slice_circles(self, id):
        if id=='start':
            p = self.p0
        elif id=='end':
            p = self.pZ
        else:
            p = self.configuration[id]
        shapes = []
        for i, r in enumerate(self.radii):
            x, y = p[:, i]
            shapes.append(dict(x0=x - r, y0=y - r, x1=x + r, y1=y + r, 
                    type="circle", fillcolor=self.color(i), opacity=0.5, line_width=0))
        shapes.append(dict(x0=-self.R, y0=-self.R, x1=self.R, y1=self.R,
                type="circle", line_color='gray', opacity=0.5, line_width=5))
        return shapes

    def show_slice(self, id, title=None):
        fig = go.Figure()
        fig.update_layout(shapes=self.get_slice_circles(id))
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

    def animate_slices(self, title=None, radii=None):
        if radii is None:
            radii = self.radii
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
    

    # OPTIMIZATION METHODS

    def select_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using device {self.device}")
        return self.device


    def optimize_slice_points(self, id, iters=200, delta=0.01, weights=None, lr=0.1, k=3):        
        if id == 'start':
            p = self.p0
        elif id == 'end':
            p = self.pZ
        
        delta = delta * self.radii.mean()
        
        default_weights = {'overlap': 1, 
                'separation': 1/self.N,
                'protrusion': self.N}
        if weights is not None:
            default_weights.update(weights)
        weights = default_weights
                
        if self.device is None:
            self.select_device()

        p = p.to(self.device)
        p.requires_grad = True
        min_d = self.min_d.to(self.device)
        radii = self.radii.to(self.device)
        
        optimizer = torch.optim.Adam([p], lr=lr)
        losses = {key: [] for key in weights.keys()}
        #progress_bar = tqdm(range(iters), bar_format=self.tqdmformat, 
        #        ncols=self.tqdmwidth)
        for iter in range(iters):  
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
            
            #progress_bar.set_description(
            #    f"Over. {overlap:.2f}, sep. {separation:.2f}, prot. {protrusion:.1f}",
            #    refresh=True)

        if id == 'start':
            self.p0 = p.detach().to('cpu')
        elif id == 'end':
            self.pZ = p.detach().to('cpu')  
      
        self.show_losses(losses)


    def optimize_configuration(self, iters=200, delta=0.01, weights=None, lr=0.1):
        delta = delta * self.radii.mean()

        if self.device is None:
            self.select_device()

        configuration = self.configuration.to(self.device)
        configuration.requires_grad = True
        min_d = self.min_d.to(self.device)
        radii = self.radii.to(self.device)
        p0 = self.p0.to(self.device)
        pZ = self.pZ.to(self.device)

        default_weights = {'overlap': 1, 
                        'crossover': 1,
                        'protrusion': self.N, 
                        'stretching': 1/self.N,
                        'bending': 1/self.N,
                        'boundary': 1}
        if weights is not None:
            default_weights.update(weights)
        weights = default_weights
    
        optimizer = torch.optim.Adam([configuration], lr=lr)
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
        self.show_losses(losses)


    # ADDITIONAL HELPING METHODS

    def analyse_configuration(self, radii=None, return_=False):
        '''Various measures about the result. Can be used with other radii.'''
        if radii is None:
            radii = self.radii
        
        min_d = minimal_distance(radii)
        self.configuration.requires_grad = False

        def get_topk_indices(a, k=5):
            k = min(k, a.shape[1]//2)
            _, j = torch.topk(a.flatten(), k)
            j = j.unsqueeze(1)
            return torch.cat((j//a.shape[1], j%a.shape[1]), dim=1)

        def indices_to_coordinates(summary):
            x, y = self.configuration[summary[:,0], :, summary[:,1]].T
            z = summary[:,0]
            if summary.shape[1] == 3:
                x2, y2 = self.configuration[summary[:,0], :, summary[:,2]].T
                x, y = 0.5 * (x + x2) , 0.5 * (y + y2)
            return x, y, z

        def overlap_for_conf(conf):
            d = pairwise_distance(conf)
            overlap_matrix = torch.relu(torch.triu(torch.relu(min_d - d), diagonal=1))
            overlap_indices = torch.nonzero(overlap_matrix)
            overlap_values = overlap_matrix[torch.unbind(overlap_indices, dim=1)]
            overlap_coordinates = indices_to_coordinates(overlap_indices)
            return overlap_values, overlap_coordinates

        overlap_values, overlap_coordinates = overlap_for_conf(self.configuration)
        overlap_mid_values, omc = overlap_for_conf(0.5 * (self.configuration[1:] 
                    + self.configuration[:-1]))
        overlap_mid_coordinates = (omc[0], omc[1], omc[2] + 0.5) 
    
        protrusion_matrix = torch.relu(self.configuration.norm(dim=-2) + radii - self.R)
        protrusion_indices = torch.nonzero(protrusion_matrix)
        protrusion_values = protrusion_matrix[torch.unbind(protrusion_indices, dim=1)]
        protrusion_coordinates = indices_to_coordinates(protrusion_indices)

        stretching_matrix = (1/2)**2 * (self.configuration[2:] 
            - 2 * self.configuration[1:-1] + self.configuration[:-2]).pow(2).sum(dim=1)
        stretching_summary = get_topk_indices(stretching_matrix) + torch.tensor([1, 0])
        stretching_coordinates = indices_to_coordinates(stretching_summary)
        
        bending_matrix = (1/6)**2 * (- self.configuration[4:] 
            + 4 * self.configuration[3:-1] - 6 * self.configuration[2:-2] 
            + 4 * self.configuration[1:-3] - self.configuration[:-4]).pow(2).sum(dim=1)
        bending_summary = get_topk_indices(bending_matrix) + torch.tensor([2, 0])
        bending_coordinates = indices_to_coordinates(bending_summary)

        analysis = {
            'overlap_values': overlap_values,
            'overlap_coordinates': overlap_coordinates,
            'overlap_mid_values': overlap_mid_values,
            'overlap_mid_coordinates': overlap_mid_coordinates,
            'protrusion_values': protrusion_values,
            'protrusion_coordinates': protrusion_coordinates,
            'stretching_coordinates': stretching_coordinates,
            'stretching_summary': stretching_summary,
            'bending_coordinates': bending_coordinates,
            'bending_summary': bending_summary
        } 
        self.show_3D_configuration_analysis(analysis, title='Location of configuration issues')
        self.show_analysis_distribution(analysis, title='Distribution of configuration issues')
        if return_:
            return analysis
    
    def show_analysis_distribution(self, analysis, title, nbins=50):
        fig = go.Figure()
        for id in ['overlap', 'overlap_mid', 'protrusion']:
            fig.add_trace(go.Histogram(x=analysis[id + '_values'], nbinsx=nbins, name=id))        
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
                    line_color = c[id], opacity=0.5, mode='lines', line=dict(width=6)))
            cx, cy, cz = analysis[id + '_coordinates']
            fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode='markers', opacity=0.75, marker={'color': c[id], 'symbol':'cross'}))

        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y',
            zaxis_title='Z', aspectmode='cube'), showlegend=False,
            width=self.figsize, height=self.figsize)
        if title:
            fig.update_layout(title=title)
        fig.show()
    
    def fix_radii(self, epsilon=1e-3):

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
        fix = self.radii - fix

        before = self.radii.pow(2).sum() / self.R**2 * 100
        after = fix.pow(2).sum() / self.R**2 * 100

        fig = go.Figure()
        x = torch.arange(self.N)
        fig.add_trace(go.Bar(x=x, y=self.radii, name='radii', opacity=0.5))
        fig.add_trace(go.Bar(x=x, y=fix, name='fix', opacity=0.5))
        fig.update_layout(title=f'Fixed radii. FVF before {before:.1f}, after {after:.1f}', 
                xaxis_title='Fibre ID', yaxis_title="Radii", 
                width=self.figsize, height=self.figsize//2)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=self.radii, name='radii', opacity=0.5))
        fig.add_trace(go.Histogram(x=fix, name='fix', opacity=0.5))
        fig.update_layout(title=f'Fixed radii. FVF before {before:.1f}, after {after:.1f}', 
            xaxis_title='Radii', yaxis_title="Count", 
            width=self.figsize, height=self.figsize//2)
        fig.show()
        return fix
    
    def save_result(self, filename, radii=None):
        if radii is None:
            radii = self.radii
        x, y = self.configuration.transpose(0, 1)
        with open(filename, 'w') as f:
            for xi, yi, ri in zip(x.T, y.T, radii):
                f.write(f"{ri:.6f}\n")
                f.write(" ".join(f"{i:.6f}" for i in xi) + "\n")
                f.write(" ".join(f"{i:.6f}" for i in yi) + "\n")
        print(f"Saved to {filename}")

    
    def save_mesh(self, filename, radii=None, n=16):
        
        if radii is None:
            radii = self.radii
        x, y = self.configuration.transpose(0, 1)
        z = torch.arange(self.Z) * self.z_multiplier

        tube = make_tube_faces(self.Z, n)

        faces = []
        vertices = []
        for i, (xi, yi, ri) in enumerate(zip(x.T, y.T, radii)):

            vertices.append(make_tube_vertices(xi, yi, z, ri, n))
            faces.append(tube + self.Z * n * i)
        
        vertices = torch.cat(vertices, dim=0)
        faces = torch.cat(faces, dim=0)
        save_obj(filename, vertices, faces + 1)

        
           

