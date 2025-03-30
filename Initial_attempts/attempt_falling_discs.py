#%%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib tk

rng = np.random.default_rng()

size = (200, 200)

r_mean = 10
r_std = 1
coverage = 80

FA = coverage/100 * size[0] * size[1] # fibre area
afa = np.pi * r_mean**2 # average fibre area
N = int(FA / afa) # number of fibres
N = int(1.5 * N) # to be on the safe side

print(f'Number of fibres: {N}')

envelope = np.zeros(size[0])
circles = []

#%%

fig, ax = plt.subplots()

n = 0
FA = 0

while n < 1.1 * N and np.any(envelope[1:-1] < size[1]): 

    radius = rng.normal(loc=r_mean, scale=r_std)
 
    if any(envelope > size[1]):
        x = np.argmin(envelope[2:-2]) + 2 + rng.uniform(low=-0.5, high=0.5)
    else:
        x = rng.uniform(low=1, high=size[0]-2)
    
    n += 1

    ax.clear()  
    ax.plot([0, 0, size[0], size[0], 0], [0, size[1], size[1], 0, 0], 'k-')
    for j, (x, y, r) in enumerate(circles):
        circle = plt.Circle((x, y), r, color='b', fill=False)
        ax.add_artist(circle)
        ax.text(x, y, j, color='b', fontsize=8, ha='center', va='center')
        ax.plot(range(size[0]), envelope)

    iter = 0
    while iter < 100:
        iter += 1

        s1 = np.maximum(np.floor(x - radius), 1)
        s2 = np.minimum(np.ceil(x + radius), size[0] - 2)
        xc = np.arange(s1, s2 + 1).astype(int)
        if xc.size == 0:
            x = rng.uniform(low=1, high=size[0]-2)
            continue
        d = radius - np.maximum((radius**2 - (xc - x)**2), 0)**0.5
    
        ye = [np.max(envelope[xc-1]-d), np.max(envelope[xc]-d), np.max(envelope[xc+1]-d)]
        y = ye[1] + radius
        ax.add_artist(plt.Circle((x, y), radius, color='r', fill=False))
    
        if ye[0] < min(ye[1], ye[2]) and x > 1:
            x = x - 1
        elif ye[2] < min(ye[0], ye[1]) and x < size[0] - 2:
            x = x + 1
        else:
            break

    circles.append((x, y, radius))
    A = np.pi * radius**2

    def segar(t):
        return radius**2 * np.arccos(1 - t / radius) - (radius - t) * (radius**2  - (radius-t)**2)**0.5
    
    h = 0
    if x - radius < 0:
        A -= segar(radius - x)
    elif x + radius > size[0] - 1:
        A -= segar(x + radius - size[0] + 1)

    if y + radius > size[1]: ## I ignore the overlap for now
        if y < size[1]:
            A -= segar(y + radius - size[1])
        if y - radius < size[1]:
            A -= (np.pi * radius**2 - segar(size[1] - y + radius))
        else:
            A = 0

    FA += max(A, 0)  
    
    yc = y + np.sqrt(np.maximum(radius**2 - (xc - x)**2, 0))
    envelope[xc] = yc

    # for j, (x, y, r) in enumerate(circles):
    #     circle = plt.Circle((x, y), r, color='b', fill=False)
    #     ax.add_artist(circle)
    #     # ax.text(x, y, j, color='b', fontsize=8, ha='center', va='center')
    #     # ax.plot(range(size[0]), envelope)

    ax.set_title(f'Fibre {n}, coverage: {100 * FA/(size[0]*size[1]):.1f}')    
    ax.set_aspect('equal')
    fig.canvas.draw()
    plt.pause(0.1)  # Removing this may result in picture not updating. So rather change to very, very small number

plt.show()
    
# %%
