#%%
'''
Demonstrates issue with skimage iradon when nr_bins is even.

`skimage` expects the center of rotation to be in bin with the index
nr_bins//2 which is off-center when nr_bins is even.

Adding empty line to sinogram does not help, as I need center of rotation 
between two bins, not in any particular bin.

'''


from skimage.transform import iradon, iradon_sart
import numpy as np
import matplotlib.pyplot as plt

nr_angles = 90
nr_bins = 100
side = np.linspace(-1, 1, nr_bins)
r = 0.5
p = np.maximum(r**2 - side**2, 0)**0.5

sinogram = np.stack(nr_angles * [p], axis=1)
theta = np.arange(nr_angles) * 180 / nr_angles
reconstruction = iradon(sinogram, theta=theta)
x, y = np.meshgrid(side, side)
test = x**2 + y**2 - (r)**2 < 0
difference = reconstruction/reconstruction.sum() - test/test.sum()

fig, ax = plt.subplots(1, 4, figsize=(12, 3))
ax[0].imshow(sinogram)
ax[1].imshow(reconstruction)
ax[2].imshow(test)
ax[3].imshow(difference)
plt.show()



#%%