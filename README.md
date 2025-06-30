# fibre_packer

Fibre packing optimized with PyTorch. This module provides tools for generating, optimizing, and analyzing 2D and 3D fibre packings with customizable fibre volume fraction (FVF) and misalignment. Achieved FVF values (shown in parentheses in the figure below) closely match the targets, even at high FVF and significant misalignment. The module is designed for use in computational materials science, especially for simulating fibre-reinforced composites. 

![Results](mosaic.png)

## Features

- Generate random fibre packings with specified FVF, domain radius, mean fibre radius, and standard deviation of radii. Alternatively, a list of radii may be provided.
- Optimize fibre positions in 2D slices by minimizing overlap, protrusion outside the domain, and (relevant for low FVF) fibre separation.
- By perturbing fibre positions in first slice of the 3D volume, generate fibre positions in the last slice. Various perturbations are possible (rotating bundles, swapping positions, or adding noise). Increasing the level of perturbations yields increasingly misaligned fibres.
- Optimize fibre positions in 3D by minimizing overlap, protrusion, stretching, and bending.
- Visualize slices, 3D configurations, and analysis results using Plotly.
- Export results and meshes for further processing.
- Generate projections and voxelizations.

## Main Classes and Functions

### [`FibrePacker`](fibre_packer.py)

The main class for creating and manipulating fibre packings.

#### Initialization

- `FibrePacker()`: Create a new packer instance.
- [`from_n`](fibre_packer.py): Create a packer with a given number of fibres.
- [`from_fvf`](fibre_packer.py): Create a packer with a target fibre volume fraction.

#### Slice and Configuration Methods

- `initialize_start_slice()`: Randomly initialize the start slice.
- `initialize_end_slice(misalignment, k=3)`: Create an end slice with specified misalignment.
- `interpolate_configuration(Z, z_multiplier=1, type='mixed')`: Interpolate between start and end slices to create a 3D configuration.

#### Optimization

- `optimize_slice(id, iters=200)`: Optimize a single slice.
- `optimize_slice_heuristic(id, iters, repetitions=10, ...)`: Heuristic optimization for a slice. Repeats optimization, possibly relaxing less important requirements, and stop when a good configuration is found.
- `optimize_configuration(iters=200)`: Optimize the full 3D configuration.
- `optimize_configuration_heuristic(iters, repetitions=10, ...)`: Heuristic optimization for the configuration.

#### Analysis

- `get_full_analysis()`: Compute overlap, protrusion, stretching, and bending metrics.
- `assess_analysis_summary(id=None)`: Assess the quality of a slice or configuration.

#### Visualization

- `show_radii_distribution(nbins=100)`: Plot histogram of fibre radii.
- `show_slice(id, title=None, show_issues=False)`: Visualize a 2D slice.
- `show_3D_configuration(title=None)`: Visualize the 3D configuration.
- `show_3D_configuration_analysis(analysis, title=None)`: Visualize analysis results in 3D.
- `animate_slices(title=None)`: Animate slices through the configuration.

#### Export

- `save_result(filename)`: Save configuration to a text file.
- `save_mesh(filename, close_ends=False, n=16)`: Export the configuration as a 3D mesh (OBJ format).

#### Utilities

- `set_radii(radii)`: Set fibre radii.
- `fix_radii(epsilon=1e-3)`: Adjust radii to remove overlaps and protrusions.
- `voxelize(pixels=None, new_z=None, transition=None)`: Generate a voxelized representation of the configuration.
- `project(thetas, bins, new_z=None)`: Compute projections (e.g., for tomography).

## Example Usage

```python
import fibre_packer as fp

# Create a packer with 70 domain radius, 50% FVF, mean radius 2, sigma 0.2
fib = fp.from_fvf(70, 50, 2, 0.2)
fib.initialize_start_slice()
fib.optimize_slice_heuristic('start', iters=100, repetitions=5)
fib.initialize_end_slice('high')
fib.optimize_slice_heuristic('end', iters=100, repetitions=5)
fib.interpolate_configuration(20, z_multiplier=25)
fib.optimize_configuration_heuristic(iters=100, repetitions=5)
fib.show_3D_configuration()
fib.save_mesh('output.obj')
```

## Dependencies

- PyTorch
- NumPy
- Plotly
- tqdm

## Author

Vedrana Andersen Dahl (vand@dtu.dk)

---

See [fibre_packer.py](fibre_packer.py) for full implementation details.
