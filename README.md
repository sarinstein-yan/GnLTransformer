# `GnLTransformer`: Classifying the Graph Topology of Non-Hermitian Energy Spectrum with Graph Transformer

<!-- [![PyPI](https://img.shields.io/pypi/v/poly2graph)](https://pypi.org/project/poly2graph/) -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2412.00568---?logo=arXiv&labelColor=b31b1b&color=grey)](https://arxiv.org/abs/.) -->

Topological physics is one of the most dynamic and rapidly advancing fields in modern physics. Conventionally, topological classification focuses on eigenstate windings, a concept central to Hermitian topological lattices (e.g., topological insulators). Beyond such notion of topology, we unravel a distinct and diverse graph topology emerging in non-Hermitian systems' energy spectra, featuring a kaleidoscope of exotic shapes like stars, kites, insects, and braids. The spectral graph solely depends on the algebraic form of characteristic polynomial.

<p align="center">
    <img src="https://raw.githubusercontent.com/sarinstein-yan/GnLTransformer/main/assets/SGs_demo.png" width="800" />
</p>

`GnLTransformer` is an explainable graph neural network that integrates multi-head attention mechanism and leverages line graphs as dual channels to explicitly capture higher-order relationships beyond node-node interactions, e.g. the interactions between edges, triplets.
- The dual channels of different orders of line graphs can be extended to learn representation of any high-order topology.
- The high-order topology is explainable via the attention weights from dual channels.
- In this work, the mutual information test shows that the high-order topological component (triplet) is more informative than the elementary topological components (node and edge) in classifying the spectral graph.

<p align="center">
    <img src="https://raw.githubusercontent.com/sarinstein-yan/GnLTransformer/main/assets/GnL_sketch.png" width="800" />
</p>

## Content of the Repository
This repository provides the code for the companion paper — *Classifying the Graph Topology of Non-Hermitian Energy Spectrum with Graph Transformer*. Specifically, it includes:

1. Dataset generation, sampled from a hypercube in a one-band characteristic polynomial space.

2. Featurization of the spatial multigraph and its line graph.
  <!-- - **The spectral graph ($\mathcal{G}$)**
    - Node features: 
      - **`pos`, 2D**: position of node, (Re(E), Im(E)), Coordinates of the vertices in the complex energy plane 
    - Edge features:
      - **`weight`, 1D**: The **length** of the edge line segment in the complex energy plane. Also serves as the edge **weight**.
      - **`pts_5`, 10D**: five equidistant points along the edge line segment that divide the edge into six equal parts. This feature is the concatenation of the five points' coordinates.
  - **The line graph of the spectral graph ($\mathcal{L}=L(\mathcal{G})$)**
    - Node features:
      - *Shared with the spectral graph's edge features, 11D*
    - Edge features:
      - **`triplet_center`, 2D**: The position of the center of the triplet formed by the two edges in the line graph. Specifically, this is the average of the three edge endpoints' coordinates in the complex energy plane.
      - **`angle`, 5D**: The angle between the two edges in the line graph, measured in the complex energy plane. -->
<table>
  <thead>
    <tr>
      <th style="text-align:center;">Graph</th>
      <th style="text-align:center;">Component</th>
      <th style="text-align:center;">Feature Name</th>
      <th style="text-align:center;">Shape</th>
      <th style="text-align:center;">Description</th>
    </tr>
  </thead>
  <tbody>
    <!-- Spectral graph group -->
    <tr>
      <td rowspan="3" style="text-align:center; vertical-align:middle;">Spectral graph (𝒢)</td>
      <td style="text-align:center;">Node</td>
      <td><code>pos</code></td>
      <td>2D</td>
      <td>Position of the node in the complex energy plane: (Re(E), Im(E)).</td>
    </tr>
    <tr>
      <!-- Graph cell spanned -->
      <td rowspan="2" style="text-align:center; vertical-align:middle;">Edge</td>
      <td><code>weight</code></td>
      <td>1D</td>
      <td>Length of the edge segment in the complex energy plane; also serves as the edge weight.</td>
    </tr>
    <tr>
      <!-- Graph and Entity cells spanned -->
      <td><code>pts_5</code></td>
      <td>10D</td>
      <td>Coordinates of five equidistant points along the edge, dividing it into six equal parts (5×2D).</td>
    </tr>
    <!-- Line graph group -->
    <tr>
      <td rowspan="3" style="text-align:center; vertical-align:middle;">Line graph (ℒ = L(𝒢))</td>
      <td style="text-align:center;">Node</td>
      <td><em>*inherits*</em></td>
      <td>11D</td>
      <td>Same as the spectral graph’s edge features (<code>weight</code> + <code>pts_5</code>).</td>
    </tr>
    <tr>
      <!-- Graph cell spanned -->
      <td rowspan="2" style="text-align:center; vertical-align:middle;">Edge</td>
      <td><code>triplet_center</code></td>
      <td>2D</td>
      <td>Center position of the triplet formed by two adjacent edges: average of the three endpoints’ coordinates.</td>
    </tr>
    <tr>
      <!-- Graph and Entity cells spanned -->
      <td><code>angle</code></td>
      <td>5D</td>
      <td>Angles between the two edges in the complex energy plane (one per segment junction, total 5).</td>
    </tr>
  </tbody>
</table>

3. Processing the `NetworkX MultiGraph` dataset to a [`PyTorch Geometric`](https://pyg.org/) Dataset.

4. The `AttentiveGnLConv` layer and `GnLTransformer` model.

5. Training, evaluation, ablation, and explainability visualization.

## Installation
The repository requires `python>=3.11` and can be installed locally.
```bash
$ conda create -n gnl python=3.12 # python>=3.11
$ conda activate gnl

$ git clone https://github.com/sarinstein-yan/GnLTransformer.git
$ cd GnLTransformer
$ pip install .
```
Or if you want to install with a specific CUDA version of PyTorch, you can set the `CUDA` environment variable to the desired version (e.g., `cu124` for CUDA 12.4):
```bash
$ export CUDA=cu124 # < On Linux or macOS
# On Windows (PowerShell): `$CUDA = "cu124"`
$ pip install . --extra-index-url https://download.pytorch.org/whl/${CUDA}
```
To install development version for faster dataset generation and visualization, you can use:
```bash
$ pip install -e .[dev]
```

## Dataset Generation

<!-- ## TODO
- [ ] Tutorials
  - [ ] dataset generation
  - [ ] `GnLTransformer`
  - [ ] explainability visualizations -->

<!-- ## Citation
If you find this work useful, please cite our paper:

```bibtex
@article{}
``` -->