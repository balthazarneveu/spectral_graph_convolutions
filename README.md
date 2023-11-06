# Sprectral graph convolutions
Student project on [Spectral Graph Convolutions for Population-based Disease Prediction](https://arxiv.org/abs/1703.03020)

## Authors
- Program: [MVA Master's degree](https://www.master-mva.com/) class on [Geometric Data Analysis](https://www.jeanfeydy.com/Teaching/index.html). ENS Paris-Saclay.
- Authors
    - [Manal Akhannous](https://github.com/ManalAkh)
    - [Ines Vati](https://github.com/InesVATI)
    - [Balthazar Neveu](https://github.com/balthazarneveu)




-----------

### Content


-----------

#### Dataset utilities
- [ABIDE dataset](/ABIDE_dataset/) 
  - [Utilities to download the dataset](/ABIDE_dataset/download_preprocess.py) *(by Ines Vati)* 
  - [Visualization tools](/ABIDE_dataset/visualize_data.ipynb)  *(by Balthazar Neveu)*


#### Notes on theory and papers 
- [Graph spectral theory](/notes-on-graph-spectral-theory) note *(by Ines Vati)*
- [Papers analyzis](/notes/) Reading notes on cited papers.


### Studies
- [Normalized adjacency matrix](/studies/normalized_adjacency.py) : getting familiar with the $D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$ matrix  *(by  Balthazar Neveu)* based on [NetworkX](https://networkx.org/).



-----------
### Getting started

```bash
git clone git@github.com:balthazarneveu/sprectral_graph_convolutions.git
python3 ABIDE_dataset/download_preprocess.py
```
- Clone repository
- Download ABIDE dataset to the default location `__ABIDE_dataset` (~300Mb after preprocessing)

| ABIDE dataset - Time series | ABIDE dataset - Connectivity matrices |
|:-----: |:-----:|
| ![time_series](/ABIDE_dataset/figures/separate_time_series.png) | ![connectivity_matrix](/ABIDE_dataset/figures/connectivity_matrix.png) |
| Time series look impossible to compare accross patients| For each patient, connectivity matrix looks much more structured and informative|