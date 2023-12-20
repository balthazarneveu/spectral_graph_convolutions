# Spectral graph convolutions
Student review on [Spectral Graph Convolutions for Population-based Disease Prediction](https://arxiv.org/abs/1703.03020)

We only deal with the ABIDE dataset dedicated to the study of ASD (*autism spectrum disorder*).


# :scroll: [Report](/report/87_Ines_VATI_Manal_AKHANNOUSS_Balthazar_NEVEU.pdf)

## Authors
- Program: [MVA Master's degree](https://www.master-mva.com/) class on [Geometric Data Analysis](https://www.jeanfeydy.com/Teaching/index.html). ENS Paris-Saclay.
- Authors
    - [Manal Akhannous](https://github.com/ManalAkh)
    - [Ines Vati](https://github.com/InesVATI)
    - [Balthazar Neveu](https://github.com/balthazarneveu)


![Method overview](report/figures/spectral_graph_convolution_graph_overview.png)

> Overview of the use of **graph convolutional networks** to predict ASD (autism spectrum disorder).
The ABIDE dataset was created to study autism and contains a set of functional MRI from 871 patients 
of different genders and captured over 17 different sites with different f-MRI devices. Input data is scarce and not totally homogeneous.
On the left, the creation of the content of a single node is shown. 111 temporal series are extracted from the f-MRI and correlation allows creating a connectivity 111×111 symmetric matrix of the brain of each patient.
A population graph is created by connecting each patient (node) to the other patients, with an edge weighted by the similarity between the patients.
It is processed by a graph convolutional network to predict the ASD status of each patient (a probability of being healthy or affected by ASD).
Nodes are partially labelled to split the dataset between training (80%), validation (10%) and test set (10%).

-----------

### Content


-----------



-----------
### Getting started

```bash
git clone git@github.com:balthazarneveu/sprectral_graph_convolutions.git
cd sprectral_graph_convolutions
pip install -e .
python ABIDE_dataset/download_preprocess.py
```

- Clone repository
- Download ABIDE dataset to the default location `__ABIDE_dataset` (~300Mb after preprocessing)

#### Training
```bash
python scripts/train_script.py -n 1000 -f rfe -m Dense Cheb-dr=0.1 -d cuda
```

Train several models:
- a baseline Dense fully connected network not taking the graph into consideration
- ChebConv (Chebconv requires something around 9Gb of GPU memory.)

![performances comparisons](report/figures/model_performances_architecture.png)
Cross validation is performed on 10 runs using 10 fixed seeds to split the dataset into training (80%), validation (10%) and test set (10%).


| ABIDE dataset - Time series | ABIDE dataset - Connectivity matrices |
|:-----: |:-----:|
| ![time_series](/ABIDE_dataset/figures/separate_time_series.png) | ![connectivity_matrix](/ABIDE_dataset/figures/connectivity_matrix.png) |
| Time series look impossible to compare accross patients| For each patient, connectivity matrix looks much more structured and informative|

Brain connectivity (Using the HO Altas)
![connectivity_brain](report/figures/ex_connectivity_pitt_ASD.png)



#### How is the graph built?
![](report/figures/spectral_graph_convolution_graph_construction.png)

:bulb: Notebook to get [intuition](studies/graph_convolution_intuition.ipynb)  on what the "graph convolution" aspect is doing:
Since edges have been built using feature similarity, convolving the graph is equivalent to denoise features the way
Non Local means would do on an image grid

------
#### :gift: Extra content

##### Dataset utilities
- [ABIDE dataset](/ABIDE_dataset/) 
  - [Utilities to download the dataset](/ABIDE_dataset/download_preprocess.py) *(by Ines Vati)* 
  - [Visualization tools](/ABIDE_dataset/visualize_data.ipynb)  *(by Balthazar Neveu)*


##### Notes on theory and papers 
- [Graph spectral theory](/notes-on-graph-spectral-theory) note *(by Ines Vati)*
- [Papers analyzis](/notes/) Reading notes on cited papers.


##### Studies
- [Normalized adjacency matrix](/studies/normalized_adjacency.py) : getting familiar with the $D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$ matrix  *(by  Balthazar Neveu)* based on [NetworkX](https://networkx.org/).

