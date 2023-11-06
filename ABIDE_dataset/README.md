# Application to ABIDE Dataset

ABIDE  database [[2]](#2) is one of the dataset used by the author of the studied article. It is freely provided py the [Preprocessed Connectome Project](http://preprocessed-connectomes-project.org/abide/) initiative and can be downloaded and prepocessed using the Python package [*nilearn*](https://nilearn.github.io/dev/introduction.html#what-is-nilearn) which allows users to apply machine learning techniques to analyse data acquired with MRI machines. 


## What are the data ?
Even if preprocessed fMRI images are available, those files are very large, i.e. ~100 Mo each. However, the extracted time series files are ~200Ko in size for each subject. We downloaded *rois-ho.1D* files which are time series extracted from [Harvard-Oxford (HO) atlas](https://nilearn.github.io/dev/auto_examples/01_plotting/plot_atlas.html#visualizing-the-harvard-oxford-atlas). 


![](/ABIDE_dataset/figures/HO_atlas.png)

In medical imaging, time series are a collection of data points that are recorded or measured over a period of time. In the case of fMRI, time series represent the intensity of brain activity within various regions at different time points as the MRI scanner captures images over time.
> The HO atlas is a set of anatomical brain regions or regions of interest (ROIs) that have been defined and labeled based on the human brain's anatomical structures.

Here, the time series data is derived from the specific regions defined in the HO atlas. In our code, they are numpy array of size (timepoints x regions), here it is (`196 x 111`).


Like the authors [[1]](#1), we use *nilearn* to preprocess the data by specifying necessary parameters :
- *pipeline* : Configurable Pipeline for the Analysis of Connectomes (C-PAC) <sup id="s1">[1](#f1)</sup>. 
- *strategy* : filt_noglobal which means band-pass filtering only without global signal regression

## How to download those data ?

The following tuto enables you to download preprocessed extracted time series and the connectivity matrix associated with each patient. Note that, in this context, the connectivity matrix is a matrix of a connectivity measure (e. g. Pearson's correlation) between two time series of the brain activity of a subject. Its size is, thereby, (number of regions x number of regions).

1. activate the virtual environment .venv
2. copy paste the following command in the terminal :
> :warning: USE ABSOLUTE PATHS (**not relative paths**) :warning:
```
python download_preprocess.py -o "path/to/data"
```
If you want to load less data (e.g. `-n 400`) than the default value (871), you can type in :
```
python download_preprocess.py -o "path/to/data" -n 400
```

### Connectivity matrices

Connectivity matrix between the signals in the 111 areas in the HO atlas.

- [Pearson's correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) $\frac{cov(X,Y)}{\sigma_X \sigma_Y} \in [-1, 1]$  *allows to normalize*
- `atanh` for [Fischer's transform](https://en.wikipedia.org/wiki/Fisher_transformation).

Sort of shows how two parts of the brain are related or not to each other.

![](/ABIDE_dataset/figures/correlation_matrix.png)

## How are the feature vectors computed ?
*Recursive feature elimination* (RFE) with scikit learn: 
- Given an external estimator (here, a Ridge Classifier) that assigns weights to features, the goal of RFE is to select features by recursively considering smaller and smaller sets of features.
- First, the estimator is trained on the initial set of features and the importance of each feature is obtained through any specific attribute. Then, the least important features are pruned from current set of features. 
- That procedure is recursively repeated until the desired number of features to select is eventually reached.

**Fisher transformed Pearson's correlation coefficient**

## What new hypotheses are we going to test ?

- a new representation of feature vector ?
- another way to classify the subject ?


---
<a id="f1">1</a> A connectome is a functional-connectivity matrix between a set of brain regions of interest (ROIs). [↩](#s1)



[niLearn tutorial on connectome](https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_inverse_covariance_connectome.html)

## Reference 
<a id="1"> [1] </a> S. Parisot et al., ‘Spectral Graph Convolutions for Population-based Disease Prediction’. arXiv, Jun. 21, 2017. doi: 10.48550/arXiv.1703.03020.

<a id="data"> [2] </a> Cameron Craddock, Yassine Benhajali, Carlton Chu, Francois Chouinard, Alan Evans, András Jakab, Budhachandra Singh Khundrakpam, John David Lewis, Qingyang Li, Michael Milham, Chaogan Yan, Pierre Bellec (2013). The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives. In Neuroinformatics 2013, Stockholm, Sweden.