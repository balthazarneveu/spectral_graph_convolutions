# Papers analyzis

## [Kipf2017](/external/papers/Kipf2017_8.pdf) SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS


[Clear and illustrated explanation](https://tkipf.github.io/graph-convolutional-networks/)

#### Semi supervision
Semi supervision comes from the fact that labels may be missing but the underlying graph structure is here.
> labels are only available for a small subset of nodes. label information is smoothed over the graph via
some form of explicit graph-based regularization

#### Convolutions versus Spectral convolutions
- localized first-order approximation of spectral graph convolutions -> simple convolutions without the spectral global aspect?




#### "Section 2. Proposed model"

$$H^{(l+1)} = \sigma \big( \mathit{\tilde{D}}^{-\frac{1}{2}} \mathit{\tilde{A}}   \mathit{\tilde{D}}^{-\frac{1}{2}} H^{(l)}W^{(l)}\big)$$

- Projecting features
  - $H^{(l)} \in \R^{N \text{x} D^{(l)}}$ features of the N nodes of dimension $D^{(l)}$ .
  - $H^{(l+1)} \in \R^{N \text{x} D^{(l+1)}}$ features of the N nodes of dimension $D^{(l+1)}$.
  - $W^{(l)} \in \R^{D^{(l)} \text{x} D^{(l+1)}}$ weight matrix.  
  - `pre_sup = dot(x, self.vars['weights_' + str(i)])`
  - $\sigma \big( W^{(l)}H^{(l)}\big) \in \R^{N \text{x} D^{(l+1)}}$ is a classical fully connected layer or `conv1x1` - we apply the same operation to all the samples, no dependance on the graph. 
- Graph convolutions
  - $\mathit{\tilde{D}}^{-\frac{1}{2}} \mathit{\tilde{A}}   \mathit{\tilde{D}}^{-\frac{1}{2}} \in \R^{N^2}$ This is the "convolutional" part which acts as an average over the neighborhood. 
  - By average, this specifically looks like a box filter.  :question: :question: to be verified 
- Multiple layers $l\in[1, 2]$:
  - In this paper, the GCN is shallow: they simply chained [2 layers](https://github.com/parisots/gcn/blob/master/gcn/models.py#L163C1-L178) only.


Note: 
- $\tilde{D}^{-1} \mathit{\tilde{A}}$ is a simple neighborhood averaging.
- $\mathit{\tilde{D}}^{-\frac{1}{2}} \mathit{\tilde{A}}   \mathit{\tilde{D}}^{-\frac{1}{2}}$ is more than a "mere averageing". 

![explanation](/notes/figures/GCN_explanations.png)


:warning: For a circulant matrix (define the adjacency matrix of a regular grid)... $\mathit{\tilde{D}}^{-\frac{1}{2}} \mathit{\tilde{A}}   \mathit{\tilde{D}}^{-\frac{1}{2}}$ is a box filter / simple neighborhood averageing. But if the graph has a non regular structure, it may be more complex.
- $\tilde{D}^{-1} \mathit{\tilde{A}}$ : divide the rows by the degree (=sum of the row)
- $\mathit{\tilde{A}} \tilde{D}^{-1}$ : divide the columns by the (sum of the row...). Counter intuitive.

[Preprocess adjacent matrix](https://github.com/parisots/gcn/blob/master/gcn/utils.py#L104-L117)


```python
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)
```


[Graph convolution application](https://github.com/parisots/gcn/blob/master/gcn/layers.py#L213-L222)
```python
supports = list()
for i in range(len(self.support)):
    # Apply conv1x1 to features 
    # (each node receives the same linear projection)
    # -> H*W
    pre_sup = dot(x, self.vars['weights_' + str(i)])
    
    # Apply the actual graph convolution 
    # Smooth over neighbors 
    # -> [D^(-1/2) * A * D^(-1/2)] * H*W
    support = dot(self.support[i], pre_sup) 
    
    supports.append(support)
output = self.act(tf.add_n(supports) + self.vars['bias']) # Relu( h.W+ bias)
```

:question: :question: :question: What is the length of the support? (what's the iteration for?)





----------

#### Modifying the graph regularity pargadim
Formulating labeling as an optimization problem with a structure regularization.
$$\mathcal{L} = \mathcal{L_0}+ \lambda \mathcal{L_{reg}}  $$

$$\mathcal{L_{reg}} = \sum_{\text{nodes} = (i,j)}A_{i,j} ||f(X_i) - f(X_j)||^2 $$



$$\mathcal{L_{reg}} = \sum_{\text{nodes} = i} \sum_{j} A_{i,j} ||f(X_i) - f(X_j)||^2 = \sum_{\text{nodes} = i} \sum_{j}  A_{i,j} \big[ f(X_i)^Tf(X_i) - 2*f(X_i)^T.f(X_j) + f(X_j)^Tf(X_j) \big]$$

$$=\sum_{\text{nodes} = i} 2*\big[ \underbrace{\large[\sum_{j}A_{i,j}\large]}_{=D_{i,i}}  f(X_i)^Tf(X_i) \big] - 2* \sum_{j} f(X_i)^T.A_{i,j}.f(X_j) $$

$$= f(X)^T \Delta f(X)$$

Quadratic form not so well justified in the calculation above but that's the idea... The $D$ terms appear 

$\Delta = D - A$ is the un-normalized laplacian matrix of the graph. 
- where $D_{i,i}= \sum_j A_{ij} = d_i$ is the diagonal degree matrix (degree of the node=number of neighbors)
- all its columns and lines sum to 0

If two nodes $i \neq j$ are connected, you want their transformed feature vectors $f(X_i)$ and $f(X_j)$ to be close to each other otherwise you'll get a high penalty. This may not be a good idea as you are forcing the graph edges to represent the similarity between nodes.

> The formulation relies on the assumption that connected nodes in the graph are likely to share the same label.

Change the paradigm so the neural network $f(X,A)$ becomes conditioned on the adjacency matrix A.

:question:  :question: :question: Still unclear why their implementation is very different. They end up smoothing features in a local neighborhood
and it seems pretty related to the minimization of the regularization term by the way.


-----------

#### Dataset evaluations
Example `Citeseer` : 
- available citations (edges) - papers on themodynamics won't cite machine learning papers.
-  Features dimensions 3.7k (bag of words)
- Sparse labeling  3.6% of themes,  to create a graph paper

Edges will act as a regularization term. Actually gradient will come from labeled data and communicate to unlabeled data.
