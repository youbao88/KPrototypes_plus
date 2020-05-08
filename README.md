# KPrototype plus (kpplus)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-informational.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.org/project/kpplus/)

## Description

K-prototype is a clustering method invented to support both categorical and numerical variables[1]

**KPrototype plus (kpplus)** is a Python 3 package that is designed to increase the performance of [nivoc's KPrototypes function](https://github.com/nicodv/kmodes) by using [Numba](http://numba.pydata.org/).

This code is part of [Stockholms diabetespreventiva program](https://www.folkhalsoguiden.se/amnesomraden1/analys-och-kartlaggning/sdpp/).

**Performance improvement**
As an [example](example/example.ipynb), I used one of the [Heart Disease Data Sets](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) from [UCI](https://archive.ics.uci.edu/ml/index.php) to test the performance.
This data set contains 4455 rows, 7 categorical variables, and 5 numerical variables.
We compare the performance between nicodv's kprototype function and k_prototype_plus.

~~~~
< nicodv's kprototype >
CPU times: user 2.14 s, sys: 18.2 ms, total: 2.16 s
Wall time: 1min 41s
~~~~
~~~~
< k_prototype_plus >
CPU times: user 298 ms, sys: 9.24 ms, total: 308 ms
Wall time: 13.4 s
~~~~

**Notice:** Only Cao initiation is supported as the initiation method[2].

## System requirement
[![Generic badge](https://img.shields.io/badge/Python-3.7.1-informational.svg)](https://www.python.org/) [![Generic badge](https://img.shields.io/badge/Pandas-0.25.3-informational.svg)](https://pandas.pydata.org/) [![Generic badge](https://img.shields.io/badge/Numpy-1.17.0-informational.svg)](https://numpy.org/) [![Generic badge](https://img.shields.io/badge/Joblib-0.13.2-informational.svg)](https://joblib.readthedocs.io/en/latest/) [![Generic badge](https://img.shields.io/badge/Numba-0.45.1-informational.svg)](http://numba.pydata.org/)

## Installiation

```
pip install kpplus
```

## Usage
```python
from kpplus import KPrototypes_plus
model = KPrototypes_plus(n_clusters = 3, n_init = 4, gamma = None, n_jobs = -1)  #initialize the model
model.fit_predict(X=df, categorical = [0,1])  #fit the data and categorical into the mdoel

model.labels_                          #return the cluster_labels
model.cluster_centroids_               #return the cluster centroid points(prototypes)
model.n_iter_                          #return the number of iterations
model.cost_                            #return the costs
```
**n_clusters:** the number of clusters

**n_init:** the number of parallel oprations by using different initializations

**gamma (optional):** A value that controls how algorithm favours categorical variables. (By default, it is the mean std of all numeric variables)

**n_jobs (optional, default=-1):** The number of parallel processors. ('-1' means using all the processor)

**X:** 2-D numpy array (dataset)

**types:** A numpy array that indicates if the variable is categorical or numerical.

For example: ```types = [1,1,0,0,0,0]``` means the first two variables are categorical and the last four variables are numerical.

## Acknowledgement
I'm extremely grateful to [Dr. Diego Yacaman Mendez](https://staff.ki.se/people/dieyac?_ga=2.70810192.1199119869.1588953123-1873461028.1579027503) and [Dr. David Ebbevi](https://www.linkedin.com/in/debbevi/?originalSubdomain=se) for their support. They are two brilliant researchers who started this project with excellent knowledge of medical science, epidemiology, statistics and programming.

## Reference
[1] Huang Z. Extensions to the k-Means Algorithm for Clustering Large Data Sets with Categorical Values. Data Mining and Knowledge Discovery. 1998;2(3):283-304.
[2] Cao F, Liang J, Bai LJESwA. A new initialization method for categorical data clustering. 2009;36(7):10223-8.
