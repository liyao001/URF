URF: Unsupervised Random Forest
==============================================

URF (Unsupervised Random Forest, or Random Forest Clustering) is a python implementation of the paper: Shi, T., & Horvath, S. (2006). Unsupervised learning with random forest predictors. *Journal of Computational and Graphical Statistics*, 15(1), 118-138.

Prerequisite
------------
::

    conda install -c bioconda pycluster

or::

  wget http://bonsai.hgc.jp/~mdehoon/software/cluster/Pycluster-1.54.tar.gz
  tar -zxvf Pycluster-1.54.tar.gz
  cd Pycluster-1.54
  python setup.py install

Installation
------------
::

  pip install URF

Usage
-----
::

  from sklearn.datasets import load_iris
  from URF.main import random_forest_cluster, plot_cluster_result
  iris = load_iris()
  X = iris.data
  y = iris.target
  print(len(list(set(y))))

  clf, prox_mat, cluster_ids = random_forest_cluster(X, k=3, max_depth=20, random_state=0)
  plot_cluster_result(prox_mat, cluster_ids, 2, y)

If you encountered an error like

  > QXcbConnection: Could not connect to display

then you need to add these codes to the very beginning of your file::

  import matplotlib as mpl
  mpl.use("Agg")

and you must assign the output file when you call `plot_cluster_result`, like this::

  plot_cluster_result(prox_mat, cluster_ids, 2, y, output="test_123.png")
