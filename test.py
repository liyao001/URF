from URF.URF import random_forest_cluster, plot_cluster_result
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

clf, prox_mat, cluster_ids = random_forest_cluster(X, k=3, max_depth=20, random_state=0)

assert len(cluster_ids) > 0
