from URF import random_forest_cluster, plot_cluster_result
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
print(len(list(set(y))))

clf, prox_mat, cluster_ids = random_forest_cluster(X, k=3, max_depth=20, random_state=0)
plot_cluster_result(prox_mat, cluster_ids, 2, y)
