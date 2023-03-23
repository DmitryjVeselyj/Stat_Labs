from svc import SupportVectorClustering as SVC
import sklearn.datasets
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from criteries import adj_rand, convert_to_clusters, silhouette, convert_to_array
from sklearn.metrics import adjusted_rand_score, silhouette_score



def calc_and_print_result(p, q, x, y):
    svc = SVC(p, q)
    svc.fit(x)
    clusters = svc.get_clusters()
    
    print(f'svs: {len(svc.svs_indx)}  bsv: {len(svc.bsv_indx)}')
    print(f'p = {p}, q = {q}')
    print(f'clusters: {len(clusters)}')
    print(f'adj_rand: {adj_rand(convert_to_clusters(y), clusters)}, silhouette: {silhouette(clusters, x, euclidean)}')
    # print(f'NOT MY adj_rand: {adjusted_rand_score(y, convert_to_array(clusters))}, sil:{silhouette_score(x, convert_to_array(clusters))}')

    mar = ['x' if i in svc.svs_indx else 'P' if i in svc.bsv_indx else '.' for i in range(len(x))]
    cmap = plt.get_cmap("brg")
    color=  cmap(np.linspace(0, 1, len(clusters)))
    col = np.zeros(len(x))
    for key,value in clusters.items():
        for i in value:
            col[i] = key

    plt.figure() 
    for i in range(len(x)):
        plt.scatter(x[i, 0], x[i, 1], color=color[int(col[i])], marker=mar[i])
    
  

if __name__ == '__main__':

    x,y = sklearn.datasets.make_blobs(n_samples=50, random_state=333, centers=3)
    
    svc = SVC(0.02, 0.1)
    svc.fit(x)
    clusters = svc.get_clusters()

    print(f'NOT MY adj_rand: {adjusted_rand_score(y, convert_to_array(clusters))}, sil:{silhouette_score(x, convert_to_array(clusters))}')
    print(f'adj_rand: {adj_rand(convert_to_clusters(y), clusters)}, silhouette: {silhouette(clusters, x, euclidean)}')

    p, q = SVC.get_begin_p(x), SVC.get_begin_q(x)
    calc_and_print_result(p, q, x, y)
    
