from svc import SupportVectorClustering as SVC
import sklearn.datasets
import numpy as np
from matplotlib import pyplot as plt

def calc_and_print_result(p, q, x):
    svc = SVC(p, q)
    svc.fit(x)
    clusters = svc.get_clusters()
    print(f'p = {p}, q = {q}')
    print(f'clusters: {len(clusters)}')

    mar = ['x' if i in svc.svs_indx else 'P' if i in svc.bsv_indx else '.' for i in range(len(x))]
    cmap = plt.get_cmap("jet")
    color=  cmap(np.linspace(0, 1, len(clusters)))
    plt.figure() 
    for i in range(len(clusters)):  
        plt.scatter(x[:, 0][clusters[i]], x[:, 1][clusters[i]], color=color[i], marker=mar[i])
  

if __name__ == '__main__':

    x,y = sklearn.datasets.make_blobs(n_samples=50, random_state=333, centers=3)

    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], color=['r' if i else 'b' for i in y ])

    p = SVC.get_optimal_p(x)
    q = SVC.get_optimal_q(x)
    svc = SVC(0.02, 0.1)
    svc.fit(x)
    clusters = svc.get_clusters()
    print(clusters)    


    
    mar = ['x' if i in svc.svs_indx else 'P' if i in svc.bsv_indx else '.' for i in range(len(x))]

    cmap = plt.get_cmap("jet")
    color=  cmap(np.linspace(0, 1, len(clusters))) 

    plt.figure()
    for i in range(len(clusters)):  
        plt.scatter(x[:, 0][clusters[i]], x[:, 1][clusters[i]], color=color[i], marker=mar[i])
    plt.show()
