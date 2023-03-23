import numpy as np
from math import comb



def convert_to_array(clusters):
    N = sum(len(lst) for lst in clusters.values())
    col = np.zeros(N)
    for key,value in clusters.items():
        for i in value:
            col[i] = key

    return col

def convert_to_clusters(y):
    clusters = {}
    for i, value in enumerate(y):
        if value not in clusters:
            clusters[value] = [i]
        else:
            clusters[value].append(i)

    return clusters

    

def adj_rand(cl_true, cl_pred):
    index = 0
    n = 0
    for i in cl_true:
        for j in cl_pred:
            n_ij = len(set(cl_true[i]) & set(cl_pred[j]))
            n+=n_ij
            index += comb(n_ij,2)

    C_2_a = sum(comb(len(cl_true[key]), 2) for key in cl_true)
    C_2_b = sum(comb(len(cl_pred[key]), 2) for key in cl_pred)
    C_2_n = comb(n, 2)

    expectedIndex =  C_2_a * C_2_b / C_2_n
    maxIndex =(C_2_a +  C_2_b) / 2
    return (index - expectedIndex) / (maxIndex - expectedIndex)       



def silhouette(clus, x, metric):
    if len(clus) == 1 :
        return 0
        
    N = len(x)
    a = lambda i, C: np.mean([metric(x[i],x[j]) for j in C if i != j])
    b = lambda i, C: min([np.mean([metric(x[i],x[j]) for j in Cl]) for Cl in clus.values() if set(Cl) != set(C)])
    res = 0
    
    for Ck in clus.values():    
        for i in Ck:
            res += np.nan_to_num((b(i, Ck) - a(i, Ck)) / max(b(i, Ck), a(i, Ck)))

    return res / N

