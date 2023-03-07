import numpy as np
from sklearn.decomposition import PCA

    
def calc_variance_ratio(x_new, x_old):
    return np.diagonal(np.cov(x_new.T)) / sum(np.diagonal(np.cov(x_old.T)))


def get_optimal_n_componets(x_centered, values):
    border = 1 / x_centered.shape[1] * np.trace(np.cov(x_centered.T))
    return len(values[values > border])


def PCA_method(x, n_components = None):
    x_centered = (x - np.mean(x, axis = 0))
    covariance = np.cov(x_centered.T)
    values, vectors = np.linalg.eig(covariance)
    eiges = sorted(zip(values, vectors.T), key=lambda x: x[0], reverse=True)

    if n_components is None:
        n_components = get_optimal_n_componets(x_centered, values)        
    transform = np.vstack(list([eiges[i][1] for i in range(n_components)]))
    x_new = x_centered @ transform.T

    # # values, vectors = np.linalg.eig(1/ len(x) * x.T @ x)
    # # eiges = sorted(zip(values, vectors.T), key=lambda x: x[0], reverse=True)
    
    # pca = PCA(n_components=2)
    # x_new = pca.fit_transform(x)
    # print(pca.explained_variance_ratio_)
    return x_new        