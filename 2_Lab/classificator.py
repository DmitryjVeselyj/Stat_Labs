import numpy as np
from scipy.stats import norm



class Classificator:
    def __init__(self) -> None:
        self._alpha = 0
        self._c = 0
        
        self.mahalanobis_d =  self.mahalanobis_d_unbias = -1
        self.prob_2_1 = self.prob_1_2 = -1
        

    def fit(self, x_train, y_train, calc_info=True):
        x1, x2 = x_train[np.where(y_train == 0)], x_train[np.where(y_train == 1)]
        n1, n2 = len(x1), len(x2)
        mu1, mu2 = np.mean(x1, axis=0), np.mean(x2, axis=0)
        cov1, cov2 = np.cov(x1.T), np.cov(x2.T)

        cov = 1 / (n1 + n2 - 2) * ((n1 - 1) * cov1 + (n2 - 1) *cov2)
        cov = np.array([[cov]]) if isinstance(cov, float) else cov
        
        self._alpha = np.linalg.inv(cov).dot(mu1 - mu2)
        z1, z2 = np.dot(self._alpha, mu1), np.dot(self._alpha, mu2)
        self._c = (z1 + z2) / 2 + np.log(n2/ n1)


        if calc_info:    
            z_var = self._alpha.T @ cov @ self._alpha
            p = x_train.shape[1]
            self._calc_mahalanobis(z1,z2,z_var ,n1, n2, p)
            self._calc_error(n1, n2)
        
    def _calc_mahalanobis(self, z1, z2, z_var, n1, n2, p):
        self.mahalanobis_d = (z1 - z2)**2 / z_var
        self.mahalanobis_d_unbias = (n1 + n2 - p - 3) / (n1 + n2 - 2) * self.mahalanobis_d  - p * (1 / n1 + 1 / n2)
       

    def _calc_error(self, n1, n2):
        K = np.log(n2 / n1)
        F = norm.cdf
        self.prob_2_1 = F((K - 1/2 * self.mahalanobis_d_unbias) / np.sqrt(self.mahalanobis_d_unbias))
        self.prob_1_2 = F((-K - 1/2 * self.mahalanobis_d_unbias) / np.sqrt(self.mahalanobis_d_unbias))


    def predict(self, x):
        return x.dot(self._alpha) < self._c


