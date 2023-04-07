import numpy as np
from scipy.stats import t


class LinearRegressor:
    def __init__(self) -> None:
        self._a = None
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = x
        self.y = y
        
        xtx = self.x.T.dot(self.x)
        self._a = np.linalg.inv( np.array([[xtx]]) if isinstance(xtx, float) else xtx).dot(self.x.T).dot(self.y)

    @property
    def a(self):
        return self._a    

    def predict(self):
        return self.x.dot(self._a)
    
    def get_variance(self):
        y_pr = self.predict()
        v = (self.y - y_pr).T.dot(self.y - y_pr) / (self.x.shape[0] - self.x.shape[1])
        return v

    def get_covariance(self):
        return self.get_variance() * np.linalg.inv(self.x.T.dot(self.x))

    def get_correlation(self):
        inv_xtx = np.linalg.inv(self.x.T.dot(self.x))
        
        corr = np.copy(inv_xtx)
        for i in range(len(corr)):
            for j in range(len(corr)):
                corr[i, j] /= np.sqrt(inv_xtx[i, i] * inv_xtx[j, j])

        return corr

    def get_standart_errors(self):
        errors = np.sqrt(np.diag(self.get_covariance()))
        return errors
       
    
    def get_determination(self, bias = True):
        sse = sum(pow(self.y - self.predict(), 2))
        sst = sum(pow(self.y - np.mean(self.y), 2))

        if bias:
            determination = 1 - sse / sst                   
        else:
            determination = 1 - (sse / (self.x.shape[0] - self.x.shape[1])) / (sst / (self.x.shape[0] - 1))

        return determination

    def get_sep_intervals(self, alpha=0.05):
        student = t(self.x.shape[0] - self.x.shape[1])
        st_val = student.ppf(1 - alpha / 2)

        sep_intervals = np.transpose(np.array([self.a - self.get_standart_errors() * st_val, self.a + self.get_standart_errors() * st_val]))
        return sep_intervals

    def get_tog_intervals(self, alpha=0.05):
        student = t(self.x.shape[0] - self.x.shape[1])
        st_val = student.ppf((2 - alpha / self.x.shape[1]) / 2)

        tog_intervals= np.transpose(np.array([self.a - self.get_standart_errors() * st_val, self.a + self.get_standart_errors() * st_val]))
        return tog_intervals