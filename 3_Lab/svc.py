import numpy as np
import cvxpy as cvx
import time
from functools import wraps
from scipy.optimize import minimize


def timeit(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        # print(f'Function: {func.__name__}\n Time: {end - start:.4f}')
        return result
    
    return wrapper



class SupportVectorClustering():
    def __init__(self, p, q, C = None,  kernel = None) -> None:
        self.p = p
        self.q = q
        self.C = C
        self.kernel = self._gaussian_kernel if kernel is None else kernel

        self._N = None
        self._beta = None
        self._x = None
        self._km = None

        self.svs_indx = 0
        self.bsv_indx = 0
        self.ovs_indx = 0
     

    def _gaussian_kernel(self, x1, x2):
        return np.exp(-self.q * np.linalg.norm(x1 - x2)**2)


    def _calc_kernel_matrix(self):
        km=np.zeros((self._N, self._N))
        
        for i in range(self._N):
            for j in range(self._N):
               km[i,j] = self.kernel(self._x[i], self._x[j])
        return km 
          
    def fit(self, x):
        self._x = x
        self._N = len(x)
        self.C = 1 / (self.p * self._N) if self.C is None else self.C 
        
        self._km = cvx.psd_wrap(self._calc_kernel_matrix())
        beta = cvx.Variable(self._N)
        
        objective = cvx.Maximize(cvx.sum(cvx.diag(self._km) @ beta)- cvx.quad_form(beta, self._km)) # 1 - cvx.quad_form(beta, km)
        constraints = [0 <= beta, beta<=self.C, cvx.sum(beta)==1]                                   # 1 - self._beta.T @self._km @ self._beta
        result = cvx.Problem(objective, constraints).solve()
    
        self._beta = beta.value


    def _line_segment_adj(self,x1,x2,R,n=10):
        res  = all([self.r_func(x1 + (x2 - x1)/(n +1) * i) < R for i in range(n)])
        return res
    
    @timeit
    def _get_adj_mtx(self, EPS = 10**-8):

        self.bsv_indx = np.where(self._beta >= (self.C - EPS))[0]
        self.svs_indx = np.where((self._beta < (self.C - EPS)) & (self._beta > EPS))[0]
        self.ovs_indx = np.where(self._beta < EPS)[0]

        print(f'svs: {len(self.svs_indx)}  bsv: {len(self.bsv_indx)}')
        R = np.mean([self.r_func(self._x[i]) for i in self.svs_indx])

        adj_mtx = np.zeros((self._N, self._N))

        for i in range(self._N):
            if i not in self.bsv_indx:
                for j in range(i, self._N):
                    if j not in self.bsv_indx:
                        adj_mtx[i,j]=adj_mtx[j,i]=self._line_segment_adj(self._x[i],self._x[j], R)

        return adj_mtx                


    def get_clusters(self):
        adj_mtx = self._get_adj_mtx()
        indices = list(range(self._N))
        clusters = {}
        num_clusters = -1
     
        while indices:
            num_clusters+=1
            clusters[num_clusters]=[]
            curr_id = indices.pop(0)
            queue = [curr_id]

            while queue:
                cid = queue.pop(0)
                for i in indices:
                    if adj_mtx[i,cid]:
                        queue.append(i)
                        indices.remove(i)
                clusters[num_clusters].append(cid)
              
        return clusters
    

    def r_func(self, x):
        return self.kernel(x,x)-2*np.sum([self._beta[i]*self.kernel(self._x[i], x) for i in range(self._N)]) + (self._beta.T @self._km @ self._beta).value
    

    @staticmethod
    def get_optimal_p(x):
        return 1 / (len(x))
    
    @staticmethod
    def get_optimal_q(x):
        return 1 / max([np.linalg.norm(x[i] - x[j])**2 for i in range(len(x)) for j in range(len(x))])
   
    


 # def another_fit(self, x):
    #     self._x = x
    #     self._N = len(x)
    #     self.C = 1 / (self.p * self._N) if self.C is None else self.C 

    #     self._km = self._calc_kernel_matrix()
    #     def func(x):
    #         return np.sum(x) - x.T @ self._km @ x

    #     con1 = {'type': 'eq', 'fun': lambda b: sum(b) - 1}
    #     con2 = [{'type': 'ineq', 'fun': lambda b: -1 * (b[i] - self.C)} for i in range(self._N)]
    #     con3 = [{'type': 'ineq', 'fun': lambda b: b[i] - 0} for i in range(self._N)]
    #     result = minimize(lambda x: -1 * func(x), np.ones(self._N), constraints=[con1, *con2, *con3])
    #     self._beta = result.x


 # r_func if use another_fit
        # return self.kernel(x,x)-2*np.sum([self._beta[i]*self.kernel(self._x[i], x) for i in range(self._N)]) + self._beta.T @self._km @ self._beta