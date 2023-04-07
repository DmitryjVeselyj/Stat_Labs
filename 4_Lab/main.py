import numpy as np
from matplotlib import pyplot as plt
from regressor import LinearRegressor
from scipy.stats import t

def load_files(n_var = 3):
    filename = '4_Lab/Y_30.txt'
    data = open(filename).readlines()[4:19]

    Y = []
    for row in data:
        Y.append(float(row.split()[n_var - 1]))

    y = np.array(Y)
    x = np.loadtxt('4_Lab/X.txt')

    return x,y


x, y = load_files()

regressor = LinearRegressor()
regressor.fit(x, y)
y_pred = regressor.predict()

# 1
print(regressor.get_variance())
print(regressor.get_covariance())
print(regressor.get_correlation())
print(regressor.get_standart_errors())
plt.figure()
plt.hist(y - y_pred, bins = 4)
plt.show()


# 2
print(regressor.get_determination())
print(regressor.get_determination(bias = False))
plt.figure()
plt.plot(range(len(y)), y, label="true")
plt.plot(range(len(y)), y_pred, label="pred")
plt.show()

# 3
print(regressor.get_separate_intervals())
print(regressor.get_together_intervals())
print(regressor.a)

alpha = 0.05
t_dist = t(x.shape[0] - x.shape[1])
t_half_interval = t_dist.ppf((2 - alpha) / 2)
for coef_idx, coef in enumerate(regressor.a):
    if abs(coef) / (np.sqrt(regressor.get_variance() * regressor.get_covariance()[coef_idx, coef_idx])) < t_half_interval:
        print("coefficient %i == 0" % (coef_idx))
# 4

coef_idx = 2
x_part = np.concatenate([x[:, :coef_idx], x[:, coef_idx + 1:]],
axis=1)
regressor.fit(x_part, y)
y_pred = regressor.predict()
plt.figure()
plt.plot(range(len(y)), y, label="true")
plt.plot(range(len(y)), y_pred, label="pred")
plt.show()


x_t, y_t = x[0, :], y[0]
x_res, y_res = x[1:, :], y[1:]
regressor.fit(x_res, y_res)
y_t_pred = x_t.T.dot(regressor.a)
variance_t = regressor.get_variance() * (x_t.T.dot(np.linalg.inv(x_res.T.dot(x_res))).dot(x_t) + 1)
print(f'y real: {y_t}\ny predicted: {y_t_pred}')
print(f'variance: {variance_t}')

alpha = 0.05
student = t(x.shape[0] - x.shape[1])
val = student.ppf(1 - alpha / 2)
interval = np.array([y_t_pred - val, y_t_pred + val])
print(f'interval: {interval}')